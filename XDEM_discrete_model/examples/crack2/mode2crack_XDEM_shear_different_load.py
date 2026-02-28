import sys
import os
sys.path.append((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Set random seed
import random
import numpy as np
import torch

# Fix random seed
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from DENNs import PINN2D
import torch.nn as nn
from utils.NodesGenerater import genMeshNodes2D
from utils.NN import stack_net,AxisScalar2D
from utils.Integral import trapz1D
from Embedding import LineCrackEmbedding,extendAxisNet
import Embedding
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import utils.Geometry as Geometry
from SIF import DispExpolation_homo, SIF_K1K2, M_integral


class Plate(PINN2D):
    def __init__(self, model: nn.Module,tau,a,b,h,q=1):
        '''x范围0-1.y范围-1到1'''
        super().__init__(model)
        self.tau = tau
        
        # 初始化 K_I 参数，确保它参与优化
        # a11 = 0.0
        # a12 = 0.0  
        # a13 = 0.0  
        # a14 = 0.0  
        # a21 = 0.0  
        # a22 = 0.0  
        # a23 = 0.0  
        # a24 = 0.0  
        # self.K_I = nn.Parameter(torch.tensor([a11,a12,a13,a14,a21,a22,a23,a24], device=self.device))

        K2_esti = tau * np.sqrt(np.pi * a)  # 这里预测的位移是mm，所以要乘以1000
        self.K_II = nn.Parameter(torch.tensor(K2_esti, device=self.device))
        self.a = a
        self.b = b
        self.h = h
        self.q = q
    def set_Optimizer(self, lr, k_i_lr=None):
        """重写优化器设置方法，为K_I参数设置不同的学习率"""
        # 获取模型参数和 K_I 参数
        model_params = list(self.model.parameters())
        k_i_params = [self.K_II]  # 直接使用整个K_I参数张量
        
        # 如果没有指定K_I的学习率，使用默认的较大学习率
        if k_i_lr is None:
            k_i_lr = lr * 10  # K_I使用10倍的学习率
        
        # 创建参数组，为不同参数设置不同学习率
        param_groups = [
            {'params': model_params, 'lr': lr},
            {'params': k_i_params, 'lr': k_i_lr}
        ]
        
        self.optimizer = torch.optim.Adam(param_groups)

    def calculate_j_integral(self, contour_points, contour_normals, radius):
        """
        Calculate J-integral along a contour around the crack tip
        
        J = ∫_Γ [W δ₁ⱼ - σᵢⱼ ∂uᵢ/∂xⱼ] nⱼ dΓ
        
        Args:
            contour_points: Points along the integration contour (N, 2)
            contour_normals: Normal vectors at each point (N, 2)
        
        Returns:
            J: J-integral value
        """
        # Convert to tensors and set as variables
        contour_xy = torch.tensor(contour_points, dtype=torch.float32, requires_grad=True).to(self.device)
        normals = torch.tensor(contour_normals, dtype=torch.float32).to(self.device)
        
        # Get displacement and stress at contour points
        u, v = self.pred_uv(contour_xy)
        
        # Calculate strain energy density
        eXX, eYY, eXY = self.compute_Strain(u, v, contour_xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY)
        W = 0.5 * (eXX * sx + eYY * sy + eXY * sxy)
        
        # Calculate displacement gradients
        dv_dx = torch.autograd.grad(v.sum(), contour_xy, create_graph=True)[0][:,0]
        
        # J-integral components
        # J1 = ∫ [W - σxx ∂u/∂x - σxy ∂v/∂x] nx dΓ
        # J2 = ∫ [-σxy ∂u/∂x - σyy ∂v/∂x] ny dΓ
        
        J1 = W * normals[:, 0] - normals[:, 0] * sx * eXX - normals[:, 0] * sxy * dv_dx -  normals[:, 1] * sxy * eXX -  normals[:, 1] * sy * dv_dx
        J1 = torch.mean(J1)*radius*2*np.pi

        return J1
    
    def generate_contour_around_crack_tip(self, crack_tip_x, crack_tip_y, radius=0.25, num_points=500):
        """
        Generate a circular contour around the crack tip for J-integral calculation
        
        Args:
            crack_tip_x, crack_tip_y: Crack tip coordinates
            radius: Contour radius
            num_points: Number of points on the contour
        
        Returns:
            contour_points: Points on the contour (N, 2)
            contour_normals: Normal vectors (N, 2)
        """
        # Generate circular contour
        theta = torch.linspace(0, 2*np.pi, num_points, device=self.device)
        x_contour = radius + radius * torch.cos(theta) # 注意这里beta角度取0
        y_contour = 0 + radius * torch.sin(theta)
        
        contour_points = torch.stack([x_contour, y_contour], dim=1)
        
        # Calculate normal vectors (pointing outward from circle center)
        normals = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        
        return contour_points.cpu().numpy(), normals.cpu().numpy()
    
    def j_integral_to_k1(self, J_value, E, nu, plane_strain=True):
        """
        Convert J-integral to stress intensity factor K1
        
        For plane strain: K1 = sqrt(J * E / (1 - nu^2))
        For plane stress: K1 = sqrt(J * E)
        
        Args:
            J_value: J-integral value
            E: Young's modulus
            nu: Poisson's ratio
            plane_strain: True for plane strain, False for plane stress
        
        Returns:
            K1: Stress intensity factor
        """
        if plane_strain:
            K1 = torch.sqrt(J_value * E / (1 - nu**2))
        else:
            K1 = torch.sqrt(J_value * E)
        
        return K1

    def hard_u(self, u, x, y):
        # Calculate polar coordinates relative to crack tip
        r_right = torch.sqrt((x - x_crackTip)**2 + (y-y_crackTip)**2)
        r_left = torch.sqrt((x + x_crackTip)**2 + (y-y_crackTip)**2) # 左右两个裂纹
        theta_right = torch.atan2(y-y_crackTip, x - x_crackTip)
        theta_left = torch.atan2(y-y_crackTip, -(x + x_crackTip)) # 左裂纹角度取反
    
        # u_analytical = self.K_I[0] * r**0.5 * torch.sin(theta/2) + \
        #                self.K_I[1] * r**0.5 * torch.sin(theta/2) * torch.sin(theta) + \
        #                self.K_I[2] * r**0.5 * torch.cos(theta/2) + \
        #                self.K_I[3] * r**0.5 * torch.cos(theta/2) * torch.sin(theta)

        # Calculate Mode I crack analytical solution
        mu = E / (2 * (1 + nu))
        kappa = 3 - 4 * nu  # Plane strain
        
        # K_I is now a trainable parameter (already initialized in __init__)
        # Analytical displacement components
        u_analytical_right = (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (2 + kappa + torch.cos(theta_right))
        v_analytical_right = (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.cos(theta_right/2) * (2 - kappa - torch.cos(theta_right))
        u_analytical_left = (self.K_II / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.sin(theta_left/2) * (2 + kappa + torch.cos(theta_left))
        v_analytical_left = (self.K_II / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.cos(theta_left/2) * (2 - kappa - torch.cos(theta_left))
        return (u + u_analytical_left * torch.exp(-self.b/self.a*r_left**self.q) + u_analytical_right * torch.exp(-self.b/self.a*r_right**self.q)) * (y+ self.h)/(self.h*2)
        # return u_analytical_left * torch.exp(-r_left*2) + u_analytical_right * torch.exp(-r_right*2)
        # return u * (y+self.h) /2

    def hard_v(self, v, x, y):
        # Calculate polar coordinates relative to crack tip
        r_right = torch.sqrt((x - x_crackTip)**2 + (y-y_crackTip)**2)
        r_left = torch.sqrt((x + x_crackTip)**2 + (y-y_crackTip)**2) # 左右两个裂纹
        theta_right = torch.atan2(y-y_crackTip, x - x_crackTip)
        theta_left = torch.atan2(y-y_crackTip, -(x + x_crackTip)) # 左裂纹角度取反
        
        # v_analytical = self.K_I[4] * r**0.5 * torch.sin(theta/2) + \
        #                self.K_I[5] * r**0.5 * torch.sin(theta/2) * torch.sin(theta) + \
        #                self.K_I[6] * r**0.5 * torch.cos(theta/2) + \
        #                self.K_I[7] * r**0.5 * torch.cos(theta/2) * torch.sin(theta)

        # Calculate Mode I crack analytical solution
        mu = E / (2 * (1 + nu))
        kappa = 3 - 4 * nu  # Plane strain
        
        # K_I is now a trainable parameter (already initialized in __init__)
        # Analytical displacement components
        u_analytical_right = (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (2 + kappa + torch.cos(theta_right))
        v_analytical_right = (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.cos(theta_right/2) * (2 - kappa - torch.cos(theta_right))
        u_analytical_left = (self.K_II / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.sin(theta_left/2) * (2 + kappa + torch.cos(theta_left))
        v_analytical_left = (self.K_II / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.cos(theta_left/2) * (2 - kappa - torch.cos(theta_left))
        return (v + v_analytical_left * torch.exp(-self.b/self.a*r_left**self.q) + v_analytical_right * torch.exp(-self.b/self.a*r_right**self.q)) \
            * (self.h-y)/(self.h*2) * (y+self.h)/(self.h*2) * (x+self.b)/(self.b*2) * (self.b-x)/(self.b*2)
        #return v_analytical_left * torch.exp(-r_left*2) + v_analytical_right * torch.exp(-r_right*2)
        # return v * (self.h-y) * (y+self.h) * (x+self.b) * (self.b-x)
    def add_BCPoints(self,num = [128]):
        x_up,y_up=genMeshNodes2D(-1,1,num[0],self.h,self.h,1)
        x_down,y_down=genMeshNodes2D(-1,1,num[0],-self.h,-self.h,1)
        x_left,y_left=genMeshNodes2D(-self.b,-self.b,1,-self.h,+self.h,num[0])
        x_right,y_right=genMeshNodes2D(self.b,self.b,1,-self.h,+self.h,num[0])
        self.x_up,self.y_up,self.xy_up = self._set_points(x_up ,y_up)
        self.x_down,self.y_down,self.xy_down = self._set_points(x_down ,y_down)
        self.x_left,self.y_left,self.xy_left = self._set_points(x_left ,y_left)
        self.x_right,self.y_right,self.xy_right = self._set_points(x_right ,y_right)
        self.up_zero = torch.zeros_like(self.x_up)
        self.down_zero = torch.zeros_like(self.x_down)
        self.left_zero = torch.zeros_like(self.x_left)
        self.right_zero = torch.zeros_like(self.x_right)

    def E_ext(self) -> torch.Tensor:
        u_up,v_up = self.pred_uv(self.xy_up)
        return trapz1D(u_up * self.tau, self.x_up)


    def train_with_k2_monitoring(self, epochs=50000, patience=10, path='test', lr=0.02, eval_sep=100,
                                 milestones=[5000,10000,10500], crack_embedding=None, 
                                 crack_surface=None, kappa=None, mu=None, early_stop_threshold=None, fem_value=None):
        """Override training method, add K_I parameter monitoring, including neural network K_I and calculated K1"""
        from utils.EarlyStopping import EarlyStopping
        from SIF import DispExpolation_homo
        import utils.Geometry as Geometry
        
        self.iter = 0
        self.set_EarlyStopping(patience=patience, verbose=True, path=path)
        self.path = path
        
        # Initialize K_I history record
        self.k2_history = []
        
        # Set optimizer (if not already set)
        if not hasattr(self, 'optimizer'):
            self.set_Optimizer(lr)
        
        # Calculate true stress intensity factor
        a_b = self.a/self.b
        normalized_Param = self.tau * np.sqrt(np.pi * self.a)
        K2_true_coefficient = (1-0.025*a_b**2+0.06*a_b**4) * np.sqrt(1/np.cos(np.pi*a_b/2))
        self.K2_true = normalized_Param * K2_true_coefficient

        scheduler = torch.optim.lr_scheduler.MultiStepLR(self
        .optimizer, milestones=milestones, gamma=0.5)
        
        print(f"Starting training, total {epochs} epochs")
        print("-" * 80)
        crack_tip_xy = [self.a, 0.0]
        m_integral = M_integral(self, self.device)           
        for i in range(epochs):
            self.train_step()
            
            if self.iter % eval_sep == 0:
                self.eval()

                current_loss = self.history[-1][0] if self.history else 0

                # Calculate K1 value through J-integral
                k2_calculated_j = None
                if self.a is not None:
                    radius_J = 0.25
                    k1_calculated_j, k2_calculated_j, MI, MII = m_integral.compute_K_via_interaction_integral(
                                                      crack_tip_xy=crack_tip_xy,
                                                      radius=radius_J,           # 取靠近裂尖的小圆，做路径无关性可多半径对比
                                                      beta=beta,          # 裂纹方向
                                                      E=E, nu=nu,
                                                      plane_strain=True,
                                                      num_points=720,
                                                      device=self.device)
                    k1_calculated_j = k1_calculated_j.cpu().detach().numpy()
                    k2_calculated_j = k2_calculated_j.cpu().detach().numpy()

                # Calculate K1 value through displacement extrapolation
                k2_calculated_disp = None
                if crack_embedding is not None and crack_surface is not None and kappa is not None and mu is not None and self.a is not None:
                    try:
                        K1_calc, K2_calc = DispExpolation_homo(self, crack_embedding, crack_surface, 5,
                                                              Geometry.LocalAxis(self.a, 0.0, 0.0), kappa, mu)
                        k2_calculated_disp = K2_calc[0]
                    except Exception as e:
                        print(f"Error calculating K1: {e}")
                        k2_calculated_disp = None
                

                self.k2_history.append((self.iter, k2_calculated_j, k2_calculated_disp))
                
                # Calculate current error - use FEM value if provided, otherwise use theoretical value
                if fem_value is not None:
                    # Use FEM value for error calculation
                    current_error = abs(k2_calculated_j - fem_value)/fem_value if k2_calculated_j is not None else float('inf')
                    reference_value = fem_value
                    reference_label = "FEM"
                else:
                    # Use theoretical value for error calculation
                    current_error = abs(k2_calculated_j - self.K2_true)/self.K2_true if k2_calculated_j is not None else float('inf')
                    reference_value = self.K2_true
                    reference_label = "True"
                
                print(f'Epoch {self.iter:6d} | K_II_J = {k2_calculated_j:.3f} | K_II_Disp = {k2_calculated_disp:.3f} | K_II_NN = {self.K_II.cpu().detach().numpy():.3f} | K_II_{reference_label} = {reference_value:.3f} | K_II_Error = {current_error:.3f} | Loss = {current_loss:.3e}')
                
                # Early stopping check based on K2 error
                if early_stop_threshold is not None and current_error > early_stop_threshold:
                    print(f'⚠️  Early stopping: K2 error {100*current_error:.2f}% > {100*early_stop_threshold:.2f}%')
                    print(f'Training stopped at epoch {self.iter} due to poor K2 performance')
                    break

            
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                print('end epoch:'+str(self.iter))
                break          
        
        self.save_hist(self.path)
        self.save(self.path)
        self.save_k2_history(self.path)

    def save_k2_history(self, name):
        """Save K_II parameter history record"""
        if hasattr(self, 'k2_history') and self.k2_history:
            import pandas as pd
            # Check if there are calculated K1 values
            if len(self.k2_history[0]) == 3:
                k2_df = pd.DataFrame(self.k2_history, columns=['epoch', 'K_II_J', 'K_II_Disp'])
            else:
                k2_df = pd.DataFrame(self.k2_history, columns=['epoch', 'K_II_NN'])
            k2_df.to_csv(name + '_k2_history.csv', index=False)
            print(f"K_II history record saved to: {name}_k2_history.csv")
    
    def plot_k2_training_curve(self, name):
        """Plot K_II parameter training curve, including neural network K_II and calculated K2"""
        if hasattr(self, 'k2_history') and self.k2_history:
            import matplotlib.pyplot as plt
            

            # Only neural network K_II values case
            epochs, k2_values_J, k2_values_Disp = zip(*self.k2_history)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, k2_values_J, 'b-', linewidth=2, label='K_II from J-Integral', marker='o', markersize=3)
            plt.plot(epochs, k2_values_Disp, 'r-', linewidth=2, label='K_II from Disp', marker='o', markersize=3)
            plt.axhline(y=self.K2_true, color='r', linestyle='--', linewidth=2, label='True K_II')
            plt.xlabel('Epoch')
            plt.ylabel('K_II')
            plt.title('K_II Parameter Training Process')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(name + '_k2_training_curve.png', dpi=300, bbox_inches='tight')
            # plt.show()
            print(f"K_II training curve saved to: {name}_k2_training_curve.png")

E=1e3 ; nu = 0.3

kappa = (3-4*nu)
mu = E/(2*(1+nu))

y_crackTip = 0.0

x_crackCenter = 0.0
y_crackCenter = 0.0
# beta = torch.pi/4
beta = 0.0

# Parameters for different load values
a = 0.5
b = 1.0
h = 3.0
q = 1
point_num = 100
x_crackTip = a

# FEM results - mapping from tau to FEM K2 values
tau_to_fem = {
    100: 141.4,
    200: 282.8, 
    300: 424.2,
    400: 565.6,
    500: 707.0,
    600: 848.4,
    700: 989.8,
    800: 1131.2,
    900: 1272.6,
    1000: 1414.0
}

# List to store results for comparison
results = []

# Loop through different tau values (100, 200, 300, ..., 900)
for tau in range(100, 601, 100):
    print(f"\n{'='*60}")
    print(f"Processing tau = {tau}")
    print(f"{'='*60}")
    
    # Early stopping threshold
    early_stop_threshold = 5.0  # 500% error threshold for early stopping
    current_seed = seed  # Use original seed
    
    model_name = f'../../result/crack2/crack_XDEM/different_load/tau_{tau}/crack'
    
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(model_name)):
        os.makedirs(os.path.dirname(model_name))
    
    # Create crack embedding
    crack_embedding = LineCrackEmbedding([x_crackCenter-x_crackTip,y_crackCenter],
                                            [x_crackTip,y_crackTip],
                                            tip = 'both')

    # Create network
    net = extendAxisNet(
            net = AxisScalar2D(
                stack_net(input=3,output=2,activation=nn.Tanh,width=30,depth=4),
                A=torch.tensor([1.0/b,1.0/h,1.0]),
                B=torch.tensor([0.0,0.0,0.0])
                ),
            extendAxis= crack_embedding)
    
    # Create PINN model
    pinn = Plate(net,tau=tau,a=a,b=b,h=h,q=q)
    pinn.add_BCPoints()
    pinn.setMaterial(E=E , nu = nu,type='plane strain')
    pinn.set_loss_func(losses=[pinn.Energy_loss,], weights=[1.0])
    pinn.set_meshgrid_inner_points(-b,b,point_num,-h,h,point_num*3)
    
    # Create crack surface
    crack_surface = Geometry.LineSegement([-x_crackTip,0.0],[x_crackTip,0.0])
    crack_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.35*a),
                                        crack_surface.clamp(dist2=0.3*a))

    # Get FEM value for this tau
    current_fem_value = tau_to_fem.get(tau, None)
    
    # Train the model with early stopping check
    print(f"Starting training for tau = {tau}...")
    if current_fem_value is not None:
        print(f"Using FEM reference value: {current_fem_value}")
    pinn.train_with_k2_monitoring(path=model_name, patience=10, epochs=5000, lr=0.001, eval_sep=100,
                                  crack_embedding=crack_embedding, crack_surface=crack_surface, 
                                  kappa=kappa, mu=mu, early_stop_threshold=early_stop_threshold, 
                                  fem_value=current_fem_value)
        

    # Calculate true K2 value
    a_b = a/b
    normalized_Param = tau * np.sqrt(np.pi * a)
    K2_true_coefficient = (1-0.025*a_b**2+0.06*a_b**4) * np.sqrt(1/np.cos(np.pi*a_b/2))
    K2_true = normalized_Param * K2_true_coefficient
    
    # Use the last epoch K2 prediction (more objective than selecting "best")
    if hasattr(pinn, 'k2_history') and pinn.k2_history:
        last_epoch_data = pinn.k2_history[-1]
        last_epoch = last_epoch_data[0]
        final_k2_j = last_epoch_data[1]
        final_k2_disp = last_epoch_data[2]
        print(f"Last epoch: {last_epoch}")
    else:
        final_k2_j = None
        final_k2_disp = None
        last_epoch = None
    
    # Calculate errors - use FEM value if available
    if current_fem_value is not None:
        error_j = abs(final_k2_j - current_fem_value) / current_fem_value  if final_k2_j is not None else float('inf')
        error_disp = abs(final_k2_disp - current_fem_value) / current_fem_value  if final_k2_disp is not None else float('inf')
    else:
        error_j = abs(final_k2_j - K2_true) / K2_true  if final_k2_j is not None else float('inf')
        error_disp = abs(final_k2_disp - K2_true) / K2_true  if final_k2_disp is not None else float('inf')
        
    # Print results with appropriate reference value
    if current_fem_value is not None:
        print(f"K2_FEM = {current_fem_value:.3f}, K2_J = {final_k2_j:.3f}, K2_Disp = {final_k2_disp:.3f}")
        print(f"Error J (vs FEM): {error_j:.2f}, Error Disp (vs FEM): {error_disp:.2f}")
    else:
        print(f"K2_true = {K2_true:.3f}, K2_J = {final_k2_j:.3f}, K2_Disp = {final_k2_disp:.3f}")
        print(f"Error J: {error_j:.2f}, Error Disp: {error_disp:.2f}")
    
    # Add result to results list
    result = {
        'tau': tau,
        'K2_true': K2_true,
        'K2_j': final_k2_j,
        'K2_disp': final_k2_disp,
        'error_j': error_j,
        'error_disp': error_disp,
        'last_epoch': last_epoch,
        'seed_used': current_seed
    }
    results.append(result)
    print(f"Result for tau={tau}: Error J = {error_j:.2f}% (epoch: {last_epoch}, seed: {current_seed})")

    # Generate field visualization for each tau
    print(f"  Generating field visualization for tau = {tau}...")

    # Create a fine grid for visualization
    test_num = 100
    x_vis = torch.linspace(-b, b, test_num, device=pinn.device)
    y_vis = torch.linspace(-h, h, test_num, device=pinn.device)
    X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
    XY_vis = torch.stack([X_vis.flatten(), Y_vis.flatten()], dim=1)
    XY_vis.requires_grad_(True)

    # Get predictions (enable gradients for visualization)
    u_pred, v_pred = pinn.pred_uv(XY_vis)
    _, _, sx_pred, sy_pred, sxy_pred = pinn.infer(XY_vis)

    # Get Gamma from embedding layer
    gamma_pred = crack_embedding.getGamma(XY_vis)

    # Convert to numpy for plotting
    X_vis_np = X_vis.cpu().numpy()
    Y_vis_np = Y_vis.cpu().numpy()
    u_pred_np = u_pred.detach().cpu().numpy().reshape(test_num, test_num)
    v_pred_np = v_pred.detach().cpu().numpy().reshape(test_num, test_num)
    sy_pred_np = sy_pred.detach().cpu().numpy().reshape(test_num, test_num)
    sxy_pred_np = sxy_pred.detach().cpu().numpy().reshape(test_num, test_num)
    sx_pred_np = sx_pred.detach().cpu().numpy().reshape(test_num, test_num)

    # Convert Gamma to numpy
    gamma_pred_np = gamma_pred.detach().cpu().numpy().reshape(test_num, test_num)

    # Save arrays to numpy files
    np.savez(model_name + '_field_data.npz',
             X_vis=X_vis_np,
             Y_vis=Y_vis_np,
             u_pred=u_pred_np,
             v_pred=v_pred_np,
             sx_pred=sx_pred_np,
             sy_pred=sy_pred_np,
             sxy_pred=sxy_pred_np,
             gamma_pred=gamma_pred_np)
    print(f"  Field data arrays saved to: {model_name}_field_data.npz")

    # Plot u2 displacement field and Gamma embedding
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_u2, label='u2 displacement (mm)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'u2 Displacement Field (tau={tau})')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    # Plot σ22 stress field
    plt.subplot(1, 3, 2)
    contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_sy, label='σ22 stress (MPa)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'σ22 Stress Field (tau={tau})')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    # Plot Gamma embedding field
    plt.subplot(1, 3, 3)
    contour_gamma = plt.contourf(X_vis_np, Y_vis_np, gamma_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_gamma, label='Embedding function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Embedding function Field (tau={tau})')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(model_name + '_displacement_stress_fields.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Displacement and stress fields saved to: {model_name}_displacement_stress_fields.png")

    # Additional detailed plots
    plt.figure(figsize=(15, 10))

    # u1 displacement
    plt.subplot(2, 3, 1)
    contour_u1 = plt.contourf(X_vis_np, Y_vis_np, u_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_u1, label='u1 displacement (mm)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'u1 Displacement Field (tau={tau})')
    plt.axis('equal')

    # u2 displacement
    plt.subplot(2, 3, 2)
    contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_u2, label='u2 displacement (mm)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'u2 Displacement Field (tau={tau})')
    plt.axis('equal')

    # σ11 stress
    plt.subplot(2, 3, 3)
    contour_sx = plt.contourf(X_vis_np, Y_vis_np, sx_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_sx, label='σ11 stress (MPa)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'σ11 Stress Field (tau={tau})')
    plt.axis('equal')

    # σ22 stress
    plt.subplot(2, 3, 4)
    contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_sy, label='σ22 stress (MPa)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'σ22 Stress Field (tau={tau})')
    plt.axis('equal')

    # σ12 stress
    plt.subplot(2, 3, 5)
    contour_sxy = plt.contourf(X_vis_np, Y_vis_np, sxy_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_sxy, label='σ12 stress (MPa)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'σ12 Stress Field (tau={tau})')
    plt.axis('equal')

    # Von Mises stress
    plt.subplot(2, 3, 6)
    von_mises = np.sqrt(sx_pred_np**2 - sx_pred_np*sy_pred_np + sy_pred_np**2 + 3*sxy_pred_np**2)
    contour_vm = plt.contourf(X_vis_np, Y_vis_np, von_mises, levels=50, cmap='jet')
    plt.colorbar(contour_vm, label='Von Mises stress (MPa)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Von Mises Stress Field (tau={tau})')
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig(model_name + '_detailed_fields.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Detailed field plots saved to: {model_name}_detailed_fields.pdf")

    # Save Von Mises stress array as well
    np.save(model_name + '_von_mises_stress.npy', von_mises)
    print(f"  Von Mises stress array saved to: {model_name}_von_mises_stress.npy")

    # =============================================================================
    # Save individual figures as PDF for paper publication
    # =============================================================================
    print(f"  Saving individual PDF figures for tau = {tau}...")

    # 1. u2 Displacement Field
    plt.figure(figsize=(5, 6))
    contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_u2, label='u2 displacement (mm)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'u2 Displacement Field (tau={tau})')
    plt.xlim(-b, b)
    plt.ylim(-h, h)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_name + '_u2_displacement.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    u2 displacement PDF saved to: {model_name}_u2_displacement.pdf")

    # 2. σ22 Stress Field
    plt.figure(figsize=(5, 6))
    contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_sy, label='σ22 stress (MPa)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'σ22 Stress Field (tau={tau})')
    plt.xlim(-b, b)
    plt.ylim(-h, h)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_name + '_sigma22_stress.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    σ22 stress PDF saved to: {model_name}_sigma22_stress.pdf")

    # 3. Gamma Embedding Field 
    plt.figure(figsize=(5, 6))
    contour_gamma = plt.contourf(X_vis_np, Y_vis_np, gamma_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_gamma, label='Embedding function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Embedding function Field (tau={tau})')
    plt.xlim(-b, b)
    plt.ylim(-h, h)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_name + '_gamma_embedding.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Gamma embedding PDF saved to: {model_name}_gamma_embedding.pdf")

    # 4. u1 Displacement Field
    plt.figure(figsize=(5, 6))
    contour_u1 = plt.contourf(X_vis_np, Y_vis_np, u_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_u1, label='u1 displacement (mm)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'u1 Displacement Field (tau={tau})')
    plt.xlim(-b, b)
    plt.ylim(-h, h)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_name + '_u1_displacement.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    u1 displacement PDF saved to: {model_name}_u1_displacement.pdf")

    # 5. σ11 Stress Field
    plt.figure(figsize=(5, 6))
    contour_sx = plt.contourf(X_vis_np, Y_vis_np, sx_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_sx, label='σ11 stress (MPa)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'σ11 Stress Field (tau={tau})')
    plt.xlim(-b, b)
    plt.ylim(-h, h)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_name + '_sigma11_stress.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    σ11 stress PDF saved to: {model_name}_sigma11_stress.pdf")

    # 6. σ12 Stress Field
    plt.figure(figsize=(5, 6))
    contour_sxy = plt.contourf(X_vis_np, Y_vis_np, sxy_pred_np, levels=50, cmap='jet')
    plt.colorbar(contour_sxy, label='σ12 stress (MPa)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'σ12 Stress Field (tau={tau})')
    plt.xlim(-b, b)
    plt.ylim(-h, h)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_name + '_sigma12_stress.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    σ12 stress PDF saved to: {model_name}_sigma12_stress.pdf")

    # 7. Von Mises Stress Field
    plt.figure(figsize=(5, 6))
    contour_vm = plt.contourf(X_vis_np, Y_vis_np, von_mises, levels=50, cmap='jet')
    plt.colorbar(contour_vm, label='Von Mises stress (MPa)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Von Mises Stress Field (tau={tau})')
    plt.xlim(-b, b)
    plt.ylim(-h, h)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_name + '_von_mises_stress.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Von Mises stress PDF saved to: {model_name}_von_mises_stress.pdf")

print(f"\n{'='*60}")
print("All tau values processed!")
print(f"{'='*60}")

# =============================================================================
# Create comparison plot for different tau values
# =============================================================================
print("Creating comparison plot for different tau values...")

# Extract data for plotting
tau_values = [r['tau'] for r in results]
k2_true_values = [r['K2_true'] for r in results]
k2_j_values = [r['K2_j'] for r in results if r['K2_j'] is not None]
k2_disp_values = [r['K2_disp'] for r in results if r['K2_disp'] is not None]
error_j_values = [r['error_j'] for r in results if r['error_j'] is not None]
error_disp_values = [r['error_disp'] for r in results if r['error_disp'] is not None]

fem_value = [141.4, 282.8, 424.2, 565.6, 707.0, 848.4, 989.8, 1131.2, 1272.6, 1414.0]

# Create comparison plot - only K2 values comparison
plt.figure(figsize=(10, 6))

# Plot: K2 values comparison (only J-integral, no displacement)
plt.plot(tau_values, k2_true_values, 'k-', linewidth=3, label='Reference K2', marker='o', markersize=6)
plt.plot(tau_values, fem_value[0:len(tau_values)], 'r-', linewidth=3, label='FEM K2', marker='s', markersize=6)
plt.plot(tau_values, k2_j_values, 'b-', linewidth=2, label='XDEM (J-integral)', marker='s', markersize=5)

# Add value annotations for Reference K2 values
for i, (tau_val, k2_val) in enumerate(zip(tau_values, k2_true_values)):
    plt.annotate(f'{k2_val:.1f}', 
                (tau_val, k2_val), 
                textcoords="offset points", 
                xytext=(-12, 5), 
                ha='center', 
                fontsize=9, 
                color='black',
                fontweight='bold')

# Add value annotations for FEM K2 values
for i, (tau_val, k2_val) in enumerate(zip(tau_values, fem_value[0:len(tau_values)])):
    plt.annotate(f'{k2_val:.1f}', 
                (tau_val, k2_val), 
                textcoords="offset points", 
                xytext=(20, -10), 
                ha='center', 
                fontsize=9, 
                color='red',
                fontweight='bold')

# Add value annotations for XDEM K2 values
for i, (tau_val, k2_val) in enumerate(zip(tau_values, k2_j_values)):
    plt.annotate(f'{k2_val:.1f}', 
                (tau_val, k2_val), 
                textcoords="offset points", 
                xytext=(0, -20), 
                ha='center', 
                fontsize=9, 
                color='blue',
                fontweight='bold')

plt.xlabel('Shear Stress (τ)')
plt.ylabel('K2 (Stress Intensity Factor)')
plt.title('K2 Prediction vs True Values for Different Shear Stresses')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the comparison plot
comparison_plot_path = '../../result/crack2/crack_XDEM/different_load/k2_comparison_different_tau.pdf'
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Comparison plot saved to: {comparison_plot_path}")

# =============================================================================
# Save results to CSV file
# =============================================================================
import pandas as pd
results_df = pd.DataFrame(results)
results_csv_path = '../../result/crack2/crack_XDEM/different_load/k2_results_different_tau.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to: {results_csv_path}")






