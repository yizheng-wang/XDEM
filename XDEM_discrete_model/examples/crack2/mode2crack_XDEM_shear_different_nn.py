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
from SIF import DispExpolation_homo


class Plate(PINN2D):
    def __init__(self, model: nn.Module,tau,a,b,h,q=1):
        '''x范围0-1.y范围-1到1'''
        super().__init__(model)
        self.tau = tau
        
        # 初始化 K_I 参数，确保它参与优化
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

    def hard_v(self, v, x, y):
        # Calculate polar coordinates relative to crack tip
        r_right = torch.sqrt((x - x_crackTip)**2 + (y-y_crackTip)**2)
        r_left = torch.sqrt((x + x_crackTip)**2 + (y-y_crackTip)**2) # 左右两个裂纹
        theta_right = torch.atan2(y-y_crackTip, x - x_crackTip)
        theta_left = torch.atan2(y-y_crackTip, -(x + x_crackTip)) # 左裂纹角度取反
        
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

        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.5)
        
        print(f"Starting training, total {epochs} epochs")
        print("-" * 80)
        
        for i in range(epochs):
            self.train_step()
            
            if self.iter % eval_sep == 0:
                self.eval()

                current_loss = self.history[-1][0] if self.history else 0

                # Calculate K1 value through J-integral
                k2_calculated_j = None
                if self.a is not None:
                    try:
                        radius_J = 0.25*(self.a+self.b)
                        # Generate contour around crack tip
                        contour_points, contour_normals = self.generate_contour_around_crack_tip(
                            crack_tip_x=self.a, crack_tip_y=0.0, radius=radius_J, num_points=50
                        )
                        
                        # Calculate J-integral
                        J_value1 = self.calculate_j_integral(contour_points, contour_normals, radius_J)
                        
                        # Convert J-integral to K1
                        k2_calculated_j = self.j_integral_to_k1(J_value1, E, nu, plane_strain=True)
                        k2_calculated_j = k2_calculated_j.cpu().detach().numpy()
                        
                    except Exception as e:
                        print(f"Error calculating K1 via J-integral: {e}")
                        k2_calculated_j = None

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

# 设置参数
E=1e3 
nu = 0.3
kappa = (3-4*nu)
mu = E/(2*(1+nu))

y_crackTip = 0.0
x_crackCenter = 0.0
y_crackCenter = 0.0
beta = 0.0

# 固定参数
tau = 100  # 固定力为100
a = 0.5
b = 1.0
h = 3.0
q = 2
point_num = 100
x_crackTip = a

# FEM参考值 (tau=100对应的FEM值)
fem_value = 141.4  # 根据tau=100估算的FEM值

# 测试不同的神经网络结构
depths = [3, 4, 5]  # 不同的深度
widths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 不同的宽度

# 存储结果
results = []

print(f"Testing different neural network architectures for tau = {tau}")
print(f"FEM reference value: {fem_value}")
print("="*80)

# 创建目录
output_dir = '../../result/crack2/crack_XDEM/different_nn_architectures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 循环测试不同的网络结构
for depth in depths:
    print(f"\nTesting depth = {depth}")
    print("-" * 40)
    
    depth_results = []
    
    for width in widths:
        print(f"  Testing width = {width}")
        
        # 设置随机种子确保可重复性
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 创建网络
        net = extendAxisNet(
            net=AxisScalar2D(
                stack_net(input=3, output=2, activation=nn.Tanh, width=width, depth=depth),
                A=torch.tensor([1.0/b, 1.0/h, 1.0]),
                B=torch.tensor([0.0, 0.0, 0.0])
            ),
            extendAxis=LineCrackEmbedding([x_crackCenter-x_crackTip, y_crackCenter],
                                        [x_crackTip, y_crackTip], tip='both')
        )
        
        # 创建PINN模型
        pinn = Plate(net, tau=tau, a=a, b=b, h=h, q=q)
        pinn.add_BCPoints()
        pinn.setMaterial(E=E, nu=nu, type='plane strain')
        pinn.set_loss_func(losses=[pinn.Energy_loss,], weights=[1.0])
        pinn.set_meshgrid_inner_points(-b, b, point_num, -h, h, point_num*3)
        
        # 创建裂纹表面
        crack_surface = Geometry.LineSegement([-x_crackTip, 0.0], [x_crackTip, 0.0])
        crack_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.35*a),
                                            crack_surface.clamp(dist2=0.3*a))
        
        # 创建裂纹嵌入
        crack_embedding = LineCrackEmbedding([x_crackCenter-x_crackTip, y_crackCenter],
                                            [x_crackTip, y_crackTip], tip='both')
        
        # 训练模型
        model_name = f'{output_dir}/depth_{depth}_width_{width}'
        pinn.train_with_k2_monitoring(path=model_name, patience=10, epochs=30000, lr=0.001, eval_sep=100,
                                      crack_embedding=crack_embedding, crack_surface=crack_surface, 
                                      kappa=kappa, mu=mu, early_stop_threshold=10.0, 
                                      fem_value=fem_value)
        
        # 计算真实K2值
        a_b = a/b
        normalized_Param = tau * np.sqrt(np.pi * a)
        K2_true_coefficient = (1-0.025*a_b**2+0.06*a_b**4) * np.sqrt(1/np.cos(np.pi*a_b/2))
        K2_true = normalized_Param * K2_true_coefficient
        
        # 找到最佳K2预测
        best_error = float('inf')
        best_k2_j = None
        best_epoch = None
        
        if hasattr(pinn, 'k2_history') and pinn.k2_history:
            for epoch_data in pinn.k2_history:
                k2_j_val = epoch_data[1]
                if k2_j_val is not None:
                    error = abs(k2_j_val - fem_value) / fem_value
                    if error < best_error:
                        best_error = error
                        best_k2_j = k2_j_val
                        best_epoch = epoch_data[0]
        
        # 存储结果
        result = {
            'depth': depth,
            'width': width,
            'best_k2_j': best_k2_j,
            'best_error': best_error,
            'best_epoch': best_epoch,
            'fem_value': fem_value,
            'K2_true': K2_true
        }
        results.append(result)
        depth_results.append(best_error)
        
        print(f"    Best error: {best_error:.4f} (epoch: {best_epoch})")
    
    print(f"  Depth {depth} completed. Best error: {min(depth_results):.4f}")

print("\n" + "="*80)
print("All network architectures tested!")
print("="*80)

# 创建对比图
plt.figure(figsize=(12, 8))

# 为每个深度绘制一条曲线
colors = ['blue', 'red', 'green']
markers = ['o', 's', '^']

for i, depth in enumerate(depths):
    # 提取该深度的数据
    depth_data = [r for r in results if r['depth'] == depth]
    widths_d = [r['width'] for r in depth_data]
    errors_d = [r['best_error'] for r in depth_data]
    
    # 绘制曲线
    plt.plot(widths_d, errors_d, color=colors[i], marker=markers[i], 
             linewidth=2, markersize=6, label=f'Depth = {depth}')

plt.xlabel('Network Width', fontsize=12)
plt.ylabel('Error vs FEM', fontsize=12)
plt.title(f'Neural Network Architecture Comparison (τ = {tau})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.yscale('log')  # 使用对数坐标轴更好地显示误差
plt.tight_layout()

# 保存图片
plot_path = f'{output_dir}/nn_architecture_comparison_tau_{tau}.pdf'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Comparison plot saved to: {plot_path}")

# 保存结果到CSV
import pandas as pd
results_df = pd.DataFrame(results)
results_csv_path = f'{output_dir}/nn_architecture_results_tau_{tau}.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to: {results_csv_path}")

# 打印最佳结果
print("\nBest results for each depth:")
for depth in depths:
    depth_results = [r for r in results if r['depth'] == depth]
    best_result = min(depth_results, key=lambda x: x['best_error'])
    print(f"  Depth {depth}: Width {best_result['width']}, Error {best_result['best_error']:.4f}")

print("\nOverall best result:")
overall_best = min(results, key=lambda x: x['best_error'])
print(f"  Depth {overall_best['depth']}, Width {overall_best['width']}, Error {overall_best['best_error']:.4f}")