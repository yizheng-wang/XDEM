import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
import matplotlib.pyplot as plt
import utils.Geometry as Geometry
from SIF import DispExpolation_homo, SIF_K1K2, M_integral


class Plate(PINN2D):
    def __init__(self, model: nn.Module,fy,a,b,h,q=1, beta = 0.0):
        '''x范围0-1.y范围-1到1'''
        super().__init__(model)
        self.fy = fy
        
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

        K1_esti = fy * np.sqrt(np.pi * a)  * np.cos(beta)**2
        K2_esti = fy * np.sqrt(np.pi * a)  * np.sin(beta) * np.cos(beta)
        self.K_I = nn.Parameter(torch.tensor(K1_esti, device=self.device))
        self.K_II = nn.Parameter(torch.tensor(K2_esti, device=self.device))
        self.a = a
        self.b = b
        self.h = h
        self.q = q
    def set_Optimizer(self, lr, k_i_lr=None):
        """重写优化器设置方法，为K_I参数设置不同的学习率"""
        # 获取模型参数和 K_I 参数
        model_params = list(self.model.parameters())
        k_i_params = [self.K_I, self.K_II]  # 直接使用整个K_I参数张量
        
        # 如果没有指定K_I的学习率，使用默认的较大学习率
        if k_i_lr is None:
            k_i_lr = lr * 10  # K_I使用10倍的学习率

        
        # 创建参数组，为不同参数设置不同学习率
        param_groups = [
            {'params': model_params, 'lr': lr},
            {'params': k_i_params, 'lr': k_i_lr}
        ]
        
        self.optimizer = torch.optim.Adam(param_groups)


        
    def hard_u(self, u, x, y):
        # Calculate polar coordinates relative to crack tip
        c = torch.cos(torch.tensor(beta, device=self.device))
        s = torch.sin(torch.tensor(beta, device=self.device))
        # 全局->局部 的旋转 R = [[c, s],[-s, c]]
        # 右/左裂尖（全局坐标）
        tip_r = torch.stack([ self.a * c,  self.a * s]).to(x.device).to(x.dtype)
        tip_l = torch.stack([-self.a * c, -self.a * s]).to(x.device).to(x.dtype)

        # 组装点坐标
        xy = torch.stack([x, y], dim=-1)  # (...,2)

        # 相对坐标（全局）
        rel_r_g = xy - tip_r
        rel_l_g = xy - tip_l

        # 旋到局部：v_local = R v_global
        R = torch.stack([torch.stack([c, s], dim=-1),
                        torch.stack([-s, c], dim=-1)], dim=0)  # (2,2)
        rel_r_l = rel_r_g @ R.T
        rel_l_l = rel_l_g @ R.T



        x1_r, x2_r = rel_r_l[..., 0], rel_r_l[..., 1]
        x1_l, x2_l = -rel_l_l[..., 0], rel_l_l[..., 1]
        r_right   = torch.sqrt(x1_r**2 + x2_r**2)
        r_left    = torch.sqrt(x1_l**2 + x2_l**2)
        theta_right = torch.atan2(x2_r, x1_r)
        theta_left  = torch.atan2(x2_l, x1_l)
    
        # u_analytical = self.K_I[0] * r**0.5 * torch.sin(theta/2) + \
        #                self.K_I[1] * r**0.5 * torch.sin(theta/2) * torch.sin(theta) + \
        #                self.K_I[2] * r**0.5 * torch.cos(theta/2) + \
        #                self.K_I[3] * r**0.5 * torch.cos(theta/2) * torch.sin(theta)

        # Calculate Mode I crack analytical solution
        mu = E / (2 * (1 + nu))
        kappa = 3 - 4 * nu  # Plane strain
        
        # K_I is now a trainable parameter (already initialized in __init__)
        # Analytical displacement components
        u_analytical_right = (self.K_I / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.cos(theta_right/2) * (kappa - 1 + 2 * torch.sin(theta_right/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (2 + kappa + torch.cos(theta_right))
        v_analytical_right = (self.K_I / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (kappa + 1 - 2 * torch.cos(theta_right/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.cos(theta_right/2) * (2 - kappa - torch.cos(theta_right))
        u_analytical_left = ((self.K_I / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.cos(theta_left/2) * (kappa - 1 + 2 * torch.sin(theta_left/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.sin(theta_left/2) * (2 + kappa + torch.cos(theta_left)))
        v_analytical_left = (self.K_I / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.sin(theta_left/2) * (kappa + 1 - 2 * torch.cos(theta_left/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.cos(theta_left/2) * (2 - kappa - torch.cos(theta_left))
        # 需要把上述局部坐标系表示的位移场放到全局坐标系中
        u_analytical_right_global = u_analytical_right * c - v_analytical_right * s
        v_analytical_right_global = u_analytical_right * s + v_analytical_right * c
        u_analytical_left_global = u_analytical_left * c - v_analytical_left * s
        v_analytical_left_global = u_analytical_left * s + v_analytical_left * c
        # return (u + u_analytical_left_global * torch.exp(-20*self.b/self.a*r_left**self.q) + u_analytical_right_global * torch.exp(-20*self.b/self.a*r_right**self.q))  * (x + self.b)/(2*self.b) * (self.b - x)/(2*self.b) 
        # return u_analytical_right_global # * torch.exp(-self.b/self.a*r_left**self.q) # + u_analytical_right * torch.exp(-self.b/self.a*r_right**self.q)
        # return (u + u_analytical_left_global * torch.exp(-self.b/self.a*r_left**self.q) + u_analytical_right_global * torch.exp(-self.b/self.a*r_right**self.q)) * (x + self.b)/(2*self.b) * (self.b - x)/(2*self.b) 
        return  u_analytical_right_global * torch.exp(-20*self.b/self.a*r_right**self.q)
    def hard_v(self, v, x, y):
        # Calculate polar coordinates relative to crack tip
        c = torch.cos(torch.tensor(beta, device=self.device))
        s = torch.sin(torch.tensor(beta, device=self.device))
        # 全局->局部 的旋转 R = [[c, s],[-s, c]]
        # 右/左裂尖（全局坐标）
        tip_r = torch.stack([ self.a * c,  self.a * s]).to(x.device).to(x.dtype)
        tip_l = torch.stack([-self.a * c, -self.a * s]).to(x.device).to(x.dtype)

        # 组装点坐标
        xy = torch.stack([x, y], dim=-1)  # (...,2)

        # 相对坐标（全局）
        rel_r_g = xy - tip_r
        rel_l_g = xy - tip_l

        # 旋到局部：v_local = R v_global
        R = torch.stack([torch.stack([c, s], dim=-1),
                        torch.stack([-s, c], dim=-1)], dim=0)  # (2,2)
        rel_r_l = rel_r_g @ R.T
        rel_l_l = rel_l_g @ R.T

        # 局部极坐标
        x1_r, x2_r = rel_r_l[..., 0], rel_r_l[..., 1]
        x1_l, x2_l = -rel_l_l[..., 0], rel_l_l[..., 1]
        r_right   = torch.sqrt(x1_r**2 + x2_r**2)
        r_left    = torch.sqrt(x1_l**2 + x2_l**2)
        theta_right = torch.atan2(x2_r, x1_r)
        theta_left  = torch.atan2(x2_l, x1_l)
        
        # v_analytical = self.K_I[4] * r**0.5 * torch.sin(theta/2) + \
        #                self.K_I[5] * r**0.5 * torch.sin(theta/2) * torch.sin(theta) + \
        #                self.K_I[6] * r**0.5 * torch.cos(theta/2) + \
        #                self.K_I[7] * r**0.5 * torch.cos(theta/2) * torch.sin(theta)

        # Calculate Mode I crack analytical solution
        mu = E / (2 * (1 + nu))
        kappa = 3 - 4 * nu  # Plane strain
        
        # K_I is now a trainable parameter (already initialized in __init__)
        # Analytical displacement components
        u_analytical_right = (self.K_I / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.cos(theta_right/2) * (kappa - 1 + 2 * torch.sin(theta_right/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (2 + kappa + torch.cos(theta_right))
        v_analytical_right = (self.K_I / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (kappa + 1 - 2 * torch.cos(theta_right/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.cos(theta_right/2) * (2 - kappa - torch.cos(theta_right))
        u_analytical_left = ((self.K_I / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.cos(theta_left/2) * (kappa - 1 + 2 * torch.sin(theta_left/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.sin(theta_left/2) * (2 + kappa + torch.cos(theta_left)))
        v_analytical_left = (self.K_I / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.sin(theta_left/2) * (kappa + 1 - 2 * torch.cos(theta_left/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left / (2 * np.pi)) * torch.cos(theta_left/2) * (2 - kappa - torch.cos(theta_left))
        u_analytical_right_global = u_analytical_right * c - v_analytical_right * s
        v_analytical_right_global = u_analytical_right * s + v_analytical_right * c
        u_analytical_left_global = u_analytical_left * c - v_analytical_left * s
        v_analytical_left_global = u_analytical_left * s + v_analytical_left * c

        
        # return (v + v_analytical_left_global * torch.exp(-20*self.b/self.a*r_left**self.q) + v_analytical_right_global * torch.exp(-20*self.b/self.a*r_right**self.q))  * (y + self.h)/(2*self.h)
        # return v_analytical_right_global # * torch.exp(-self.b/self.a*r_left**self.q) # + v_analytical_right * torch.exp(-self.b/self.a*r_right**self.q)
        # return v_analytical_left_global * torch.exp(-2*r_left**2)
        return    v_analytical_left_global * torch.exp(-20*self.b/self.a*r_left**self.q) + v_analytical_right_global * torch.exp(-20*self.b/self.a*r_right**self.q)
    def add_BCPoints(self,num = [128]):
        x_up,y_up=genMeshNodes2D(-self.b,self.b,num[0],self.h,self.h,1)
        x_down,y_down=genMeshNodes2D(-self.b,self.b,num[0],-self.h,-self.h,1)
        self.x_up,self.y_up,self.xy_up = self._set_points(x_up ,y_up)
        self.x_down,self.y_down,self.xy_down = self._set_points(x_down ,y_down)
        self.up_zero = torch.zeros_like(self.x_up)
        self.down_zero = torch.zeros_like(self.x_down)

    def E_ext(self) -> torch.Tensor:
        u_up,v_up = self.pred_uv(self.xy_up)
        u_down,v_down = self.pred_uv(self.xy_down)

        return trapz1D(v_up * self.fy, self.x_up) # + trapz1D(v_down * -self.fy, self.x_down)

    def train_with_k12_monitoring(self, epochs=50000, patience=10, path='test', lr=0.02, eval_sep=100,
                                 milestones=[10000, 15000], crack_embedding=None, 
                                 crack_surface=None, kappa=None, mu=None, beta = 0.0):
        """Override training method, add K_I parameter monitoring, including neural network K_I and calculated K1"""
        from utils.EarlyStopping import EarlyStopping
        from SIF import DispExpolation_homo
        import utils.Geometry as Geometry
        
        self.iter = 0
        self.set_EarlyStopping(patience=patience, verbose=True, path=path)
        self.path = path
        
        # Initialize K_I history record
        self.k1_history = []
        
        # Initialize best error tracking
        self.best_k1_error = float('inf')
        self.best_k2_error = float('inf')
        self.best_total_error = float('inf')
        self.best_k1_epoch = 0
        self.best_k2_epoch = 0
        self.best_total_epoch = 0
        
        # Set optimizer (if not already set)
        if not hasattr(self, 'optimizer'):
            self.set_Optimizer(lr)
        
        # Calculate true stress intensity factor - 使用预定义的理论值
        # 请根据您的理论计算结果填入以下数值
        K1_theory_values = {
            15: 136.3,
            30: 117.6,
            45: 92.45,
            60: 67.2,
            75: 49.23 
        }
        
        K2_theory_values = {
            15: 17.26,
            30: 33.1,
            45: 40.15,
            60: 35.32,
            75: 20.30
        }
        
        # 获取当前角度的理论值
        angle_deg = int(beta * 180 / np.pi + 0.2) # 四舍五入
        self.K1_true = K1_theory_values.get(angle_deg, 0.0)
        self.K2_true = K2_theory_values.get(angle_deg, 0.0)
        
        # 如果理论值为0，则使用原来的公式计算（作为备用）
        if self.K1_true == 0.0:
            normalized_Param = self.fy * np.sqrt(np.pi * self.a)
            self.K1_true = normalized_Param * np.cos(beta)**2
        if self.K2_true == 0.0:
            normalized_Param = self.fy * np.sqrt(np.pi * self.a)
            self.K2_true = normalized_Param * np.sin(beta) * np.cos(beta)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.3)
        
        print(f"Starting training for beta={beta*180/np.pi:.1f}°, total {epochs} epochs")
        print("-" * 80)
        
        crack_tip_xy = [self.a*np.cos(beta), self.a*np.sin(beta)]

        m_integral = M_integral(self, self.device)
        for i in range(epochs):
            self.train_step()
            
            if self.iter % eval_sep == 0:
                self.eval()

                current_loss = self.history[-1][0] if self.history else 0

                # Calculate K1 value through J-integral
                k1_calculated_j = None
                if self.a is not None:
                    radius_J = 0.25 # (self.a+self.b)/4
                    # Generate contour around crack tip
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


                

                # Calculate strain energy and external work separately
                strain_energy = self.E_int().cpu().detach().numpy()
                external_work = self.E_ext().cpu().detach().numpy()
                
                # Calculate errors
                k1_error = abs(k1_calculated_j - self.K1_true) / self.K1_true * 100
                k2_error = abs(k2_calculated_j - self.K2_true) / self.K2_true * 100
                total_error = k1_error + k2_error
                
                # Track best errors
                if k1_error < self.best_k1_error:
                    self.best_k1_error = k1_error
                    self.best_k1_epoch = self.iter
                
                if k2_error < self.best_k2_error:
                    self.best_k2_error = k2_error
                    self.best_k2_epoch = self.iter
                
                if total_error < self.best_total_error:
                    self.best_total_error = total_error
                    self.best_total_epoch = self.iter
                
                self.k1_history.append((self.iter, k1_calculated_j,  k2_calculated_j, strain_energy, external_work))
                print(f'Epoch {self.iter:6d} | K_I_J = {k1_calculated_j:.3f} | K_I_NN = {self.K_I.cpu().detach().numpy():.3f} | K_I_True = {self.K1_true:.3f} | K_I_Error = {k1_error:.3f}% | \
                | K_II_J = {k2_calculated_j:.3f} | K_II_NN = {self.K_II.cpu().detach().numpy():.3f} | K_II_True = {self.K2_true:.3f} | K_II_Error = {k2_error:.3f}% \
                | Strain_Energy = {strain_energy:.3e} | External_Work = {external_work:.3e} | Total_Loss = {current_loss:.3e}')

            
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                print('end epoch:'+str(self.iter))
                break          
        
        # Print best error statistics
        print(f"\n{'='*80}")
        print("Best Error Statistics:")
        print(f"{'='*80}")
        print(f"Best K1 Error: {self.best_k1_error:.3f}% at Epoch {self.best_k1_epoch}")
        print(f"Best K2 Error: {self.best_k2_error:.3f}% at Epoch {self.best_k2_epoch}")
        print(f"Best Total Error: {self.best_total_error:.3f}% at Epoch {self.best_total_epoch}")
        print(f"{'='*80}")
        
        self.save_hist(self.path)
        self.save(self.path)
        self.save_k12_history(self.path)
        self.save_best_error_stats(self.path)

    def save_k12_history(self, name):
        """Save K_I and K_II parameter history record with energy components for mixed-mode crack"""
        if hasattr(self, 'k1_history') and self.k1_history:
            import pandas as pd
            # Check if there are calculated K1, K2 values and energy components
            
            k1_df = pd.DataFrame(self.k1_history, columns=['epoch', 'K_I_J', 'K_II_J', 'Strain_Energy', 'External_Work'])
            k1_df.to_csv(name + '_k1_history.csv', index=False)
            print(f"K_I and K_II history record with energy components saved to: {name}_k1_history.csv")
    
    def save_best_error_stats(self, name):
        """Save best error statistics to a text file"""
        with open(name + '_best_error_stats.txt', 'w') as f:
            f.write("Best Error Statistics\n")
            f.write("="*50 + "\n")
            f.write(f"Best K1 Error: {self.best_k1_error:.3f}% at Epoch {self.best_k1_epoch}\n")
            f.write(f"Best K2 Error: {self.best_k2_error:.3f}% at Epoch {self.best_k2_epoch}\n")
            f.write(f"Best Total Error: {self.best_total_error:.3f}% at Epoch {self.best_total_epoch}\n")
            f.write("="*50 + "\n")
        print(f"Best error statistics saved to: {name}_best_error_stats.txt")
    
    def plot_k12_training_curve(self, name):
        """Plot K_I and K_II parameter training curve, including energy components"""
        if hasattr(self, 'k1_history') and self.k1_history:
            import matplotlib.pyplot as plt
            

            # With K1, K2 and energy data
            epochs, k1_values_J, k2_values_J, strain_energy, external_work = zip(*self.k1_history)
            
            # Create subplots
            fig, (ax1) = plt.subplots(1, 1, figsize=(15, 10))
            
            # Plot K_I values
            ax1.plot(epochs, k1_values_J, 'b-', linewidth=2, label='K_I from J-Integral', marker='o', markersize=3)
            ax1.plot(epochs, k2_values_J, 'r-', linewidth=2, label='K_II from J-Integral', marker='o', markersize=3)
            ax1.axhline(y=self.K1_true, color='g', linestyle='--', linewidth=2, label='True K_I')
            ax1.axhline(y=self.K2_true, color='r', linestyle='--', linewidth=2, label='True K_II')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('K_I and K_II')
            ax1.set_title('K_I and K_II Parameter Training Process')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
                    
            plt.tight_layout()
            plt.savefig(name + '_k12_training_curve.png', dpi=300, bbox_inches='tight')
            print(f"K_I and K_II training curve with energy components saved to: {name}_k12_training_curve.png")


E=1e3 ; nu = 0.3
fy=100.0

kappa = (3-4*nu)
mu = E/(2*(1+nu))

y_crackTip = 0.0

x_crackCenter = 0.0
y_crackCenter = 0.0
beta =    torch.pi/12*3 # torch.pi/12

a = 0.5
b = 1.0
h = 1.0
q = 1 
point_num = 100
epoch_num = 100
x_crackTip = a
model_name = f'../result/crack_mix/crack_XDEM/a_{a:.1f}/crack'
if not os.path.exists(model_name):
    os.makedirs(model_name)
crack_embedding = LineCrackEmbedding([np.cos(beta)*(x_crackCenter-x_crackTip), np.sin(beta)*(x_crackCenter-x_crackTip)],
                                            [np.cos(beta)*(x_crackCenter+x_crackTip), np.sin(beta)*(x_crackCenter+x_crackTip)],
                                            tip = 'both')


net = extendAxisNet(
        net = AxisScalar2D(
            stack_net(input=3,output=2,activation=nn.Tanh,width=30,depth=4),
            A=torch.tensor([1.0/b,1.0/h,1.0]),
            B=torch.tensor([0.0,0.0,0.0])
            ),
        extendAxis= crack_embedding)
pinn = Plate(net,fy=fy,a=a,b=b,h=h,q=q, beta=beta)

pinn.add_BCPoints()

pinn.setMaterial(E=E , nu = nu,type='plane strain')

pinn.set_loss_func(losses=[pinn.Energy_loss,
                                    ],
                            weights=[1.0]
                                    )


pinn.set_meshgrid_inner_points(-b,b,point_num,-h,h,point_num)
crack_surface = Geometry.LineSegement.init_theta([np.cos(beta)*(x_crackCenter-x_crackTip), np.sin(beta)*(x_crackCenter-x_crackTip)], beta)
crack_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.35*a),
                                    crack_surface.clamp(dist2=0.3*a))


pinn.train_with_k12_monitoring(path=model_name, patience=30, epochs=epoch_num, lr=0.001, eval_sep=100,
                              crack_embedding=crack_embedding, crack_surface=crack_surface, 
                              kappa=kappa, mu=mu, beta=beta)

pinn.load(path=model_name)

# pinn.readData('reference/mode1crack.txt')

print(pinn.Energy_loss().cpu().detach().numpy())

pinn.plot_k12_training_curve(model_name)

# Plot displacement and stress fields
print("Plotting displacement and stress fields...")

# Create a fine grid for visualization
test_num = 300

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
print(f"Field data arrays saved to: {model_name}_field_data.npz")
# Plot u2 displacement field and Gamma embedding
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_u2, label='u2 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u2 Displacement Field')
plt.axis('equal')
plt.grid(True, alpha=0.3)

# Plot σ22 stress field
plt.subplot(1, 3, 2)
contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sy, label='σ22 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ22 Stress Field')
plt.axis('equal')
plt.grid(True, alpha=0.3)

# Plot Gamma embedding field
plt.subplot(1, 3, 3)
contour_gamma = plt.contourf(X_vis_np, Y_vis_np, gamma_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_gamma, label='Gamma Embedding')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gamma Embedding Field')
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(model_name + '_displacement_stress_fields.pdf', dpi=300, bbox_inches='tight')
# plt.show()
print(f"Displacement and stress fields saved to: {model_name}_displacement_stress_fields.pdf")

# Additional detailed plots
plt.figure(figsize=(15, 8))

# u1 displacement
plt.subplot(2, 3, 1)
contour_u1 = plt.contourf(X_vis_np, Y_vis_np, u_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_u1, label='u1 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u1 Displacement Field')
plt.axis('equal')

# u2 displacement
plt.subplot(2, 3, 2)
contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_u2, label='u2 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u2 Displacement Field')
plt.axis('equal')

# σ11 stress
plt.subplot(2, 3, 3)
contour_sx = plt.contourf(X_vis_np, Y_vis_np, sx_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sx, label='σ11 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ11 Stress Field')
plt.axis('equal')

# σ22 stress
plt.subplot(2, 3, 4)
contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sy, label='σ22 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ22 Stress Field')
plt.axis('equal')

# σ12 stress
plt.subplot(2, 3, 5)
contour_sxy = plt.contourf(X_vis_np, Y_vis_np, sxy_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sxy, label='σ12 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ12 Stress Field')
plt.axis('equal')

# Von Mises stress
plt.subplot(2, 3, 6)
von_mises = np.sqrt(sx_pred_np**2 - sx_pred_np*sy_pred_np + sy_pred_np**2 + 3*sxy_pred_np**2)
contour_vm = plt.contourf(X_vis_np, Y_vis_np, von_mises, levels=20, cmap='RdBu_r')
plt.colorbar(contour_vm, label='Von Mises stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Von Mises Stress Field')
plt.axis('equal')

plt.tight_layout()
plt.savefig(model_name + '_detailed_fields.pdf', dpi=300, bbox_inches='tight')
# plt.show()
print(f"Detailed field plots saved to: {model_name}_detailed_fields.pdf")

# Save Von Mises stress array as well
np.save(model_name + '_von_mises_stress.npy', von_mises)
print(f"Von Mises stress array saved to: {model_name}_von_mises_stress.npy")

# =============================================================================
# Save individual figures as PDF for paper publication
# =============================================================================
print("Saving individual PDF figures for paper publication...")


# 2. u2 Displacement Field
plt.figure(figsize=(5, 4))
contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=300, cmap='RdBu_r')
plt.colorbar(contour_u2, label='u2 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u2 Displacement Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_u2_displacement.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"u2 displacement PDF saved to: {model_name}_u2_displacement.pdf")

# 3. σ22 Stress Field
plt.figure(figsize=(5, 6))
contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sy, label='σ22 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ22 Stress Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_sigma22_stress.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"σ22 stress PDF saved to: {model_name}_sigma22_stress.pdf")

# 4. Gamma Embedding Field 
plt.figure(figsize=(5, 4))
contour_gamma = plt.contourf(X_vis_np, Y_vis_np, gamma_pred_np, levels=300, cmap='RdBu_r')
plt.colorbar(contour_gamma, label='Crack function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Crack function Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_gamma_crack.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"Gamma embedding PDF saved to: {model_name}_gamma_embedding.pdf")

# 5. u1 Displacement Field
plt.figure(figsize=(5, 6))
contour_u1 = plt.contourf(X_vis_np, Y_vis_np, u_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_u1, label='u1 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u1 Displacement Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_u1_displacement.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"u1 displacement PDF saved to: {model_name}_u1_displacement.pdf")

# 6. σ11 Stress Field
plt.figure(figsize=(5, 6))
contour_sx = plt.contourf(X_vis_np, Y_vis_np, sx_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sx, label='σ11 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ11 Stress Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_sigma11_stress.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"σ11 stress PDF saved to: {model_name}_sigma11_stress.pdf")

# 7. σ12 Stress Field
plt.figure(figsize=(5, 6))
contour_sxy = plt.contourf(X_vis_np, Y_vis_np, sxy_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sxy, label='σ12 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ12 Stress Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_sigma12_stress.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"σ12 stress PDF saved to: {model_name}_sigma12_stress.pdf")

# 8. Von Mises Stress Field
plt.figure(figsize=(5, 6))
contour_vm = plt.contourf(X_vis_np, Y_vis_np, von_mises, levels=20, cmap='RdBu_r')
plt.colorbar(contour_vm, label='Von Mises stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Von Mises Stress Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_von_mises_stress.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"Von Mises stress PDF saved to: {model_name}_von_mises_stress.pdf")

# =============================================================================
# Generate mesh point distribution plot
# =============================================================================
print("Generating mesh point distribution plot...")

# Generate mesh points according to the same rules as in the code
# x: point_num points from -b to b
# y: point_num*3 points from -h to h
x_mesh = np.linspace(-b, b, point_num)
y_mesh = np.linspace(-h, h, point_num * 3)
X_mesh, Y_mesh = np.meshgrid(x_mesh, y_mesh, indexing='ij')

# Flatten for scatter plot
mesh_x = X_mesh.flatten()
mesh_y = Y_mesh.flatten()

plt.figure(figsize=(3, 6))
plt.scatter(mesh_x, mesh_y, s=1, alpha=0.6, c='blue', label=f'Inner points ({point_num}×{point_num*3})')
plt.axis('tight')
# Add boundary points
# Upper boundary: y = h
# x_up_boundary = np.linspace(-b, b, point_num)
# y_up_boundary = np.full_like(x_up_boundary, h)
# plt.scatter(x_up_boundary, y_up_boundary, s=3, alpha=0.8, c='red', label='Upper boundary')

# # Lower boundary: y = -h
# x_down_boundary = np.linspace(-b, b, point_num)
# y_down_boundary = np.full_like(x_down_boundary, -h)
# plt.scatter(x_down_boundary, y_down_boundary, s=3, alpha=0.8, c='green', label='Lower boundary')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Mesh Point Distribution')
plt.axis('equal')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(model_name + '_mesh_distribution.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"Mesh distribution PDF saved to: {model_name}_mesh_distribution.pdf")

print("All individual PDF figures saved successfully!")




