import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set random seed
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time

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
import utils.Geometry as Geometry
from SIF import DispExpolation_homo, SIF_K1K2, M_integral


class Plate(PINN2D):
    def __init__(self, model: nn.Module,fy,a,b,h,q=1, beta = 0.0):
        '''x范围0-1.y范围-1到1'''
        super().__init__(model)
        self.fy = fy
        
        K1_esti = fy * np.sqrt(np.pi * a)  * np.cos(beta)**2
        K2_esti = fy * np.sqrt(np.pi * a)  * np.sin(beta) * np.cos(beta)
        self.K_I = nn.Parameter(torch.tensor(K1_esti, device=self.device))
        self.K_II = nn.Parameter(torch.tensor(K2_esti, device=self.device))
        self.a = a
        self.b = b
        self.h = h
        self.q = q
        self.beta = beta
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
        c = torch.cos(torch.tensor(self.beta, device=self.device))
        s = torch.sin(torch.tensor(self.beta, device=self.device))
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
        return (u + u_analytical_left_global * torch.exp(-20*self.b/self.a*r_left**self.q) + u_analytical_right_global * torch.exp(-20*self.b/self.a*r_right**self.q))  * (x + self.b)/(2*self.b) * (self.b - x)/(2*self.b) 

    def hard_v(self, v, x, y):
        # Calculate polar coordinates relative to crack tip
        c = torch.cos(torch.tensor(self.beta, device=self.device))
        s = torch.sin(torch.tensor(self.beta, device=self.device))
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

        return (v + v_analytical_left_global * torch.exp(-20*self.b/self.a*r_left**self.q) + v_analytical_right_global * torch.exp(-20*self.b/self.a*r_right**self.q))  * (y + self.h)/(2*self.h)

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
        return trapz1D(v_up * self.fy, self.x_up)

    def train_with_convergence_tracking(self, epochs=20000, patience=50, path='test', lr=0.001, eval_sep=100,
                                       milestones=[5000, 10000, 15000], crack_embedding=None, 
                                       crack_surface=None, kappa=None, mu=None, beta=0.0,
                                       convergence_threshold=1.0):
        """训练方法，添加收敛跟踪功能"""
        from utils.EarlyStopping import EarlyStopping
        
        self.iter = 0
        self.set_EarlyStopping(patience=patience, verbose=True, path=path)
        self.path = path
        
        # Initialize K_I history record
        self.k1_history = []
        
        # Initialize convergence tracking
        self.converged = False
        self.convergence_epoch = None
        self.convergence_threshold = convergence_threshold
        
        # Initialize timing
        self.start_time = time.time()
        self.convergence_time = None
        
        # Set optimizer (if not already set)
        if not hasattr(self, 'optimizer'):
            self.set_Optimizer(lr)
        
        # Calculate true stress intensity factor - 使用预定义的理论值
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
        angle_deg = int(self.beta * 180 / np.pi + 0.2) # 四舍五入
        self.K1_true = K1_theory_values.get(angle_deg, 0.0)
        self.K2_true = K2_theory_values.get(angle_deg, 0.0)
        
        # 如果理论值为0，则使用原来的公式计算（作为备用）
        if self.K1_true == 0.0:
            normalized_Param = self.fy * np.sqrt(np.pi * self.a)
            self.K1_true = normalized_Param * np.cos(self.beta)**2
        if self.K2_true == 0.0:
            normalized_Param = self.fy * np.sqrt(np.pi * self.a)
            self.K2_true = normalized_Param * np.sin(self.beta) * np.cos(self.beta)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.3)
        
        print(f"Starting training for beta={self.beta*180/np.pi:.1f}°, total {epochs} epochs")
        print(f"Convergence threshold: {convergence_threshold}%")
        print("-" * 80)
        
        crack_tip_xy = [self.a*np.cos(self.beta), self.a*np.sin(self.beta)]
        m_integral = M_integral(self, self.device)
        
        for i in range(epochs):
            self.train_step()
            
            if self.iter % eval_sep == 0:
                self.eval()
                current_loss = self.history[-1][0] if self.history else 0

                # Calculate K1 value through J-integral
                k1_calculated_j = None
                k2_calculated_j = None
                if self.a is not None:
                    radius_J = 0.25
                    k1_calculated_j, k2_calculated_j, MI, MII = m_integral.compute_K_via_interaction_integral(
                                                      crack_tip_xy=crack_tip_xy,
                                                      radius=radius_J,
                                                      beta=self.beta,
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
                
                # Check convergence
                if not self.converged and total_error <= self.convergence_threshold:
                    self.converged = True
                    self.convergence_epoch = self.iter
                    self.convergence_time = time.time() - self.start_time
                    print(f"*** CONVERGED at epoch {self.iter} with total error {total_error:.3f}% ***")
                    print(f"*** Convergence time: {self.convergence_time:.2f} seconds ***")
                
                self.k1_history.append((self.iter, k1_calculated_j, k2_calculated_j, strain_energy, external_work, k1_error, k2_error, total_error))
                
                print(f'Epoch {self.iter:6d} | K_I_J = {k1_calculated_j:.3f} | K_I_Error = {k1_error:.3f}% | K_II_J = {k2_calculated_j:.3f} | K_II_Error = {k2_error:.3f}% | Total_Error = {total_error:.3f}%')

            scheduler.step()
            if (self.EarlyStopping.early_stop):
                print('Early stopping at epoch: '+str(self.iter))
                break          
        
        # Calculate total training time
        total_time = time.time() - self.start_time
        
        # Print convergence statistics
        print(f"\n{'='*80}")
        print("Convergence Statistics:")
        print(f"{'='*80}")
        if self.converged:
            print(f"Converged at epoch: {self.convergence_epoch}")
            print(f"Convergence time: {self.convergence_time:.2f} seconds")
            print(f"Final total error: {self.k1_history[-1][-1]:.3f}%")
        else:
            print(f"Did not converge within {epochs} epochs")
            print(f"Final total error: {self.k1_history[-1][-1]:.3f}%")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"{'='*80}")
        
        self.save_hist(self.path)
        self.save(self.path)
        self.save_k12_history(self.path)

    def save_k12_history(self, name):
        """Save K_I and K_II parameter history record with convergence tracking"""
        if hasattr(self, 'k1_history') and self.k1_history:
            k1_df = pd.DataFrame(self.k1_history, columns=['epoch', 'K_I_J', 'K_II_J', 'Strain_Energy', 'External_Work', 'K1_Error', 'K2_Error', 'Total_Error'])
            k1_df.to_csv(name + '_k1_history.csv', index=False)
            print(f"K_I and K_II history record saved to: {name}_k1_history.csv")
    
    def get_timing_info(self):
        """获取时间统计信息"""
        total_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        return {
            'convergence_time': self.convergence_time,
            'total_time': total_time,
            'converged': self.converged
        }


def run_analysis():
    """运行不同角度和配点数的分析"""
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 全局参数
    global E, nu, fy, kappa, mu
    E = 1e3
    nu = 0.3
    fy = 100.0
    kappa = (3-4*nu)
    mu = E/(2*(1+nu))
    
    # 分析参数
    angles =   [15, 30, 45, 60, 75]  # 角度（度）
    point_nums = [30, 50, 80]  # 配点数

    # 其他固定参数
    a = 0.5
    b = 1.0
    h = 1.0
    q = 1
    epoch_num = 20000  # 增加训练轮数
    convergence_threshold = 5.0  # 放宽收敛阈值（%）
    
    # 存储所有结果
    all_results = []
    convergence_data = []
    
    # 创建结果目录
    base_result_dir = f'../../result/crack_mix/crack_XDEM/a_{a:.1f}/angle_point_analysis'
    if not os.path.exists(base_result_dir):
        os.makedirs(base_result_dir)
    
    print("="*80)
    print("Mixed-Mode Crack Analysis: Different Angles and Point Numbers")
    print("="*80)
    print(f"Angles: {angles}°")
    print(f"Point numbers: {point_nums}")
    print(f"Convergence threshold: {convergence_threshold}%")
    print("="*80)
    
    # 遍历所有角度和配点数组合
    for angle_deg in angles:
        beta = angle_deg * np.pi / 180
        for point_num in point_nums:
            print(f"\n{'='*60}")
            print(f"Processing: Angle = {angle_deg}°, Points = {point_num}")
            print(f"{'='*60}")
            
            # 设置当前配置的参数
            beta = angle_deg * np.pi / 180
            x_crackTip = a
            x_crackCenter = 0.0
            y_crackCenter = 0.0
            
            # 创建模型名称
            model_name = f'{base_result_dir}/angle_{angle_deg}_points_{point_num}'
            if not os.path.exists(model_name):
                os.makedirs(model_name)
            
            # 创建裂纹嵌入
            crack_embedding = LineCrackEmbedding([np.cos(beta)*(x_crackCenter-x_crackTip), np.sin(beta)*(x_crackCenter-x_crackTip)],
                                                [np.cos(beta)*(x_crackCenter+x_crackTip), np.sin(beta)*(x_crackCenter+x_crackTip)],
                                                tip='both')
            
            # 创建神经网络
            net = extendAxisNet(
                net=AxisScalar2D(
                    stack_net(input=3, output=2, activation=nn.Tanh, width=30, depth=4),
                    A=torch.tensor([1.0/b, 1.0/h, 1.0]),
                    B=torch.tensor([0.0, 0.0, 0.0])
                ),
                extendAxis=crack_embedding
            )
            
            # 创建PINN模型
            pinn = Plate(net, fy=fy, a=a, b=b, h=h, q=q, beta=beta)
            pinn.add_BCPoints()
            pinn.setMaterial(E=E, nu=nu, type='plane strain')
            pinn.set_loss_func(losses=[pinn.Energy_loss], weights=[1.0])
            pinn.set_meshgrid_inner_points(-b, b, point_num, -h, h, point_num)
            
            # 创建裂纹表面
            crack_surface = Geometry.LineSegement.init_theta([np.cos(beta)*(x_crackCenter-x_crackTip), np.sin(beta)*(x_crackCenter-x_crackTip)], beta)
            crack_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.35*a), crack_surface.clamp(dist2=0.3*a))
            
            # 训练模型
            pinn.train_with_convergence_tracking(
                path=model_name, 
                patience=50, 
                epochs=epoch_num, 
                lr=0.001, 
                eval_sep=100,
                crack_embedding=crack_embedding, 
                crack_surface=crack_surface, 
                kappa=kappa, 
                mu=mu, 
                beta=beta,
                convergence_threshold=convergence_threshold
            )
            
            # 获取时间信息
            timing_info = pinn.get_timing_info()
            
            # 记录收敛信息
            convergence_info = {
                'angle': angle_deg,
                'point_num': point_num,
                'converged': pinn.converged,
                'convergence_epoch': pinn.convergence_epoch if pinn.converged else None,
                'convergence_time': timing_info['convergence_time'],
                'total_time': timing_info['total_time'],
                'final_k1_predicted': pinn.k1_history[-1][1] if pinn.k1_history else None,  # 最后预测的K1值
                'final_k2_predicted': pinn.k1_history[-1][2] if pinn.k1_history else None,  # 最后预测的K2值
                'final_total_error': pinn.k1_history[-1][-1] if pinn.k1_history else None,
                'final_k1_error': pinn.k1_history[-1][-3] if pinn.k1_history else None,
                'final_k2_error': pinn.k1_history[-1][-2] if pinn.k1_history else None
            }
            convergence_data.append(convergence_info)
            
            # 存储历史数据用于绘图
            all_results.append({
                'angle': angle_deg,
                'point_num': point_num,
                'history': pinn.k1_history,
                'model_name': model_name
            })
            
            print(f"Completed: Angle = {angle_deg}°, Points = {point_num}")
            if pinn.converged:
                print(f"  Converged at epoch: {pinn.convergence_epoch}")
                print(f"  Convergence time: {timing_info['convergence_time']:.2f} seconds")
                print(f"  Final K1 predicted: {pinn.k1_history[-1][1]:.3f}")
                print(f"  Final K2 predicted: {pinn.k1_history[-1][2]:.3f}")
                print(f"  Final total error: {pinn.k1_history[-1][-1]:.3f}%")
            else:
                print(f"  Did not converge, final error: {pinn.k1_history[-1][-1]:.3f}%")
                print(f"  Final K1 predicted: {pinn.k1_history[-1][1]:.3f}")
                print(f"  Final K2 predicted: {pinn.k1_history[-1][2]:.3f}")
            print(f"  Total training time: {timing_info['total_time']:.2f} seconds")
    
    # 保存收敛统计
    convergence_df = pd.DataFrame(convergence_data)
    convergence_df.to_csv(f'{base_result_dir}/convergence_summary.csv', index=False)
    print(f"\nConvergence summary saved to: {base_result_dir}/convergence_summary.csv")
    
    
    # 生成汇总统计图
    plot_convergence_summary(convergence_data, base_result_dir)
    
    # 生成时间统计图
    plot_timing_analysis(convergence_data, base_result_dir)
    
    # 生成K1和K2预测值对比图
    plot_k1_k2_predictions(convergence_data, base_result_dir)
    
    return convergence_data, all_results




def plot_convergence_summary(convergence_data, base_result_dir):
    """绘制收敛汇总统计图"""
    print("\nGenerating convergence summary plots...")
    
    # 转换为DataFrame
    df = pd.DataFrame(convergence_data)
    
    # 创建汇总图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Convergence Analysis Summary', fontsize=16, fontweight='bold')
    
    # 1. 收敛迭代数热力图
    ax1 = axes[0, 0]
    pivot_epochs = df.pivot(index='angle', columns='point_num', values='convergence_epoch')
    
    # 处理NaN值，将NaN替换为0（表示未收敛）
    pivot_epochs_clean = pivot_epochs.fillna(0)
    
    # 确保数据类型为float
    pivot_epochs_clean = pivot_epochs_clean.astype(float)
    
    im1 = ax1.imshow(pivot_epochs_clean.values, cmap='viridis', aspect='auto')
    ax1.set_xticks(range(len(pivot_epochs_clean.columns)))
    ax1.set_xticklabels(pivot_epochs_clean.columns)
    ax1.set_yticks(range(len(pivot_epochs_clean.index)))
    ax1.set_yticklabels(pivot_epochs_clean.index)
    ax1.set_xlabel('Point Number')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Convergence Epochs')
    
    # 添加数值标注
    for i in range(len(pivot_epochs_clean.index)):
        for j in range(len(pivot_epochs_clean.columns)):
            value = pivot_epochs.iloc[i, j]  # 使用原始数据检查NaN
            if pd.notna(value):
                ax1.text(j, i, f'{int(value)}', ha='center', va='center', color='white', fontweight='bold')
            else:
                ax1.text(j, i, 'NC', ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, label='Epochs')
    
    # 2. 最终总误差热力图
    ax2 = axes[0, 1]
    pivot_errors = df.pivot(index='angle', columns='point_num', values='final_total_error')
    
    # 处理NaN值，将NaN替换为最大值+1（表示未收敛）
    max_error = pivot_errors.max().max()
    if pd.notna(max_error):
        pivot_errors_clean = pivot_errors.fillna(max_error + 10)
    else:
        pivot_errors_clean = pivot_errors.fillna(100)  # 默认值
    
    # 确保数据类型为float
    pivot_errors_clean = pivot_errors_clean.astype(float)
    
    im2 = ax2.imshow(pivot_errors_clean.values, cmap='Reds', aspect='auto')
    ax2.set_xticks(range(len(pivot_errors_clean.columns)))
    ax2.set_xticklabels(pivot_errors_clean.columns)
    ax2.set_yticks(range(len(pivot_errors_clean.index)))
    ax2.set_yticklabels(pivot_errors_clean.index)
    ax2.set_xlabel('Point Number')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Final Total Error (%)')
    
    # 添加数值标注
    for i in range(len(pivot_errors_clean.index)):
        for j in range(len(pivot_errors_clean.columns)):
            value = pivot_errors.iloc[i, j]  # 使用原始数据检查NaN
            if pd.notna(value):
                ax2.text(j, i, f'{value:.1f}%', ha='center', va='center', color='white', fontweight='bold')
            else:
                ax2.text(j, i, 'NC', ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, label='Error (%)')
    
    # 3. 收敛成功率
    ax3 = axes[1, 0]
    convergence_rate = df.groupby('angle')['converged'].mean() * 100
    bars = ax3.bar(convergence_rate.index, convergence_rate.values, color='skyblue', alpha=0.7)
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Convergence Rate (%)')
    ax3.set_title('Convergence Rate by Angle')
    ax3.set_ylim(0, 100)
    
    # 添加数值标注
    for bar, rate in zip(bars, convergence_rate.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. 平均收敛迭代数
    ax4 = axes[1, 1]
    # 只计算收敛配置的平均值
    converged_df = df[df['converged'] == True]
    if len(converged_df) > 0:
        avg_epochs = converged_df.groupby('point_num')['convergence_epoch'].mean()
        bars = ax4.bar(avg_epochs.index, avg_epochs.values, color='lightcoral', alpha=0.7)
        ax4.set_xlabel('Point Number')
        ax4.set_ylabel('Average Convergence Epochs')
        ax4.set_title('Average Convergence Epochs by Point Number (Converged Only)')
        
        # 添加数值标注
        for bar, epochs in zip(bars, avg_epochs.values):
            if pd.notna(epochs):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                        f'{epochs:.0f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Converged Configurations', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_xlabel('Point Number')
        ax4.set_ylabel('Average Convergence Epochs')
        ax4.set_title('Average Convergence Epochs by Point Number')
    
    plt.tight_layout()
    plt.savefig(f'{base_result_dir}/convergence_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence summary saved to: {base_result_dir}/convergence_summary.pdf")
    
    # 打印详细统计
    print("\n" + "="*80)
    print("DETAILED CONVERGENCE STATISTICS")
    print("="*80)
    print("\nConvergence Summary Table:")
    print(df.to_string(index=False))
    
    print(f"\nOverall Statistics:")
    print(f"Total configurations: {len(df)}")
    print(f"Converged configurations: {df['converged'].sum()}")
    print(f"Overall convergence rate: {df['converged'].mean()*100:.1f}%")
    
    # 安全计算平均收敛迭代数
    converged_epochs = df[df['converged'] == True]['convergence_epoch']
    if len(converged_epochs) > 0:
        avg_conv_epoch = converged_epochs.mean()
        print(f"Average convergence epoch (converged only): {avg_conv_epoch:.0f}")
    else:
        print("Average convergence epoch (converged only): N/A (no converged configurations)")
    
    # 安全计算平均最终误差
    final_errors = df['final_total_error'].dropna()
    if len(final_errors) > 0:
        avg_final_error = final_errors.mean()
        print(f"Average final error: {avg_final_error:.2f}%")
    else:
        print("Average final error: N/A (no valid error data)")


def plot_timing_analysis(convergence_data, base_result_dir):
    """绘制时间分析图"""
    print("\nGenerating timing analysis plots...")
    
    # 转换为DataFrame
    df = pd.DataFrame(convergence_data)
    
    # 创建时间分析图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Time Analysis', fontsize=16, fontweight='bold')
    
    # 1. 收敛时间热力图
    ax1 = axes[0, 0]
    pivot_conv_time = df.pivot(index='angle', columns='point_num', values='convergence_time')
    
    # 处理NaN值
    pivot_conv_time_clean = pivot_conv_time.fillna(0)
    pivot_conv_time_clean = pivot_conv_time_clean.astype(float)
    
    im1 = ax1.imshow(pivot_conv_time_clean.values, cmap='plasma', aspect='auto')
    ax1.set_xticks(range(len(pivot_conv_time_clean.columns)))
    ax1.set_xticklabels(pivot_conv_time_clean.columns)
    ax1.set_yticks(range(len(pivot_conv_time_clean.index)))
    ax1.set_yticklabels(pivot_conv_time_clean.index)
    ax1.set_xlabel('Point Number')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Convergence Time (seconds)')
    
    # 添加数值标注
    for i in range(len(pivot_conv_time_clean.index)):
        for j in range(len(pivot_conv_time_clean.columns)):
            value = pivot_conv_time.iloc[i, j]
            if pd.notna(value) and value > 0:
                ax1.text(j, i, f'{value:.1f}s', ha='center', va='center', color='white', fontweight='bold')
            else:
                ax1.text(j, i, 'NC', ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, label='Time (seconds)')
    
    # 2. 总训练时间热力图
    ax2 = axes[0, 1]
    pivot_total_time = df.pivot(index='angle', columns='point_num', values='total_time')
    pivot_total_time_clean = pivot_total_time.astype(float)
    
    im2 = ax2.imshow(pivot_total_time_clean.values, cmap='viridis', aspect='auto')
    ax2.set_xticks(range(len(pivot_total_time_clean.columns)))
    ax2.set_xticklabels(pivot_total_time_clean.columns)
    ax2.set_yticks(range(len(pivot_total_time_clean.index)))
    ax2.set_yticklabels(pivot_total_time_clean.index)
    ax2.set_xlabel('Point Number')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Total Training Time (seconds)')
    
    # 添加数值标注
    for i in range(len(pivot_total_time_clean.index)):
        for j in range(len(pivot_total_time_clean.columns)):
            value = pivot_total_time_clean.iloc[i, j]
            ax2.text(j, i, f'{value:.1f}s', ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, label='Time (seconds)')
    
    # 3. 按角度的平均收敛时间
    ax3 = axes[1, 0]
    conv_times_by_angle = df[df['converged'] == True].groupby('angle')['convergence_time'].mean()
    if len(conv_times_by_angle) > 0:
        bars = ax3.bar(conv_times_by_angle.index, conv_times_by_angle.values, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Angle (degrees)')
        ax3.set_ylabel('Average Convergence Time (seconds)')
        ax3.set_title('Average Convergence Time by Angle (Converged Only)')
        
        # 添加数值标注
        for bar, time_val in zip(bars, conv_times_by_angle.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Converged Configurations', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_xlabel('Angle (degrees)')
        ax3.set_ylabel('Average Convergence Time (seconds)')
        ax3.set_title('Average Convergence Time by Angle')
    
    # 4. 按配点数的平均训练时间
    ax4 = axes[1, 1]
    total_times_by_points = df.groupby('point_num')['total_time'].mean()
    bars = ax4.bar(total_times_by_points.index, total_times_by_points.values, color='orange', alpha=0.7)
    ax4.set_xlabel('Point Number')
    ax4.set_ylabel('Average Total Training Time (seconds)')
    ax4.set_title('Average Total Training Time by Point Number')
    
    # 添加数值标注
    for bar, time_val in zip(bars, total_times_by_points.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{base_result_dir}/timing_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Timing analysis saved to: {base_result_dir}/timing_analysis.pdf")
    
    # 打印时间统计
    print("\n" + "="*80)
    print("TIMING STATISTICS")
    print("="*80)
    
    # 收敛时间统计
    converged_df = df[df['converged'] == True]
    if len(converged_df) > 0:
        print(f"\nConvergence Time Statistics (for {len(converged_df)} converged configurations):")
        print(f"Average convergence time: {converged_df['convergence_time'].mean():.2f} seconds")
        print(f"Minimum convergence time: {converged_df['convergence_time'].min():.2f} seconds")
        print(f"Maximum convergence time: {converged_df['convergence_time'].max():.2f} seconds")
        print(f"Standard deviation: {converged_df['convergence_time'].std():.2f} seconds")
        
        # 按角度统计
        print(f"\nConvergence time by angle:")
        for angle in sorted(converged_df['angle'].unique()):
            angle_data = converged_df[converged_df['angle'] == angle]
            avg_time = angle_data['convergence_time'].mean()
            print(f"  Angle {angle}°: {avg_time:.2f} seconds (avg)")
        
        # 按配点数统计
        print(f"\nConvergence time by point number:")
        for point_num in sorted(converged_df['point_num'].unique()):
            point_data = converged_df[converged_df['point_num'] == point_num]
            avg_time = point_data['convergence_time'].mean()
            print(f"  {point_num} points: {avg_time:.2f} seconds (avg)")
    else:
        print("\nNo converged configurations found.")
    
    # 总训练时间统计
    print(f"\nTotal Training Time Statistics:")
    print(f"Average total time: {df['total_time'].mean():.2f} seconds")
    print(f"Minimum total time: {df['total_time'].min():.2f} seconds")
    print(f"Maximum total time: {df['total_time'].max():.2f} seconds")
    print(f"Standard deviation: {df['total_time'].std():.2f} seconds")
    
    # 效率分析
    if len(converged_df) > 0:
        print(f"\nEfficiency Analysis:")
        efficiency = (converged_df['convergence_time'] / converged_df['total_time'] * 100).mean()
        print(f"Average convergence efficiency: {efficiency:.1f}% (convergence_time/total_time)")
    
    print("="*80)


def plot_k1_k2_predictions(convergence_data, base_result_dir):
    """绘制K1和K2预测值对比图"""
    print("\nGenerating K1 and K2 prediction comparison plots...")
    
    # 转换为DataFrame
    df = pd.DataFrame(convergence_data)
    
    # 理论值
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
    
    # 创建K1和K2预测值对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('K1 and K2 Prediction Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. K1预测值热力图
    ax1 = axes[0, 0]
    pivot_k1_pred = df.pivot(index='angle', columns='point_num', values='final_k1_predicted')
    pivot_k1_pred_clean = pivot_k1_pred.astype(float)
    
    im1 = ax1.imshow(pivot_k1_pred_clean.values, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(len(pivot_k1_pred_clean.columns)))
    ax1.set_xticklabels(pivot_k1_pred_clean.columns)
    ax1.set_yticks(range(len(pivot_k1_pred_clean.index)))
    ax1.set_yticklabels(pivot_k1_pred_clean.index)
    ax1.set_xlabel('Point Number')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Final K1 Predictions')
    
    # 添加数值标注
    for i in range(len(pivot_k1_pred_clean.index)):
        for j in range(len(pivot_k1_pred_clean.columns)):
            value = pivot_k1_pred_clean.iloc[i, j]
            if pd.notna(value):
                ax1.text(j, i, f'{value:.1f}', ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, label='K1 (MPa√mm)')
    
    # 2. K2预测值热力图
    ax2 = axes[0, 1]
    pivot_k2_pred = df.pivot(index='angle', columns='point_num', values='final_k2_predicted')
    pivot_k2_pred_clean = pivot_k2_pred.astype(float)
    
    im2 = ax2.imshow(pivot_k2_pred_clean.values, cmap='Reds', aspect='auto')
    ax2.set_xticks(range(len(pivot_k2_pred_clean.columns)))
    ax2.set_xticklabels(pivot_k2_pred_clean.columns)
    ax2.set_yticks(range(len(pivot_k2_pred_clean.index)))
    ax2.set_yticklabels(pivot_k2_pred_clean.index)
    ax2.set_xlabel('Point Number')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Final K2 Predictions')
    
    # 添加数值标注
    for i in range(len(pivot_k2_pred_clean.index)):
        for j in range(len(pivot_k2_pred_clean.columns)):
            value = pivot_k2_pred_clean.iloc[i, j]
            if pd.notna(value):
                ax2.text(j, i, f'{value:.1f}', ha='center', va='center', color='white', fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, label='K2 (MPa√mm)')
    
    # 3. K1预测值 vs 理论值对比
    ax3 = axes[1, 0]
    angles = sorted(df['angle'].unique())
    point_nums = sorted(df['point_num'].unique())
    
    # 为每个角度绘制K1预测值
    for i, angle in enumerate(angles):
        angle_data = df[df['angle'] == angle]
        k1_theory = K1_theory_values.get(angle, 0)
        
        # 绘制理论值线
        ax3.axhline(y=k1_theory, color=f'C{i}', linestyle='--', alpha=0.7, 
                   label=f'K1 Theory (β={angle}°)' if i == 0 else "")
        
        # 绘制预测值
        for j, point_num in enumerate(point_nums):
            point_data = angle_data[angle_data['point_num'] == point_num]
            if len(point_data) > 0 and pd.notna(point_data['final_k1_predicted'].iloc[0]):
                k1_pred = point_data['final_k1_predicted'].iloc[0]
                ax3.scatter(point_num, k1_pred, color=f'C{i}', s=100, alpha=0.8, 
                           marker='o' if j == 0 else 's' if j == 1 else '^')
    
    ax3.set_xlabel('Point Number')
    ax3.set_ylabel('K1 (MPa√mm)')
    ax3.set_title('K1 Predictions vs Theory')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. K2预测值 vs 理论值对比
    ax4 = axes[1, 1]
    
    # 为每个角度绘制K2预测值
    for i, angle in enumerate(angles):
        angle_data = df[df['angle'] == angle]
        k2_theory = K2_theory_values.get(angle, 0)
        
        # 绘制理论值线
        ax4.axhline(y=k2_theory, color=f'C{i}', linestyle='--', alpha=0.7, 
                   label=f'K2 Theory (β={angle}°)' if i == 0 else "")
        
        # 绘制预测值
        for j, point_num in enumerate(point_nums):
            point_data = angle_data[angle_data['point_num'] == point_num]
            if len(point_data) > 0 and pd.notna(point_data['final_k2_predicted'].iloc[0]):
                k2_pred = point_data['final_k2_predicted'].iloc[0]
                ax4.scatter(point_num, k2_pred, color=f'C{i}', s=100, alpha=0.8, 
                           marker='o' if j == 0 else 's' if j == 1 else '^')
    
    ax4.set_xlabel('Point Number')
    ax4.set_ylabel('K2 (MPa√mm)')
    ax4.set_title('K2 Predictions vs Theory')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{base_result_dir}/k1_k2_predictions.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"K1 and K2 predictions comparison saved to: {base_result_dir}/k1_k2_predictions.pdf")
    
    # 打印K1和K2预测值统计
    print("\n" + "="*80)
    print("K1 AND K2 PREDICTION STATISTICS")
    print("="*80)
    
    print("\nK1 Prediction Statistics:")
    k1_predictions = df['final_k1_predicted'].dropna()
    if len(k1_predictions) > 0:
        print(f"Average K1 prediction: {k1_predictions.mean():.2f} MPa√mm")
        print(f"Minimum K1 prediction: {k1_predictions.min():.2f} MPa√mm")
        print(f"Maximum K1 prediction: {k1_predictions.max():.2f} MPa√mm")
        print(f"Standard deviation: {k1_predictions.std():.2f} MPa√mm")
    
    print("\nK2 Prediction Statistics:")
    k2_predictions = df['final_k2_predicted'].dropna()
    if len(k2_predictions) > 0:
        print(f"Average K2 prediction: {k2_predictions.mean():.2f} MPa√mm")
        print(f"Minimum K2 prediction: {k2_predictions.min():.2f} MPa√mm")
        print(f"Maximum K2 prediction: {k2_predictions.max():.2f} MPa√mm")
        print(f"Standard deviation: {k2_predictions.std():.2f} MPa√mm")
    
    # 按角度统计预测值
    print(f"\nPredictions by angle:")
    for angle in sorted(df['angle'].unique()):
        angle_data = df[df['angle'] == angle]
        k1_theory = K1_theory_values.get(angle, 0)
        k2_theory = K2_theory_values.get(angle, 0)
        
        k1_preds = angle_data['final_k1_predicted'].dropna()
        k2_preds = angle_data['final_k2_predicted'].dropna()
        
        if len(k1_preds) > 0 and len(k2_preds) > 0:
            avg_k1 = k1_preds.mean()
            avg_k2 = k2_preds.mean()
            k1_error = abs(avg_k1 - k1_theory) / k1_theory * 100 if k1_theory != 0 else 0
            k2_error = abs(avg_k2 - k2_theory) / k2_theory * 100 if k2_theory != 0 else 0
            
            print(f"  Angle {angle}°:")
            print(f"    K1: Theory={k1_theory:.1f}, Avg_Pred={avg_k1:.1f}, Error={k1_error:.1f}%")
            print(f"    K2: Theory={k2_theory:.1f}, Avg_Pred={avg_k2:.1f}, Error={k2_error:.1f}%")
    
    print("="*80)


if __name__ == "__main__":
    print("Starting Mixed-Mode Crack Analysis with Different Angles and Point Numbers")
    print("="*80)
    
    # 运行分析
    convergence_data, all_results = run_analysis()
    
    print("\n" + "="*80)
    print("Analysis completed successfully!")
    print("="*80)
    print("Generated files:")
    print("1. convergence_summary.csv - Detailed convergence statistics with timing and K1/K2 predictions")
    print("2. convergence_curves_by_angle.pdf - Convergence curves grouped by angle")
    print("3. convergence_curves_by_points.pdf - Convergence curves grouped by point number")
    print("4. convergence_summary.pdf - Summary statistics and heatmaps")
    print("5. timing_analysis.pdf - Training time analysis and statistics")
    print("6. k1_k2_predictions.pdf - K1 and K2 prediction comparison analysis")
    print("7. Individual result folders for each angle-point combination")
