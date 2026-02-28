import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
from utils.NN import stack_net, AxisScalar2D
from utils.Integral import trapz1D
from Embedding import LineCrackEmbedding, extendAxisNet
import Embedding
import matplotlib.pyplot as plt
import utils.Geometry as Geometry
from SIF import DispExpolation_homo, SIF_K1K2, M_integral
import pandas as pd


class Plate(PINN2D):
    def __init__(self, model: nn.Module, fy, a, b, h, q=1, beta=0.0):
        '''x范围0-1.y范围-1到1'''
        super().__init__(model)
        self.fy = fy
        
        # 初始化 K_I 和 K_II 参数
        K1_esti = fy * np.sqrt(np.pi * a) * np.cos(beta)**2
        K2_esti = fy * np.sqrt(np.pi * a) * np.sin(beta) * np.cos(beta)
        self.K_I = nn.Parameter(torch.tensor(K1_esti, device=self.device))
        self.K_II = nn.Parameter(torch.tensor(K2_esti, device=self.device))
        self.a = a
        self.b = b
        self.h = h
        self.q = q
        self.beta = beta
        
    def set_Optimizer(self, lr, k_i_lr=None):
        """重写优化器设置方法，为K_I和K_II参数设置不同的学习率"""
        model_params = list(self.model.parameters())
        k_i_params = [self.K_I, self.K_II]
        
        if k_i_lr is None:
            k_i_lr = lr * 10

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
        return (u + u_analytical_left_global * torch.exp(-20*self.b/self.a*r_left**self.q) + u_analytical_right_global * torch.exp(-20*self.b/self.a*r_right**self.q))  * (x + self.b)/(2*self.b) * (self.b - x)/(2*self.b) 
        # return u_analytical_right_global # * torch.exp(-self.b/self.a*r_left**self.q) # + u_analytical_right * torch.exp(-self.b/self.a*r_right**self.q)
        # return (u + u_analytical_left_global * torch.exp(-self.b/self.a*r_left**self.q) + u_analytical_right_global * torch.exp(-self.b/self.a*r_right**self.q)) * (x + self.b)/(2*self.b) * (self.b - x)/(2*self.b) 
        # return (u) # * (x + self.b)/(2*self.b) * (self.b - x)/(2*self.b) 
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

        
        return (v + v_analytical_left_global * torch.exp(-20*self.b/self.a*r_left**self.q) + v_analytical_right_global * torch.exp(-20*self.b/self.a*r_right**self.q))  * (y + self.h)/(2*self.h)
        # return v_analytical_right_global # * torch.exp(-self.b/self.a*r_left**self.q) # + v_analytical_right * torch.exp(-self.b/self.a*r_right**self.q)
        # return v_analytical_left_global * torch.exp(-2*r_left**2)
        # return  v  * (y + self.h)/(2*self.h)

    def add_BCPoints(self, num=[128]):
        x_up, y_up = genMeshNodes2D(-self.b, self.b, num[0], self.h, self.h, 1)
        x_down, y_down = genMeshNodes2D(-self.b, self.b, num[0], -self.h, -self.h, 1)
        self.x_up, self.y_up, self.xy_up = self._set_points(x_up, y_up)
        self.x_down, self.y_down, self.xy_down = self._set_points(x_down, y_down)
        self.up_zero = torch.zeros_like(self.x_up)
        self.down_zero = torch.zeros_like(self.x_down)

    def E_ext(self) -> torch.Tensor:
        u_up, v_up = self.pred_uv(self.xy_up)
        return trapz1D(v_up * self.fy, self.x_up)

    def train_with_k12_monitoring(self, epochs=1000, patience=30, path='test', lr=0.001, eval_sep=100,
                                 milestones=[10000, 15000], crack_embedding=None, 
                                 crack_surface=None, kappa=None, mu=None, beta=0.0):
        from utils.EarlyStopping import EarlyStopping
        
        self.iter = 0
        self.set_EarlyStopping(patience=patience, verbose=True, path=path)
        self.path = path
        
        self.k1_history = []
        
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
        angle_deg = int(beta * 180 / np.pi+0.2) # 四舍五入
        self.K1_true = K1_theory_values.get(angle_deg, 0.0)
        self.K2_true = K2_theory_values.get(angle_deg, 0.0)
        
        # 如果理论值为0，则使用原来的公式计算（作为备用）
        if self.K1_true == 0.0:
            normalized_Param = self.fy * np.sqrt(np.pi * self.a)
            self.K1_trradius_J = (self.a+self.b)/np.cos(beta)**2
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

                # Calculate K1 and K2 values through J-integral
                k1_calculated_j = None
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

                strain_energy = self.E_int().cpu().detach().numpy()
                external_work = self.E_ext().cpu().detach().numpy()
                
                # Calculate errors for display
                k1_error = abs(k1_calculated_j - self.K1_true) / self.K1_true * 100 if k1_calculated_j is not None else 0
                k2_error = abs(k2_calculated_j - self.K2_true) / self.K2_true * 100 if k2_calculated_j is not None and self.K2_true != 0 else 0
                
                self.k1_history.append((self.iter, k1_calculated_j,  k2_calculated_j, strain_energy, external_work))
                print(f'Epoch {self.iter:6d} | K_I_J = {k1_calculated_j:.3f} | K_I_NN = {self.K_I.cpu().detach().numpy():.3f} | K_I_True = {self.K1_true:.3f} | K_I_Error = {k1_error:.3f}% | \
                | K_II_J = {k2_calculated_j:.3f} | K_II_NN = {self.K_II.cpu().detach().numpy():.3f} | K_II_True = {self.K2_true:.3f} | K_II_Error = {k2_error:.3f}% \
                | Strain_Energy = {strain_energy:.3e} | External_Work = {external_work:.3e} | Total_Loss = {current_loss:.3e}')

            scheduler.step()
            if (self.EarlyStopping.early_stop):
                print('end epoch:'+str(self.iter))
                break          
        
        self.save_hist(self.path)
        self.save(self.path)
        return self.k1_history

    def save_k12_history(self, name):
        if hasattr(self, 'k1_history') and self.k1_history:
            k1_df = pd.DataFrame(self.k1_history, columns=['epoch', 'K_I_J', 'K_II_J', 'Strain_Energy', 'External_Work'])
            k1_df.to_csv(name + '_k1_history.csv', index=False)
            print(f"K_I and K_II history record saved to: {name}_k1_history.csv")


# 主程序
if __name__ == "__main__":
    # 材料参数
    E = 1e3
    nu = 0.3
    kappa = (3-4*nu)
    mu = E/(2*(1+nu))

    # 几何参数
    fy = 100.0
    a = 0.5
    b = 1.0
    h = 1.0
    q = 1
    point_num = 100
    epoch_num = 20000
    test_num = 300
    # 测试不同的beta角度
    beta_angles =  [15,30,45,60,75]  # 度
    beta_radians = [np.pi * angle / 180 for angle in beta_angles]

    # 存储所有结果
    all_results = {}
    
    print("="*80)
    print("测试不同beta角度的混合模式裂纹")
    print("="*80)
    print(f"材料参数: E = {E}, nu = {nu}")
    print(f"几何参数: a = {a}, b = {b}, h = {h}")
    print(f"载荷: fy = {fy}")
    print(f"测试角度: {beta_angles} 度")
    print("="*80)

    for i, (angle, beta) in enumerate(zip(beta_angles, beta_radians)):
        print(f"\n{'='*60}")
        print(f"测试角度 {i+1}/{len(beta_angles)}: {angle}° (beta = {beta:.3f} rad)")
        print(f"{'='*60}")
        
        # 创建模型名称
        model_name = f'../../result/crack_mix/crack_XDEM/a_{a:.1f}/different_beta/beta_{angle}/crack'
        if not os.path.exists(model_name):
            os.makedirs(model_name)
        
        # 创建裂纹嵌入
        x_crackCenter = 0.0
        y_crackCenter = 0.0
        x_crackTip = a
        crack_embedding = LineCrackEmbedding(
            [np.cos(beta)*(x_crackCenter-x_crackTip), np.sin(beta)*(x_crackCenter-x_crackTip)],
            [np.cos(beta)*(x_crackCenter+x_crackTip), np.sin(beta)*(x_crackCenter+x_crackTip)],
            tip='both'
        )

        # 创建网络
        net = extendAxisNet(
            net=AxisScalar2D(
                stack_net(input=3, output=2, activation=nn.Tanh, width=30, depth=4),
                A=torch.tensor([1.0/b, 1.0/h, 1.0]),
                B=torch.tensor([0.0, 0.0, 0.0])
            ),
            extendAxis=crack_embedding
        )
        
        # 创建Plate对象
        pinn = Plate(net, fy=fy, a=a, b=b, h=h, q=q, beta=beta)
        
        # 添加边界点
        pinn.add_BCPoints()
        
        # 设置材料参数
        pinn.setMaterial(E=E, nu=nu, type='plane strain')
        
        # 设置损失函数
        pinn.set_loss_func(losses=[pinn.Energy_loss], weights=[1.0])
        
        # 设置内部点
        pinn.set_meshgrid_inner_points(-b, b, point_num, -h, h, point_num)
        
        # 创建裂纹表面
        crack_surface = Geometry.LineSegement.init_theta(
            [np.cos(beta)*(x_crackCenter-x_crackTip), np.sin(beta)*(x_crackCenter-x_crackTip)], beta)
        crack_surface = Geometry.LineSegement(
            crack_surface.clamp(dist2=0.35*a),
            crack_surface.clamp(dist2=0.3*a)
        )

        # 训练
        history = pinn.train_with_k12_monitoring(
            path=model_name, patience=20, epochs=epoch_num, lr=0.001, eval_sep=100,
            crack_embedding=crack_embedding, crack_surface=crack_surface, 
            kappa=kappa, mu=mu, beta=beta
        )

        # 保存结果
        pinn.save_k12_history(model_name)
        
        # 计算理论值 - 使用预定义的理论值列表
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
        K1_theory = K1_theory_values[angle]
        K2_theory = K2_theory_values[angle]
        
        # 如果理论值为0，则使用原来的公式计算（作为备用）
        if K1_theory == 0.0:
            K1_theory = fy * np.sqrt(np.pi * a) * np.cos(beta)**2
        if K2_theory == 0.0:
            K2_theory = fy * np.sqrt(np.pi * a) * np.sin(beta) * np.cos(beta)
        
        # 使用最后一个迭代步的结果（更客观）
        if history:
            # 获取最后一个epoch的结果
            last_epoch, last_K1_J, last_K2_J, last_strain_energy, last_external_work = history[-1]
            K1_error = abs(last_K1_J - K1_theory) / K1_theory * 100 if last_K1_J is not None else 0
            K2_error = abs(last_K2_J - K2_theory) / K2_theory * 100 if last_K2_J is not None and K2_theory != 0 else 0
            total_error = K1_error + K2_error
            
            print(f"  最后结果来自第 {last_epoch} 个epoch，总误差: {total_error:.2f}%")
        else:
            last_K1_J = last_K2_J = last_strain_energy = last_external_work = 0
            K1_error = K2_error = total_error = 0

        # 存储结果
        all_results[angle] = {
            'beta_rad': beta,
            'K1_theory': K1_theory,
            'K2_theory': K2_theory,
            'K1_calculated': last_K1_J,
            'K2_calculated': last_K2_J,
            'K1_error_%': K1_error,
            'K2_error_%': K2_error,
            'Strain_Energy': last_strain_energy,
            'External_Work': last_external_work,
            'Total_Loss': last_strain_energy - last_external_work,
            'Last_Epoch': last_epoch if history else 0,
            'Total_Error_%': total_error if history else 0
        }

        print(f"\n角度 {angle}° 结果:")
        print(f"  理论值: K1 = {K1_theory:.3f}, K2 = {K2_theory:.3f}")
        print(f"  计算值: K1 = {last_K1_J:.3f}, K2 = {last_K2_J:.3f}")
        print(f"  误差:   K1 = {K1_error:.1f}%, K2 = {K2_error:.1f}%")
        print(f"  应变能: {last_strain_energy:.3e}")
        print(f"  外力功: {last_external_work:.3e}")

        # 生成云图可视化
        print(f"  生成角度 {angle}° 的云图...")
        
        # 加载训练好的模型
        pinn.load(path=model_name)
        
        # Create a fine grid for visualization
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
        contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=20, cmap='RdBu_r')
        plt.colorbar(contour_u2, label='u2 displacement (mm)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'u2 Displacement Field (β={angle}°)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

        # Plot σ22 stress field
        plt.subplot(1, 3, 2)
        contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=20, cmap='RdBu_r')
        plt.colorbar(contour_sy, label='σ22 stress (MPa)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'σ22 Stress Field (β={angle}°)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

        # Plot Gamma embedding field
        plt.subplot(1, 3, 3)
        contour_gamma = plt.contourf(X_vis_np, Y_vis_np, gamma_pred_np, levels=20, cmap='RdBu_r')
        plt.colorbar(contour_gamma, label='Gamma Embedding')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Gamma Embedding Field (β={angle}°)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(model_name + '_displacement_stress_fields.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Displacement and stress fields saved to: {model_name}_displacement_stress_fields.png")

        # Additional detailed plots
        plt.figure(figsize=(15, 8))

        # u1 displacement
        plt.subplot(2, 3, 1)
        contour_u1 = plt.contourf(X_vis_np, Y_vis_np, u_pred_np, levels=20, cmap='RdBu_r')
        plt.colorbar(contour_u1, label='u1 displacement (mm)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'u1 Displacement Field (β={angle}°)')
        plt.axis('equal')

        # u2 displacement
        plt.subplot(2, 3, 2)
        contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=20, cmap='RdBu_r')
        plt.colorbar(contour_u2, label='u2 displacement (mm)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'u2 Displacement Field (β={angle}°)')
        plt.axis('equal')

        # σ11 stress
        plt.subplot(2, 3, 3)
        contour_sx = plt.contourf(X_vis_np, Y_vis_np, sx_pred_np, levels=20, cmap='RdBu_r')
        plt.colorbar(contour_sx, label='σ11 stress (MPa)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'σ11 Stress Field (β={angle}°)')
        plt.axis('equal')

        # σ22 stress
        plt.subplot(2, 3, 4)
        contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=20, cmap='RdBu_r')
        plt.colorbar(contour_sy, label='σ22 stress (MPa)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'σ22 Stress Field (β={angle}°)')
        plt.axis('equal')

        # σ12 stress
        plt.subplot(2, 3, 5)
        contour_sxy = plt.contourf(X_vis_np, Y_vis_np, sxy_pred_np, levels=20, cmap='RdBu_r')
        plt.colorbar(contour_sxy, label='σ12 stress (MPa)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'σ12 Stress Field (β={angle}°)')
        plt.axis('equal')

        # Von Mises stress
        plt.subplot(2, 3, 6)
        von_mises = np.sqrt(sx_pred_np**2 - sx_pred_np*sy_pred_np + sy_pred_np**2 + 3*sxy_pred_np**2)
        contour_vm = plt.contourf(X_vis_np, Y_vis_np, von_mises, levels=20, cmap='RdBu_r')
        plt.colorbar(contour_vm, label='Von Mises stress (MPa)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Von Mises Stress Field (β={angle}°)')
        plt.axis('equal')

        plt.tight_layout()
        plt.savefig(model_name + '_detailed_fields.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Detailed field plots saved to: {model_name}_detailed_fields.pdf")

        # Save Von Mises stress array as well
        np.save(model_name + '_von_mises_stress.npy', von_mises)
        print(f"  Von Mises stress array savdf to: {model_name}_von_mises_stress.npy")

    # 创建结果汇总表
    print(f"\n{'='*80}")
    print("结果汇总")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_df = results_df.round(3)
    
    print("\n详细结果表:")
    print(results_df.to_string())
    
    # 保存结果到CSV
    results_df.to_csv(f'../../result/crack_mix/crack_XDEM/a_{a:.1f}/different_beta/beta_comparison_results.csv')
    print(f"\n结果已保存到: ../../result/crack_mix/crack_XDEM/a_{a:.1f}/beta_comparison_results.csv")
    
    # 创建K1和K2预测对比图 - 四条曲线在一张图上
    plt.figure(figsize=(12, 8))
    
    # 绘制四条曲线：K1和K2的解析值与XDEM预测值对比
    plt.plot(beta_angles, results_df['K1_theory'], 'o-', label='K1 Analytical', linewidth=3, markersize=8, color='blue')
    plt.plot(beta_angles, results_df['K1_calculated'], 's--', label='K1 XDEM Prediction', linewidth=3, markersize=8, color='red')
    plt.plot(beta_angles, results_df['K2_theory'], 'o-', label='K2 Analytical', linewidth=3, markersize=8, color='green')
    plt.plot(beta_angles, results_df['K2_calculated'], 's--', label='K2 XDEM Prediction', linewidth=3, markersize=8, color='orange')
    
    # Add value annotations for K1 Analytical
    for i, (angle, k1_val) in enumerate(zip(beta_angles, results_df['K1_theory'])):
        plt.annotate(f'{k1_val:.1f}', 
                    (angle, k1_val), 
                    textcoords="offset points", 
                    xytext=(-15, 8), 
                    ha='center', 
                    fontsize=9, 
                    color='blue',
                    fontweight='bold')
    
    # Add value annotations for K1 XDEM Prediction
    for i, (angle, k1_val) in enumerate(zip(beta_angles, results_df['K1_calculated'])):
        plt.annotate(f'{k1_val:.1f}', 
                    (angle, k1_val), 
                    textcoords="offset points", 
                    xytext=(15, -15), 
                    ha='center', 
                    fontsize=9, 
                    color='red',
                    fontweight='bold')
    
    # Add value annotations for K2 Analytical
    for i, (angle, k2_val) in enumerate(zip(beta_angles, results_df['K2_theory'])):
        plt.annotate(f'{k2_val:.1f}', 
                    (angle, k2_val), 
                    textcoords="offset points", 
                    xytext=(-15, -20), 
                    ha='center', 
                    fontsize=9, 
                    color='green',
                    fontweight='bold')
    
    # Add value annotations for K2 XDEM Prediction
    for i, (angle, k2_val) in enumerate(zip(beta_angles, results_df['K2_calculated'])):
        plt.annotate(f'{k2_val:.1f}', 
                    (angle, k2_val), 
                    textcoords="offset points", 
                    xytext=(15, 8), 
                    ha='center', 
                    fontsize=9, 
                    color='orange',
                    fontweight='bold')
    plt.xlabel('Beta Angle (degrees)', fontsize=12)
    plt.ylabel('K1 (MPa√mm)', fontsize=12)
    plt.title('K1 Comparison: Analytical vs XDEM Prediction', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=10)
    
    # 设置X轴范围，确保显示左右边界值
    plt.xlim(min(beta_angles) - 2, max(beta_angles) + 2)
    
    # 设置X轴刻度，确保显示所有角度值
    plt.xticks(beta_angles)
    
    plt.tight_layout()
    plt.savefig(f'../../result/crack_mix/crack_XDEM/a_{a:.1f}/different_beta/K1_K2_prediction_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"\nK1和K2预测对比图已保存到: ../../result/crack_mix/crack_XDEM/a_{a:.1f}/K1_K2_prediction_comparison.pdf")
    
    # 创建详细的对比图
    plt.figure(figsize=(15, 10))
    
    # K1对比
    plt.subplot(2, 3, 1)
    plt.plot(beta_angles, results_df['K1_theory'], 'o-', label='K1 Theory', linewidth=2, markersize=8)
    plt.plot(beta_angles, results_df['K1_calculated'], 's-', label='K1 Calculated', linewidth=2, markersize=8)
    
    # Add value annotations for K1 Theory
    for i, (angle, k1_val) in enumerate(zip(beta_angles, results_df['K1_theory'])):
        plt.annotate(f'{k1_val:.1f}', 
                    (angle, k1_val), 
                    textcoords="offset points", 
                    xytext=(-10, 5), 
                    ha='center', 
                    fontsize=8, 
                    color='blue',
                    fontweight='bold')
    
    # Add value annotations for K1 Calculated
    for i, (angle, k1_val) in enumerate(zip(beta_angles, results_df['K1_calculated'])):
        plt.annotate(f'{k1_val:.1f}', 
                    (angle, k1_val), 
                    textcoords="offset points", 
                    xytext=(10, -10), 
                    ha='center', 
                    fontsize=8, 
                    color='red',
                    fontweight='bold')
    
    plt.xlabel('Beta Angle (degrees)')
    plt.ylabel('K1 (MPa√mm)')
    plt.title('K1 Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # K2对比
    plt.subplot(2, 3, 2)
    plt.plot(beta_angles, results_df['K2_theory'], 'o-', label='K2 Theory', linewidth=2, markersize=8)
    plt.plot(beta_angles, results_df['K2_calculated'], 's-', label='K2 Calculated', linewidth=2, markersize=8)
    
    # Add value annotations for K2 Theory
    for i, (angle, k2_val) in enumerate(zip(beta_angles, results_df['K2_theory'])):
        plt.annotate(f'{k2_val:.1f}', 
                    (angle, k2_val), 
                    textcoords="offset points", 
                    xytext=(-10, 5), 
                    ha='center', 
                    fontsize=8, 
                    color='green',
                    fontweight='bold')
    
    # Add value annotations for K2 Calculated
    for i, (angle, k2_val) in enumerate(zip(beta_angles, results_df['K2_calculated'])):
        plt.annotate(f'{k2_val:.1f}', 
                    (angle, k2_val), 
                    textcoords="offset points", 
                    xytext=(10, -10), 
                    ha='center', 
                    fontsize=8, 
                    color='orange',
                    fontweight='bold')
    
    plt.xlabel('Beta Angle (degrees)')
    plt.ylabel('K2 (MPa√mm)')
    plt.title('K2 Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 误差分析
    plt.subplot(2, 3, 3)
    plt.plot(beta_angles, results_df['K1_error_%'], 'o-', label='K1 Error %', linewidth=2, markersize=8)
    plt.plot(beta_angles, results_df['K2_error_%'], 's-', label='K2 Error %', linewidth=2, markersize=8)
    
    # Add value annotations for K1 Error
    for i, (angle, error_val) in enumerate(zip(beta_angles, results_df['K1_error_%'])):
        plt.annotate(f'{error_val:.1f}%', 
                    (angle, error_val), 
                    textcoords="offset points", 
                    xytext=(-10, 5), 
                    ha='center', 
                    fontsize=8, 
                    color='blue',
                    fontweight='bold')
    
    # Add value annotations for K2 Error
    for i, (angle, error_val) in enumerate(zip(beta_angles, results_df['K2_error_%'])):
        plt.annotate(f'{error_val:.1f}%', 
                    (angle, error_val), 
                    textcoords="offset points", 
                    xytext=(10, -10), 
                    ha='center', 
                    fontsize=8, 
                    color='red',
                    fontweight='bold')
    
    plt.xlabel('Beta Angle (degrees)')
    plt.ylabel('Error (%)')
    plt.title('Error Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 能量分析
    plt.subplot(2, 3, 4)
    plt.plot(beta_angles, results_df['Strain_Energy'], 'o-', label='Strain Energy', linewidth=2, markersize=8)
    plt.plot(beta_angles, results_df['External_Work'], 's-', label='External Work', linewidth=2, markersize=8)
    plt.xlabel('Beta Angle (degrees)')
    plt.ylabel('Energy')
    plt.title('Energy Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # K1/K2比值
    plt.subplot(2, 3, 5)
    K1_K2_ratio_theory = results_df['K1_theory'] / results_df['K2_theory']
    K1_K2_ratio_calculated = results_df['K1_calculated'] / results_df['K2_calculated']
    plt.plot(beta_angles, K1_K2_ratio_theory, 'o-', label='K1/K2 Theory', linewidth=2, markersize=8)
    plt.plot(beta_angles, K1_K2_ratio_calculated, 's-', label='K1/K2 Calculated', linewidth=2, markersize=8)
    plt.xlabel('Beta Angle (degrees)')
    plt.ylabel('K1/K2 Ratio')
    plt.title('K1/K2 Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 总损失
    plt.subplot(2, 3, 6)
    plt.plot(beta_angles, results_df['Total_Loss'], 'o-', label='Total Loss', linewidth=2, markersize=8, color='red')
    plt.xlabel('Beta Angle (degrees)')
    plt.ylabel('Total Loss')
    plt.title('Total Loss vs Beta Angle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    last_epochs = [all_results[angle]['Last_Epoch'] for angle in beta_angles]
    plt.bar(beta_angles, last_epochs, color='skyblue', alpha=0.7, edgecolor='navy')
    plt.xlabel('Beta Angle (degrees)', fontsize=12)
    plt.ylabel('Last Epoch', fontsize=12)
    plt.title('Last Epoch for Each Angle', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 总误差分析
    plt.subplot(2, 2, 2)
    total_errors = [all_results[angle]['Total_Error_%'] for angle in beta_angles]
    plt.plot(beta_angles, total_errors, 'o-', linewidth=3, markersize=8, color='red')
    plt.xlabel('Beta Angle (degrees)', fontsize=12)
    plt.ylabel('Total Error (%)', fontsize=12)
    plt.title('Total Error (K1 + K2) for Last Results', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 子图3: K1误差分析
    plt.subplot(2, 2, 3)
    k1_errors = [all_results[angle]['K1_error_%'] for angle in beta_angles]
    plt.plot(beta_angles, k1_errors, 'o-', linewidth=3, markersize=8, color='blue', label='K1 Error')
    plt.xlabel('Beta Angle (degrees)', fontsize=12)
    plt.ylabel('K1 Error (%)', fontsize=12)
    plt.title('K1 Error for Last Results', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 子图4: K2误差分析
    plt.subplot(2, 2, 4)
    k2_errors = [all_results[angle]['K2_error_%'] for angle in beta_angles]
    plt.plot(beta_angles, k2_errors, 'o-', linewidth=3, markersize=8, color='green', label='K2 Error')
    plt.xlabel('Beta Angle (degrees)', fontsize=12)
    plt.ylabel('K2 Error (%)', fontsize=12)
    plt.title('K2 Error for Last Results', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../../result/crack_mix/crack_XDEM/a_{a:.1f}/different_beta/last_results_analysis.png', dpi=300, bbox_inches='tight')
    print(f"最后结果分析图已保存到: ../../result/crack_mix/crack_XDEM/a_{a:.1f}/last_results_analysis.png")
    
    print(f"\n{'='*80}")
    print("所有测试完成！")
    print(f"{'='*80}")
