import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, os, math, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ====== 工程依赖（与你给的工程保持一致）======
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DENNs import PINN2D                          # 你的底座类
from utils.NodesGenerater import genMeshNodes2D   # 边界/内点
from utils.NN import stack_net, AxisScalar2D      # 标量网络+坐标尺度
from utils.get_grad import get_grad
from Embedding import LineCrackEmbedding, extendAxisNet  # 裂纹嵌入、扩展坐标
import utils.Geometry as Geometry
from utils.Integral import trapz1D                # 若需要1D数值积分
# ============================================

# -------- 随机种子 --------
seed = 2025
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# =========================
#       工具函数
# =========================
def cartesian_to_polar(x, y, tip=(0.0, 0.0)):
    """以裂尖为极点的极坐标；theta范围(-pi, pi]，与atan2一致"""
    X = x - tip[0]
    Y = y - tip[1]
    r = torch.sqrt(X*X + Y*Y + 1e-32)   # 避免0
    theta = torch.atan2(Y, X)
    return r, theta

def modeIII_analytical_w(xy, KIII, mu, crack_tip=(0.0, 0.0)):
    """
    解析位移： w = (KIII/mu)*sqrt(2r/pi)*sin(theta/2)
    说明：mu = G (剪切模量)
    """
    x = xy[..., 0]
    y = xy[..., 1]
    r, theta = cartesian_to_polar(x, y, tip=crack_tip)
    w = (KIII / mu) * torch.sqrt(2.0 * r / np.pi) * torch.sin(0.5 * theta)
    return w

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
# =========================
#     Mode III 主类
# =========================
class AntiplanePlate(PINN2D):
    """
    仅1个自由度：w(x,y) = u3
    应变(工程量)：gamma13 = dw/dx, gamma23 = dw/dy
    应力：tau13 = mu*gamma13, tau23 = mu*gamma23
    能量密度：W = 0.5*mu*(wx^2 + wy^2)
    """
    def __init__(self, model: nn.Module, KIII_init, square_half=1.0, a_crack=1.0, points_num=10):
        super().__init__(model)
        self.points_num = points_num
        self.half = square_half               # 几何半长 = 1
        self.a = a_crack                      # 裂纹半长（这里等于1，裂纹[-1,0]）
        self.crack_tip = (0.0, 0.0)           # 裂纹尖端 (0,0)
        # 可训练的 KIII（可选）：也可以固定外推
        self.KIII_init = KIII_init
        self.KIII = nn.Parameter(torch.tensor(float(KIII_init), device=self.device))
        # 材料参数
        self.E = None
        self.nu = None
        self.mu = 1.0   # G
        self.err_history = []   # (iter, L2_w, H1_energy)  —— H1这里按能量密度W的L2相对误差来度量
        # 训练监控
        self.history = []
        self.k_history = []   # (iter, J, KIII_from_J)

    # ====== 材料 ======
    def setMaterial(self, E, nu):
        self.E = float(E); self.nu = float(nu)
        self.mu = self.E/(2.0*(1.0+self.nu))
        return self


    # ====== hard w ======
    def hard_w(self, xy: torch.Tensor):
        """
        约定：模型输出为标量 w(x,y)
        """
        out =  self.model(xy)[0].squeeze(-1)
        r, theta = cartesian_to_polar(xy[..., 0], xy[..., 1], tip=(0.0, 0.0))
        bound_w = modeIII_analytical_w(xy, self.KIII_init, self.mu, crack_tip=(0.0, 0.0)) + (self.half-xy[..., 0])*(self.half+xy[..., 0]) * (self.half-xy[..., 1])*(self.half+xy[..., 1])
        # bound_w = self.net_special(xy).squeeze(-1)
        return bound_w + (out + torch.exp(-r*2)*self.KIII/self.mu*(2*r/3.14159)**0.5*torch.sin(theta/2)) * (self.half-xy[..., 0])*(self.half+xy[..., 0]) * (self.half-xy[..., 1])*(self.half+xy[..., 1])

    # ====== 导数/本构/能量 ======
    def grads(self, w, xy):
        dw_dxy = get_grad(w, xy)
        wx = dw_dxy[..., 0]
        wy = dw_dxy[..., 1]
        return wx, wy

    def constitutive(self, wx, wy):
        tau13 = self.mu * wx
        tau23 = self.mu * wy
        return tau13, tau23

    def energy_density(self, wx, wy):
        return 0.5*self.mu*(wx*wx + wy*wy)


    # ====== 外功（本问题位移边界为主：取0）======
    def E_ext(self) -> torch.Tensor:
        return torch.zeros((), device=self.device)

    # ====== 总能（内部能）======
    def E_int_c3(self) -> torch.Tensor:
        """
        用你工程里 set_meshgrid_inner_points 产生的 self.inner_xy 做能量平均
        如果你工程已有更精确的积分法，替换这里即可。
        """
        x = torch.linspace(-self.half, self.half, self.points_num)
        y = torch.linspace(-self.half, self.half, self.points_num)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1).to(self.device)
        xy.requires_grad_(True)
        # 下面加上hard constraint
        w = self.hard_w(xy)
        wx, wy = self.grads(w, xy)
        W = self.energy_density(wx, wy)
        return W.mean() * (2*self.half)*(2*self.half)  # 乘区域面积，近似积分

    def Energy_loss(self) -> torch.Tensor:
        return self.E_int_c3() - self.E_ext()

    # ====== 边界点与Dirichlet位移 (解析解) ======
    def add_BCPoints(self, n_each=201, KIII_BC=None):
        """
        在四条边采样，并用解析解做Dirichlet损失
        KIII_BC若为None，则使用当前 self.KIII
        """
        L = self.half
        # 上边 y=+L
        x_up, y_up = genMeshNodes2D(-L, L, n_each,  L, L, 1)
        # 下边 y=-L
        x_dn, y_dn = genMeshNodes2D(-L, L, n_each, -L,-L, 1)
        # 左边 x=-L
        x_le, y_le = genMeshNodes2D(-L,-L, 1, -L, L, n_each)
        # 右边 x=+L
        x_ri, y_ri = genMeshNodes2D( L, L, 1, -L, L, n_each)

        def _stack(x, y):
            xy = torch.stack([x, y], dim=-1).to(self.device)
            return xy

        self.xy_up = _stack(x_up, y_up)
        self.xy_dn = _stack(x_dn, y_dn)
        self.xy_le = _stack(x_le, y_le)
        self.xy_ri = _stack(x_ri, y_ri)

        self.bc_all = torch.cat([self.xy_up, self.xy_dn, self.xy_le, self.xy_ri], dim=0)

        if KIII_BC is None:
            KIII_BC = self.KIII
        self.w_bc = modeIII_analytical_w(self.bc_all, KIII=KIII_BC, mu=self.mu, crack_tip=self.crack_tip).detach()


    def BC_loss(self) -> torch.Tensor:
        w_pred = self.hard_w(self.bc_all)
        return torch.mean((w_pred - self.w_bc)**2)

    # ====== J-积分（Mode III 专用）======
    def generate_contour(self, radius=0.25, num_points=720, center=None):
        if center is None: center = self.crack_tip
        theta = torch.linspace(0, 2*np.pi, num_points, device=self.device)
        x = center[0] + radius*torch.cos(theta)
        y = center[1] + radius*torch.sin(theta)
        pts = torch.stack([x, y], dim=-1)
        # 外法向（圆）：n = (cos, sin)
        normals = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        # 弧长权重（均匀角度采样）：dΓ ≈ R * dθ -> 用平均后再* 2πR
        return pts, normals, radius

    def J_integral(self, radius=0.25, num_points=720):
        xy, n, R = self.generate_contour(radius, num_points)
        xy = xy.clone().detach().requires_grad_(True)
        n = n.detach()

        w = self.hard_w(xy)
        wx, wy = self.grads(w, xy)
        tau13, tau23 = self.constitutive(wx, wy)
        W = self.energy_density(wx, wy)

        # integrand = [ W - tau13*wx ] * n_x  - tau23*wx * n_y
        integrand = (W - tau13*wx) * n[:, 0] - (tau23*wx) * n[:, 1]
        J = integrand.mean() * (2*np.pi*R)  # 平均值乘总弧长
        return J

    def KIII_from_J(self, J):
        # Antiplane: J = KIII^2 / (2*mu)
        return torch.sqrt(2.0*self.mu*torch.clamp(J, min=0.0))

    # ====== 训练 ======
    def set_Optimizer(self, lr_net=2e-3, lr_K=1e-2):
        params = [
            {"params": self.model.parameters(), "lr": lr_net},
            {"params": [self.KIII], "lr": lr_K},
        ]
        self.optimizer = torch.optim.Adam(params)

    def set_loss_func(self, weights=(1.0, 1.0)):
        self.weight_energy, self.weight_bc = weights

    def train_loop(self, epochs=20000, eval_every=200, contour_radius=0.2, save_dir="result_modeIII"):
        os.makedirs(save_dir, exist_ok=True)
        if not hasattr(self, "optimizer"):
            self.set_Optimizer()

        for it in range(1, epochs+1):
            self.optimizer.zero_grad()
            loss_energy = self.Energy_loss()
            loss_bc = self.BC_loss()
            loss = self.weight_energy*loss_energy + self.weight_bc*loss_bc
            loss.backward()
            self.optimizer.step()

            if it % eval_every == 0:
                # 计算 J / KIII
                J = self.J_integral(radius=contour_radius, num_points=720)
                KJ = self.KIII_from_J(J)
                self.k_history.append((it, float(J.detach().cpu()), float(KJ.detach().cpu()), float(self.KIII.detach().cpu())))

                # 计算 L2 / H1(以能量密度度量)
                L2_w, H1_energy = self.compute_L2_H1_errors(n=128)
                self.err_history.append((it, L2_w, H1_energy))

                print(f"[{it:6d}] E_int={loss_energy.item():.4e}  BC={loss_bc.item():.4e}  "
                    f"J={J.item():.4e}  KIII(J)={KJ.item():.4f}  KIII(nn)={self.KIII.item():.4f}  "
                    f"L2_w={L2_w:.3e}  H1_energy={H1_energy:.3e}")

        # ===== 保存监控曲线 =====
        import pandas as pd
        # K 历史
        df_k = pd.DataFrame(self.k_history, columns=["iter","J","KIII_from_J","KIII_nn"])
        df_k.to_csv(os.path.join(save_dir, "KIII_history.csv"), index=False)
        # 误差历史
        if len(self.err_history) > 0:
            df_err = pd.DataFrame(self.err_history, columns=["iter","L2_w","H1_energy"])
            df_err.to_csv(os.path.join(save_dir, "error_history.csv"), index=False)

            # 画误差曲线
            iters = [t[0] for t in self.err_history]
            L2s   = [t[1] for t in self.err_history]
            H1s   = [t[2] for t in self.err_history]

            plt.figure(figsize=(5.0, 5.0))
            plt.plot(iters, L2s, label=r"$L^2$ error of $w$")
            plt.plot(iters, H1s, label=r"$H^1$ error (via energy density $W$)")
            plt.xlabel("Iterations")
            plt.ylabel("Relative Error")
            plt.yscale("log")   # 误差更清晰
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "error_curves.png"), dpi=300)
            plt.close()

    def train_special(self, net_special, epochs=1000, train_if = True):
        self.net_special = net_special
        if train_if == False:
            return
        # 训练特解网络，根据边界条件进行训练
        # 首先在边界上生成很多点
        XY = torch.cat([self.xy_up, self.xy_dn, self.xy_le, self.xy_ri], dim=0)

        # 构建优化器
        optimizer = torch.optim.Adam(self.net_special.parameters(), lr=1e-3)
        for i in range(epochs):
            optimizer.zero_grad()
            # 然后计算特解网络的输出
            w = self.net_special(XY).squeeze(-1)
            # 根据解析解确定边界条件是多少
            w_bc = modeIII_analytical_w(XY, self.KIII_init, self.mu, crack_tip=(0.0, 0.0))
            # 计算损失
            loss = torch.mean((w - w_bc)**2)
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            # 打印损失
            print(f"[{i:6d}] loss={loss.item():.4e}")
        save_dir = './special_net'
        torch.save(self.net_special, save_dir)

    # ====== 可视化 ======
    def visualize_fields(self, n=200, save_dir="result_modeIII"):
        """
        可视化：
        - 解析解 w_a, tau13_a, tau23_a 云图
        - XDEM 预测 w, tau13, tau23 云图
        - 绝对误差 |.| 云图
        - 界面 y=0 的奇异应变 eps_{zθ} 1D 对比（预测 vs 解析）
        """
        os.makedirs(save_dir, exist_ok=True)

        # --- 网格 ---
        x = torch.linspace(-self.half, self.half, n, device=self.device)
        y = torch.linspace(-self.half, self.half, n, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1).requires_grad_(True)

        # --- XDEM 预测 ---
        w_pred = self.hard_w(XY)
        wx_pred, wy_pred = self.grads(w_pred, XY)
        tau13_pred, tau23_pred = self.constitutive(wx_pred, wy_pred)

        # --- 解析解（位移 + 自动微分求导得到应力） ---
        w_ana = modeIII_analytical_w(XY, KIII=self.KIII_init, mu=self.mu, crack_tip=(0.0, 0.0))
        wx_ana, wy_ana = self.grads(w_ana, XY)
        tau13_ana, tau23_ana = self.constitutive(wx_ana, wy_ana)

        # --- 误差 ---
        w_err    = torch.abs(w_pred - w_ana)
        tau13_err = torch.abs(tau13_pred - tau13_ana)
        tau23_err = torch.abs(tau23_pred - tau23_ana)

        # --- 转 numpy ---
        def to_np(z): return z.detach().cpu().reshape(n, n).numpy()
        Xnp, Ynp = X.detach().cpu().numpy(), Y.detach().cpu().numpy()

        wP   = to_np(w_pred);      t13P = to_np(tau13_pred);  t23P = to_np(tau23_pred)
        wA   = to_np(w_ana);       t13A = to_np(tau13_ana);   t23A = to_np(tau23_ana)
        wE   = to_np(w_err);       t13E = to_np(tau13_err);   t23E = to_np(tau23_err)

        # ====== 云图：解析 vs 预测 vs 误差 ======
        def imsave(Z, title, fname, cmap='jet'):
            plt.figure(figsize=(6,5))
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.set_xlim(-self.half, self.half); ax.set_ylim(-self.half, self.half)
            c = plt.contourf(Xnp, Ynp, Z, levels=60, cmap=cmap)
            plt.colorbar(c)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname), dpi=300)
            plt.close()

        # 解析
        imsave(wA,   'Analytical w',           'w_field_analytical.png')
        imsave(t13A, 'Analytical tau13',       'tau13_field_analytical.png')
        imsave(t23A, 'Analytical tau23',       'tau23_field_analytical.png')
        # 预测（XDEM）
        imsave(wP,   'XDEM w (prediction)',    'w_field_XDEM.png')
        imsave(t13P, 'XDEM tau13 (prediction)','tau13_field_XDEM.png')
        imsave(t23P, 'XDEM tau23 (prediction)','tau23_field_XDEM.png')
        # 绝对误差
        imsave(wE,   '|w - w_a|',              'w_field_abs_error.png')
        imsave(t13E, '|tau13 - tau13_a|',      'tau13_field_abs_error.png')
        imsave(t23E, '|tau23 - tau23_a|',      'tau23_field_abs_error.png')

        # ====== y=0
        plt.figure(figsize=(5,5))
        r_vals = np.linspace(0.05, 1.0, 50)  # 避开裂尖奇异区 r<0.05
        x_line = torch.tensor(r_vals, dtype=torch.float32, device=self.device)
        y_line = torch.zeros_like(x_line)
        xy_line = torch.stack([x_line, y_line], dim=-1).requires_grad_(True)

        w_line = self.hard_w(xy_line)
        dw_dy = get_grad(w_line, xy_line)[...,1]  # 对y求导
        eps_zθ_pred = dw_dy.detach().cpu().numpy()
        eps_zθ_exact = (self.KIII_init/self.mu*(2/np.pi)**0.5)*0.5*np.sqrt(1.0/(r_vals))

        plt.plot(r_vals, eps_zθ_exact, 'r-', lw=2, label='Exact Solution')
        plt.scatter(r_vals, eps_zθ_pred, color='r', label='XDEM Prediction')

        # 计算下不同的r，J积分计算的KIII是不是准的
        KIII_vals = []
        r_vals_J = np.linspace(0.20, 0.80, 30)
        for r in r_vals_J:
            J = self.J_integral(radius=r, num_points=720)
            KIII = self.KIII_from_J(J)
            KIII_vals.append(KIII.detach().cpu().numpy())
        KIII_vals = np.array(KIII_vals)
        
        
        # K3的精确值是1.0
        plt.plot(r_vals_J, np.ones_like(r_vals_J), 'b-', lw=2, label='Exact Value of KIII')
        plt.scatter(r_vals_J, KIII_vals, color='b', s=10, label='KIII from J Integration')



        plt.xlabel('r (distance from crack tip)')
        # plt.ylabel(r'$\varepsilon_{z\theta}$')
        plt.title('Comparison of Singular Strain along y=0')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "singular_strain_compare.png"), dpi=300)
        plt.close()

    def compute_L2_H1_errors(self, n=128):
        """
        计算：
        - L2_w: 位移 w 的相对 L2 误差
        - H1_energy: 用能量密度 W 的相对 L2 误差来表示的“H1”误差（按你的要求）
        积分用均匀网格的均值近似（相对误差里面积因子抵消）
        """
        # 网格
        x = torch.linspace(-self.half, self.half, n, device=self.device)
        y = torch.linspace(-self.half, self.half, n, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1).requires_grad_(True)

        # 预测
        w_pred = self.hard_w(XY)
        wx_pred, wy_pred = self.grads(w_pred, XY)
        W_pred = self.energy_density(wx_pred, wy_pred)

        # 解析
        w_ana = modeIII_analytical_w(XY, KIII=self.KIII_init, mu=self.mu, crack_tip=(0.0, 0.0))
        wx_ana, wy_ana = self.grads(w_ana, XY)
        W_ana = self.energy_density(wx_ana, wy_ana)

        # 相对 L2 误差（加入极小值防止除零）
        eps = 1e-16
        L2_w = torch.mean((w_pred - w_ana)**2) / (torch.mean(w_ana**2) + eps)
        H1_energy = torch.mean((W_pred - W_ana)**2) / (torch.mean(W_ana**2) + eps)

        return float(L2_w.detach().cpu()), float(H1_energy.detach().cpu())
       
# =========================
#          主程序
# =========================
if __name__ == "__main__":
    # ----- 材料参数 -----

    mu = 1.0

    # ----- 解析KIII（用来给外边界BC和对比）-----
    # 你也可以指定为目标KIII_true，然后让KIII可训练去贴合
    KIII_true = 1.0

    # ----- 几何 -----
    L = 1.0       # 方域半长
    a = 1.0       # 裂纹半长，使裂纹区间[-1,0]
    crack_tip    = (0.0, 0.0)

    # ----- 裂纹嵌入（使网络具备跨裂纹的不连续能力）-----
    crack_embedding = LineCrackEmbedding([-a,0.0],[0.0,0.0], tip='right')
    # 标量网络（只输出 w）
    net_core = AxisScalar2D(
        stack_net(input=3, output=1, activation=nn.Tanh, width=40, depth=5),
        A=torch.tensor([1.0/L, 1.0/L, 1.0]),   # 坐标尺度
        B=torch.tensor([0.0,   0.0,   0.0])
    )
    net = extendAxisNet(net=net_core, extendAxis=crack_embedding)

    # ----- PINN 实例 -----
    pinn = AntiplanePlate(model=net, KIII_init=KIII_true, square_half=L, a_crack=a, points_num=100)

    # # ----- 采样内点（与您的工程接口一致）-----
    # # x: [-L,L] 分 Nx， y: [-L,L] 分 Ny
    # Nx, Ny = 64, 64
    # pinn.set_meshgrid_inner_points(-L, L, Nx, -L, L, Ny)

    # ----- 边界点 & 解析位移BC -----
    pinn.add_BCPoints(n_each=201, KIII_BC=KIII_true)

    # ----- 损失与优化器 -----
    pinn.set_loss_func(weights=(1.0, 10.0))   # 能量:边界 = 1:10，可按需要调整
    pinn.set_Optimizer(lr_net=2e-3, lr_K=5e-3)

    # 构建特解网络
    # net_special = MLP(input_size=2, hidden_size=40, output_size=1).to(pinn.device)
    # net_special = torch.load('./special_net')
    # pinn.train_special(net_special, epochs=100000, train_if = False)

    # ----- 训练 -----
    out_dir = "../../result/crack3/crack_XDEM"
    pinn.train_loop(epochs=1000, eval_every=100, contour_radius=0.2, save_dir=out_dir)

    # ----- 可视化 -----
    pinn.visualize_fields(n=200, save_dir=out_dir)

    # ----- 最终J与KIII -----
    J = pinn.J_integral(radius=0.2, num_points=1440)
    KJ = pinn.KIII_from_J(J)
    print("\n========== SUMMARY ==========")
    print(f"True   KIII = {KIII_true:.6f}")
    print(f"From J KIII = {KJ.item():.6f}")
    print(f"       J    = {J.item():.6e}")

    
