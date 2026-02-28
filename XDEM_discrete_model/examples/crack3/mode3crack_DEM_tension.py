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
        # bound_w = self.net_special(xy).squeeze(-1)
        bound_w = modeIII_analytical_w(xy, self.KIII_init, self.mu, crack_tip=(0.0, 0.0)) + (self.half-xy[..., 0])*(self.half+xy[..., 0]) * (self.half-xy[..., 1])*(self.half+xy[..., 1])
        return bound_w + (out) * (self.half-xy[..., 0])*(self.half+xy[..., 0]) * (self.half-xy[..., 1])*(self.half+xy[..., 1])

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
                J = self.J_integral(radius=contour_radius, num_points=720)
                KJ = self.KIII_from_J(J)
                self.k_history.append((it, float(J.detach().cpu()), float(KJ.detach().cpu()), float(self.KIII.detach().cpu())))
                print(f"[{it:6d}] E_int={loss_energy.item():.4e}  BC={loss_bc.item():.4e}  "
                        f"J={J.item():.4e}  KIII(J)={KJ.item():.4f}  KIII(nn)={self.KIII.item():.4f}")

        # 保存监控曲线
        import pandas as pd
        df = pd.DataFrame(self.k_history, columns=["iter","J","KIII_from_J","KIII_nn"])
        df.to_csv(os.path.join(save_dir, "KIII_history.csv"), index=False)

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
        os.makedirs(save_dir, exist_ok=True)
        x = torch.linspace(-self.half, self.half, n, device=self.device)
        y = torch.linspace(-self.half, self.half, n, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1).requires_grad_(True)

        w = self.hard_w(XY)
        wx, wy = self.grads(w, XY)
        tau13, tau23 = self.constitutive(wx, wy)

        W = self.energy_density(wx, wy)

        def to_np(z): return z.detach().cpu().reshape(n, n).numpy()
        Wnp   = to_np(W)
        wnp   = to_np(w)
        t13np = to_np(tau13)
        t23np = to_np(tau23)

        Xnp, Ynp = X.detach().cpu().numpy(), Y.detach().cpu().numpy()

        # w
        plt.figure(figsize=(6,5))
        plt.gca().set_aspect('equal')
        plt.xlim(-self.half,self.half)
        plt.ylim(-self.half,self.half)
        c = plt.contourf(Xnp, Ynp, wnp, levels=60, cmap='jet')
        plt.colorbar(c,label='w (u3)')
        plt.title('Antiplane displacement w')
        plt.axis('equal'); plt.xlim(-self.half,self.half); plt.ylim(-self.half,self.half)
        plt.tight_layout(); plt.savefig(os.path.join(save_dir,"dis_field.png"),dpi=300)

        # tau13
        plt.figure(figsize=(6,5))
        plt.gca().set_aspect('equal')
        plt.xlim(-self.half,self.half)
        plt.ylim(-self.half,self.half)
        c = plt.contourf(Xnp, Ynp, t13np, levels=60, cmap='jet')
        plt.colorbar(c,label='tau13')
        plt.title('Shear stress tau13')
        plt.axis('equal'); plt.xlim(-self.half,self.half); plt.ylim(-self.half,self.half)
        plt.tight_layout(); plt.savefig(os.path.join(save_dir,"tau13_field.png"),dpi=300)

        # tau23
        plt.figure(figsize=(6,5))
        plt.gca().set_aspect('equal')
        plt.xlim(-self.half,self.half)
        plt.ylim(-self.half,self.half)
        c = plt.contourf(Xnp, Ynp, t23np, levels=60, cmap='jet')
        plt.colorbar(c,label='tau23')
        plt.title('Shear stress tau23')
        plt.axis('equal'); plt.xlim(-self.half,self.half); plt.ylim(-self.half,self.half)
        plt.tight_layout(); plt.savefig(os.path.join(save_dir,"tau23_field.png"),dpi=300)

        # energy density
        plt.figure(figsize=(6,5))
        plt.gca().set_aspect('equal')
        plt.xlim(-self.half,self.half)
        plt.ylim(-self.half,self.half)
        c = plt.contourf(Xnp, Ynp, Wnp, levels=60, cmap='jet')
        plt.colorbar(c,label='W')
        plt.title('Strain energy density W')
        plt.axis('equal'); plt.xlim(-self.half,self.half); plt.ylim(-self.half,self.half)
        plt.tight_layout(); plt.savefig(os.path.join(save_dir,"W_field.png"),dpi=300)
        plt.close('all')


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
    pinn = AntiplanePlate(model=net, KIII_init=KIII_true, square_half=L, a_crack=a, points_num=30)

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
    net_special = torch.load('./special_net')
    pinn.train_special(net_special, epochs=100000, train_if = False)

    # ----- 训练 -----
    out_dir = "../../result/crack3/crack_DEM"
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
