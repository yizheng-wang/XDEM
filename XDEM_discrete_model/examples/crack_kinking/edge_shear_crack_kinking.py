# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 基础库
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from torch.nn import functional as F
# 你的项目组件
from DENNs import PINN2D
from utils.NodesGenerater import genMeshNodes2D
from utils.NN import stack_net, AxisScalar2D
from utils.Integral import trapz1D
import utils.Geometry as Geometry
from utils.Geometry import LineSegement
import Embedding_old
from Embedding_old import LineCrackEmbedding, extendAxisNet
from SIF import DispExpolation_homo, SIF_K1K2, M_integral, max_stress_theta

# ---------------- 随机种子 ----------------
seed = 2025
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- 材料与几何参数 ----------------
E = 210.0e3       # MPa
nu = 0.30
plane_strain = True
kappa = (3 - 4*nu) if plane_strain else (3 - nu)/(1 + nu)
mu = E / (2*(1+nu))

# 几何域与裂纹初值（以水平初裂纹为例）
a0 = 0.25        # 初始半长（或末端到中心距离）
b  = 0.5        # x方向半宽
h  = 0.5        # y方向半高
beta0 = 0.0     # 初始裂纹方向角（弧度），0=水平向右
q = 1
point_num = 100
decay_alpha_num = 50
# ---------------- 位移控制加载参数 ----------------
# Umax = 0.01        # 顶部位移最大量（mm）
# 前面4步用0.001间隔，后面用0.0001间隔
U_schedule = np.concatenate([
    np.linspace(0.001, 0.006, num=6, endpoint=True), np.linspace(0.0061, 0.0160, num=100, endpoint=True)
])
# ---------------- 裂纹扩展参数 ----------------
a_increment = 0.05    # 每次扩展长度
Gc = 2.7              # 临界能量释放率（单位：同J的单位，N/mm = MPa*mm）
max_propagations_per_load = 30  # 每个载荷步最多扩展次数，防无限循环

# ---------------- 训练参数 ----------------
epochs_per_state = 3000
patience = 10
lr_init = 0.01
milestones = [8000]
eval_sep = 200
tol_early_stop = 0.00001
# ---------------- 终止判据（建议值，可按需调整） ----------------
J_abs_cutoff = 1000        # 绝对阈值：J 超过该值认为已失效（单位：N/mm）


# 输出目录
out_dir = '../../result/crack_kinking/displacement_control_growth'
os.makedirs(out_dir, exist_ok=True)


# =========================
# 工具：由 K 计算 J（后备）
# =========================
def J_from_K(K1, K2, E, nu, plane_strain=True):
    """
    线弹性各向同性材料下，J 与 K 的关系：
    - 平面应力： J = (K_I^2 + K_II^2) / E
    - 平面应变： J = (1 - nu^2) / E * (K_I^2 + K_II^2)
    """
    Keff2 = (K1**2 + K2**2)
    if plane_strain:
        return (1.0 - nu**2) / E * Keff2
    else:
        return (1.0) / E * Keff2


# ---------- LoRA 适配器 ----------
class LoRALinear(nn.Module):
    """
    将 nn.Linear 替换为： y = x W^T + (alpha/r) * x B^T A^T + bias
      - 冻结 W,bias，只训练 A,B
      - r: rank, alpha: 缩放
    """
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-12 else nn.Identity()

        # 原始权重复制并冻结
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        if self.r > 0:
            # B: out_features x r, A: in_features x r   （实现中用转置更高效）
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            # r=0 退化为冻结线性层
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def forward(self, x: torch.Tensor):
        # 基础输出
        y = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            # LoRA 分支
            x_d = self.dropout(x)
            # (N, in) @ (in, r)^T -> (N, r);  (N, r) @ (r, out)^T -> (N, out)
            update = x_d @ self.lora_A.t() @ self.lora_B.t()
            y = y + self.scaling * update
        return y


def _replace_linear_with_lora(module: nn.Module, r=4, alpha=8.0, dropout=0.0):
    """
    递归将 module 内的 nn.Linear 全部换成 LoRALinear；返回被替换的层数。
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lora = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, lora)
            count += 1
        else:
            count += _replace_linear_with_lora(child, r=r, alpha=alpha, dropout=dropout)
    return count


def attach_lora(model: nn.Module, r=4, alpha=8.0, dropout=0.0):
    """
    在 model（你的 AxisScalar2D/stack_net 的线性层）上挂 LoRA。
    """
    n = _replace_linear_with_lora(model, r=r, alpha=alpha, dropout=dropout)
    print(f"[LoRA] attached to {n} Linear layers (r={r}, alpha={alpha}, dropout={dropout}).")


def lora_parameters(model: nn.Module):
    """仅返回 LoRA 可训练参数（A、B），用于优化器。"""
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.r > 0:
            params += [m.lora_A, m.lora_B]
    return params


def save_lora(model: nn.Module, path: str):
    """仅保存 LoRA 参数（轻量干净）。"""
    state = {}
    idx = 0
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.r > 0:
            state[f"{idx}.A"] = m.lora_A.detach().cpu()
            state[f"{idx}.B"] = m.lora_B.detach().cpu()
            state[f"{idx}.shape"] = (m.in_features, m.out_features, m.r, m.alpha)
            idx += 1
    torch.save(state, path)
    print(f"[LoRA] saved adapters -> {path}")


def load_lora(model: nn.Module, path: str, strict_shape: bool = False):
    """加载 LoRA 参数到当前模型结构。"""
    state = torch.load(path, map_location="cpu")
    idx = 0
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.r > 0:
            keyA, keyB, keyS = f"{idx}.A", f"{idx}.B", f"{idx}.shape"
            if keyA in state and keyB in state:
                if strict_shape and keyS in state:
                    in_f, out_f, r, alpha = state[keyS]
                    assert (in_f == m.in_features and out_f == m.out_features and r == m.r), "LoRA shape mismatch"
                m.lora_A.data.copy_(state[keyA].to(m.lora_A.dtype))
                m.lora_B.data.copy_(state[keyB].to(m.lora_B.dtype))
            idx += 1
    print(f"[LoRA] loaded adapters from {path}")

# =========================
# 模型：位移控制的 hard-BC
# =========================
class Plate(PINN2D):
    def __init__(self, model: nn.Module, a, b, h, plane_strain=True):
        super().__init__(model)
        self.a = a; self.b = b; self.h = h
        self.plane_strain = plane_strain
        # 当前载荷步的顶边位移幅值（外部在每个载荷步前设置）
        self.Ubar = torch.tensor(0.0, device=self.device, dtype=torch.get_default_dtype())
        self.beta = 0.0
        self.x_crackTip = 0.0
        self.y_crackTip = 0.0
        K1_esti = 0.0
        K2_esti = 0.001*E * np.sqrt(np.pi * a) 
        self.K_I = nn.Parameter(torch.tensor(K1_esti, device=self.device))
        self.K_II = nn.Parameter(torch.tensor(K2_esti, device=self.device))
        self.q = 1.0
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

    # 顶部 y=+h 规定 v = Ubar；底部 y=-h 规定 v = 0
    # u 不作强约束，仅用形函数“软约束”抑制刚体漂移
    def hard_u(self, u, x, y):
        # Calculate polar coordinates relative to crack tip
        c = torch.cos(torch.tensor(self.beta, device=self.device))
        s = torch.sin(torch.tensor(self.beta, device=self.device))
        # 全局->局部 的旋转 R = [[c, s],[-s, c]]
        # 右/左裂尖（全局坐标）
        tip_r = torch.tensor([self.x_crackTip,  self.y_crackTip]).to(x.device).to(x.dtype)

        # 组装点坐标
        xy = torch.stack([x, y], dim=-1)  # (...,2)

        # 相对坐标（全局）
        rel_r_g = xy - tip_r

        # 旋到局部：v_local = R v_global
        R = torch.stack([torch.stack([c, s], dim=-1),
                        torch.stack([-s, c], dim=-1)], dim=0)  # (2,2)
        rel_r_l = rel_r_g @ R.T



        x1_r, x2_r = rel_r_l[..., 0], rel_r_l[..., 1]
        r_right   = torch.sqrt(x1_r**2 + x2_r**2)
        theta_right = torch.atan2(x2_r, x1_r)
    

        # Calculate Mode I crack analytical solution
        mu = E / (2 * (1 + nu))
        kappa = 3 - 4 * nu  # Plane strain
        
        # K_I is now a trainable parameter (already initialized in __init__)
        # Analytical displacement components
        u_analytical_right = (self.K_I / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.cos(theta_right/2) * (kappa - 1 + 2 * torch.sin(theta_right/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (2 + kappa + torch.cos(theta_right))
        v_analytical_right = (self.K_I / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (kappa + 1 - 2 * torch.cos(theta_right/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.cos(theta_right/2) * (2 - kappa - torch.cos(theta_right))
        # 需要把上述局部坐标系表示的位移场放到全局坐标系中
        u_analytical_right_global = u_analytical_right * c - v_analytical_right * s
        # return (u/1000 + u_analytical_right_global * torch.exp(-5*self.b/self.a*r_right**self.q)) * (1 + y/self.h)/2 #  除以1000，不要让U太小，标准化一下
        return (u/30 + u_analytical_right_global * torch.exp(-5*self.b/self.a*r_right**self.q)) * (1 + y/self.h)*(1 - y/self.h)/2 + self.Ubar * (1 + y/self.h)/2
        # return u_analytical_left * torch.exp(-r_left*2) + u_analytical_right * torch.exp(-r_right*2)
        #return (u_analytical_left * torch.exp(-5*self.b/self.a*r_left**self.q) + u_analytical_right * torch.exp(-5*self.b/self.a*r_right**self.q)) * x
    def hard_v(self, v, x, y):
        # Calculate polar coordinates relative to crack tip
        c = torch.cos(torch.tensor(self.beta, device=self.device))
        s = torch.sin(torch.tensor(self.beta, device=self.device))
        # 全局->局部 的旋转 R = [[c, s],[-s, c]]
        # 右/左裂尖（全局坐标）
        tip_r = torch.tensor([self.x_crackTip,  self.y_crackTip]).to(x.device).to(x.dtype)

        # 组装点坐标
        xy = torch.stack([x, y], dim=-1)  # (...,2)

        # 相对坐标（全局）
        rel_r_g = xy - tip_r

        # 旋到局部：v_local = R v_global
        R = torch.stack([torch.stack([c, s], dim=-1),
                        torch.stack([-s, c], dim=-1)], dim=0)  # (2,2)
        rel_r_l = rel_r_g @ R.T

        # 局部极坐标
        x1_r, x2_r = rel_r_l[..., 0], rel_r_l[..., 1]
        r_right   = torch.sqrt(x1_r**2 + x2_r**2)
        theta_right = torch.atan2(x2_r, x1_r)
        

        # Calculate Mode I crack analytical solution
        mu = E / (2 * (1 + nu))
        kappa = 3 - 4 * nu  # Plane strain
        
        # K_I is now a trainable parameter (already initialized in __init__)
        # Analytical displacement components
        u_analytical_right = (self.K_I / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.cos(theta_right/2) * (kappa - 1 + 2 * torch.sin(theta_right/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (2 + kappa + torch.cos(theta_right))
        v_analytical_right = (self.K_I / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (kappa + 1 - 2 * torch.cos(theta_right/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.cos(theta_right/2) * (2 - kappa - torch.cos(theta_right))
        v_analytical_right_global = u_analytical_right * s + v_analytical_right * c
        return (v/30  + v_analytical_right_global * torch.exp(-5*self.b/self.a*r_right**self.q)) * (1-y/self.h) * (1 + y/self.h) * (1 + x/self.b) * (1 - x/self.b)
        # return (v/1000 + v_analytical_right_global * torch.exp(-5*self.b/self.a*r_right**self.q))  * (1-y/self.h) * (1 + y/self.h) + self.Ubar * (1 + y/self.h)/2 #  除以1000，不要让U太小，标准化一下  
        #return v_analytical_left * torch.exp(-r_left*2) + v_analytical_right * torch.exp(-r_right*2)
        #return (148 / (2 * mu)) * torch.sqrt(r_right / (2 * np.pi)) * torch.sin(theta_right/2) * (kappa + 1 - 2 * torch.cos(theta_right/2)**2)
        #return (v_analytical_left * torch.exp(-5*self.b/self.a*r_left**self.q) + v_analytical_right * torch.exp(-5*self.b/self.a*r_right**self.q)) * (y+ self.h)/2

    def add_BCPoints(self, num=[128]):
        # 仅用于可视化或能量外功=0时无所谓；位移控制不需要外功
        x_up, y_up = genMeshNodes2D(-self.b, self.b, num[0], self.h, self.h, 1)
        x_dn, y_dn = genMeshNodes2D(-self.b, self.b, num[0], -self.h, -self.h, 1)
        self.x_up, self.y_up, self.xy_up = self._set_points(x_up, y_up)
        self.x_dn, self.y_dn, self.xy_dn = self._set_points(x_dn, y_dn)

    def E_ext(self) -> torch.Tensor:
        # 位移控制：不通过外功项求解，取0即可（只最小化内部能）
        return torch.zeros((), device=self.device)


# =========================
# 初始化网络与 PINN
# =========================
def make_net_with_embedding(crack_points):
    # 多段裂纹嵌入：建议使用你示例里的 multiLineCrackEmbedding
    try:
        embedding = Embedding_old.multiLineCrackEmbedding(crack_points, tip='right')
    except Exception:
        # 若没有 multiLineCrackEmbedding，也可退化为末段 LineCrackEmbedding
        p0, p1 = crack_points[-2], crack_points[-1]
        embedding = LineCrackEmbedding(p0, p1, tip='right')

    net = extendAxisNet(
        net=AxisScalar2D(
            stack_net(input=3, output=2, activation=nn.Tanh, width=30, depth=4),
            A=torch.tensor([1.0/b, 1.0/h, 1.0]),
            B=torch.tensor([0.0, 0.0, 0.0]),
        ),
        extendAxis=embedding,
    )
    return net, embedding


def init_pinn(crack_points):
    net, embedding = make_net_with_embedding(crack_points)
    # === 在模型上挂 LoRA（只对 Linear 层） ===
    attach_lora(net, r=4, alpha=8.0, dropout=0.0)

    pinn = Plate(net, a=a0, b=b, h=h, plane_strain=plane_strain)
    pinn.add_BCPoints()
    pinn.setMaterial(E=E, nu=nu, type='plane strain' if plane_strain else 'plane stress')
    pinn.set_loss_func(losses=[pinn.Energy_loss], weights=[1.0])
    return pinn, embedding


def _finite_or_nan_guard(x):
    x = float(x)
    if not np.isfinite(x):
        return np.nan
    return x


# =========================
# 评估：计算 KI, KII 与 J
# =========================
def estimate_KJ(pinn, embedding, tip_point, beta_local, radius=0.25, ntheta=720):
    """
    输入：
      - tip_point: [x_tip, y_tip]
      - beta_local: 当地裂纹切向角（弧度）
    返回：
      - K1, K2, J
    """
    mtool = M_integral(pinn, pinn.device)
    try:
        K1_t, K2_t, MI, MII = mtool.compute_K_via_interaction_integral(
            crack_tip_xy=tip_point,
            radius=radius,
            beta=beta_local,
                                                      E=E, nu=nu,
            plane_strain=plane_strain,
            num_points=ntheta,
            device=pinn.device,
        )
        K1 = float(K1_t.detach().cpu().numpy())
        K2 = float(K2_t.detach().cpu().numpy())
        # 由 K 计算 J（不直接用 MI/MII 避免符号差异）
        J = float(J_from_K(K1, K2, E, nu, plane_strain))
        K1 = _finite_or_nan_guard(K1_t.detach().cpu().numpy())
        K2 = _finite_or_nan_guard(K2_t.detach().cpu().numpy())
        J  = _finite_or_nan_guard(J_from_K(K1, K2, E, nu, plane_strain))
        return K1, K2, J
    except Exception:
        # 后备：用位移外推获取 K，再由 K 算 J
        seg = LineSegement(embedding.points[-2], embedding.points[-1])
        seg = LineSegement(seg.clamp(dist2=0.12), seg.clamp(dist2=0.08))
        loc_axis = Geometry.LocalAxis(tip_point[0], tip_point[1], beta=beta_local)
        K1_t, K2_t = DispExpolation_homo(pinn, embedding, seg, 5, loc_axis, kappa, mu)
        K1 = float(K1_t); K2 = float(K2_t)
        J = float(J_from_K(K1, K2, E, nu, plane_strain))
        K1 = _finite_or_nan_guard(K1_t)
        K2 = _finite_or_nan_guard(K2_t)
        J  = _finite_or_nan_guard(J_from_K(K1, K2, E, nu, plane_strain))
        return K1, K2, J


# =========================
# 主流程：位移步 + 裂纹扩展
# =========================
def run_displacement_loading_with_growth():
    # 初始裂纹：左端固定在 (-1, 0)，沿 beta0 方向，长度 = 2*a0
    p0 = np.array([-0.5, 0.0], dtype=float)  # 起点（左端）
    d  = np.array([np.cos(beta0), np.sin(beta0)], dtype=float)  # 方向单位向量

    # 裂尖（右端）
    p_tip = p0 + (2.0 * a0) * d

    # 为了兼容你示例中“三点”的嵌入实现，给出一个靠近裂尖的前一点
    # （避免与 p_tip 重合，默认比总长短一个 a_increment）
    pre_len = max(2.0 * a0 - a_increment, 1e-6)  # 防止过短或负值
    p_prev = p0 + pre_len * d

    # 三点序列（与你示例保持一致：前一点、当前尖端）
    points = [
        p0.tolist(),         # 左端（固定在 -1,0）
        [-0.15, 0.0],     # 靠近裂尖的前一点
        p_tip.tolist()       # 裂尖（右端）
    ]
    # points = [
    #     p0.tolist(),         # 左端（固定在 -1,0）
    #     p_tip.tolist()       # 裂尖（右端）
    # ]


    # 初始化模型
    pinn, embedding = init_pinn(points)
    # 训练配置
    def set_optimizer(pinn, lr):
        # 只优化 LoRA 适配器参数（A、B），冻结基座权重
        params = lora_parameters(pinn.model)
        pinn.optimizer = torch.optim.Adam(params, lr=lr) if len(params) > 0 else torch.optim.Adam([], lr=lr)
        return torch.optim.lr_scheduler.MultiStepLR(pinn.optimizer, milestones=milestones, gamma=0.3)


    # 网格与可视化点（训练时会重设密度）
    pinn.set_meshgrid_inner_points(-b, b, point_num, -h, h, point_num)

    # 载荷步循环
    all_records = []  # (load_step, prop_iter, Ubar, K1,K2,J, tipx, tipy)
    ld_records = []   # (Ubar, Fy) 载荷-位移对
    pinn.set_Optimizer(lr_init)
    for il, Ubar in enumerate(U_schedule, start=1):
        # 为每个载荷步创建子文件夹
        step_dir = os.path.join(out_dir, f'step_{il:02d}_Ubar_{Ubar:.6f}')
        os.makedirs(step_dir, exist_ok=True)
        
        # ---- LoRA 热启动：从上一个加载步的 LoRA 继续 ----
        if il > 1:
            # prev_lora = os.path.join(out_dir, f"lora_step{il-1:02d}.pth")
            prev_lora = os.path.join(out_dir, f"lora_step01.pth") # 用第一步骤的神经网络参数，用后面会出现失败
            if os.path.exists(prev_lora):
                print(f"load loar from {prev_lora}")
                load_lora(pinn.model, prev_lora)  # 只加载 A/B，不覆盖基座
            else:
                print(f"[LoRA] previous adapters not found: {prev_lora} (start fresh)")
        embedding = Embedding_old.multiLineCrackEmbedding(points,tip = 'right')
        print(f"\n====== Load step {il}/{len(U_schedule)}: Ubar = {Ubar:.6f} mm ======")
        pinn.model.set_extend_axis(embedding)   # 确保当前嵌入
        pinn.Ubar = torch.tensor(float(Ubar), device=pinn.device)

        # 每个载荷步先训练一次"不扩展"的当前几何
        pinn.set_meshgrid_inner_points(-b,b,point_num,-h,h,point_num)
        save_tag0 = os.path.join(step_dir, f'prop0')
        os.makedirs(os.path.dirname(save_tag0), exist_ok=True)
        pinn.train(path=save_tag0,
                   epochs=epochs_per_state, patience=patience, lr=lr_init, eval_sep=eval_sep,milestones=milestones, tol_early_stop=tol_early_stop)
        ckpt0 = save_tag0 + '.pth'
        if os.path.exists(ckpt0):
            pinn.load(path=save_tag0)
        else:
            print(f"[warn] checkpoint not found: {ckpt0}, skip loading.")

        # 内部“扩展”循环
        prop_cnt = 0
        while True:
            # 裂尖与方向
            tipx, tipy = points[-1]
            # 当前末段的切向
            last_seg = LineSegement(points[-2], points[-1])
            beta_local = float(last_seg.tangent_theta.numpy())

            # 评估 K 与 J
            K1, K2, J = estimate_KJ(pinn, embedding, [tipx, tipy], beta_local, radius=0.25*a0)
            print(f"[step {il} | prop {prop_cnt}]  K_I={K1:.4f}, K_II={K2:.4f},  J={J:.6f}  (Gc={Gc:.6f})")
            print(f"  [step {il}, the current K1 and K2 in NN are {pinn.K_I.cpu().detach().numpy():.3f} and {pinn.K_II.cpu().detach().numpy():.3f}")
            all_records.append((il, prop_cnt, Ubar, K1, K2, J, tipx, tipy))

            # === 简化策略：只要 J 过大就立即停止整个过程 ===
            if J >= J_abs_cutoff:
                print(f"  [stop-simple] J 超过阈值: J={J:.6f} >= {J_abs_cutoff}. 立即结束全部迭代。")
                # 可选：保存一张场图
                try:
                    img_path = os.path.join(step_dir, f'break_fields.png')
                    plot_comprehensive_fields(pinn, embedding, out_path=img_path, test_num=300)
                except Exception:
                    pass

                # 记录当前载荷点
                try:
                    Fy = compute_top_reaction(pinn, nx=600)
                    ld_records.append((float(Ubar), float(Fy)))
                    save_load_displacement(out_dir, ld_records)
                except Exception:
                    pass

                # 持久化轨迹与点集
                torch.save(points, os.path.join(step_dir, f'points.pt'))
                np.save(os.path.join(out_dir, 'all_records.npy'), np.array(all_records, dtype=float))
                print("\n=== 因 J 超阈值，提前结束 ===")
                plot_load_displacement_curve(ld_records, out_dir)
                return
            # 否则扩展：方向用最大环向拉应力
            dtheta = float(max_stress_theta(K1, K2))  # 相对当前切向角的偏转
            new_theta = beta_local + dtheta
            
            print(f"    [扩展] dtheta = {dtheta:.6f} rad ({np.degrees(dtheta):.2f}°), new_theta = {new_theta:.6f} rad ({np.degrees(new_theta):.2f}°)")


            # 判据：J <= Gc => 不扩展（本载荷步稳定），跳出
            if J <= Gc or prop_cnt >= max_propagations_per_load:
                if prop_cnt >= max_propagations_per_load:
                    print(f"  Reached max propagation iterations ({max_propagations_per_load}) at load step {il}.")
                print(f"  Stable for load step {il}: J={J:.6f} <= Gc={Gc:.6f}. Move to next load step.")
                # 保存本步的 LoRA 适配器，供下一步热启动
                cur_lora = os.path.join(out_dir, f"lora_step{il:02d}.pth")
                save_lora(pinn.model, cur_lora)
                # ✅ 载荷步稳定后，存一张2×3综合图
                img_path = os.path.join(step_dir, f'stable_fields.png')
                plot_comprehensive_fields(pinn, embedding, out_path=img_path, test_num=300)

                # ✅ 计算顶边合力，并记录 (Ubar, Fy)
                Fy = compute_top_reaction(pinn, nx=600)
                ld_records.append((float(Ubar), float(Fy)))
                print(f"  Recorded L-D point: Ubar={float(Ubar):.6f} mm,  Fy={Fy:.6f} N/mm")

                # 立刻持久化一次，避免后续异常/提前返回丢数据
                save_load_displacement(out_dir, ld_records)
                break


            new_tip = [tipx + a_increment*np.cos(new_theta),
                       tipy + a_increment*np.sin(new_theta)]
            print(f"    [扩展] new_tip = {new_tip}")
            # 更新一下裂纹尖端位置，用于构造u和v的extend function
            pinn.x_crackTip = new_tip[0]
            pinn.y_crackTip = new_tip[1]
            pinn.beta = new_theta

            # 更新几何（添加新端点）
            points.append(new_tip)
            # 重新构建/设置 embedding
            try:
                embedding = Embedding_old.multiLineCrackEmbedding(points, tip='right')
            except Exception:
                embedding = LineCrackEmbedding(points[-2], points[-1], tip='right')
            pinn.model.set_extend_axis(embedding)

            # 在新几何下再训练一次（同一载荷步）
            prop_cnt += 1
            save_tag = os.path.join(step_dir, f'prop{prop_cnt}')
            scheduler = set_optimizer(pinn, lr_init)
            pinn.set_meshgrid_inner_points(-b, b, point_num, -h, h, point_num)
            pinn.train(path=save_tag, epochs=epochs_per_state, patience=patience, lr=lr_init, eval_sep=eval_sep, tol_early_stop=tol_early_stop)
            ckpt = save_tag + '.pth'
            if os.path.exists(ckpt):
                pinn.load(path=save_tag)
            else:
                print(f"[warn] checkpoint not found: {ckpt}, skip loading.")
            # === 每次裂纹扩展（prop）结束后，保存本次的 LoRA 适配器 ===
            cur_lora_prop = os.path.join(out_dir, f"lora_step{il:02d}_prop{prop_cnt}.pth")
            save_lora(pinn.model, cur_lora_prop)

            # 当前载荷步结束，缓存裂纹点集
            torch.save(points, os.path.join(step_dir, f'points.pt'))
            if points[-1][0] > b or points[-1][0] < -b or points[-1][1] > h or points[-1][1] < -h:
                print('裂纹扩展超出边界，立即结束全部迭代。')
                img_path = os.path.join(step_dir, f'stable_fields.png')
                plot_comprehensive_fields(pinn, embedding, out_path=img_path, test_num=300)
                # ✅ 计算顶边合力直接赋予0，因为断了
                Fy = 0.0
                ld_records.append((float(Ubar), float(Fy)))
                    # 最终保存轨迹
                np.save(os.path.join(out_dir, 'all_records.npy'), np.array(all_records, dtype=float))
                print("\n=== 全部载荷步完成 ===")
                # === 导出载荷-位移曲线数据并画图 ===
                import csv
                ld_path_csv = os.path.join(out_dir, 'load_displacement.csv')
                with open(ld_path_csv, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['Ubar_mm', 'Fy_N_per_mm'])  # 二维单位厚度
                    for Ubar, Fy in ld_records:
                        w.writerow([Ubar, Fy])
                print(f"载荷-位移数据已保存: {ld_path_csv}")

                # 画载荷-位移曲线
                plot_load_displacement_curve(ld_records, out_dir)
                print(f"记录保存至: {out_dir}")
                print('裂纹扩展超出边界，立即结束全部迭代。')
                    # 最终保存轨迹
                return
        # 当前载荷步结束，缓存裂纹点集
        torch.save(points, os.path.join(step_dir, f'points.pt'))
    # 最终保存轨迹
    np.save(os.path.join(out_dir, 'all_records.npy'), np.array(all_records, dtype=float))
    print("\n=== 全部载荷步完成 ===")
    # === 导出载荷-位移曲线数据并画图 ===
    import csv
    ld_path_csv = os.path.join(out_dir, 'load_displacement.csv')
    with open(ld_path_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Ubar_mm', 'Fy_N_per_mm'])  # 二维单位厚度
        for Ubar, Fy in ld_records:
            w.writerow([Ubar, Fy])
    print(f"载荷-位移数据已保存: {ld_path_csv}")

    # 画载荷-位移曲线
    plot_load_displacement_curve(ld_records, out_dir)
    print(f"记录保存至: {out_dir}")


def compute_top_reaction(pinn, nx=400):
    """
    计算顶边( y = +h )的竖向合力（单位厚度），返回 Fy。
    Fy = ∫_{x=-b}^{b} σ_yy(x, y=+h) dx
    """
    # 采样顶边点
    x = torch.linspace(-pinn.b, pinn.b, nx, device=pinn.device, dtype=torch.get_default_dtype())
    y = torch.full_like(x, pinn.h)
    XY = torch.stack([x, y], dim=1)

    # **必须**开启对坐标的梯度，这样 infer() 里才能算应变/应力
    XY.requires_grad_(True)

    # 确保梯度开着（以防外部用了 torch.no_grad）
    with torch.set_grad_enabled(True):
        # 只需应力
        _, _, sx, sy, sxy = pinn.infer(XY)

        # 数值积分（单位：MPa*mm = N/mm，按单位厚度）
        Fy = torch.trapz(sxy, x)

    # 转 numpy
    Fy_val = float(Fy.detach().cpu().numpy())
    if not np.isfinite(Fy_val):
        Fy_val = 0.0
    return Fy_val

def save_load_displacement(out_dir, ld_records, fname='load_displacement'):
    """
    将载荷-位移数据保存为 .csv / .npy / .txt 三种格式，便于后续画图或复用。
    ld_records: List[Tuple[Ubar(float), Fy(float)]]
    """
    import csv
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    arr = np.array(ld_records, dtype=float) if len(ld_records) > 0 else np.zeros((0,2), dtype=float)

    # 1) CSV（带表头）
    csv_path = os.path.join(out_dir, f'{fname}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Ubar_mm', 'Fy_N_per_mm'])
        for Ubar, Fy in ld_records:
            w.writerow([float(Ubar), float(Fy)])

    # 2) NPY（numpy 原生）
    npy_path = os.path.join(out_dir, f'{fname}.npy')
    np.save(npy_path, arr)

    # 3) TXT（两列，便于 gnuplot / matlab / 直接读）
    txt_path = os.path.join(out_dir, f'{fname}.txt')
    np.savetxt(txt_path, arr, fmt='%.10e', header='Ubar_mm Fy_N_per_mm', comments='')

    print(f'[save_load_displacement] saved -> {csv_path}, {npy_path}, {txt_path}')
    return csv_path, npy_path, txt_path


def plot_load_displacement_curve(ld_records, out_dir, fname='load_displacement_curve'):
    """
    绘制载荷-位移曲线并保存为图片
    ld_records: List[Tuple[Ubar(float), Fy(float)]] 载荷-位移数据
    out_dir: 输出目录
    fname: 文件名（不含扩展名）
    """
    if len(ld_records) == 0:
        print("[plot_load_displacement_curve] 无数据，跳过绘图")
        return
    
    # 添加 (0,0) 起始点
    Us = np.concatenate([[0.0], np.array([p[0] for p in ld_records], dtype=float)])
    Fs = np.concatenate([[0.0], np.array([p[1] for p in ld_records], dtype=float)])
    
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(Us, Fs, marker='o', linewidth=2)
    plt.xlabel('Top displacement UY bar (mm)')
    plt.ylabel('Resultant FY (N/mm)')   # 二维单位厚度
    plt.title('Load–Displacement Curve')
    plt.grid(True, alpha=0.3)
    
    fig_path = os.path.join(out_dir, f'{fname}.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[plot_load_displacement_curve] 曲线图已保存: {fig_path}")




# =========================
# 可视化（简要）
# =========================
def quick_plot_fields(pinn, embedding, out_path=None, test_num=300):
    """
    画三联图：u2、σ22、Gamma。若 out_path 给出则直接保存（.png/.pdf随你传入的扩展名）。
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x_vis = torch.linspace(-pinn.b, pinn.b, test_num, device=pinn.device)
    y_vis = torch.linspace(-pinn.h, pinn.h, test_num, device=pinn.device)
    X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
    XY_vis = torch.stack([X_vis.flatten(), Y_vis.flatten()], dim=1)
    XY_vis.requires_grad_(True)

    # 前向推理
    u_pred, v_pred = pinn.pred_uv(XY_vis)
    _, _, sx, sy, sxy = pinn.infer(XY_vis)
    gamma_pred = embedding.getGamma(XY_vis)

    # to numpy
    def to_np(t): return t.detach().cpu().numpy().reshape(test_num, test_num)
    Xn = X_vis.detach().cpu().numpy()
    Yn = Y_vis.detach().cpu().numpy()
    vn  = to_np(v_pred)
    syn = to_np(sy)
    gn  = to_np(gamma_pred)

    # 画图
    plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, 3, 1)
    c1 = plt.contourf(Xn, Yn, vn, levels=30, cmap='RdBu_r')
    plt.colorbar(c1, ax=ax, label='u2 displacement')
    plt.title('u2'); plt.axis('equal')

    ax = plt.subplot(1, 3, 2)
    c2 = plt.contourf(Xn, Yn, syn, levels=30, cmap='RdBu_r')
    plt.colorbar(c2, ax=ax, label='σ22 stress')
    plt.title('σ22'); plt.axis('equal')

    ax = plt.subplot(1, 3, 3)
    c3 = plt.contourf(Xn, Yn, gn, levels=30, cmap='RdBu_r')
    plt.colorbar(c3, ax=ax, label='Gamma')
    plt.title('Gamma'); plt.axis('equal')

    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[quick_plot_fields] saved -> {out_path}")
    else:
        plt.show()


def plot_comprehensive_fields(pinn, embedding, out_path=None, test_num=300):
    """
    画2×3综合图：u1, u2, σ11, σ22, σ12, von Mises应力, crack function
    若 out_path 给出则直接保存（.png/.pdf随你传入的扩展名）。
    """
    import numpy as np
    import matplotlib.pyplot as plt

    x_vis = torch.linspace(-pinn.b, pinn.b, test_num, device=pinn.device)
    y_vis = torch.linspace(-pinn.h, pinn.h, test_num, device=pinn.device)
    X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
    XY_vis = torch.stack([X_vis.flatten(), Y_vis.flatten()], dim=1)
    XY_vis.requires_grad_(True)

    # 前向推理
    u_pred, v_pred = pinn.pred_uv(XY_vis)
    _, _, sx, sy, sxy = pinn.infer(XY_vis)
    gamma_pred = embedding.getGamma(XY_vis)

    # 计算von Mises应力
    von_mises = torch.sqrt(sx**2 + sy**2 - sx*sy + 3*sxy**2)

    # to numpy
    def to_np(t): return t.detach().cpu().numpy().reshape(test_num, test_num)
    Xn = X_vis.detach().cpu().numpy()
    Yn = Y_vis.detach().cpu().numpy()
    un  = to_np(u_pred)
    vn  = to_np(v_pred)
    sxn = to_np(sx)
    syn = to_np(sy)
    sxyn = to_np(sxy)
    von_mises_n = to_np(von_mises)
    gn  = to_np(gamma_pred)

    # 画2×3图
    plt.figure(figsize=(18, 8))
    
    # 第一行：位移场
    ax = plt.subplot(2, 3, 1)
    c1 = plt.contourf(Xn, Yn, un, levels=30, cmap='RdBu_r')
    plt.colorbar(c1, ax=ax, label='u1 displacement')
    plt.title('u1 Displacement'); plt.axis('equal')

    ax = plt.subplot(2, 3, 2)
    c2 = plt.contourf(Xn, Yn, vn, levels=30, cmap='RdBu_r')
    plt.colorbar(c2, ax=ax, label='u2 displacement')
    plt.title('u2 Displacement'); plt.axis('equal')

    ax = plt.subplot(2, 3, 3)
    c3 = plt.contourf(Xn, Yn, sxn, levels=30, cmap='RdBu_r')
    plt.colorbar(c3, ax=ax, label='σ11 stress')
    plt.title('σ11 Stress'); plt.axis('equal')

    # 第二行：应力场
    ax = plt.subplot(2, 3, 4)
    c4 = plt.contourf(Xn, Yn, syn, levels=30, cmap='RdBu_r')
    plt.colorbar(c4, ax=ax, label='σ22 stress')
    plt.title('σ22 Stress'); plt.axis('equal')

    ax = plt.subplot(2, 3, 5)
    c5 = plt.contourf(Xn, Yn, sxyn, levels=30, cmap='RdBu_r')
    plt.colorbar(c5, ax=ax, label='σ12 stress')
    plt.title('σ12 Stress'); plt.axis('equal')

    ax = plt.subplot(2, 3, 6)
    c6 = plt.contourf(Xn, Yn, von_mises_n, levels=30, cmap='RdBu_r')
    plt.colorbar(c6, ax=ax, label='von Mises stress')
    plt.title('von Mises Stress'); plt.axis('equal')

    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[plot_comprehensive_fields] saved -> {out_path}")
        
        # 单独保存每个子图
        base_path = out_path.replace('.png', '')
        
        # u1 displacement
        plt.figure(figsize=(8, 6))
        c1 = plt.contourf(Xn, Yn, un, levels=30, cmap='RdBu_r')
        plt.colorbar(c1, label='u1 displacement')
        plt.title('u1 Displacement')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{base_path}_u1_displacement.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # u2 displacement
        plt.figure(figsize=(8, 6))
        c2 = plt.contourf(Xn, Yn, vn, levels=30, cmap='RdBu_r')
        plt.colorbar(c2, label='u2 displacement')
        plt.title('u2 Displacement')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{base_path}_u2_displacement.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # σ11 stress
        plt.figure(figsize=(8, 6))
        c3 = plt.contourf(Xn, Yn, sxn, levels=30, cmap='RdBu_r')
        plt.colorbar(c3, label='σ11 stress')
        plt.title('σ11 Stress')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{base_path}_sigma11_stress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # σ22 stress
        plt.figure(figsize=(8, 6))
        c4 = plt.contourf(Xn, Yn, syn, levels=30, cmap='RdBu_r')
        plt.colorbar(c4, label='σ22 stress')
        plt.title('σ22 Stress')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{base_path}_sigma22_stress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # σ12 stress
        plt.figure(figsize=(8, 6))
        c5 = plt.contourf(Xn, Yn, sxyn, levels=30, cmap='RdBu_r')
        plt.colorbar(c5, label='σ12 stress')
        plt.title('σ12 Stress')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{base_path}_sigma12_stress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # von Mises stress
        plt.figure(figsize=(8, 6))
        c6 = plt.contourf(Xn, Yn, von_mises_n, levels=30, cmap='RdBu_r')
        plt.colorbar(c6, label='von Mises stress')
        plt.title('von Mises Stress')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{base_path}_von_mises_stress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[plot_comprehensive_fields] individual plots saved with prefix: {base_path}")
    else:
        plt.show()

    # 单独保存crack function图
    if out_path is not None:
        crack_path = out_path.replace('.png', '_crack_function.png')
        plt.figure(figsize=(8, 6))
        c7 = plt.contourf(Xn, Yn, gn, levels=30, cmap='RdBu_r')
        plt.colorbar(c7, label='Crack Function')
        plt.title('Crack Function (Gamma)')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(crack_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[plot_comprehensive_fields] crack function saved -> {crack_path}")



# =========================
# 入口
# =========================
if __name__ == "__main__":
    run_displacement_loading_with_growth()
