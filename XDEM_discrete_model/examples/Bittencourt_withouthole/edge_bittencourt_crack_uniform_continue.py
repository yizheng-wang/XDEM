# -*- coding: utf-8 -*-
"""
Bittencourt 多孔板 + 初始缺口 + 顶边监测点竖向点载（载荷控制）
依赖你的工程：
- DENNs.PINN2D
- utils.NodesGenerater.genMeshNodes2D
- utils.NN.stack_net / AxisScalar2D
- utils.Integral.trapz1D
- utils.Geometry.LineSegement / LocalAxis
- Embedding_old.LineCrackEmbedding / extendAxisNet / multiLineCrackEmbedding
- SIF.DispExpolation_homo / SIF_K1K2 / M_integral / max_stress_theta
"""
import sys, os, math, random, csv
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# === 工程依赖 ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DENNs import PINN2D
from utils.NodesGenerater import genMeshNodes2D
from utils.NN import stack_net, AxisScalar2D
from utils.Integral import trapz1D
import utils.Geometry as Geometry
from utils.Geometry import LineSegement
import Embedding_bit
from Embedding_bit import LineCrackEmbedding, extendAxisNet
from SIF import DispExpolation_homo, SIF_K1K2, M_integral, max_stress_theta

# ---------------- 随机种子 ----------------
seed = 2025
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- 材料与几何参数（按图） ----------------
E  = 3000.0      # N/mm^2
nu = 0.35
plane_strain = True
kappa = (3 - 4*nu) if plane_strain else (3 - nu)/(1 + nu)
mu = E / (2*(1+nu))

W, H = 20.0, 8.0                     # 板尺寸
b, h = W/2.0, H/2.0                  # 半宽/半高（你的PINN域）

# 试件 #1 or #2
specimen_id = 1     # 改成 2 即 a=2,b=1
if specimen_id == 1:
    a_dim, b_dim = 1.0, 1.5
else:
    a_dim, b_dim = 2.0, 1.0

# —— 初始裂纹方向与长度（修正：定义缺失的 beta0/a0）——
a0 = float(b_dim)/2            # 初始“半长”参数，下面使用 2*a0 作为根到尖距离
beta0 = 3.14159/2                  # 沿 +x 方向

# 初始缺口：从左下角内角处（-b + 6 - a_dim, -h）向 +x 切入 a_dim
notch_root = np.array([-b + 6.0 - a_dim, -h], dtype=float)

# ---------------- 孔洞参数（按图） ----------------
x_holes = -b + 6.0
R_hole = 0.25
y_top =  h - 1.25
y1    =  y_top - 2.0
y2    =  y1   - 2.0
HOLES = [
    (x_holes, y_top, R_hole),
    (x_holes, y1,    R_hole),
    (x_holes, y2,    R_hole)
]

# ---------------- 载荷（力控）与训练参数 ----------------
# 点载强度序列（单位 N），向下为正；可按需要加密
Q_schedule = np.linspace(500.0, 2500.0, num=5)*(-1)

epochs_per_state = 5000
patience = 10
lr_init = 1e-3
milestones = [8000]
eval_sep = 200
tol_early_stop = 1e-5

point_num = 150         # 域内基础网格密度
a_increment = 1.0       # 裂纹步长
# a_increment_list = [1.0, 1.0, 2.0]
Gc = 0.0001             # 临界能量释放率（N/mm）
max_propagations_per_load = 30
J_abs_cutoff = 1000.0

# 输出目录
out_dir = '../../result/bittencourt_load_control'
os.makedirs(out_dir, exist_ok=True)


# ===== 继续训练工具（保存/恢复优化器） =====
CONTINUE_STEP0 = True          # ← 打开/关闭：是否在第1步(未扩展)继续训练
EXTRA_EPOCHS_STEP0 = 3000      # ← 继续训练多少个epoch
OPT_SUFFIX = ".opt.pth"        # 优化器状态的后缀

def save_checkpoint_with_opt(pinn, path_no_ext: str):
    """兼容你已有的 pinn.train/pinn.load 的保存方式，再额外单独存 optimizer。"""
    # 1) 你自己的模型权重（pinn.train里已经保存 .pth，这里确保再存一次）
    pinn.save(path=path_no_ext)  # 等价于写 path_no_ext + ".pth"
    # 2) 保存优化器
    if getattr(pinn, "optimizer", None) is not None:
        torch.save(pinn.optimizer.state_dict(), path_no_ext + OPT_SUFFIX)

def load_checkpoint_with_opt(pinn, path_no_ext: str, lr_for_recreate: float):
    """先加载模型，再重建并恢复优化器（以保留动量/自适应统计）"""
    # 1) 模型
    pinn.load(path=path_no_ext)
    # 2) 重新创建并恢复优化器
    pinn.set_Optimizer(lr_for_recreate)  # 先用原来的学习率重建
    opt_path = path_no_ext + OPT_SUFFIX
    if os.path.exists(opt_path):
        state = torch.load(opt_path, map_location=pinn.device)
        try:
            pinn.optimizer.load_state_dict(state)
            print(f"[resume] optimizer state loaded from {opt_path}")
        except Exception as e:
            print(f"[resume] optimizer state mismatch, reinit optimizer. Reason: {e}")


# =========================
# 网格采样（剔孔）
# =========================
def set_meshgrid_inner_points_with_holes(pinn, nx, ny):
    """
    生成 [-b,b]×[-h,h] 的均匀网格，剔除落在圆孔内的点；
    然后用 pinn.set_inner_points(internal_points, internal_points_pdf, variable=True)
    设置到 PINN。
    """
    # 1) 均匀网格
    X = np.linspace(-b, b, nx)
    Y = np.linspace(-h, h, ny)
    xx, yy = np.meshgrid(X, Y, indexing='xy')

    # 2) 剔孔
    keep = np.ones_like(xx, dtype=bool)
    for (xc, yc, R) in HOLES:
        keep &= ((xx - xc)**2 + (yy - yc)**2 >= R**2 - 1e-12)

    # 3) 组装为 (N,2)
    pts_np = np.stack([xx[keep], yy[keep]], axis=1)        # (N,2)

    # 4) 变成张量（走你工程的 device / dtype）
    internal_points = torch.tensor(
        pts_np, device=pinn.device, dtype=torch.get_default_dtype()
    )                                                      # (N,2)

    # 5) 权重（均匀）并归一
    internal_points_pdf = torch.ones(
        internal_points.shape[0], device=pinn.device, dtype=torch.get_default_dtype()
    )
    internal_points_pdf = internal_points_pdf / internal_points_pdf.sum()

    # 6) 设置
    pinn.set_inner_points(internal_points, internal_points_pdf, variable=True)
    
def set_triangle_quadrature_points_with_holes(pinn, nx, ny, scheme='tri3'):
    """
    在 [-b,b]×[-h,h] 上生成 (nx × ny) 结点的结构化网格，
    每个小矩形切成两个三角形；对每个三角形用 3 点二阶规则积分。
    对于落入圆孔的求积点直接丢弃（相当于对孔做裁剪），
    最后把所有求积点及其（面积×权重）汇总并归一化，喂给 pinn.set_inner_points。
    """
    # 1) 网格结点
    X = np.linspace(-b, b, nx)
    Y = np.linspace(-h, h, ny)
    # 2) 每个小矩形 -> 两个三角形
    # 参考三角形顶点： (x_i,y_j),(x_{i+1},y_j),(x_{i+1},y_{j+1}) 以及 (x_i,y_j),(x_{i+1},y_{j+1}),(x_i,y_{j+1})
    tris = []
    for i in range(nx-1):
        for j in range(ny-1):
            x0, x1 = X[i], X[i+1]
            y0, y1 = Y[j], Y[j+1]
            # tri A: (x0,y0)-(x1,y0)-(x1,y1)
            tris.append(np.array([[x0,y0],[x1,y0],[x1,y1]], dtype=float))
            # tri B: (x0,y0)-(x1,y1)-(x0,y1)
            tris.append(np.array([[x0,y0],[x1,y1],[x0,y1]], dtype=float))

    # 3) 二阶对称三点求积（参考三角形重心坐标）
    #   (λ1,λ2,λ3) = (2/3,1/6,1/6) 及其循环置换；每点权重 = 1/3
    if scheme == 'tri3':
        bary = np.array([
            [2/3, 1/6, 1/6],
            [1/6, 2/3, 1/6],
            [1/6, 1/6, 2/3],
        ], dtype=float)
        w_loc = np.array([1/3, 1/3, 1/3], dtype=float)
    else:
        # 简单：重心一点（一次规则），可选
        bary = np.array([[1/3, 1/3, 1/3]], dtype=float)
        w_loc = np.array([1.0], dtype=float)

    quad_pts = []
    quad_wts = []

    # 4) 遍历三角形，生成全局求积点与权重（面积×局部权）
    for T in tris:
        # 三角形顶点
        x1, y1 = T[0]; x2, y2 = T[1]; x3, y3 = T[2]
        # 面积（正值）
        area = abs(0.5 * ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)))
        if area < 1e-14:
            continue
        # 全局坐标
        V = np.array([[x1,y1],[x2,y2],[x3,y3]])  # (3,2)
        for (l1, l2, l3), wl in zip(bary, w_loc):
            xy = l1*V[0] + l2*V[1] + l3*V[2]
            xq, yq = float(xy[0]), float(xy[1])

            # 5) 剔孔：若求积点落入任一圆孔则跳过
            inside_hole = False
            for (xc, yc, R) in HOLES:
                if (xq-xc)**2 + (yq-yc)**2 < R**2 - 1e-12:
                    inside_hole = True
                    break
            if inside_hole:
                continue

            quad_pts.append([xq, yq])
            quad_wts.append(area * wl)

    if len(quad_pts) == 0:
        raise RuntimeError("Triangle quadrature produced zero points—check nx, ny or geometry.")

    # 6) 转为张量；权重归一化（和=1），以适配现有 E_int() 的“pdf”接口
    internal_points = torch.tensor(quad_pts, device=pinn.device, dtype=torch.get_default_dtype())
    w = torch.tensor(quad_wts, device=pinn.device, dtype=torch.get_default_dtype())
    w = w / w.sum()

    # 7) 设置到 PINN
    pinn.set_inner_points(internal_points, w, variable=True)

# =========================
# 实用函数
# =========================
def J_from_K(K1, K2, E, nu, plane_strain=True):
    keff2 = K1**2 + K2**2
    if plane_strain:
        return (1 - nu**2)/E * keff2
    else:
        return (1.0/E) * keff2

def _finite_or_nan_guard(x):
    x = float(x)
    if not np.isfinite(x): return np.nan
    return x

def tip_out_of_domain(pt):
    x, y = pt
    if x < -b or x > b or y < -h or y > h: return True
    for (xc, yc, R) in HOLES:
        print("The distance is", np.sqrt((x-xc)**2 + (y-yc)**2))
        if (x-xc)**2 + (y-yc)**2 < R**2 - 1e-10:
            return True
    return False

# =========================
# LoRA 轻适配
# =========================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r); self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-12 else nn.Identity()
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            x_d = self.dropout(x)
            update = x_d @ self.lora_A.t() @ self.lora_B.t()
            y = y + self.scaling * update
        return y

def _replace_linear_with_lora(module: nn.Module, r=4, alpha=8.0, dropout=0.0):
    cnt = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            cnt += 1
        else:
            cnt += _replace_linear_with_lora(child, r=r, alpha=alpha, dropout=dropout)
    return cnt

def attach_lora(model: nn.Module, r=4, alpha=8.0, dropout=0.0):
    n = _replace_linear_with_lora(model, r=r, alpha=alpha, dropout=dropout)
    print(f"[LoRA] attached to {n} Linear layers (r={r}, alpha={alpha}, dropout={dropout}).")

def lora_parameters(model: nn.Module):
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.r > 0:
            params += [m.lora_A, m.lora_B]
    return params

def save_lora(model: nn.Module, path: str):
    state = {}; idx = 0
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.r > 0:
            state[f"{idx}.A"] = m.lora_A.detach().cpu()
            state[f"{idx}.B"] = m.lora_B.detach().cpu()
            state[f"{idx}.shape"] = (m.in_features, m.out_features, m.r, m.alpha)
            idx += 1
    torch.save(state, path)
    print(f"[LoRA] saved adapters -> {path}")

def load_lora(model: nn.Module, path: str, strict_shape: bool=False):
    state = torch.load(path, map_location="cpu")
    idx = 0
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.r > 0:
            keyA, keyB, keyS = f"{idx}.A", f"{idx}.B", f"{idx}.shape"
            if keyA in state and keyB in state:
                if strict_shape and keyS in state:
                    in_f, out_f, r, alpha = state[keyS]
                    assert (in_f == m.in_features and out_f == m.out_features and r == m.r), "LoRA shape mismatch"
                m.lora_A.data.copy_((state[keyA].to(m.lora_A.dtype)))
                m.lora_B.data.copy_(state[keyB].to(m.lora_B.dtype))
            idx += 1
    print(f"[LoRA] loaded adapters from {path}")

# =========================
# 网络 + 裂纹嵌入
# =========================
def make_net_with_embedding(crack_points):
    try:
        embedding = Embedding_bit.multiLineCrackEmbedding(crack_points, tip='right')
    except Exception:
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

# =========================
# 载荷控制板类（外功=点载）
# =========================
class Plate(PINN2D):
    def __init__(self, model: nn.Module, a, b, h, plane_strain=True):
        super().__init__(model)
        self.a, self.b, self.h = a, b, h
        self.plane_strain = plane_strain
        self.Q = torch.tensor(0.0, device=self.device, dtype=torch.get_default_dtype())

        # 近场嵌入参数（可训练）——初值占位，运行中会由几何更新
        self.beta = 0.0
        self.x_crackTip = 0.0
        self.y_crackTip = 0.0
        K1_esti = 1.0
        K2_esti = 1.0
        self.K_I = nn.Parameter(torch.tensor(K1_esti, device=self.device))
        self.K_II = nn.Parameter(torch.tensor(K2_esti, device=self.device))
        self.q = 1.0

    def set_Optimizer(self, lr, k_i_lr=None):
        model_params = list(self.model.parameters())
        k_i_params = [self.K_I, self.K_II]
        if k_i_lr is None: k_i_lr = lr * 10.0
        param_groups = [{'params': model_params, 'lr': lr},
                        {'params': k_i_params, 'lr': k_i_lr}]
        self.optimizer = torch.optim.Adam(param_groups)

    # —— 支座：左下 1×1 mm 固定；底边右端滚子（v=0）——
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

        w1 = (9+x)**2 + (4+y)**2
        w2 = (-9+x)**2 + (4+y)**2
        return (5 * u + u_analytical_right_global * torch.exp(-5*self.b/self.a*r_right**self.q))* w2/(w2+1)

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


        w1 = (9+x)**2 + (4+y)**2
        w2 = (-9+x)**2 + (4+y)**2
        return (5 * v + v_analytical_right_global * torch.exp(-5*r_right**self.q)) * w1*w2/(w1+1)/(w2+1)

    def add_BCPoints(self, num=[600]):
        x_up, y_up = genMeshNodes2D(-self.b, self.b, num[0], self.h, self.h, 1)
        x_dn, y_dn = genMeshNodes2D(-self.b, self.b, num[0], -self.h, -self.h, 1)
        self.x_up, self.y_up, self.xy_up = self._set_points(x_up, y_up)
        self.x_dn, self.y_dn, self.xy_dn = self._set_points(x_dn, y_dn)

    # 外功势：把竖向点力施加在“监测点” (-b+6, h)
    def E_ext(self) -> torch.Tensor:
        XY = torch.tensor([[0.0, h]], device=self.device, dtype=torch.get_default_dtype())
        u_pred, v_pred = self.pred_uv(XY)
        Wext = v_pred * self.Q  # 向下为正
        return Wext[0]

    def Energy_loss(self)->torch.Tensor:
        return torch.stack([(20*8 - 3*0.25**2*3.14159)*self.E_int() - self.E_ext()])
        # return torch.stack([14000*self.E_int() - self.E_ext()])

# =========================
# 评估 K/J
# =========================
def estimate_KJ(pinn, embedding, tip_point, beta_local, radius=0.25, ntheta=720):
    mtool = M_integral(pinn, pinn.device)
    try:
        K1_t, K2_t, MI, MII = mtool.compute_K_via_interaction_integral(
            crack_tip_xy=tip_point, radius=radius, beta=beta_local,
            E=E, nu=nu, plane_strain=plane_strain, num_points=ntheta, device=pinn.device
        )
        K1 = _finite_or_nan_guard(K1_t.detach().cpu().numpy())
        K2 = _finite_or_nan_guard(K2_t.detach().cpu().numpy())
        J  = _finite_or_nan_guard(J_from_K(K1, K2, E, nu, plane_strain))
        return K1, K2, J
    except Exception:
        seg = LineSegement(embedding.points[-2], embedding.points[-1])
        seg = LineSegement(seg.clamp(dist2=0.12), seg.clamp(dist2=0.08))
        loc_axis = Geometry.LocalAxis(tip_point[0], tip_point[1], beta=beta_local)
        K1_t, K2_t = DispExpolation_homo(pinn, embedding, seg, 5, loc_axis, kappa, mu)
        K1 = _finite_or_nan_guard(K1_t)
        K2 = _finite_or_nan_guard(K2_t)
        J  = _finite_or_nan_guard(J_from_K(K1, K2, E, nu, plane_strain))
        return K1, K2, J


def monitor_top_displacement(pinn):
    XY = torch.tensor([[-b + 6.0, h]], device=pinn.device, dtype=torch.get_default_dtype()).view(1,2)
    u, v = pinn.pred_uv(XY)
    return float(v.squeeze().detach().cpu().numpy())


def compute_top_reaction(pinn, nx=600):
    # 这里保留供对比（积分顶边 σ_y）——力控不必须
    x = torch.linspace(-pinn.b, pinn.b, nx, device=pinn.device, dtype=torch.get_default_dtype())
    y = torch.full_like(x, pinn.h)
    XY = torch.stack([x, y], dim=1).requires_grad_(True)
    with torch.set_grad_enabled(True):
        _, _, sx, sy, sxy = pinn.infer(XY)
        Fy = torch.trapz(sy, x)  # 合力（单位厚度）
    Fy_val = float(Fy.detach().cpu().numpy())
    if not np.isfinite(Fy_val): Fy_val = 0.0
    return Fy_val

# =========================
# 初始化
# =========================
def init_pinn(crack_points):
    net, embedding = make_net_with_embedding(crack_points)
    attach_lora(net, r=4, alpha=8.0, dropout=0.0)
    pinn = Plate(net, a=a0, b=b, h=h, plane_strain=plane_strain)  # 修正：a 传 a0
    pinn.add_BCPoints()
    pinn.setMaterial(E=E, nu=nu, type='plane strain' if plane_strain else 'plane stress')
    pinn.set_loss_func(losses=[pinn.Energy_loss], weights=[1.0])
    return pinn, embedding

# =========================
# 主流程（载荷步 + 裂纹扩展）
# =========================
def run_load_control_with_growth(a_increment):
    # 初始裂纹三点：根部 -> 前一点 -> 裂尖
    d  = np.array([0.0, 1.0], dtype=float)
    p_tip  = notch_root + (2.0*a0)*d
    p_prev = notch_root + max(2.0*a0 - a_increment, 1e-6)*d
    points = [notch_root.tolist(), p_prev.tolist(), p_tip.tolist()]

    pinn, embedding = init_pinn(points)

    # 训练仅更新 LoRA
    def set_optimizer_lora_only(pinn, lr):
        params = lora_parameters(pinn.model)
        if len(params) == 0:
            params = []
        pinn.optimizer = torch.optim.Adam(params, lr=lr)
        return torch.optim.lr_scheduler.MultiStepLR(pinn.optimizer, milestones=milestones, gamma=0.3)

    # 初次剔孔采样
    set_triangle_quadrature_points_with_holes(pinn, point_num, point_num)

    all_records = []  # (load_step, prop_iter, Q, K1,K2,J, tipx, tipy)
    ld_records  = []  # (Q, v_monitored)
    pinn.set_Optimizer(lr_init)
    for il, Q in enumerate(Q_schedule, start=1):
        step_dir = os.path.join(out_dir, f'step_{il:02d}_Q_{Q:.3f}')
        os.makedirs(step_dir, exist_ok=True)

        if il > 1:
            prev_lora = os.path.join(out_dir, f"lora_step{il-1:02d}.pth")
            if os.path.exists(prev_lora):
                load_lora(pinn.model, prev_lora)

        embedding = Embedding_bit.multiLineCrackEmbedding(points, tip='right')
        pinn.model.set_extend_axis(embedding)
        pinn.Q = torch.tensor(float(Q), device=pinn.device)

        set_triangle_quadrature_points_with_holes(pinn, point_num, point_num)
        save_tag0 = os.path.join(step_dir, f'prop0')

        if os.path.exists(save_tag0 + '.pth'):
            pinn.load(path=save_tag0)
        cur_lora = os.path.join(out_dir, f"lora_step{il:02d}.pth")
        save_lora(pinn.model, cur_lora)
        plot_comprehensive_fields(pinn, embedding, out_path=os.path.join(step_dir, 'fields_initial.png'), test_num=300, levels=40, cmap='RdBu_r')
        prop_cnt = 0
        while True:
            prev_lora = os.path.join(out_dir, f"lora_step01.pth") # 用第一步骤的神经网络参数，用后面会出现失败
            load_lora(pinn.model, prev_lora)
            print(f"load loar from {prev_lora}")
            tipx, tipy = points[-1]
            last_seg = LineSegement(points[-2], points[-1])
            beta_local = float(last_seg.tangent_theta.numpy())

            K1, K2, J = estimate_KJ(pinn, embedding, [tipx, tipy], beta_local, radius=a0) # 改成半径为a0
#  用张开位移算一下
            crack_surface = Geometry.LineSegement(points[-2],points[-1])
            '''计算裂纹张开位移时略微远离裂尖端'''
            crack_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.12),
                                                    crack_surface.clamp(dist2=0.08))


            K1 , K2 = DispExpolation_homo(pinn,
                                    embedding,
                                    crack_surface,5,
                                    Geometry.LocalAxis(points[-1][0],points[-1][1],
                                    beta = crack_surface.tangent_theta),
                                    kappa,mu)
            K1 = K1[0]
            K2 = K2[0]


            print(f"[step {il} | prop {prop_cnt}]  Q={Q:.3f} N;  K_I={K1:.4f}, K_II={K2:.4f},  J={J:.6f}  (Gc={Gc:.6f})")
            all_records.append((il, prop_cnt, Q, K1, K2, J, tipx, tipy))

            if J >= J_abs_cutoff:
                plot_comprehensive_fields(pinn, embedding, out_path=os.path.join(step_dir, f"fields_step{il:02d}_prop{prop_cnt}.png"), test_num=300, levels=40, cmap='RdBu_r')
                print(f"  [stop] J={J:.6f} >= {J_abs_cutoff}, 结束。")
                v_mon = monitor_top_displacement(pinn)
                ld_records.append((float(Q), float(v_mon)))
                np.save(os.path.join(out_dir, 'all_records.npy'), np.array(all_records, dtype=float))
                break

            # 判据：稳定（不扩展）
            if (J <= Gc) or (prop_cnt >= max_propagations_per_load):
                if prop_cnt >= max_propagations_per_load:
                    print(f"  Reached max prop per load ({max_propagations_per_load})")
                print(f"  Stable at load step {il}: J={J:.6f} <= Gc={Gc:.6f}.")
                cur_lora = os.path.join(out_dir, f"lora_step{il:02d}.pth")
                save_lora(pinn.model, cur_lora)

                v_mon = monitor_top_displacement(pinn)
                ld_records.append((float(Q), float(v_mon)))
                plot_comprehensive_fields(pinn, embedding, out_path=os.path.join(step_dir, f"fields_step{il:02d}_prop{prop_cnt}.png"), test_num=300, levels=40, cmap='RdBu_r')
                break
            
            # a_increment = a_increment_list[prop_cnt]
            # 需要扩展：最大环向拉应力方向
            dtheta = float(max_stress_theta(K1, K2))
            new_theta = beta_local + dtheta
            new_tip = [tipx + a_increment*np.cos(new_theta),
                       tipy + a_increment*np.sin(new_theta)]
            print(f"    [prop] dtheta={np.degrees(dtheta):.2f}°, theta={np.degrees(new_theta):.2f}°, new_tip={new_tip}")

            # 更新一下裂纹尖端位置，用于构造u和v的extend function
            pinn.x_crackTip = new_tip[0]
            pinn.y_crackTip = new_tip[1]
            pinn.beta = new_theta


            # 边界/孔碰撞判定
            if tip_out_of_domain(new_tip):
                plot_comprehensive_fields(pinn, embedding, out_path=os.path.join(out_dir, 'fields.png'), test_num=300, levels=40, cmap='RdBu_r')
                print("  裂纹将越界/入孔，停止扩展并结束。")
                v_mon = monitor_top_displacement(pinn)
                ld_records.append((float(Q), float(v_mon)))
                return

            # 更新裂纹几何
            pinn.x_crackTip, pinn.y_crackTip, pinn.beta = new_tip[0], new_tip[1], new_theta
            points.append(new_tip)
            embedding = Embedding_bit.multiLineCrackEmbedding(points, tip='right')
            pinn.model.set_extend_axis(embedding)

            # 在新几何下继续训练（仅 LoRA）
            prop_cnt += 1
            save_tag = os.path.join(step_dir, f'prop{prop_cnt}')
            set_optimizer_lora_only(pinn, lr_init)
            set_triangle_quadrature_points_with_holes(pinn, point_num, point_num)
            pinn.train(path=save_tag, epochs=epochs_per_state, patience=patience,
                       lr=lr_init, eval_sep=eval_sep, tol_early_stop=tol_early_stop)
            if os.path.exists(save_tag + '.pth'):
                pinn.load(path=os.path.join(step_dir, f'prop{prop_cnt}'))
            save_lora(pinn.model, os.path.join(out_dir, f"lora_step{il:02d}_prop{prop_cnt}.pth"))
            plot_comprehensive_fields(pinn, embedding, out_path=os.path.join(step_dir, f"fields_step{il:02d}_prop{prop_cnt}.png"), test_num=300, levels=40, cmap='RdBu_r')

            # 缓存点集
            torch.save(points, os.path.join(step_dir, f'points.pt'))

    # 结果导出
    np.save(os.path.join(out_dir, 'all_records.npy'), np.array(all_records, dtype=float))
    # 导出力—位移
    csv_path = os.path.join(out_dir, 'Q_vs_v_monitor.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['Q_N', 'v_monitor_mm(negative=down)'])
        for Q, v in ld_records: w.writerow([Q, v])
    print(f"[done] 力—位移数据已保存：{csv_path}")

# =========================
# 云图绘制工具
# =========================

def plot_comprehensive_fields(pinn, embedding, out_path=None, test_num=300, levels=30, cmap='RdBu_r', draw_crack=True, draw_holes=True):
    """
    画 2×3 综合图：u1, u2, σ11, σ22, σ12, von Mises，并另外输出 crack function Γ。
    - `out_path` 形如 '.../fields.png'，若提供则保存；否则 plt.show()。
    - `embedding.points` 用于叠加裂纹折线。
    - 自动剔除孔洞显示（置 NaN）。
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # 可视网格（与训练域一致）
    x_vis = torch.linspace(-pinn.b, pinn.b, test_num, device=pinn.device, dtype=torch.get_default_dtype())
    y_vis = torch.linspace(-pinn.h, pinn.h, test_num, device=pinn.device, dtype=torch.get_default_dtype())
    Xv, Yv = torch.meshgrid(x_vis, y_vis, indexing='ij')
    XY = torch.stack([Xv.reshape(-1), Yv.reshape(-1)], dim=1).requires_grad_(True)

    # 前向推理（一次拿全量）
    with torch.set_grad_enabled(True):
        u, v, sx, sy, sxy = pinn.infer(XY)
    # 裂纹函数（若提供）
    try:
        gamma = embedding.getGamma(XY)
    except Exception:
        gamma = torch.zeros_like(u)

    # von Mises（平面问题惯用等效应力）
    von_mises = torch.sqrt(torch.clamp(sx**2 + sy**2 - sx*sy + 3.0*sxy**2, min=0.0))

    # to numpy & reshape
    def T(t):
        return t.detach().cpu().numpy().reshape(test_num, test_num)
    Xn, Yn = Xv.detach().cpu().numpy(), Yv.detach().cpu().numpy()
    Un, Vn = T(u), T(v)
    Sxx, Syy, Sxy = T(sx), T(sy), T(sxy)
    VM = T(von_mises)
    Gn = T(gamma)


    # 掩膜孔洞（置 NaN）
    if draw_holes:
        mask = np.zeros_like(Xn, dtype=bool)
        for (xc, yc, R) in HOLES:
            mask |= ((Xn - xc)**2 + (Yn - yc)**2) < (R**2 - 1e-12)
        for arr in (Un, Vn, Sxx, Syy, Sxy, VM, Gn):
            arr[mask] = np.nan

    # 统一绘图助手
    def _cont(ax, Z, title, label):
        cs = ax.contourf(Xn, Yn, Z, levels=levels, cmap=cmap)
        cb = plt.colorbar(cs, ax=ax, label=label)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlim([-pinn.b, pinn.b])
        ax.set_ylim([-pinn.h, pinn.h])
        # 叠加孔洞与裂纹
        if draw_holes:
            for (xc, yc, R) in HOLES:
                th = np.linspace(0, 2*np.pi, 200)
                ax.plot(xc + R*np.cos(th), yc + R*np.sin(th), lw=1.0)
        if draw_crack and hasattr(embedding, 'points'):
            pts = np.array(embedding.points)
            if pts.ndim == 2 and pts.shape[1] == 2:
                ax.plot(pts[:,0], pts[:,1], 'k-', lw=2.0)
                ax.plot(pts[-1,0], pts[-1,1], 'ko', ms=4)
        return cs

    # 2×3 综合图
    fig = plt.figure(figsize=(18, 8))
    ax1 = plt.subplot(2,3,1); _cont(ax1, Un,  'u1 Displacement', 'u1')
    ax2 = plt.subplot(2,3,2); _cont(ax2, Vn,  'u2 Displacement', 'u2')
    ax3 = plt.subplot(2,3,3); _cont(ax3, Sxx, 'σ11 Stress',      'σ11')
    ax4 = plt.subplot(2,3,4); _cont(ax4, Syy, 'σ22 Stress',      'σ22')
    ax5 = plt.subplot(2,3,5); _cont(ax5, Sxy, 'σ12 Stress',      'σ12')
    ax6 = plt.subplot(2,3,6); _cont(ax6, VM,  'von Mises Stress','σ_vm')
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[plot] saved -> {out_path}")
        # 另外单独保存 Γ
        base = os.path.splitext(out_path)[0]
        figG = plt.figure(figsize=(8,6))
        axG = figG.add_subplot(111); _cont(axG, Gn, 'Crack Function (Gamma)', 'Gamma')
        figG.tight_layout(); figG.savefig(base + '_gamma.png', dpi=300, bbox_inches='tight')
        plt.close(figG)
        print(f"[plot] saved -> {base + '_gamma.png'}")
    else:
        plt.show()

# =========================
# 入口
# =========================
if __name__ == "__main__":
    run_load_control_with_growth(a_increment)
    # 用法示例（训练后手动加载最新一步模型/裂纹后，再调用）
    # pinn, embedding = init_pinn(points)  # 或从 checkpoint 读回
    # plot_comprehensive_fields(pinn, embedding, out_path=os.path.join(out_dir, 'fields.png'), test_num=300)
