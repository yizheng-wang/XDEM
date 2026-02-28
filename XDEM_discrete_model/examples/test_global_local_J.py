# We'll compute the J-integral and (K_I, K_II) from the user's K-field using autograd.
# This cell defines corrected utilities, evaluates J on a circular contour, estimates the phase angle,
# and prints the results.

import torch
import numpy as np
import matplotlib.pyplot as plt
# ===== Settings =====
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Material & crack params
E  = torch.tensor(1000.0, device=device)   # MPa
nu = torch.tensor(0.30,   device=device)
plane_strain = True
mu = E / (2 * (1 + nu))                    # shear modulus
kappa = 3 - 4 * nu if plane_strain else (3 - nu) / (1 + nu)

KI  = torch.tensor(100.0, device=device)   # MPa*sqrt(mm)
KII = torch.tensor(0.0,  device=device)    # unused in K_field below (pure mode I)
beta = torch.tensor(torch.pi/4, device=device)    # crack angle
a = torch.tensor(0.5, device=device)       # half length
xx, yy = torch.meshgrid(torch.linspace(-1.0, 1.0, 100), torch.linspace(-1.0, 1.0, 100))
# ===== K-field displacement (Mode I) =====
def K_field(x, y, a, beta):
    """
    Inputs:
      x, y: coordinates (broadcastable), tensors on device
      a: half crack length (scalar tensor)
      beta: crack orientation angle (scalar tensor, radians)

    Returns: u(x,y), v(x,y) in global axes
    """
    x = x.to(device); y = y.to(device)
    beta_t = beta.to(x.dtype)

    tip_x = a*torch.cos(beta_t)
    tip_y = a*torch.sin(beta_t)

    X = x - tip_x
    Y = y - tip_y

    # rotate into local crack coords (x_local aligned with crack)
    c = torch.cos(beta_t); s = torch.sin(beta_t)
    xloc =  X*c + Y*s
    yloc = -X*s + Y*c

    r    = torch.sqrt(xloc**2 + yloc**2).clamp_min(1e-18)
    th   = torch.atan2(yloc, xloc)

    # Displacements (Mode I near-tip field)
    fac  = (KI / (2*mu)) * torch.sqrt(r/(2*torch.pi))
    common = kappa - torch.cos(th)  # == (kappa - 1 + 2 sin^2(th/2))
    uloc = fac * torch.cos(th/2) * common
    vloc = fac * torch.sin(th/2) * common

    # rotate back to global
    u =  uloc*c - vloc*s
    v =  uloc*s + vloc*c
    return u, v

# ===== Utilities: grad, strain (tensorial), constitutive, rotations =====
def get_grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

def compute_Strain(u, v, xy):
    du = get_grad(u, xy)  # [du/dx, du/dy]
    dv = get_grad(v, xy)  # [dv/dx, dv/dy]
    eXX = du[...,0]
    eYY = dv[...,1]
    eXY = 0.5*(du[...,1] + dv[...,0])  # tensorial shear
    return eXX, eYY, eXY, du, dv

def constitutive_tensorial(eXX, eYY, eXY, E, nu, plane_strain=True):
    if plane_strain:
        lam = E*nu/((1+nu)*(1-2*nu))
        mu  = E/(2*(1+nu))
    else:
        lam = E*nu/(1 - nu**2)
        mu  = E/(2*(1+nu))
    tr = eXX + eYY
    sx  = 2*mu*eXX + lam*tr
    sy  = 2*mu*eYY + lam*tr
    sxy = 2*mu*eXY
    return sx, sy, sxy

def rotate_vec_to_local(vec_g, beta):
    """ vec_g: (...,2) """
    c = torch.cos(beta); s = torch.sin(beta)
    R = torch.stack([torch.stack([c, s]), torch.stack([-s, c])])  # 2x2
    return (vec_g @ R.T)  # (...,2)

def rotate_sig_to_local(sig_g, beta):
    """ sig_g: (...,2,2) """
    c = torch.cos(beta); s = torch.sin(beta)
    R = torch.stack([torch.stack([c, s]), torch.stack([-s, c])])  # 2x2
    RT = R.T
    # (...,2,2) -> R * sig * R^T
    return R @ sig_g @ RT

# ===== J-integral (local x1-direction) =====
def calculate_j_integral_local(crack_tip_xy, radius, beta, num_points=720):
# 生成从 0 开始、步长均匀的角度（不包含 2π，避免首尾重复）
    theta0 = torch.linspace(0, 2*np.pi, num_points, device=device, dtype=DTYPE)[:-1]
    # 平移到以 beta 开始，再映射回 [0, 2π)
    theta = (theta0 + beta) % (2*np.pi)
    x = crack_tip_xy[0] + radius*torch.cos(theta)
    y = crack_tip_xy[1] + radius*torch.sin(theta)
    xy = torch.stack([x, y], dim=1).requires_grad_(True)

    # displacements at contour
    u, v = K_field(xy[:,0], xy[:,1], a, beta)

    # strains & grads
    eXX, eYY, eXY, du, dv = compute_Strain(u, v, xy)
    sx, sy, sxy = constitutive_tensorial(eXX, eYY, eXY, E, nu, plane_strain=plane_strain)
    W = 0.5*(sx*eXX + sy*eYY + 2 * sxy*eXY)

    # pack strain and rotate to local
    e_g = torch.stack([torch.stack([eXX,  eXY], dim=-1),
                         torch.stack([eXY, eYY ], dim=-1)], dim=-2)  # (N,2,2)
    e_l = rotate_sig_to_local(e_g, beta)

    # pack stress and rotate to local
    sig_g = torch.stack([torch.stack([sx,  sxy], dim=-1),
                         torch.stack([sxy, sy ], dim=-1)], dim=-2)  # (N,2,2)
    sig_l = rotate_sig_to_local(sig_g, beta)

    # 计算下局部坐标系下的应变能密度
    W_l = 0.5*(e_l[:,0,0]*sig_l[:,0,0] + e_l[:,1,1]*sig_l[:,1,1] + 2*e_l[:,0,1]*sig_l[:,1,0])

    F_g = torch.stack([torch.stack([du[:,0],  du[:,1]], dim=-1),
                         torch.stack([dv[:,0], dv[:,1]], dim=-1)], dim=-2)  # (N,2,2)
    F_l = rotate_sig_to_local(F_g, beta)
    # gradients as vectors, rotate to local to get u_{i,1}
    grad_u_l =  F_l[:,0] # (N,2)
    grad_v_l = F_l[:,1]  # (N,2)
    u1x = grad_u_l[:,0]
    u2x = grad_v_l[:,0]

    # normals: outward unit normal on circle
    n_g = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    n_l = rotate_vec_to_local(n_g, beta)
    n1, n2 = n_l[:,0], n_l[:,1]

    s11 = sig_l[:,0,0]; s12 = sig_l[:,0,1]
    s21 = sig_l[:,1,0]; s22 = sig_l[:,1,1]

    integrand = W_l*n1 - (s11*u1x + s21*u2x)*n1 - (s12*u1x + s22*u2x)*n2
    J = torch.sum(integrand) * radius * (2*np.pi/num_points)
    return J

def calculate_j_integral_global(crack_tip_xy, radius, beta, num_points=720):
# 生成从 0 开始、步长均匀的角度（不包含 2π，避免首尾重复）
    theta0 = torch.linspace(0, 2*np.pi, num_points, device=device, dtype=DTYPE)[:-1]
    # 平移到以 beta 开始，再映射回 [0, 2π)
    theta = (theta0 + beta) % (2*np.pi)
    x = crack_tip_xy[0] + radius*torch.cos(theta)
    y = crack_tip_xy[1] + radius*torch.sin(theta)
    xy = torch.stack([x, y], dim=1).requires_grad_(True)

    # displacements at contour
    u, v = K_field(xy[:,0], xy[:,1], a, beta)

    # strains & grads
    eXX, eYY, eXY, du, dv = compute_Strain(u, v, xy)
    sx, sy, sxy = constitutive_tensorial(eXX, eYY, eXY, E, nu, plane_strain=plane_strain)
    W = 0.5*(sx*eXX + sy*eYY + 2 * sxy*eXY)
    # normals: outward unit normal on circle
    n_g = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    n1, n2 = n_g[:,0], n_g[:,1]

    u1x = du[:,0]
    u2x = dv[:,0]

    integrand = W*n1 - (sx*u1x + sxy*u2x)*n1 - (sxy*u1x + sy*u2x)*n2
    J = torch.sum(integrand) * radius * (2*np.pi/num_points)
    return J


# ===== Phase angle estimate near theta≈0 (local) =====
def estimate_phase_angle(crack_tip_xy, sample_r, beta=beta, n_samples=201):
    thetas = torch.linspace(-0.01, 0.01, n_samples, device=device, dtype=DTYPE)
    # local points
    x1 = sample_r*torch.cos(thetas)
    x2 = sample_r*torch.sin(thetas)
    pts_local = torch.stack([x1, x2], dim=1)

    # rotate back to global & shift by tip
    c = torch.cos(beta); s = torch.sin(beta)
    R_T = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])  # R^T
    pts_global = (pts_local @ R_T.T)
    pts_global = pts_global + torch.tensor(crack_tip_xy, device=device, dtype=DTYPE)

    pts_global.requires_grad_(True)
    u, v = K_field(pts_global[:,0], pts_global[:,1], a, beta)

    eXX, eYY, eXY, du, dv = compute_Strain(u, v, pts_global)
    sx, sy, sxy = constitutive_tensorial(eXX, eYY, eXY, E, nu, plane_strain=plane_strain)

    sig_g = torch.stack([torch.stack([sx,  sxy], dim=-1),
                         torch.stack([sxy, sy ], dim=-1)], dim=-2)
    sig_l = rotate_sig_to_local(sig_g, beta)

    # convert to polar (local) to get σ_rθ / σ_θθ at theta≈0
    x = pts_local[:,0]; y = pts_local[:,1]
    r = torch.sqrt(x**2 + y**2).clamp_min(1e-18)
    cth = x/r; sth = y/r
    sxx = sig_l[:,0,0]; sxy = sig_l[:,0,1]; syy = sig_l[:,1,1]

    # σ_rr, σ_rθ, σ_θθ (standard transform in local coords)
    srr   = sxx*cth*cth + 2*sxy*cth*sth + syy*sth*sth
    srth  = -sxx*cth*sth + sxy*(cth*cth - sth*sth) + syy*cth*sth
    sthth = sxx*sth*sth - 2*sxy*cth*sth + syy*cth*cth

    w = torch.cos(thetas).clamp_min(0)
    num = torch.sum(w * srth)
    den = torch.sum(w * sthth.clamp_min(1e-18))
    psi = torch.atan2(num, den)
    return psi

# ===== Convert J & phase to (K_I, K_II) =====
def j_to_k1_k2_from_phase(J_value, E, nu, psi, plane_strain=True):
    Eprime = E/(1-nu**2) if plane_strain else E
    K_eff = torch.sqrt(torch.clamp(J_value * Eprime, min=0.0))
    KI  = K_eff*torch.cos(psi)
    KII = K_eff*torch.sin(psi)
    return KI, KII

# 我们实验下，位移场生成的对不对
u1, v1 = K_field(xx, yy, a, beta)
xx = xx.cpu().numpy()
yy = yy.cpu().numpy()
u1 = u1.cpu().numpy()
v1 = v1.cpu().numpy()

# 计算位移的绝对值
u1_abs = np.abs(u1)
v1_abs = np.abs(v1)

# 找到位移绝对值接近零的位置（使用阈值）
threshold = 0.001  # 可以根据需要调整阈值
zero_u_mask = u1_abs < threshold
zero_v_mask = v1_abs < threshold

plt.figure(figsize=(15, 5))

# u1位移场
plt.subplot(1, 3, 1)
contour_u = plt.contourf(xx, yy, u1, levels=20, cmap='RdBu_r')
plt.colorbar(contour_u, label='u1 displacement')
# 标记零位移点
#zero_u_points = np.where(zero_u_mask)
#plt.scatter(xx[zero_u_points], yy[zero_u_points], c='red', s=10, alpha=0.7, label=f'|u1| < {threshold}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u1 Displacement Field')
plt.legend()
plt.axis('equal')

# v1位移场
plt.subplot(1, 3, 2)
contour_v = plt.contourf(xx, yy, v1, levels=20, cmap='RdBu_r')
plt.colorbar(contour_v, label='v1 displacement')
# 标记零位移点
#zero_v_points = np.where(zero_v_mask)
#plt.scatter(xx[zero_v_points], yy[zero_v_points], c='red', s=10, alpha=0.7, label=f'|v1| < {threshold}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('v1 Displacement Field')
plt.legend()
plt.axis('equal')

# 合位移场
plt.subplot(1, 3, 3)
total_displacement = np.sqrt(u1**2 + v1**2)
contour_total = plt.contourf(xx, yy, total_displacement, levels=20, cmap='viridis')
plt.colorbar(contour_total, label='Total displacement magnitude')
# 标记合位移为零的点
zero_total_mask = total_displacement < threshold
#zero_total_points = np.where(zero_total_mask)
#plt.scatter(xx[zero_total_points], yy[zero_total_points], c='red', s=10, alpha=0.7, label=f'|u| < {threshold}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Total Displacement Magnitude')
plt.legend()
plt.axis('equal')

plt.tight_layout()
plt.show()

# ===== Run computation =====
crack_tip_xy = (a*torch.cos(beta), a*torch.sin(beta))
R = torch.tensor(0.15, device=device, dtype=DTYPE)

J = calculate_j_integral_local(crack_tip_xy, R, beta, num_points=720)
psi = estimate_phase_angle(crack_tip_xy, sample_r=0.02*R, beta=beta, n_samples=21)
KI_est, KII_est = j_to_k1_k2_from_phase(J, E, nu, psi, plane_strain=plane_strain)

# Theoretical J for Mode I in plane strain: J = KI^2 / E'
Eprime = E/(1-nu**2) if plane_strain else E
J_theory = KI**2 / Eprime

J.item(), psi.item(), KI_est.item(), KII_est.item(), J_theory.item()

print("J: ", J.item(), "psi: ", psi.item(), "KI_est: ", KI_est.item(), "KII_est: ", KII_est.item(), "J_theory: ", J_theory.item())