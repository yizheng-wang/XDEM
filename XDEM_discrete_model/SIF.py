import torch
import DENNs
from Embedding import extendAxisNet,multiEmbedding,Embedding
from utils.Geometry import Geometry1D,LocalAxis
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.get_grad import get_grad
from utils import Integral
from utils import NodesGenerater

def get_delta_u(model:DENNs.PINN2D,
                embedding:Embedding,
                crack_surface:Geometry1D,num,
                local_axis:LocalAxis):
    x , y  = crack_surface.generate_linespace_points(num)
    x , y , xy = model._set_points(x,y)

    '''下表面位移直接求出来''' 

    embedding.set_ls(-1.0) # 把heaviside设置为-1,这样输出的gamma就是负数了
    u_low,v_low = model.pred_uv(xy)
    '''上表面位移需要反一下裂纹面extension的符号'''

    embedding.set_ls(1.0) # 把heaviside设置为1,这样输出的gamma就是正数了
    u_up,v_up = model.pred_uv(xy)
    embedding.restore_ls()

    delta_u = u_up - u_low
    delta_v = v_up - v_low

    # 坐标转换
    delta_u , delta_v = local_axis.cartesianVariableToLocal(delta_u,delta_v)

    r = local_axis.getR(x,y).unsqueeze(-1)

    return r , delta_u.unsqueeze(-1) , delta_v.unsqueeze(-1)



def DispExpolation_homo(model:DENNs.PINN2D,
                        embedding:Embedding,
                    crack_surface:Geometry1D,num,
                    local_axis:LocalAxis,
                    kappa,mu):
    

    '''SIF单位:MPa*sqrt(mm)'''
    r , delta_u , delta_v = get_delta_u(model,embedding,crack_surface,num,local_axis)
    r_sqrt = torch.sqrt(r)

    material_coefficient = mu/(kappa+1) * np.sqrt(np.pi*2)
    K1_bar = material_coefficient * delta_v / r_sqrt
    K2_bar = material_coefficient * delta_u / r_sqrt

    K1_bar = K1_bar.cpu().detach().numpy()
    K2_bar = K2_bar.cpu().detach().numpy()
    r      = r.cpu().detach().numpy()

    K1_model = LinearRegression() 
    K1_model.fit(r, K1_bar)  
    K1 = K1_model.intercept_

    K2_model = LinearRegression()  
    K2_model.fit(r, K2_bar)  
    K2 = K2_model.intercept_
    return K1,K2
                                        

def DispExpolation_bimaterial(model:DENNs.PINN2D,
                    embedding:Embedding,
                    crack_surface:Geometry1D,num,
                    local_axis:LocalAxis,
                    kappa_up,mu_up,kappa_low,mu_low):
    

    '''SIF单位:MPa*sqrt(mm)'''
    r , delta_u , delta_v = get_delta_u(model,embedding,crack_surface,num,local_axis)
    r_sqrt = torch.sqrt(r)

 
    eps = np.log( (kappa_up / mu_up + 1 / mu_low) / (kappa_low/ mu_low + 1 / mu_up) ) / (2*np.pi)

    Q = eps * torch.log(r/1000) #Q=eps*ln(r/2a)

    C = 2 * np.cosh(eps * np.pi)  * np.sqrt(np.pi * 2) / (kappa_up / mu_up + 1 / mu_low + kappa_low/ mu_low + 1 / mu_up)

    cosQ = torch.cos(Q)
    sinQ = torch.sin(Q)

    e1 = cosQ + 2 * eps * sinQ
    e2 = sinQ - 2 * eps * cosQ

    K1_bar = C * (delta_v * e1 + delta_u * e2) / r_sqrt 
    K2_bar = C * (delta_u * e1 - delta_v * e2) / r_sqrt

    K1_bar = K1_bar.cpu().detach().numpy()
    K2_bar = K2_bar.cpu().detach().numpy()
    r      = r.cpu().detach().numpy()

    K1_model = LinearRegression()  
    K1_model.fit(r, K1_bar)  
    K1 = K1_model.intercept_

    K2_model = LinearRegression()  
    K2_model.fit(r, K2_bar) 
    K2 = K2_model.intercept_

    return K1,K2

def max_stress_theta(K1,K2):
    '''
    最大环向应力计算扩展角度
    '''
    
    '''环向应力极值方向,包含最大值与最小值'''
    theta = np.arctan(np.array([
                (K1 + np.sqrt(K1**2+8*K2**2)) / (4*K2) ,
                (K1 - np.sqrt(K1**2+8*K2**2)) / (4*K2) ,
            ]))
    theta = np.arccos(np.array([
        (3*K2**2 + np.sqrt(K1**4+8 * K1**2 * K2**2)) / (K1**2 + 9 * K2**2),
        (3*K2**2 - np.sqrt(K1**4+8 * K1**2 * K2**2)) / (K1**2 + 9 * K2**2)
    ]))
    theta = theta[theta<np.pi/2]
    theta = np.concatenate((theta,-theta),0)
    
    '''计算环向应力'''
    stress_theta = np.cos(theta/2) * (K1*(1+np.cos(theta)) - 3*K2*np.sin(theta))
    
    return theta[np.argmax(stress_theta)]



class SIF_K1K2(object):
    def __init__(self, model:DENNs.PINN2D, device:torch.device):
        self.model = model
        self.device = device

    def _rotate_to_local(self, tens, beta):
        # tens: (..., 2) 或 (..., 2, 2)
        c = torch.cos(torch.tensor(beta, device=self.device))
        s = torch.sin(torch.tensor(beta, device=self.device))
        R = torch.stack([torch.stack([c, s], -1),
                        torch.stack([-s, c], -1)], -2)  # 2x2
        if tens.dim() >= 2 and tens.size(-1) == 2 and tens.size(-2) == 2:
            # 二阶张量: σ' = R σ R^T
            return R @ tens @ R.transpose(-1, -2)
        elif tens.size(-1) == 2:  # 向量: v' = R v
            return (R @ tens.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError("Shape not supported")

    def calculate_j_integral_local(self, contour_points, contour_normals, crack_tip_xy, radius, beta=0.0):
        """
        在裂纹局部坐标系 (x1: 沿裂纹, x2: 法向) 计算 J = ∮ [W δ1j - σij u_{i,1}] n_j dΓ
        contour_points/normals: numpy (N,2), 为以 crack_tip_xy 为圆心、半径 radius 的圆
        """
        # 准备张量
        contour_xy = torch.tensor(contour_points, dtype=torch.float32, requires_grad=True).to(self.device)
        normals_g  = torch.tensor(contour_normals, dtype=torch.float32).to(self.device)

        # 预测位移 (全局)
        u, v = self.model.pred_uv(contour_xy)                    # (N,)
        # 应变->应力 (全局)
        eXX, eYY, eXY = self.model.compute_Strain(u, v, contour_xy)  # 假定 eXY 为张量剪切应变 ε_xy
        sx, sy, sxy   = self.model.constitutive(eXX, eYY, eXY)

        # 位移梯度 (全局)
        grads = torch.autograd.grad(u.sum()+v.sum(), contour_xy, create_graph=True)[0]  # (N,2), du/dx, du/dy + dv/dx, dv/dy 混在一起不方便
        du_dx = torch.autograd.grad(u.sum(), contour_xy, create_graph=True)[0]          # (N,2)
        dv_dx = torch.autograd.grad(v.sum(), contour_xy, create_graph=True)[0]          # (N,2)

        # 组装二阶张量与向量 (全局)
        sig_g = torch.stack([torch.stack([sx,  sxy], -1),
                            torch.stack([sxy, sy ], -1)], -2)          # (N,2,2)
        # u_{i,1} = ∂u_i/∂x1 (先在全局有 du/dx,du/dy; dv/dx,dv/dy，再旋转到局部并取对 x1 的分量)
        # 先把 ∇u, ∇v 旋转成局部基，等价于把 (du_dx, du_dy) 这个向量旋转
        grad_u_g = du_dx                                                      # (N,2)
        grad_v_g = dv_dx                                                      # (N,2)


        F_g = torch.stack([torch.stack([grad_u_g[:,0],  grad_u_g[:,1]], dim=-1),
                            torch.stack([grad_v_g[:,0], grad_v_g[:,1]], dim=-1)], dim=-2)  # (N,2,2)
        F_l = self._rotate_to_local(F_g, beta)
        # gradients as vectors, rotate to local to get u_{i,1}
        grad_u_l =  F_l[:,0] # (N,2)
        grad_v_l = F_l[:,1]  # (N,2)
        u1x = grad_u_l[:,0]
        u2x = grad_v_l[:,0]

        # 旋转：法向、应力 到局部
        n_l   = self._rotate_to_local(normals_g, beta)                        # (N,2) -> [n1, n2]
        sig_l = self._rotate_to_local(sig_g, beta)                            # (N,2,2)

        n1, n2 = n_l[:,0], n_l[:,1]
        s11 = sig_l[:,0,0]; s12 = sig_l[:,0,1]
        s21 = sig_l[:,1,0]; s22 = sig_l[:,1,1]

        # 局部能量密度 W = 1/2 σ:ε ；注意 eXY 与 sxy 是张量剪切
        W = 0.5*(eXX * sx + eYY * sy + eXY * sxy)
        # W = 0.5*(u1x * s11 + u2y * s22 + (u1y + u2x) * s12)

        # 线积分离散：J = ∮ ( W n1 - (σ11 u1x + σ21 u2x) n1 - (σ12 u1x + σ22 u2x) n2 ) dΓ
        # 对圆：dΓ = R dθ；等角度点：Δθ = 2π/N
        N  = contour_xy.shape[0]
        dtheta = 2*np.pi / N
        integrand = ( W * n1
                    - (s11 * u1x + s21 * u2x) * n1
                    - (s12 * u1x + s22 * u2x) * n2 )                      # (N,)
        J = torch.sum(integrand) * radius * dtheta
        return J

    def calculate_j_integral_global(self, contour_points, contour_normals, crack_tip_xy, radius, beta=0.0):
        """
        在裂纹全局坐标系 (x1: 沿裂纹, x2: 法向) 计算 J = ∮ [W δ1j - σij u_{i,1}] n_j dΓ
        contour_points/normals: numpy (N,2), 为以 crack_tip_xy 为圆心、半径 radius 的圆
        """
        contour_xy = torch.tensor(contour_points, dtype=torch.float32, requires_grad=True).to(self.device)
        normals = torch.tensor(contour_normals, dtype=torch.float32).to(self.device)
        
        # Get displacement and stress at contour points
        u, v = self.model.pred_uv(contour_xy)
        
        # Calculate strain energy density
        eXX, eYY, eXY = self.model.compute_Strain(u, v, contour_xy)
        sx, sy, sxy = self.model.constitutive(eXX, eYY, eXY)
        W = 0.5 * (eXX * sx + eYY * sy + eXY * sxy)
        
        # Calculate displacement gradients
        dv_dx = torch.autograd.grad(v.sum(), contour_xy, create_graph=True)[0][:,0]
        
        # J-integral components
        # J1 = ∫ [W - σxx ∂u/∂x - σxy ∂v/∂x] nx dΓ
        # J2 = ∫ [-σxy ∂u/∂x - σyy ∂v/∂x] ny dΓ
        
        J1 = W * normals[:, 0] - normals[:, 0] * sx * eXX - normals[:, 0] * sxy * dv_dx -  normals[:, 1] * sxy * eXX -  normals[:, 1] * sy * dv_dx
        J1 = torch.mean(J1)*radius*2*np.pi

        return J1

    def _cartesian_stress_to_polar_local(self, sig_l, rel_xy_l):
        """
        把局部直角坐标下的应力 σ_ij 转成极坐标分量 σ_rr, σ_rθ, σ_θθ
        rel_xy_l: (N,2) 为相对裂尖坐标(局部)
        """
        x = rel_xy_l[:,0]; y = rel_xy_l[:,1]
        r = torch.clamp(torch.sqrt(x**2 + y**2), min=1e-9)
        c = x / r; s = y / r   # cosθ, sinθ

        c2 = c*c; s2 = s*s; cs = c*s
        sxx = sig_l[:,0,0]; sxy = sig_l[:,0,1]; syy = sig_l[:,1,1]

        srr   = sxx*c2 + 2*sxy*cs + syy*s2
        stt   = sxx*s2 - 2*sxy*cs + syy*c2
        srtt  = -sxx*cs + sxy*(c2 - s2) + syy*cs
        return srr, srtt, stt, r, c, s

    def estimate_phase_angle(self, crack_tip_xy, sample_r, beta=0.0, n_samples=16):
        """
        在 θ≈0 的一条小扇区上用应力比估计 ψ=atan(σ_rθ/σ_θθ)|_{θ=0}
        """
        # 取一个很小的环扇区，θ ∈ [-Δ, Δ]
        thetas = torch.linspace(-0, 0, n_samples, device=self.device)  # 小角度
        x1 = sample_r * torch.cos(thetas)
        x2 = sample_r * torch.sin(thetas)
        pts_local = torch.stack([x1, x2], -1)                              # (n,2) 局部
        # 旋回全局坐标点
        c = torch.cos(torch.tensor(beta, device=self.device)); s = torch.sin(torch.tensor(beta, device=self.device))
        R_T = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # R^T
        pts_global = (R_T @ pts_local.unsqueeze(-1)).squeeze(-1)
        pts_global = pts_global + torch.tensor(crack_tip_xy, device=self.device, dtype=torch.float32)
        pts_global.requires_grad = True
        u, v = self.model.pred_uv(pts_global)
        eXX, eYY, eXY = self.model.compute_Strain(u, v, pts_global)
        sx, sy, sxy   = self.model.constitutive(eXX, eYY, eXY)

        sig_g = torch.stack([torch.stack([sx,  sxy], -1),
                            torch.stack([sxy, sy ], -1)], -2)
        sig_l = self._rotate_to_local(sig_g, beta)

        srr, srtheta, stheta, r, c0, s0 = self._cartesian_stress_to_polar_local(sig_l, pts_local)

        # 在 θ≈0 处取加权平均降低噪声
        w = torch.cos(thetas).clamp(min=0)  # 0附近权重大
        num = torch.sum(w * srtheta)
        den = torch.sum(w * stheta.clamp(min=1e-9))
        psi = torch.atan2(num, den)
        return psi

    def j_to_k1_k2_from_phase(self, J_value, E, nu, psi, plane_strain=True):
        if plane_strain:
            E_prime = E / (1.0 - nu**2)
        else:
            E_prime = E
        K_eff = torch.sqrt(torch.clamp(J_value * E_prime, min=0.0))
        KI  = K_eff * torch.cos(psi)
        KII = K_eff * torch.sin(psi)
        return KI, KII

    def compute_k1_k2_via_j(self, crack_tip_xy, radius, beta, E, nu, plane_strain=True,
                            num_points=720, phase_sample_r=None):
        """
        先算 J（局部坐标线积分），再估计 ψ，最后拆出 KI,KII
        """
        # 生成圆形轮廓（全局）
        theta = torch.linspace(0, 2*np.pi, num_points, device=self.device)
        x_contour = radius*torch.cos(torch.tensor(beta)) + radius * torch.cos(theta)
        y_contour = radius*torch.sin(torch.tensor(beta)) + radius * torch.sin(theta)
        # x_contour = torch.tensor(crack_tip_xy[0], device=self.device) + radius * torch.cos(theta)
        # y_contour = torch.tensor(crack_tip_xy[1], device=self.device) + radius * torch.sin(theta)
        contour_points = torch.stack([x_contour, y_contour], dim=1).detach().cpu().numpy()
        normals = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1).detach().cpu().numpy()

        # J（局部）
        J = self.calculate_j_integral_local(contour_points, normals, crack_tip_xy, radius, beta)

        # 估计 ψ：若未给定采样半径，就取比积分半径更小的一点（比如 R/5）
        if phase_sample_r is None:
            phase_sample_r = 0.2 * radius
        psi = self.estimate_phase_angle(crack_tip_xy, phase_sample_r, beta=beta, n_samples=21)

        # 拆出 KI,KII
        KI, KII = self.j_to_k1_k2_from_phase(J, E, nu, psi, plane_strain=plane_strain)
        return KI, KII, J, psi

class M_integral(object):
    def __init__(self, model:DENNs.PINN2D, device:torch.device):
        self.model = model
        self.device = device
    def _rot_mat(self, beta: float, device):
        c = torch.cos(torch.tensor(beta, device=device))
        s = torch.sin(torch.tensor(beta, device=device))
        R = torch.stack([torch.stack([c,  s], -1),
                        torch.stack([-s, c], -1)], -2)  # 2x2
        return R

    def _strain_vec_to_tensor(self, exx, eyy, exy):
        # exy 为张量剪切应变 ε_xy（非工程应变）
        # 返回 2x2 应变张量
        e12 = exy
        e21 = exy
        E = torch.stack([
            torch.stack([exx, e12], -1),
            torch.stack([e21, eyy], -1)
        ], -2)
        return E

    def _tensor_to_strain_vec(self, E):
        # 2x2 -> (exx, eyy, exy)
        exx = E[...,0,0]
        eyy = E[...,1,1]
        exy = 0.5*(E[...,0,1] + E[...,1,0])  # 张量剪切
        return exx, eyy, exy

    def _rotate_tensor(self, T, beta, device):
        R = self._rot_mat(beta, device)
        return R @ T @ R.transpose(-1, -2)

    def _rotate_vector(self, v, beta, device):
        R = self._rot_mat(beta, device)
        return (R @ v.unsqueeze(-1)).squeeze(-1)

    def _aux_stress_local_unitKI(self, r, th):
        # 近场 Williams 展开，单位 KI=1（单位 MPa*sqrt(mm)）
        # 返回 σ_xx, σ_yy, σ_xy（局部直角坐标, x1 沿裂纹）
        root = torch.sqrt(2*np.pi*r).clamp(min=1e-12)
        c = torch.cos(th/2.0); s = torch.sin(th/2.0)
        c3 = torch.cos(3*th/2.0); s3 = torch.sin(3*th/2.0)

        sxx = (c*(1 - s*s3)) / root
        syy = (c*(1 + s*s3)) / root
        sxy = (s*c*c3) / root
        return sxx, syy, sxy

    def _aux_stress_local_unitKII(self, r, th):
        # 单位 KII=1
        root = torch.sqrt(2*np.pi*r).clamp(min=1e-12)
        c = torch.cos(th/2.0); s = torch.sin(th/2.0)
        c3 = torch.cos(3*th/2.0); s3 = torch.sin(3*th/2.0)

        sxx = -(s*(2 + c*c3)) / root
        syy =  (s*c*c3) / root
        sxy =   c*(1 - s*s3)  / root
        return sxx, syy, sxy

    def aux_u_col1_I(self, r, theta, E, nu, plane_strain=True):
        """
        单位 KI=1 辅助场的 u_{i,1}^a (i=1,2)
        r, theta: 张量 (N,)
        返回: (N,2) = [u1,1^a, u2,1^a]
        """
        mu = E / (2*(1+nu))
        if plane_strain:
            kappa = 3 - 4*nu
        else:
            kappa = (3 - nu) / (1 + nu)

        fac = 1.0 / (2*mu) * torch.sqrt(r/(2*np.pi))

        # 辅助场位移 (KI=1)
        u1 = fac * torch.cos(theta/2) * (kappa - 1 + 2*torch.sin(theta/2)**2)
        u2 = fac * torch.sin(theta/2) * (kappa + 1 - 2*torch.cos(theta/2)**2)

        # 位移对 r,θ 的导数
        du1_dr = (1/(4*mu)) * (1/torch.sqrt(2*np.pi*r)) * torch.cos(theta/2) * (kappa - 1 + 2*torch.sin(theta/2)**2)
        du2_dr = (1/(4*mu)) * (1/torch.sqrt(2*np.pi*r)) * torch.sin(theta/2) * (kappa + 1 - 2*torch.cos(theta/2)**2)

        du1_dtheta = fac * (torch.sin(theta)*torch.cos(theta/2) + (kappa-torch.cos(theta))*(-1)*(torch.sin(theta/2))/2)
        du2_dtheta = fac * (torch.sin(theta)*torch.sin(theta/2) + (kappa - torch.cos(theta))*torch.cos(theta/2)/2)

        # 换算到 x1 导数
        cosT, sinT = torch.cos(theta), torch.sin(theta)
        u1x = cosT*du1_dr - (sinT/r)*du1_dtheta
        u2x = cosT*du2_dr - (sinT/r)*du2_dtheta

        return torch.stack([u1x, u2x], -1)


    def aux_u_col1_II(self, r, theta, E, nu, plane_strain=True):
        """
        单位 KII=1 辅助场的 u_{i,1}^a (i=1,2)
        """
        mu = E / (2*(1+nu))
        if plane_strain:
            kappa = 3 - 4*nu
        else:
            kappa = (3 - nu) / (1 + nu)

        fac = 1.0 / (2*mu) * torch.sqrt(r/(2*np.pi))

        # 辅助场位移 (KII=1)
        u1 = fac * 0.5*((2*kappa+3)*torch.sin(theta/2)+torch.sin(theta*1.5))
        u2 = fac * 0.5*(-(2*kappa-3)*torch.cos(theta/2)-torch.cos(theta*1.5))
        # 位移对 r,θ 的导数
        du1_dr = (1/(4*mu)) * (1/torch.sqrt(2*np.pi*r)) * 0.5*((2*kappa+3)*torch.sin(theta/2)+torch.sin(theta*1.5))
        du2_dr = (1/(4*mu)) * (1/torch.sqrt(2*np.pi*r)) * 0.5*(-(2*kappa-3)*torch.cos(theta/2)-torch.cos(theta*1.5))

        du1_dtheta = fac * 0.5 * ((2*kappa+3)*torch.cos(theta/2)*0.5 + (torch.cos(theta*1.5)*1.5))
        du2_dtheta = fac * 0.5 * ((2*kappa-3)*torch.sin(theta/2)*0.5 + (torch.sin(theta*1.5)*1.5))

        # 换算到 x1 导数
        cosT, sinT = torch.cos(theta), torch.sin(theta)
        u1x = cosT*du1_dr - (sinT/r)*du1_dtheta
        u2x = cosT*du2_dr - (sinT/r)*du2_dtheta

        return torch.stack([u1x, u2x], -1)



    def _compliance_eps_from_sigma(self, sxx, syy, sxy, E, nu, plane_strain: bool):
        # 由应力得到应变（张量剪切 exy）
        if plane_strain:
            # ε = S_ps * σ, 其中
            # exx = ((1-ν)/E')*sxx - (ν/E')*syy
            # eyy = - (ν/E')*sxx + ((1-ν)/E')*syy
            # γ_xy = 2*exy = τ_xy / G * 2 = 2 τ_xy / (E/(2(1+ν))) = 4(1+ν)/E * τ_xy
            Eprime = E/(1-nu**2)
            exx = sxx/Eprime - (nu/Eprime)/(1-nu) * syy
            eyy = syy/Eprime - (nu/Eprime)/(1-nu) * sxx
            exy = (1.0/(2.0*E/(2*(1+nu)))) * sxy  # exy = τ/(2G) ; G = E/(2(1+ν))
        else:
            # 平面应力
            exx = (1.0/E)*(sxx - nu*syy)
            eyy = (1.0/E)*(syy - nu*sxx)
            exy = (1.0/(2.0*E/(2*(1+nu)))) * sxy  # 同上：exy = τ/(2G)

        return exx, eyy, exy

    def compute_K_via_interaction_integral(self,
                                        crack_tip_xy,      # (2,)
                                        radius: float,
                                        beta: float,
                                        E: float, nu: float,
                                        plane_strain: bool = True,
                                        num_points: int = 720,
                                        device=None):
        """
        基于交错积分计算 K1, K2
        - 取单位辅助场 KI_aux=1 与 KII_aux=1
        - 线积分 (圆)：dΓ = R dθ
        """
        if device is None:
            device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")
        crack_tip_xy = np.array(crack_tip_xy)
        crack_tip_xy = crack_tip_xy.astype(np.float32) 
        # 1) 圆形轮廓（全局）
        theta = torch.linspace(0, 2*np.pi, num_points, device=device)
        cx = torch.tensor(crack_tip_xy[0], device=device)
        cy = torch.tensor(crack_tip_xy[1], device=device)
        xg = cx + radius * torch.cos(theta)
        yg = cy + radius * torch.sin(theta)
        contour_xy = torch.stack([xg, yg], -1).requires_grad_(True)

        # 外法向（全局）
        normals_g = torch.stack([torch.cos(theta), torch.sin(theta)], -1)  # 圆的外法向

        # 2) 模型实际场（全局 -> 局部）
        u, v = self.model.pred_uv(contour_xy)
        eXX, eYY, eXY = self.model.compute_Strain(u, v, contour_xy)  # eXY 为工程应变
        sx, sy, sxy = self.model.constitutive(eXX, eYY, eXY)

        # 组装 σ, ε 张量（全局）
        sig_g = torch.stack([torch.stack([sx,  sxy], -1),
                            torch.stack([sxy, sy ], -1)], -2)  # (N,2,2)
        eps_g = self._strain_vec_to_tensor(eXX, eYY, eXY/2)            # (N,2,2) eXY是工程应变

        # 旋到局部
        sig_l = self._rotate_tensor(sig_g, beta, device)             # (N,2,2)
        eps_l = self._rotate_tensor(eps_g, beta, device)             # (N,2,2)
        n_l   = self._rotate_vector(normals_g, beta, device)         # (N,2)

        # 3) 计算局部相对坐标与极坐标
        # 全局点 -> 局部点
        pts_local = self._rotate_vector(contour_xy - torch.tensor(crack_tip_xy, device=device), beta, device)
        x1 = pts_local[:,0]; x2 = pts_local[:,1]
        r  = torch.sqrt(torch.clamp(x1**2 + x2**2, min=1e-18))
        th = torch.atan2(x2, x1)

        # 4) 单位辅助场：应力 (局部) 以及由应力反推的辅助应变
        sxx_I, syy_I, sxy_I = self._aux_stress_local_unitKI(r, th)
        sxx_II, syy_II, sxy_II = self._aux_stress_local_unitKII(r, th)

        exx_I, eyy_I, exy_I = self._compliance_eps_from_sigma(sxx_I, syy_I, sxy_I, E, nu, plane_strain)
        exx_II, eyy_II, exy_II = self._compliance_eps_from_sigma(sxx_II, syy_II, sxy_II, E, nu, plane_strain)

        sig_aux_I = torch.stack([torch.stack([sxx_I,  sxy_I], -1),
                                torch.stack([sxy_I, syy_I ], -1)], -2)
        sig_aux_II = torch.stack([torch.stack([sxx_II,  sxy_II], -1),
                                torch.stack([sxy_II, syy_II ], -1)], -2)

        eps_aux_I = self._strain_vec_to_tensor(exx_I, eyy_I, exy_I)
        eps_aux_II = self._strain_vec_to_tensor(exx_II, eyy_II, exy_II)

        # 5) 混合能量密度 W_mix = 0.5(σ:ε_aux + σ_aux:ε)
        def double_contraction(A, B):
            return (A*B).sum(dim=(-1,-2))

        Wmix_I  = 0.5*(double_contraction(sig_l, eps_aux_I)  + double_contraction(sig_aux_I,  eps_l))
        Wmix_II = 0.5*(double_contraction(sig_l, eps_aux_II) + double_contraction(sig_aux_II, eps_l))

        # 6) 切向力 t = σ n 以及 ε_{\bullet 1}
        t_l       = sig_l @ n_l.unsqueeze(-1)         # (N,2,1)
        t_l_I_aux = sig_aux_I @ n_l.unsqueeze(-1)
        t_l_II_aux= sig_aux_II @ n_l.unsqueeze(-1)

        # ε_{•1} = [ε11, ε21]^T
        # 6) 位移梯度 ∇u，在局部坐标系下
        # F_g = [ [∂u1/∂x, ∂u1/∂y],
        #         [∂u2/∂x, ∂u2/∂y] ]
        du_dx = torch.autograd.grad(u.sum(), contour_xy, create_graph=True)[0]  # (N,2)
        dv_dx = torch.autograd.grad(v.sum(), contour_xy, create_graph=True)[0]  # (N,2)

        F_g = torch.stack([torch.stack([du_dx[:,0], du_dx[:,1]], -1),
                        torch.stack([dv_dx[:,0], dv_dx[:,1]], -1)], -2)  # (N,2,2)

        # 旋转到局部
        F_l = self._rotate_tensor(F_g, beta, device)  # (N,2,2)

        # 取 ∂u_i/∂x1 = F_l[:, i, 0]
        u1x = F_l[:,0,0]   # ∂u1/∂x1
        u2x = F_l[:,1,0]   # ∂u2/∂x1
        u_col1 = torch.stack([u1x, u2x], -1)  # (N,2)

        # 辅助场位移梯度列
        u_col1_I  = self.aux_u_col1_I(r, th, E, nu, plane_strain)
        u_col1_II = self.aux_u_col1_II(r, th, E, nu, plane_strain)

        # 7) 交错积分核：q = Wmix*n1 - t·eps_aux_col1 - t_aux·eps_col1
        n1 = n_l[:,0]
        integrand_I  = Wmix_I  * n1 - (t_l.squeeze(-1)*u_col1_I).sum(dim=-1)  - (t_l_I_aux.squeeze(-1)*u_col1).sum(dim=-1)
        integrand_II = Wmix_II * n1 - (t_l.squeeze(-1)*u_col1_II).sum(dim=-1) - (t_l_II_aux.squeeze(-1)*u_col1).sum(dim=-1)

        # 8) 线积分（等角度）：∮ q dΓ, 其中 dΓ = R dθ
        dtheta = 2*np.pi / num_points
        M_I  = torch.sum(integrand_I)  * radius * dtheta
        M_II = torch.sum(integrand_II) * radius * dtheta

        # 9) 由 M -> K
        Eprime = E/(1-nu**2) if plane_strain else E
        KI  = Eprime * M_I/2
        KII = Eprime * M_II/2
        return KI, KII, M_I, M_II




