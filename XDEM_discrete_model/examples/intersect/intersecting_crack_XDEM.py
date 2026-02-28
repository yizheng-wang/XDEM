import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set random seed
import random
import numpy as np
import torch

from DENNs import PINN2D
import torch
import torch.nn as nn
from utils.NodesGenerater import genMeshNodes2D
from utils.NN import stack_net
from utils.Integral import trapz1D
import numpy as np
import Embedding
import utils.Geometry as Geometry
from Embedding import LineCrackEmbedding
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Fix random seed
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Plate(PINN2D):
    def __init__(self, model: nn.Module,fy,a,b,h,q, beta1, beta2):
        '''x范围0-1.y范围-1到1'''
        super().__init__(model)
        self.fy = fy
        K1_esti = 1.
        K2_esti = 1.
        self.K_I = nn.Parameter(torch.tensor(K1_esti, device=self.device))
        self.K_II = nn.Parameter(torch.tensor(K2_esti, device=self.device))
        self.a = a
        self.b = b
        self.h = h
        self.q = q
    def hard_u(self, u, x, y):
        # Calculate polar coordinates relative to crack tip
        c1 = torch.cos(torch.tensor(beta1, device=self.device))
        s1 = torch.sin(torch.tensor(beta1, device=self.device))
        c2 = torch.cos(torch.tensor(beta2, device=self.device))
        s2 = torch.sin(torch.tensor(beta2, device=self.device))
        # 全局->局部 的旋转 R = [[c, s],[-s, c]]
        # 右/左裂尖（全局坐标）
        tip_r1 = torch.stack([ self.a * c1,  self.a * s1]).to(x.device).to(x.dtype)
        tip_l1 = torch.stack([-self.a * c1, -self.a * s1]).to(x.device).to(x.dtype)
        tip_r2 = torch.stack([ self.a * c2,  self.a * s2]).to(x.device).to(x.dtype)
        tip_l2 = torch.stack([-self.a * c2, -self.a * s2]).to(x.device).to(x.dtype)

        # 组装点坐标
        xy = torch.stack([x, y], dim=-1)  # (...,2)

        # 相对坐标（全局）
        rel_r_g1 = xy - tip_r1
        rel_l_g1 = xy - tip_l1
        rel_r_g2 = xy - tip_r2
        rel_l_g2 = xy - tip_l2

        # 旋到局部：v_local = R v_global
        R1 = torch.stack([torch.stack([c1, s1], dim=-1),
                        torch.stack([-s1, c1], dim=-1)], dim=0)  # (2,2)
        R2 = torch.stack([torch.stack([c2, s2], dim=-1),
                        torch.stack([-s2, c2], dim=-1)], dim=0)  # (2,2)
        rel_r_l1 = rel_r_g1 @ R1.T
        rel_l_l1 = rel_l_g1 @ R1.T
        rel_r_l2 = rel_r_g2 @ R2.T
        rel_l_l2 = rel_l_g2 @ R2.T



        x1_r1, x2_r1 = rel_r_l1[..., 0], rel_r_l1[..., 1]
        x1_l1, x2_l1 = -rel_l_l1[..., 0], rel_l_l1[..., 1]
        x1_r2, x2_r2 = rel_r_l2[..., 0], rel_r_l2[..., 1]
        x1_l2, x2_l2 = -rel_l_l2[..., 0], rel_l_l2[..., 1]
        r_right1   = torch.sqrt(x1_r1**2 + x2_r1**2)
        r_left1    = torch.sqrt(x1_l1**2 + x2_l1**2)
        r_right2   = torch.sqrt(x1_r2**2 + x2_r2**2)
        r_left2    = torch.sqrt(x1_l2**2 + x2_l2**2)
        theta_right1 = torch.atan2(x2_r1, x1_r1)
        theta_left1  = torch.atan2(x2_l1, x1_l1)
        theta_right2 = torch.atan2(x2_r2, x1_r2)
        theta_left2  = torch.atan2(x2_l2, x1_l2)
    
        # u_analytical = self.K_I[0] * r**0.5 * torch.sin(theta/2) + \
        #                self.K_I1[1] * r**0.5 * torch.sin(theta/2) * torch.sin(theta) + \
        #                self.K_I1[2] * r**0.5 * torch.cos(theta/2) + \
        #                self.K_I1[3] * r**0.5 * torch.cos(theta/2) * torch.sin(theta)

        # Calculate Mode I crack analytical solution
        mu = E / (2 * (1 + nu))
        kappa = 3 - 4 * nu  # Plane strain
        
        # K_I is now a trainable parameter (already initialized in __init__)
        # Analytical displacement components
        u_analytical_right1 = (self.K_I / (2 * mu)) * torch.sqrt(r_right1 / (2 * np.pi)) * torch.cos(theta_right1/2) * (kappa - 1 + 2 * torch.sin(theta_right1/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right1 / (2 * np.pi)) * torch.sin(theta_right1/2) * (2 + kappa + torch.cos(theta_right1))
        v_analytical_right1 = (self.K_I / (2 * mu)) * torch.sqrt(r_right1 / (2 * np.pi)) * torch.sin(theta_right1/2) * (kappa + 1 - 2 * torch.cos(theta_right1/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right1 / (2 * np.pi)) * torch.cos(theta_right1/2) * (2 - kappa - torch.cos(theta_right1))
        u_analytical_left1 = ((self.K_I / (2 * mu)) * torch.sqrt(r_left1 / (2 * np.pi)) * torch.cos(theta_left1/2) * (kappa - 1 + 2 * torch.sin(theta_left1/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left1 / (2 * np.pi)) * torch.sin(theta_left1/2) * (2 + kappa + torch.cos(theta_left1)))
        v_analytical_left1 = (self.K_I / (2 * mu)) * torch.sqrt(r_left1 / (2 * np.pi)) * torch.sin(theta_left1/2) * (kappa + 1 - 2 * torch.cos(theta_left1/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left1 / (2 * np.pi)) * torch.cos(theta_left1/2) * (2 - kappa - torch.cos(theta_left1))
        # 需要把上述局部坐标系表示的位移场放到全局坐标系中
        u_analytical_right_global1 = u_analytical_right1 * c1 - v_analytical_right1 * s1
        v_analytical_right_global1 = u_analytical_right1 * s1 + v_analytical_right1 * c1
        u_analytical_left_global1 = u_analytical_left1 * c1 - v_analytical_left1 * s1
        v_analytical_left_global1 = u_analytical_left1 * s1 + v_analytical_left1 * c1

        u_analytical_right2 = (self.K_I / (2 * mu)) * torch.sqrt(r_right2 / (2 * np.pi)) * torch.cos(theta_right2/2) * (kappa - 1 + 2 * torch.sin(theta_right2/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right2 / (2 * np.pi)) * torch.sin(theta_right2/2) * (2 + kappa + torch.cos(theta_right2))
        v_analytical_right2 = (self.K_I / (2 * mu)) * torch.sqrt(r_right2 / (2 * np.pi)) * torch.sin(theta_right2/2) * (kappa + 1 - 2 * torch.cos(theta_right2/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right2 / (2 * np.pi)) * torch.cos(theta_right2/2) * (2 - kappa - torch.cos(theta_right2))
        u_analytical_left2 = ((self.K_I / (2 * mu)) * torch.sqrt(r_left2 / (2 * np.pi)) * torch.cos(theta_left2/2) * (kappa - 1 + 2 * torch.sin(theta_left2/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left2 / (2 * np.pi)) * torch.sin(theta_left2/2) * (2 + kappa + torch.cos(theta_left2)))
        v_analytical_left2 = (self.K_I / (2 * mu)) * torch.sqrt(r_left2 / (2 * np.pi)) * torch.sin(theta_left2/2) * (kappa + 1 - 2 * torch.cos(theta_left2/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left2 / (2 * np.pi)) * torch.cos(theta_left2/2) * (2 - kappa - torch.cos(theta_left2))
        u_analytical_right_global2 = u_analytical_right2 * c2 - v_analytical_right2 * s2
        v_analytical_right_global2 = u_analytical_right2 * s2 + v_analytical_right2 * c2
        u_analytical_left_global2 = u_analytical_left2 * c2 - v_analytical_left2 * s2
        v_analytical_left_global2 = u_analytical_left2 * s2 + v_analytical_left2 * c2

        return (u + u_analytical_left_global1 * torch.exp(-20*self.b/self.a*r_left1**self.q) \
            + u_analytical_right_global1 * torch.exp(-20*self.b/self.a*r_right1**self.q) \
                + u_analytical_left_global2 * torch.exp(-20*self.b/self.a*r_left2**self.q) \
                    + u_analytical_right_global2 * torch.exp(-20*self.b/self.a*r_right2**self.q))  \
                        * (x + self.b)/(2*self.b) * (self.b - x)/(2*self.b) 
        # return u_analytical_right_global # * torch.exp(-self.b/self.a*r_left**self.q) # + u_analytical_right * torch.exp(-self.b/self.a*r_right**self.q)
        # return (u + u_analytical_left_global * torch.exp(-self.b/self.a*r_left**self.q) + u_analytical_right_global * torch.exp(-self.b/self.a*r_right**self.q)) * (x + self.b)/(2*self.b) * (self.b - x)/(2*self.b) 
        # return (u) * (x + self.b)/(2*self.b) * (self.b - x)/(2*self.b) 
    def hard_v(self, v, x, y):
        # Calculate polar coordinates relative to crack tip
        c1 = torch.cos(torch.tensor(beta1, device=self.device))
        s1 = torch.sin(torch.tensor(beta1, device=self.device))
        c2 = torch.cos(torch.tensor(beta2, device=self.device))
        s2 = torch.sin(torch.tensor(beta2, device=self.device))
        # 全局->局部 的旋转 R = [[c, s],[-s, c]]
        # 右/左裂尖（全局坐标）
        tip_r1 = torch.stack([ self.a * c1,  self.a * s1]).to(x.device).to(x.dtype)
        tip_l1 = torch.stack([-self.a * c1, -self.a * s1]).to(x.device).to(x.dtype)
        tip_r2 = torch.stack([ self.a * c2,  self.a * s2]).to(x.device).to(x.dtype)
        tip_l2 = torch.stack([-self.a * c2, -self.a * s2]).to(x.device).to(x.dtype)

        # 组装点坐标
        xy = torch.stack([x, y], dim=-1)  # (...,2)

        # 相对坐标（全局）
        rel_r_g1 = xy - tip_r1
        rel_l_g1 = xy - tip_l1
        rel_r_g2 = xy - tip_r2
        rel_l_g2 = xy - tip_l2

        # 旋到局部：v_local = R v_global
        R1 = torch.stack([torch.stack([c1, s1], dim=-1),
                        torch.stack([-s1, c1], dim=-1)], dim=0)  # (2,2)
        R2 = torch.stack([torch.stack([c2, s2], dim=-1),
                        torch.stack([-s2, c2], dim=-1)], dim=0)  # (2,2)
        rel_r_l1 = rel_r_g1 @ R1.T
        rel_l_l1 = rel_l_g1 @ R1.T
        rel_r_l2 = rel_r_g2 @ R2.T
        rel_l_l2 = rel_l_g2 @ R2.T



        x1_r1, x2_r1 = rel_r_l1[..., 0], rel_r_l1[..., 1]
        x1_l1, x2_l1 = -rel_l_l1[..., 0], rel_l_l1[..., 1]
        x1_r2, x2_r2 = rel_r_l2[..., 0], rel_r_l2[..., 1]
        x1_l2, x2_l2 = -rel_l_l2[..., 0], rel_l_l2[..., 1]
        r_right1   = torch.sqrt(x1_r1**2 + x2_r1**2)
        r_left1    = torch.sqrt(x1_l1**2 + x2_l1**2)
        r_right2   = torch.sqrt(x1_r2**2 + x2_r2**2)
        r_left2    = torch.sqrt(x1_l2**2 + x2_l2**2)
        theta_right1 = torch.atan2(x2_r1, x1_r1)
        theta_left1  = torch.atan2(x2_l1, x1_l1)
        theta_right2 = torch.atan2(x2_r2, x1_r2)
        theta_left2  = torch.atan2(x2_l2, x1_l2)
    
        # u_analytical = self.K_I[0] * r**0.5 * torch.sin(theta/2) + \
        #                self.K_I1[1] * r**0.5 * torch.sin(theta/2) * torch.sin(theta) + \
        #                self.K_I1[2] * r**0.5 * torch.cos(theta/2) + \
        #                self.K_I1[3] * r**0.5 * torch.cos(theta/2) * torch.sin(theta)

        # Calculate Mode I crack analytical solution
        mu = E / (2 * (1 + nu))
        kappa = 3 - 4 * nu  # Plane strain
        
        # K_I is now a trainable parameter (already initialized in __init__)
        # Analytical displacement components
        u_analytical_right1 = (self.K_I / (2 * mu)) * torch.sqrt(r_right1 / (2 * np.pi)) * torch.cos(theta_right1/2) * (kappa - 1 + 2 * torch.sin(theta_right1/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right1 / (2 * np.pi)) * torch.sin(theta_right1/2) * (2 + kappa + torch.cos(theta_right1))
        v_analytical_right1 = (self.K_I / (2 * mu)) * torch.sqrt(r_right1 / (2 * np.pi)) * torch.sin(theta_right1/2) * (kappa + 1 - 2 * torch.cos(theta_right1/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right1 / (2 * np.pi)) * torch.cos(theta_right1/2) * (2 - kappa - torch.cos(theta_right1))
        u_analytical_left1 = ((self.K_I / (2 * mu)) * torch.sqrt(r_left1 / (2 * np.pi)) * torch.cos(theta_left1/2) * (kappa - 1 + 2 * torch.sin(theta_left1/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left1 / (2 * np.pi)) * torch.sin(theta_left1/2) * (2 + kappa + torch.cos(theta_left1)))
        v_analytical_left1 = (self.K_I / (2 * mu)) * torch.sqrt(r_left1 / (2 * np.pi)) * torch.sin(theta_left1/2) * (kappa + 1 - 2 * torch.cos(theta_left1/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left1 / (2 * np.pi)) * torch.cos(theta_left1/2) * (2 - kappa - torch.cos(theta_left1))
        # 需要把上述局部坐标系表示的位移场放到全局坐标系中
        u_analytical_right_global1 = u_analytical_right1 * c1 - v_analytical_right1 * s1
        v_analytical_right_global1 = u_analytical_right1 * s1 + v_analytical_right1 * c1
        u_analytical_left_global1 = u_analytical_left1 * c1 - v_analytical_left1 * s1
        v_analytical_left_global1 = u_analytical_left1 * s1 + v_analytical_left1 * c1

        u_analytical_right2 = (self.K_I / (2 * mu)) * torch.sqrt(r_right2 / (2 * np.pi)) * torch.cos(theta_right2/2) * (kappa - 1 + 2 * torch.sin(theta_right2/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right2 / (2 * np.pi)) * torch.sin(theta_right2/2) * (2 + kappa + torch.cos(theta_right2))
        v_analytical_right2 = (self.K_I / (2 * mu)) * torch.sqrt(r_right2 / (2 * np.pi)) * torch.sin(theta_right2/2) * (kappa + 1 - 2 * torch.cos(theta_right2/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_right2 / (2 * np.pi)) * torch.cos(theta_right2/2) * (2 - kappa - torch.cos(theta_right2))
        u_analytical_left2 = ((self.K_I / (2 * mu)) * torch.sqrt(r_left2 / (2 * np.pi)) * torch.cos(theta_left2/2) * (kappa - 1 + 2 * torch.sin(theta_left2/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left2 / (2 * np.pi)) * torch.sin(theta_left2/2) * (2 + kappa + torch.cos(theta_left2)))
        v_analytical_left2 = (self.K_I / (2 * mu)) * torch.sqrt(r_left2 / (2 * np.pi)) * torch.sin(theta_left2/2) * (kappa + 1 - 2 * torch.cos(theta_left2/2)**2) + \
                             (self.K_II / (2 * mu)) * torch.sqrt(r_left2 / (2 * np.pi)) * torch.cos(theta_left2/2) * (2 - kappa - torch.cos(theta_left2))
        u_analytical_right_global2 = u_analytical_right2 * c2 - v_analytical_right2 * s2
        v_analytical_right_global2 = u_analytical_right2 * s2 + v_analytical_right2 * c2
        u_analytical_left_global2 = u_analytical_left2 * c2 - v_analytical_left2 * s2
        v_analytical_left_global2 = u_analytical_left2 * s2 + v_analytical_left2 * c2

        # return (v + v_analytical_left_global1 * torch.exp(-20*self.b/self.a*r_left1**self.q) + v_analytical_right_global1 * torch.exp(-20*self.b/self.a*r_right1**self.q) + v_analytical_left_global2 * torch.exp(-20*self.b/self.a*r_left2**self.q) + v_analytical_right_global2 * torch.exp(-20*self.b/self.a*r_right2**self.q))  * (y + self.h)/(2*self.h) 


        
        return (v + v_analytical_left_global1 * torch.exp(-20*self.b/self.a*r_left1**self.q) \
            + v_analytical_right_global1 * torch.exp(-20*self.b/self.a*r_right1**self.q) \
                + v_analytical_left_global2 * torch.exp(-20*self.b/self.a*r_left2**self.q) \
                    + v_analytical_right_global2 * torch.exp(-20*self.b/self.a*r_right2**self.q))  \
                        * (y + self.h)/(2*self.h)
        # return v_analytical_right_global # * torch.exp(-self.b/self.a*r_left**self.q) # + v_analytical_right * torch.exp(-self.b/self.a*r_right**self.q)
        # return v_analytical_left_global * torch.exp(-2*r_left**2)
        # return  v  * (y + self.h)/(2*self.h)
    
    def add_BCPoints(self,num = [256]):
        x_up,y_up=genMeshNodes2D(-1,1,num[0],1,1,1)
        self.x_up,self.y_up,self.xy_up = self._set_points(x_up ,y_up)
        self.up_zero = torch.zeros_like(self.x_up)

    def E_ext(self) -> torch.Tensor:
        u_up,v_up = self.pred_uv(self.xy_up)


        return trapz1D(v_up * self.fy, self.x_up) 
    

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
'''两条互相交叉的裂纹'''
'''平面应力'''
E=1e3 ; nu = 0.3
fy=100.0

kappa = (3-4*nu)
mu = E/(2*(1+nu))

y_crackTip = 0.0

x_crackCenter = 0.0
y_crackCenter = 0.0
beta1 =    torch.pi/12*4 # torch.pi/12
beta2 =    -beta1
a = 0.5
b = 1.0
h = 1.0
q = 1 
point_num = 100
epoch_num = 200
x_crackTip = a
embedding_1 = LineCrackEmbedding([np.cos(beta1)*a, np.sin(beta1)*a],
                                            [-np.cos(beta1)*a, -np.sin(beta1)*a],
                                            tip = 'both')
embedding_2 = LineCrackEmbedding([np.cos(beta1)*a, -np.sin(beta1)*a],
                                            [-np.cos(beta1)*a, +np.sin(beta1)*a],
                                            tip = 'both')


multiEmbedding = Embedding.multiEmbedding([embedding_1,embedding_2])

net = Embedding.extendAxisNet(
        net = stack_net(input=4,output=2,activation=nn.Tanh,
                          width=30,depth=4),
        extendAxis= multiEmbedding)

pinn = Plate(net,fy=fy,a=a,b=b,h=h,q=q,beta1=beta1,beta2=beta2)
pinn.add_BCPoints()

pinn.setMaterial(E=E , nu = nu)

pinn.set_meshgrid_inner_points(-b,b,point_num,-h,h,point_num)

pinn.set_loss_func(losses=[pinn.Energy_loss,
                                      ],
                              weights=[1000.0]
                                       )


# # Evaluate
# pinn.readData('result/cross_crack/model_in_paper/cross_crack.txt')
# crack_line_1 = Geometry.LineSegement([0.49,0.49],[-0.49,-0.49])
# index_1 = crack_line_1.is_on_geometry(pinn.labeled_xy,eps=1e-4)
# crack_line_2 = Geometry.LineSegement([0.49,-0.49],[-0.49,0.49])
# index_2 = crack_line_2.is_on_geometry(pinn.labeled_xy,eps=1e-4)

# index = index_1 | index_2

# labeled_xy = pinn.labeled_xy[~index]
# labeled_u,labeled_v = pinn.labeled_u[~index] , pinn.labeled_v[~index]
# labeled_sx,labeled_sy,labeled_sxy = pinn.labeled_sx[~index] , pinn.labeled_sy[~index] , pinn.labeled_sxy[~index]


# u_disp_ref = pinn.displacement(labeled_u,labeled_v).cpu().detach().numpy()
# mises_ref = pinn.stressToMises(labeled_sx,labeled_sy,labeled_sxy).cpu().detach().numpy()

# def record_item():
#     u,v,sx,sy,sxy = pinn.infer(labeled_xy)
#     u_disp = pinn.displacement(u,v).cpu().detach().numpy()
#     mises = pinn.stressToMises(sx,sy,sxy).cpu().detach().numpy()
#     hist = [pinn.rmse(u_disp,u_disp_ref) , pinn.rmse(mises,mises_ref)]
#     print(hist)
#     return hist
# pinn.record_item = record_item

model_name = f'../../result/crack_intersect/crack_XDEM/crack'
if not os.path.exists(model_name):
    os.makedirs(model_name)

pinn.train(path=model_name,patience=10,epochs=epoch_num,lr=0.02)

pinn.load(path=model_name)
# pinn.evaluate(model_name,levels=100)
# pinn.evaluate(name=None,levels=100)
pinn.showPrediction(pinn.XY)
print(pinn.Energy_loss())


# =============================
# Field visualization (cloud maps)
# =============================
print("Generating field visualizations and embedding plot...")

# Create grid for visualization
test_num = 200
x_vis = torch.linspace(-1.0, 1.0, test_num, device=pinn.device)
y_vis = torch.linspace(-1.0, 1.0, test_num, device=pinn.device)
X_vis, Y_vis = torch.meshgrid(x_vis, y_vis, indexing='ij')
XY_vis = torch.stack([X_vis.flatten(), Y_vis.flatten()], dim=1)
XY_vis.requires_grad_(True)

# Predict fields
u_pred, v_pred = pinn.pred_uv(XY_vis)
_, _, sx_pred, sy_pred, sxy_pred = pinn.infer(XY_vis)

# Embedding function (gamma)
with torch.no_grad():
    gamma_pred = multiEmbedding.getGamma(XY_vis)

# Convert to numpy arrays
X_vis_np = X_vis.detach().cpu().numpy()
Y_vis_np = Y_vis.detach().cpu().numpy()
u_pred_np = u_pred.detach().cpu().numpy().reshape(test_num, test_num)
v_pred_np = v_pred.detach().cpu().numpy().reshape(test_num, test_num)
sx_pred_np = sx_pred.detach().cpu().numpy().reshape(test_num, test_num)
sy_pred_np = sy_pred.detach().cpu().numpy().reshape(test_num, test_num)
sxy_pred_np = sxy_pred.detach().cpu().numpy().reshape(test_num, test_num)
gamma_pred_np = gamma_pred.detach().cpu().numpy()[:,0].reshape(test_num, test_num)

# Von Mises stress
von_mises = np.sqrt(sx_pred_np**2 - sx_pred_np*sy_pred_np + sy_pred_np**2 + 3*sxy_pred_np**2)

# Save raw field data
np.savez(model_name + '_field_data.npz',
         X_vis=X_vis_np,
         Y_vis=Y_vis_np,
         u_pred=u_pred_np,
         v_pred=v_pred_np,
         sx_pred=sx_pred_np,
         sy_pred=sy_pred_np,
         sxy_pred=sxy_pred_np,
         gamma_pred=gamma_pred_np,
         von_mises=von_mises)


level_num = 50
# Additional detailed plots
plt.figure(figsize=(15, 10))

# Plot u2 displacement field and Gamma embedding
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_u2, label='u2 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'u2 Displacement Field (β={beta1}°)')
plt.axis('equal')
plt.grid(True, alpha=0.3)

# Plot σ22 stress field
plt.subplot(1, 3, 2)
contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sy, label='σ22 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'σ22 Stress Field (β={beta1}°)')
plt.axis('equal')
plt.grid(True, alpha=0.3)

# Plot Gamma embedding field
plt.subplot(1, 3, 3)
contour_gamma = plt.contourf(X_vis_np, Y_vis_np, gamma_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_gamma, label='Gamma Embedding')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Gamma Embedding Field (β={beta1}°)')
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
plt.title(f'u1 Displacement Field (β={beta1}°)')
plt.axis('equal')

# u2 displacement
plt.subplot(2, 3, 2)
contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_u2, label='u2 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'u2 Displacement Field (β={beta1}°)')
plt.axis('equal')

# σ11 stress
plt.subplot(2, 3, 3)
contour_sx = plt.contourf(X_vis_np, Y_vis_np, sx_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sx, label='σ11 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'σ11 Stress Field (β={beta1}°)')
plt.axis('equal')

# σ22 stress
plt.subplot(2, 3, 4)
contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sy, label='σ22 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'σ22 Stress Field (β={beta1}°)')
plt.axis('equal')

# σ12 stress
plt.subplot(2, 3, 5)
contour_sxy = plt.contourf(X_vis_np, Y_vis_np, sxy_pred_np, levels=20, cmap='RdBu_r')
plt.colorbar(contour_sxy, label='σ12 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'σ12 Stress Field (β={beta1}°)')
plt.axis('equal')

# Von Mises stress
plt.subplot(2, 3, 6)
von_mises = np.sqrt(sx_pred_np**2 - sx_pred_np*sy_pred_np + sy_pred_np**2 + 3*sxy_pred_np**2)
contour_vm = plt.contourf(X_vis_np, Y_vis_np, von_mises, levels=20, cmap='RdBu_r')
plt.colorbar(contour_vm, label='Von Mises stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Von Mises Stress Field (β={beta1}°)')
plt.axis('equal')

plt.tight_layout()
plt.savefig(model_name + '_detailed_fields.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Detailed field plots saved to: {model_name}_detailed_fields.pdf")

# Save Von Mises stress array as well
np.save(model_name + '_von_mises_stress.npy', von_mises)
print(f"  Von Mises stress array savdf to: {model_name}_von_mises_stress.npy")
