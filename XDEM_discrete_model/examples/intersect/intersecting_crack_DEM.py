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
seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Plate(PINN2D):
    def __init__(self, model: nn.Module,fy):
        '''x范围0-1.y范围-1到1'''
        super().__init__(model)
        self.fy = fy

    def hard_u(self, u, x, y):
        return u * (1 - x) * (1 + x)/4
    
    def hard_v(self, v, x, y):
        return v * (y+1)/2
    
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
E=1000 ; nu = 0.3
fy=100.0
b=1.0
h=1.0
point_num=100
epoch_num=20000
embedding_1 = LineCrackEmbedding(xy0=[0.5,0.5],xy1=[-0.5,-0.5],tip='both')
embedding_2 = LineCrackEmbedding(xy0=[0.5,-0.5],xy1=[-0.5,0.5],tip='both')


multiEmbedding = Embedding.multiEmbedding([embedding_1,embedding_2])

net = Embedding.extendAxisNet(
        net = stack_net(input=4,output=2,activation=nn.Tanh,
                          width=30,depth=4),
        extendAxis= multiEmbedding)

pinn = Plate(net,fy=fy)
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

model_name = f'../../result/crack_intersect/crack_DEM/crack'
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
gamma_pred_np = gamma_pred.detach().cpu().numpy().reshape(test_num, test_num)

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

level_num = 300
# Additional detailed plots
plt.figure(figsize=(15, 10))

# u1 displacement
plt.subplot(2, 3, 1)
contour_u1 = plt.contourf(X_vis_np, Y_vis_np, u_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_u1, label='u1 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u1 Displacement Field')
plt.axis('equal')

# u2 displacement
plt.subplot(2, 3, 2)
contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_u2, label='u2 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u2 Displacement Field')
plt.axis('equal')

# σ11 stress
plt.subplot(2, 3, 3)
contour_sx = plt.contourf(X_vis_np, Y_vis_np, sx_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_sx, label='σ11 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ11 Stress Field')
plt.axis('equal')

# σ22 stress
plt.subplot(2, 3, 4)
contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_sy, label='σ22 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ22 Stress Field')
plt.axis('equal')

# σ12 stress
plt.subplot(2, 3, 5)
contour_sxy = plt.contourf(X_vis_np, Y_vis_np, sxy_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_sxy, label='σ12 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ12 Stress Field')
plt.axis('equal')

# Von Mises stress
plt.subplot(2, 3, 6)
von_mises = np.sqrt(sx_pred_np**2 - sx_pred_np*sy_pred_np + sy_pred_np**2 + 3*sxy_pred_np**2)
contour_vm = plt.contourf(X_vis_np, Y_vis_np, von_mises, levels=level_num, cmap='jet')
plt.colorbar(contour_vm, label='Von Mises stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Von Mises Stress Field')
plt.axis('equal')

plt.tight_layout()
plt.savefig(model_name + '_detailed_fields.pdf', dpi=150, bbox_inches='tight')
plt.close()

# Save Von Mises stress array as well
np.save(model_name + '_von_mises_stress.npy', von_mises)
print(f"Von Mises stress array saved to: {model_name}_von_mises_stress.npy")

# =============================================================================
# Save individual figures as PDF for paper publication
# =============================================================================
print("Saving individual PDF figures for paper publication...")


# 2. u2 Displacement Field
plt.figure(figsize=(5, 6))
contour_u2 = plt.contourf(X_vis_np, Y_vis_np, v_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_u2, label='u2 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u2 Displacement Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_u2_displacement.pdf', dpi=150, bbox_inches='tight')
plt.close()
print(f"u2 displacement PDF saved to: {model_name}_u2_displacement.pdf")

# 3. σ22 Stress Field
plt.figure(figsize=(5, 6))
contour_sy = plt.contourf(X_vis_np, Y_vis_np, sy_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_sy, label='σ22 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ22 Stress Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_sigma22_stress.pdf', dpi=150, bbox_inches='tight')
plt.close()
print(f"σ22 stress PDF saved to: {model_name}_sigma22_stress.pdf")

# 4. Gamma Embedding Field 
plt.figure(figsize=(5, 6))
contour_gamma = plt.contourf(X_vis_np, Y_vis_np, gamma_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_gamma, label='Embedding function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Embedding function Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_gamma_embedding.pdf', dpi=150, bbox_inches='tight')
plt.close()
print(f"Gamma embedding PDF saved to: {model_name}_gamma_embedding.pdf")

# 5. u1 Displacement Field
plt.figure(figsize=(5, 6))
contour_u1 = plt.contourf(X_vis_np, Y_vis_np, u_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_u1, label='u1 displacement (mm)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('u1 Displacement Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_u1_displacement.pdf', dpi=150, bbox_inches='tight')
plt.close()
print(f"u1 displacement PDF saved to: {model_name}_u1_displacement.pdf")

# 6. σ11 Stress Field
plt.figure(figsize=(5, 6))
contour_sx = plt.contourf(X_vis_np, Y_vis_np, sx_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_sx, label='σ11 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ11 Stress Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_sigma11_stress.pdf', dpi=150, bbox_inches='tight')
plt.close()
print(f"σ11 stress PDF saved to: {model_name}_sigma11_stress.pdf")

# 7. σ12 Stress Field
plt.figure(figsize=(5, 6))
contour_sxy = plt.contourf(X_vis_np, Y_vis_np, sxy_pred_np, levels=level_num, cmap='jet')
plt.colorbar(contour_sxy, label='σ12 stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('σ12 Stress Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_sigma12_stress.pdf', dpi=150, bbox_inches='tight')
plt.close()
print(f"σ12 stress PDF saved to: {model_name}_sigma12_stress.pdf")

# 8. Von Mises Stress Field
plt.figure(figsize=(5, 6))
contour_vm = plt.contourf(X_vis_np, Y_vis_np, von_mises, levels=level_num, cmap='jet')
plt.colorbar(contour_vm, label='Von Mises stress (MPa)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Von Mises Stress Field')
plt.xlim(-b, b)
plt.ylim(-h, h)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(model_name + '_von_mises_stress.pdf', dpi=150, bbox_inches='tight')
plt.close()
print(f"Von Mises stress PDF saved to: {model_name}_von_mises_stress.pdf")


