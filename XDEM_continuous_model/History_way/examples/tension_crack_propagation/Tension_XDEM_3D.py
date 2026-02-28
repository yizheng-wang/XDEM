# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import time
import os
import numpy as np
import scipy.io
from scipy.spatial import Delaunay
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.dpi"] = 200
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from utils.kan_efficiency import *

from utils.gridPlot3D import (scatterPlot, genGrid, plotDispStrainEnerg_uni,
                              plotPhiStrainEnerg_uni, plotConvergence,
                              createFolder, plot1dPhi, plotForceDisp)

# -----------------------------------------------------------------------------
# Helper for deterministic runs (equivalent to np.random.seed / tf.set_random_seed)
# -----------------------------------------------------------------------------
seed = 2021
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
#   Neural‑network core (unchanged public API, now inheriting nn.Module)
# -----------------------------------------------------------------------------

class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        yt = x
        y1 = torch.tanh(self.linear1(yt))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2)) + y1
        y4 = torch.tanh(self.linear4(y3)) + y2
        y =  self.linear5(y4)
        return y
    

# -----------------------------------------------------------------------------
#  Problem‑specific subclass with boundary conditions (name unchanged)
# -----------------------------------------------------------------------------
class DEM_PF(nn.Module):
    """Tension‑plate PINN with symmetry boundary conditions (PyTorch)."""

    def __init__(self, model, modelNN_U, modelNN_phi): # modelNN_U, modelNN_phi):
        super(DEM_PF, self).__init__()
        self.model = model
        # self.modelNN_U = modelNN_U
        # self.modelNN_phi = modelNN_phi
        self.modelNN_U = modelNN_U
        self.modelNN_phi = modelNN_phi
        self.crackTip = 0.5  # moved here so it’s available immediately

        self.E = model["E"]
        self.nu = model["nu"]
        # Elastic stiffness components (orthotropic form retained)
        denom = (1 + self.nu) * (1 - 2 * self.nu)
        self.c11 = self.E * (1 - self.nu) / denom
        self.c22 = self.c11
        self.c33 = self.c11
        self.c12 = self.E * self.nu / denom
        self.c13 = self.c12
        self.c21 = self.c12
        self.c23 = self.c12
        self.c31 = self.c12
        self.c32 = self.c12
        self.c44 = self.E / (2 * (1 + self.nu))
        self.c55 = self.c44
        self.c66 = self.c44
        self.lamda = self.E * self.nu / ((1 - 2 * self.nu) * (1 + self.nu))
        self.mu = 0.5 * self.E / (1 + self.nu)
        self.K = self.E/(3*(1-2*self.nu))
        self.cEnerg = 2.7
        self.B = 1000
        self.l = model["l"]

        self.lb = torch.tensor(model["lb"], dtype=torch.float32, device=device)
        self.ub = torch.tensor(model["ub"], dtype=torch.float32, device=device)


    # net_uv is already implemented identically in base class; override here to
    # keep attribute order if you had custom logic.  Keep original code 1‑to‑1.
    def net_uv(self, x, y, z, wdelta):
        X = torch.cat([x, y, z], dim=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0 # 归一化到-1到1
        uv = self.modelNN_U(H)
        
        # Combine analytical solution with neural network prediction
        uNN = uv[:, 0:1]
        vNN = uv[:, 1:2]
        wNN = uv[:, 2:3]
        
        # Add analytical solution to neural network prediction
        u = z * (uNN)
        v = z * (vNN)
        w = z * (z - 1) * (wNN) + z * wdelta 
     
        return u, v, w

    def net_phi(self, x, y, z):
        X = torch.cat([x, y, z], dim=1)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0 # 归一化到-1到1
        phi = self.modelNN_phi(H)

        return phi

    # net_hist identical to base but keeps crackTip; re‑implemented for clarity
    def net_hist(self, x, y, z):
        shape = x.shape
        init_hist = torch.zeros(shape, dtype=torch.float32, device=device)
        dist = torch.where(x > self.crackTip,
                           torch.sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2),
                           torch.abs(z - 0.5))
        mask = dist < 0.5 * self.l
        init_hist[mask] = self.B * self.cEnerg * 0.5 * (1 - (2 * dist[mask] / self.l)) / self.l
        return init_hist

    # --- update history tensor -----------------------------------------------------
    #二维有拉压能量分解啊
    def net_update_hist(self, x, y, z, u_x, v_y, w_z, u_xy, u_yz, u_xz, hist):
        init_hist = self.net_hist(x, y, z)
        # tensile strain energy density
        u_xy = 0.5 * u_xy
        u_yz = 0.5 * u_yz
        u_xz = 0.5 * u_xz

        theta = u_x + v_y + w_z
        
        # 计算特征值
        strain_psi = torch.stack([
            torch.stack([u_x-1/3*theta, u_xy, u_xz], dim=-1),
            torch.stack([u_xy, v_y-1/3*theta, u_yz], dim=-1),
            torch.stack([u_xz, u_yz, w_z-1/3*theta], dim=-1)
        ], dim=-2)[:,0,:,:]  # shape: (..., 3, 3)



        sEnergy_pos = 0.125 * self.K * (theta + torch.abs(theta)) ** 2 + \
                       self.mu * (strain_psi[:,0,0].unsqueeze(1)**2+strain_psi[:,0,1].unsqueeze(1)**2+strain_psi[:,0,2].unsqueeze(1)**2+\
                                  strain_psi[:,1,0].unsqueeze(1)**2+strain_psi[:,1,1].unsqueeze(1)**2+strain_psi[:,1,2].unsqueeze(1)**2+\
                                  strain_psi[:,2,0].unsqueeze(1)**2+strain_psi[:,2,1].unsqueeze(1)**2+strain_psi[:,2,2].unsqueeze(1)**2)
        
        hist_temp = torch.maximum(init_hist, sEnergy_pos)
        hist = torch.maximum(hist, hist_temp)
        return hist

    def net_energy(self, x, y, z, hist, vdelta):
        # enable gradients
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        u, v, w = self.net_uv(x, y, z, vdelta)
        phi = self.net_phi(x, y, z)
        g = (1 - phi) ** 2
        phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        phi_z = torch.autograd.grad(phi, z, grad_outputs=torch.ones_like(phi), create_graph=True)[0]

        nabla = phi_x ** 2 + phi_y ** 2 + phi_z ** 2

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xy = (u_y + v_x)
        u_yz = (w_y + v_z)
        u_xz = (w_x + u_z)
        
        # if iStep == 1: # 只在第一步迭代步下调用初始场
        #     hist = self.net_hist(x, y)
        # hist = self.net_update_hist(x, y, u_x, v_y, u_xy, hist) # 不让历史场更新

        # sigmaX = self.c11 * u_x + self.c12 * v_y
        # sigmaY = self.c21 * u_x + self.c22 * v_y
        # tauXY = self.c33 * u_xy

        u_xy = 0.5 * u_xy
        u_yz = 0.5 * u_yz
        u_xz = 0.5 * u_xz
        theta = u_x + v_y + w_z
        
        # 计算特征值
        strain_psi = torch.stack([
            torch.stack([u_x-1/3*theta, u_xy, u_xz], dim=-1),
            torch.stack([u_xy, v_y-1/3*theta, u_yz], dim=-1),
            torch.stack([u_xz, u_yz, w_z-1/3*theta], dim=-1)
        ], dim=-2)[:,0,:,:]  # shape: (..., 3, 3)



        sEnergy_pos = 0.125 * self.K * (theta + torch.abs(theta)) ** 2 + \
                       self.mu * (strain_psi[:,0,0].unsqueeze(1)**2+strain_psi[:,0,1].unsqueeze(1)**2+strain_psi[:,0,2].unsqueeze(1)**2+\
                                  strain_psi[:,1,0].unsqueeze(1)**2+strain_psi[:,1,1].unsqueeze(1)**2+strain_psi[:,1,2].unsqueeze(1)**2+\
                                  strain_psi[:,2,0].unsqueeze(1)**2+strain_psi[:,2,1].unsqueeze(1)**2+strain_psi[:,2,2].unsqueeze(1)**2)
        sEnergy_neg = 0.125 * self.K * (theta - torch.abs(theta)) ** 2

        energy_u = g * sEnergy_pos + sEnergy_neg
        energy_phi = 0.5 * self.cEnerg * (phi ** 2 / self.l + self.l * nabla) + g * hist
        return energy_u, energy_phi, hist.detach()  # detach hist to stop gradient here


    def net_energy_update_his(self, x, y, z, hist, vdelta):
        # enable gradients
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        u, v, w = self.net_uv(x, y, z, vdelta)
        phi = self.net_phi(x, y, z)
        g = (1 - phi) ** 2
        phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        phi_z = torch.autograd.grad(phi, z, grad_outputs=torch.ones_like(phi), create_graph=True)[0]

        nabla = phi_x ** 2 + phi_y ** 2 + phi_z ** 2


        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xy = (u_y + v_x)
        u_yz = (w_y + v_z)
        u_xz = (w_x + u_z)
        hist = self.net_update_hist(x, y, z, u_x, v_y, w_z, u_xy, u_yz, u_xz, hist) 

        # sigmaX = self.c11 * u_x + self.c12 * v_y
        # sigmaY = self.c21 * u_x + self.c22 * v_y
        # tauXY = self.c33 * u_xy

        u_xy = 0.5 * u_xy
        u_yz = 0.5 * u_yz
        u_xz = 0.5 * u_xz
        theta = u_x + v_y + w_z
        
        # 计算特征值
        strain_psi = torch.stack([
            torch.stack([u_x-1/3*theta, u_xy, u_xz], dim=-1),
            torch.stack([u_xy, v_y-1/3*theta, u_yz], dim=-1),
            torch.stack([u_xz, u_yz, w_z-1/3*theta], dim=-1)
        ], dim=-2)[:,0,:,:]  # shape: (..., 3, 3)



        sEnergy_pos = 0.125 * self.K * (theta + torch.abs(theta)) ** 2 + \
                       self.mu * (strain_psi[:,0,0].unsqueeze(1)**2+strain_psi[:,0,1].unsqueeze(1)**2+strain_psi[:,0,2].unsqueeze(1)**2+\
                                  strain_psi[:,1,0].unsqueeze(1)**2+strain_psi[:,1,1].unsqueeze(1)**2+strain_psi[:,1,2].unsqueeze(1)**2+\
                                  strain_psi[:,2,0].unsqueeze(1)**2+strain_psi[:,2,1].unsqueeze(1)**2+strain_psi[:,2,2].unsqueeze(1)**2)
        sEnergy_neg = 0.125 * self.K * (theta - torch.abs(theta)) ** 2

        energy_u = g * sEnergy_pos + sEnergy_neg
        energy_phi = 0.5 * self.cEnerg * (phi ** 2 / self.l + self.l * nabla) + g * hist
        return energy_u, energy_phi, hist.detach()  # detach hist to stop gradient here

    def net_traction(self, x, y, z, vdelta):
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        u, v, w = self.net_uv(x, y, z, vdelta)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]


        traction = self.c31 * u_x + self.c32 * v_y + self.c33 * w_z
        return traction

    def train_model(self, X_f, v_delta, hist_f, nIter, his_deal):
        # keep identical attribute names used in plotting utils
        self.loss_adam_buff = np.zeros(nIter)
        self.lbfgs_buffer = []
        # convert data to torch tensors on device
        x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, device=device)
        y_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, device=device)
        z_f = torch.tensor(X_f[:, 2:3], dtype=torch.float32, device=device)
        wt_f = torch.tensor(X_f[:, 3:4], dtype=torch.float32, device=device)
        hist = torch.tensor(hist_f, dtype=torch.float32, device=device)
        vdelta = torch.tensor(v_delta, dtype=torch.float32, device=device)

        def loss_fn(his_deal):
            if his_deal == 'fix':
                energy_u_pred, energy_phi_pred, _ = self.net_energy(x_f, y_f, z_f, hist, vdelta)
            if his_deal == 'update':
                energy_u_pred, energy_phi_pred, _ = self.net_energy_update_his(x_f, y_f, z_f, hist, vdelta)
            loss_energy_u = torch.sum(energy_u_pred * wt_f)
            loss_energy_phi = torch.sum(energy_phi_pred * wt_f)
            return loss_energy_u + loss_energy_phi, loss_energy_u, loss_energy_phi

        # ----------------- Adam phase -----------------------------   
        optimizer_U_adam = optim.Adam(self.modelNN_U.parameters(), lr=1e-3)
        optimizer_phi_adam = optim.Adam(self.modelNN_phi.parameters(), lr=1e-3)
        t0 = time.time()
        for it in range(nIter):
            optimizer_U_adam.zero_grad()
            optimizer_phi_adam.zero_grad()  
            loss, e_u, e_phi = loss_fn(his_deal)
            loss.backward()
            optimizer_U_adam.step()
            optimizer_phi_adam.step()
            self.loss_adam_buff[it] = loss.item()
            if it % 100 == 0:
                dt = time.time() - t0
                print(f"It: {it}, Total Loss: {loss.item():.3e}, Energy U: {e_u.item():.3e}, "
                      f"Energy Phi: {e_phi.item():.3e}, Time: {dt:.2f}")
                t0 = time.time()


    def predict(self, X_star, Hist_star, w_delta):
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        y = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        z = torch.tensor(X_star[:, 2:3], dtype=torch.float32, device=device)
        
        hist = torch.tensor(Hist_star, dtype=torch.float32, device=device)
        wdelta = torch.tensor(w_delta, dtype=torch.float32, device=device)

        u_pred, v_pred, w_pred = self.net_uv(x, y, z, wdelta)   
        phi_pred = self.net_phi(x, y, z)
        energy_u_pred, energy_phi_pred, hist_pred = self.net_energy_update_his(x, y, z, hist, wdelta)

        return (u_pred.detach().cpu().numpy(), v_pred.detach().cpu().numpy(), w_pred.detach().cpu().numpy(),
                         phi_pred.detach().cpu().numpy(),
                         energy_u_pred.detach().cpu().numpy(),
                         energy_phi_pred.detach().cpu().numpy(),
                         hist_pred.detach().cpu().numpy())


    def predict_traction(self, X_star, w_delta):
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        y = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        z = torch.tensor(X_star[:, 2:3], dtype=torch.float32, device=device)
        wdelta = torch.tensor(w_delta, dtype=torch.float32, device=device)
        trac = self.net_traction(x, y, z, wdelta)
        return trac.detach().cpu().numpy()


    def predict_phi(self, X_star):
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
        y = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
        z = torch.tensor(X_star[:, 2:3], dtype=torch.float32, device=device)
        phi = self.net_phi(x, y, z)
        return phi.detach().cpu().numpy()


class RBFNetwork(nn.Module):
    def __init__(self, input_dim, num_centers, output_dim):
        super(RBFNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.output_dim = output_dim
        
        # 高斯布点
        # # Initialize centers randomly 
        # self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        
        # # Initialize widths (beta) for each RBF
        # self.beta = nn.Parameter(torch.ones(num_centers)/model['l'])
        
        # # Initialize output weights
        # self.weights = nn.Parameter(torch.randn(num_centers, output_dim))
        
        # # Initialize bias
        # self.bias = nn.Parameter(torch.zeros(output_dim))
   
        self.linear = torch.nn.Linear(num_centers, 1, bias=False)  # 设置bias=False        


        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.1)

   
#        # 均匀布点
#        # Calculate number of points per dimension to achieve total num_centers
#        points_per_dim = int(np.ceil(num_centers ** (1/input_dim)))
        
#        # Generate grid points for each dimension
#        grid_points = []
#        for i in range(input_dim):
#            grid_points.append(torch.linspace(-1, 1, points_per_dim))

        # 均匀布点
        # Calculate number of points per dimension to achieve total num_centers
        
        # Generate grid points for each dimension
        grid_points = []
        grid_points.append(torch.linspace(-1, 1, 20))
        grid_points.append(torch.linspace(-1, 1, 5))
        grid_points.append(torch.linspace(-1, 1, 20))


        # Create meshgrid
        mesh = torch.meshgrid(grid_points, indexing='ij')
        
        # Stack and reshape to get all combinations
        centers = torch.stack(mesh, dim=-1).reshape(-1, input_dim)
        
        # If we generated more points than needed, randomly select num_centers points
        if centers.shape[0] > num_centers:
            indices = torch.randperm(centers.shape[0])[:num_centers]
            centers = centers[indices]
            
        self.centers = nn.Parameter(centers, requires_grad=True)  # Make centers non-trainable
        
        # Initialize widths (beta) for each RBF
        self.beta = nn.Parameter(torch.ones(num_centers)/model['l'])
        
        # Initialize output weights
        self.weights = nn.Parameter(torch.randn(num_centers, output_dim))
        
        # Initialize bias
        self.bias = nn.Parameter(torch.zeros(output_dim))
    def forward(self, x):
        # Calculate distances between input and centers
        x_expanded = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        centers_expanded = self.centers.unsqueeze(0)  # [1, num_centers, input_dim]
        distances = torch.sum((x_expanded - centers_expanded) ** 2, dim=2)  # [batch_size, num_centers]
        
        # Calculate RBF activations
        rbf = torch.exp(-self.beta * distances)  # [batch_size, num_centers]
        sum_rbf = torch.sum(rbf, axis = 1).unsqueeze(1)
        output = self.linear(rbf)/sum_rbf
        
        return output

def freeze_layers(model: nn.Module, train_layer_keywords: list, lr=1e-3):
    """
    冻结 model 中除含有指定关键字的层以外的所有参数
    
    参数:
        model (nn.Module): 你的 PyTorch 模型
        train_layer_keywords (list): 需要更新(微调)的层名称关键字的列表。
            比如 ["linear3", "linear4"]，表示只更新名中含 "linear3" 或 "linear4" 的层。
        lr (float): 优化器学习率

    """
    # 1) 遍历所有参数，判断其所属层名字是否包含在指定的层名称关键字
    for name, param in model.named_parameters():
        # 如果关键词命中，就保持 requires_grad=True，否则 requires_grad=False
        if any(keyword in name for keyword in train_layer_keywords):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
def activate_layers(model: nn.Module):
    """
    冻结 model 中除含有指定关键字的层以外的所有参数
    
    参数:
        model (nn.Module): 你的 PyTorch 模型
        train_layer_keywords (list): 需要更新(微调)的层名称关键字的列表。
            比如 ["linear3", "linear4"]，表示只更新名中含 "linear3" 或 "linear4" 的层。
        lr (float): 优化器学习率

    """
    # 1) 遍历所有参数，判断其所属层名字是否包含在指定的层名称关键字
    for name, param in model.named_parameters():
        # 如果关键词命中，就保持 requires_grad=True，否则 requires_grad=False

        param.requires_grad = True



# -----------------------------------------------------------------------------
#  Main driver (verbatim structure from original TF script)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    originalDir = os.getcwd()
    foldername =  "../../results/Tension_3D/"
    
    createFolder(foldername)

    figHeight = 5
    figWidth = 5
    nSteps = 30    # Total number of steps
    deltaV = 10.0e-4   # Displacement increment per step

    model = {
        "E": 210.0 * 1e3,
        "nu": 0.3,
        "L": 1.0,
        "W": 0.2,
        "H": 1.0,
        "l": 0.0313,
        "lb": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "ub": np.array([1.0, 0.2, 1.0], dtype=np.float32),
    }

    # Generate uniform training points
    n_points = 70 # Number of points in each direction to generate
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 0.2, 8)    
    z = np.linspace(0, 1, n_points)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    
    # Create Delaunay triangulation
    tri = Delaunay(points)


    # Calculate tetrahedron centroids and volumes
    centroids = np.mean(points[tri.simplices], axis=1)
    
    # Calculate volumes of tetrahedra using determinant formula
    # Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
    volumes = []
    for simplex in tri.simplices:
        # Get the four vertices of the tetrahedron
        v0 = points[simplex[0]]
        v1 = points[simplex[1]]
        v2 = points[simplex[2]]
        v3 = points[simplex[3]]
        
        # Create the matrix [v1-v0, v2-v0, v3-v0]
        matrix = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
        
        # Calculate volume using determinant
        volume = np.abs(np.linalg.det(matrix)) / 6.0
        volumes.append(volume)
    
    volumes = np.array(volumes)
    
    # Combine centroids and volumes as weights
    X_f = np.hstack((centroids, volumes.reshape(-1,1)))
    # hist_f = np.zeros((X_f.shape[0], 1), dtype=np.float32)

    # scatter of training points (unchanged)
    os.chdir(os.path.join(originalDir, "./" + foldername + "/"))
#    scatterPlot(X_f, figHeight, figWidth, "Training_scatter")

    # Boundary points ---------------------------------------------------------
    N_b = 100
    x_bottomEdge = np.linspace(0.0, model["L"], int(N_b), dtype=np.float32)[:, None]
    y_bottomEdge = np.linspace(0.0, model["W"], int(N_b), dtype=np.float32)[:, None]
    z_bottomEdge = 0.1
    xx_bottomEdge, yy_bottomEdge, zz_bottomEdge = np.meshgrid(x_bottomEdge, y_bottomEdge, z_bottomEdge)  
    xBottomEdge = np.column_stack((xx_bottomEdge.ravel(), yy_bottomEdge.ravel(), zz_bottomEdge.ravel()))

    # Prediction grid ---------------------------------------------------------
    # Generate uniform grid points in 0-1 square
    n_points = 50  # Number of points along each axis
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 0.2, n_points//5)
    z = np.linspace(0, 1, n_points)
    xx, yy, zz = np.meshgrid(x, y, z)   
    Grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    xGrid = xx
    yGrid = yy
    zGrid = zz
    hist_grid = np.zeros((Grid.shape[0], 1), dtype=np.float32)
#    scatterPlot(Grid, figHeight, figWidth, "Prediction_scatter")

    fdGraph = np.zeros((nSteps+1, 2), dtype=np.float32)
    phi_pred_old = hist_grid.copy()

    # instantiate network -----------------------------------------------------
    modelNN_U_lora = KANLoRA([3, 5,5,5, 3], base_activation=torch.nn.SiLU, grid_size=15, grid_range=[-1.0,  1.0], spline_order=3,\
                    lora_rank_base = 1, lora_rank_spline = 1).cuda()        
    activate_layers(modelNN_U_lora)
    # modelNN_U = MultiLayerNet(2, 50, 2).cuda()
    modelNN_phi = RBFNetwork(3, 2000, 1).to(device)  # 100 centers for better approximation
    # modelNN_phi = MultiLayerNet(2, 50, 1).cuda()
    # instantiate network -----------------------------------------------------
    modelNN = DEM_PF(model, modelNN_U_lora , modelNN_phi).to(device)

    
    X_f_t = torch.tensor(X_f[:,:3], dtype=torch.float32, device=device)
    hist_f = modelNN.net_hist(X_f_t[:, 0:1], X_f_t[:, 1:2], X_f_t[:, 2:3])
    w_delta = 0.0
    for iStep in range(nSteps):
        iStep = iStep + 1
        num_train_its = 3000 if iStep == 1 else 1000


        # Set deltaV based on the step number
        if iStep <= 6:
            deltaV = 0.001  # First 4 steps with larger increment
            hist_deal = 'fix'
        if iStep >= 7 and iStep <= 10:
            deltaV = 0.00025  # First 4 steps with larger increment
            hist_deal = 'fix'
        if iStep >= 11:
            deltaV = 0.0001  # Remaining 60 steps with smaller increment
            hist_deal = 'fix'

        w_delta = w_delta + deltaV

        if iStep != 1: # transfer learning
            freeze_layers(modelNN_U_lora, train_layer_keywords=['0.base_weight_lora', '0.spline_weight_lora', '1.base_weight_lora','1.spline_weight_lora',\
                                                                '2.base_weight_lora', '2.spline_weight_lora', '3.base_weight_lora', '3.spline_weight_lora'], lr=0.001)

        start = time.time()
        modelNN.train_model(X_f, w_delta, hist_f, num_train_its, hist_deal)
        _, _,_, phi_f, _, _, hist_f = modelNN.predict(X_f[:, 0:3], hist_f, w_delta)
        elapsed = time.time() - start
        print(f"Training time: {elapsed:.4f}")

        u_pred, v_pred, w_pred, phi_pred, elas_energy_pred, frac_energy_pred, hist_grid = \
            modelNN.predict(Grid, hist_grid, w_delta)
        phi_pred = np.maximum(phi_pred, phi_pred_old)
        phi_pred_old = phi_pred

        fname = str(iStep)
        plotPhiStrainEnerg_uni(n_points, xGrid, yGrid, zGrid, phi_pred, frac_energy_pred, hist_grid, fname)
        plotDispStrainEnerg_uni(n_points, xGrid, yGrid, zGrid, u_pred, v_pred, w_pred, elas_energy_pred, fname)

        adam_buff = modelNN.loss_adam_buff
        lbfgs_buff = np.array(modelNN.lbfgs_buffer)
        plotConvergence(num_train_its, adam_buff, lbfgs_buff, iStep, figHeight, figWidth)

        traction_pred = modelNN.predict_traction(xBottomEdge, w_delta) # 下边的力，统计上边的是不是更加合理？
        fdGraph[iStep, 0] = w_delta
        fdGraph[iStep, 1] = np.mean(traction_pred, axis=0) * 0.2 # 计算合力

        # 1‑D phase‑field profile -----------------------------------------
        xVal = 0.75
        nPredY = 2000
        xPred = xVal * np.ones((nPredY, 1))
        yPred = 0.1 * np.ones((nPredY, 1))
        zPred = np.linspace(0, model["H"], nPredY)[None, :]
        xyzPred = np.concatenate([xPred, yPred, zPred.T], axis=1)
        phi_pred_1d = modelNN.predict_phi(xyzPred)
        phi_exact = np.exp(-np.abs(zPred - 0.5) / model["l"])
        plot1dPhi(zPred, phi_pred_1d, phi_exact, iStep, figHeight, figWidth)

        error_phi = np.linalg.norm(phi_exact - phi_pred_1d, 2) / np.linalg.norm(phi_exact, 2)
        print(f"Relative error phi: {error_phi:e}")
        print(f"Completed {iStep + 1} of {nSteps}.")
        createFolder("./model/")
        # 保存模型
        torch.save(modelNN.modelNN_U.state_dict(), "./model/modelNN_U_step_" + str(iStep) + ".pth")
        torch.save(modelNN.modelNN_phi.state_dict(), "./model/modelNN_phi_step_" + str(iStep) + ".pth")

        # 我想对RBF网络的中心进行保存
        centers = modelNN.modelNN_phi.centers.detach().cpu().numpy()
        np.save("./model/centers_step_" + str(iStep) + ".npy", centers) 
        # 我想对RBF网络的宽度进行保存
        widths = modelNN.modelNN_phi.beta.detach().cpu().numpy()
        np.save("./model/widths_step_" + str(iStep) + ".npy", widths)   



    plotForceDisp(fdGraph, figHeight, figWidth)
    np.save("./fdGraph.npy", fdGraph)
    os.chdir(originalDir)
