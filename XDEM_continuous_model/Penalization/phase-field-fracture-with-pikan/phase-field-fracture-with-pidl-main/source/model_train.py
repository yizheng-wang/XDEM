import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path

from input_data_from_mesh import prep_input_data
from fit import fit, fit_with_early_stopping
from optim import *
from plotting import plot_field

def train(field_comp, disp, pffmodel, matprop, crack_dict, numr_dict, optimizer_dict, training_dict, coarse_mesh_file, fine_mesh_file, device, trainedModel_path, intermediateModel_path, writer, L_uniform=False):
    '''
    Neural network training: pretraining with a coarser mesh in the first stage before the main training proceeds.
    
    Input is prepared from the .msh file.

    Network training to learn the solution of the BVP in step wise loading.
    Trained network from the previous load step is used for learning the solution
    in the current load step.

    Trained models and loss data are saved in the trainedModel_path directory.
    '''
    
    ## #############################################################################
    # Initial training #############################################################
    # Prepare initial input data
    
    if L_uniform:
        inp_np, T_conn_np, area_T_np, _ = generate_L_mesh(nx=100, ny=100, remove_square=(0.0, -0.5, 0.5, 0.0)) # 自己生成一个L形区域
        hist_alpha_np = np.zeros(inp_np.shape[0])
        # 把inp_np转换为torch.Tensor
        inp = torch.tensor(inp_np, dtype=torch.float32, device=device)
        T_conn = torch.tensor(T_conn_np, dtype=torch.int64, device=device)
        area_T = torch.tensor(area_T_np, dtype=torch.float32, device=device)
        hist_alpha = torch.tensor(hist_alpha_np, dtype=torch.float32, device=device)
    else:
        inp, T_conn, area_T, hist_alpha = prep_input_data(matprop, pffmodel, crack_dict, numr_dict, mesh_file=coarse_mesh_file, device=device)
    outp = torch.zeros(inp.shape[0], 1).to(device)
    training_set = DataLoader(torch.utils.data.TensorDataset(inp, outp), batch_size=inp.shape[0], shuffle=False)
    field_comp.lmbda = torch.tensor(disp[0]).to(device)

    loss_data = list()
    start = time.time()

    n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
    NNparams = field_comp.net.parameters()
    optimizer = get_optimizer(NNparams, "LBFGS")
    loss_data1 = fit(field_comp, training_set, T_conn, area_T, hist_alpha, matprop, pffmodel,
                     optimizer_dict["weight_decay"], num_epochs=n_epochs, optimizer=optimizer, 
                     intermediateModel_path=None, writer=writer, training_dict=training_dict)
    loss_data = loss_data + loss_data1

    n_epochs = optimizer_dict["n_epochs_RPROP"]
    NNparams = field_comp.net.parameters()
    optimizer = get_optimizer(NNparams, "RPROP")
    loss_data2 = fit_with_early_stopping(field_comp, training_set, T_conn, area_T, hist_alpha, matprop, pffmodel,
                                         optimizer_dict["weight_decay"], num_epochs=n_epochs, optimizer=optimizer, min_delta=optimizer_dict["optim_rel_tol_pretrain"], 
                                         intermediateModel_path=None, writer=writer, training_dict=training_dict)
    loss_data = loss_data + loss_data2

    end = time.time()
    print(f"Execution time: {(end-start)/60:.03f}minutes")

    torch.save(field_comp.net.state_dict(), trainedModel_path/Path('trained_1NN_initTraining.pt'))
    with open(trainedModel_path/Path('trainLoss_1NN_initTraining.npy'), 'wb') as file:
        np.save(file, np.asarray(loss_data))

    ## #############################################################################


    ## #############################################################################
    # Main training ################################################################

    # Prepare input data
    
    
    if L_uniform:
        inp_np, T_conn_np, area_T_np, _ = generate_L_mesh(nx=100, ny=100, remove_square=(0.0, -0.5, 0.5, 0.0)) # 自己生成一个L形区域
        hist_alpha_np = np.zeros(inp_np.shape[0])
        # 把inp_np转换为torch.Tensor
        inp = torch.tensor(inp_np, dtype=torch.float32, device=device)
        T_conn = torch.tensor(T_conn_np, dtype=torch.int64, device=device)
        area_T = torch.tensor(area_T_np, dtype=torch.float32, device=device)
        hist_alpha = torch.tensor(hist_alpha_np, dtype=torch.float32, device=device)
    else:
        inp, T_conn, area_T, hist_alpha = prep_input_data(matprop, pffmodel, crack_dict, numr_dict, mesh_file=fine_mesh_file, device=device)


    outp = torch.zeros(inp.shape[0], 1).to(device)
    training_set = DataLoader(torch.utils.data.TensorDataset(inp, outp), batch_size=inp.shape[0], shuffle=False)

    # solve BVP by step wise loading.
    for j, disp_i in enumerate(disp):
        field_comp.lmbda = torch.tensor(disp_i).to(device)
        print(f'idx: {j}; displacement: {field_comp.lmbda}')
        loss_data = list()

        start = time.time()
        
        if j == 0 or optimizer_dict["n_epochs_LBFGS"] > 0:
            n_epochs = max(optimizer_dict["n_epochs_LBFGS"], 1)
            NNparams = field_comp.net.parameters()
            optimizer = get_optimizer(NNparams, "LBFGS")
            loss_data1 = fit(field_comp, training_set, T_conn, area_T, hist_alpha, matprop, pffmodel,
                             optimizer_dict["weight_decay"], num_epochs=n_epochs, optimizer=optimizer,
                             intermediateModel_path=None, writer=writer, training_dict=training_dict)
            loss_data = loss_data + loss_data1

        if optimizer_dict["n_epochs_RPROP"] > 0:
            n_epochs = optimizer_dict["n_epochs_RPROP"]
            NNparams = field_comp.net.parameters()
            optimizer = get_optimizer(NNparams, "RPROP")
            loss_data2 = fit_with_early_stopping(field_comp, training_set, T_conn, area_T, hist_alpha, matprop, pffmodel,
                                                 optimizer_dict["weight_decay"], num_epochs=n_epochs, optimizer=optimizer, min_delta=optimizer_dict["optim_rel_tol"],
                                                 intermediateModel_path=intermediateModel_path, writer=writer, training_dict=training_dict)
            loss_data = loss_data + loss_data2

        end = time.time()
        print(f"Execution time: {(end-start)/60:.03f}minutes")

        hist_alpha = field_comp.update_hist_alpha(inp)

        torch.save(field_comp.net.state_dict(), trainedModel_path/Path('trained_1NN_' + str(j) + '.pt'))
        with open(trainedModel_path/Path('trainLoss_1NN_' + str(j) + '.npy'), 'wb') as file:
            np.save(file, np.asarray(loss_data))


def generate_L_mesh(nx=40, ny=40, remove_square=(0.0, 0.0, 0.5, 0.5)):
    """
    生成 L 形区域的均匀三角网格

    参数
    ----
    nx, ny : int
        x、y 方向等分段数（区间 [0,1] 被等分为 nx、ny 段）
    remove_square : (x_min, y_min, x_max, y_max)
        从单位方域中剔除的凹口方形区域

    返回
    ----
    nodes : (N,2) float64
    tris  : (M,3) int64
    areas : (M,)  float64
    meta  : dict  包含 dx, dy, nx, ny 等信息
    """
    x_min_r, y_min_r, x_max_r, y_max_r = remove_square

    xs = np.linspace(-0.5, 0.5, nx + 1)
    ys = np.linspace(-0.5, 0.5, ny + 1)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # full grid nodes (for temporary indexing)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    all_nodes = np.column_stack([X.ravel(), Y.ravel()])

    def node_id(i, j):
        return j * (nx + 1) + i

    tris_full = []
    used = set()

    for j in range(ny):
        for i in range(nx):
            x0, y0 = xs[i], ys[j]
            cx, cy = x0 + 0.5 * dx, y0 + 0.5 * dy

            # 若单元中心落入剔除方块，则跳过
            if (x_min_r <= cx <= x_max_r) and (y_min_r <= cy <= y_max_r):
                continue

            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n11 = node_id(i + 1, j + 1)
            n01 = node_id(i, j + 1)

            # 统一采用对角线 (n00 -> n11) 进行剖分
            tris_full.append((n00, n10, n11))
            tris_full.append((n00, n11, n01))
            used.update([n00, n10, n11, n01])

    tris_full = np.asarray(tris_full, dtype=np.int64)

    # 压缩节点索引，移除未用节点
    used_sorted = np.array(sorted(used), dtype=np.int64)
    old2new = -np.ones((nx + 1) * (ny + 1), dtype=np.int64)
    old2new[used_sorted] = np.arange(used_sorted.size, dtype=np.int64)

    nodes = all_nodes[used_sorted]
    tris = old2new[tris_full]

    # 计算三角形面积（向量化）
    P = nodes[tris]              # (M,3,2)
    v1 = P[:, 1, :] - P[:, 0, :]
    v2 = P[:, 2, :] - P[:, 0, :]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

    meta = dict(nx=nx, ny=ny, dx=dx, dy=dy, remove_square=remove_square)
    return nodes, tris, areas, meta