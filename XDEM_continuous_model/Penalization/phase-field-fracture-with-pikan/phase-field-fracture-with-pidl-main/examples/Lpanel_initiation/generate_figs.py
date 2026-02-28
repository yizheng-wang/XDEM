from config import *

PATH_SOURCE = Path(__file__).parents[2]
sys.path.insert(0, str(PATH_SOURCE/Path('source')))

from field_computation import FieldComputation
from construct_model import construct_model
from input_data_from_mesh import prep_input_data
from plotting import plot_mesh, plot_field, img_plot, plot_energy

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

# prescribe the index of disp to generate plot
disp_idx = 75

# run as: python .\generate_figs.py hidden_layers neurons seed activation init_coeff
# for example: python .\generate_figs.py 8 400 1 TrainableReLU 3.0


device = 'cpu'
## ############################################################################
## Model construction #########################################################
## ############################################################################
pffmodel, matprop, network = construct_model(PFF_model_dict, mat_prop_dict, 
                                             network_dict, domain_extrema, device)
field_comp = FieldComputation(net = network,
                              domain_extrema = domain_extrema, 
                              lmbda = torch.tensor([0.0], device = device), 
                              theta = loading_angle, 
                              alpha_constraint = numr_dict["alpha_constraint"])
field_comp.net = field_comp.net.to(device)
field_comp.domain_extrema = field_comp.domain_extrema.to(device)
field_comp.theta = field_comp.theta.to(device)


# # Prepare input data
# inp, T_conn, area_T, hist_alpha = prep_input_data(matprop, pffmodel, crack_dict, numr_dict, 
#                                                          mesh_file=fine_mesh_file, device=device)


inp_np, T_conn_np, area_T_np, _ = generate_L_mesh(nx=100, ny=100, remove_square=(0.0, -0.5, 0.5, 0.0)) # 自己生成一个L形区域
hist_alpha_np = np.zeros(inp_np.shape[0])
# 把inp_np转换为torch.Tensor
inp = torch.tensor(inp_np, dtype=torch.float32, device=device)
T_conn = torch.tensor(T_conn_np, dtype=torch.int64, device=device)
area_T = torch.tensor(area_T_np, dtype=torch.float32, device=device)
hist_alpha = torch.tensor(hist_alpha_np, dtype=torch.float32, device=device)

## ############################################################################
## Setting up fig directory ###################################################
## ############################################################################
if Path.is_dir(model_path):
    figfiles = model_path/Path('figfiles')
    figfiles.mkdir(parents=True, exist_ok=True)
    pngfigs = figfiles/Path('pngfigs')
    pngfigs.mkdir(parents=True, exist_ok=True)
    pdffigs = figfiles/Path('pdffigs')
    pdffigs.mkdir(parents=True, exist_ok=True)
    arrayfigs = figfiles/Path('arrayfigs')
    arrayfigs.mkdir(parents=True, exist_ok=True)
    figdir = {"png": pngfigs, "pdf": pdffigs, "array": arrayfigs}

    print(f"tensorboard logdir = {model_path/Path('TBruns')}")

    # plot mesh
    plot_mesh(mesh_file=fine_mesh_file, figdir=figdir)

    # plot initial phase field
    plot_field(inp, hist_alpha, T_conn, figname='Initial-phase-field', figdir=figdir)


    # generate fields at prescribed displacement = disp[disp_idx]
    model = trainedModel_path/Path('trained_1NN_'+str(disp_idx)+'.pt')
    if Path.is_file(model):
        print(f"generating plots for prescribed displacement: {disp[disp_idx]}")
        field_comp.net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
        field_comp.lmbda = torch.tensor(disp[disp_idx]).to(device)
        img_plot(field_comp, pffmodel, matprop, inp, T_conn, area_T, figdir, dpi=600)
    else:
        print(f"No trained network available with filename: {model}")


    # plot energy vs prescribed displacement
    plot_energy(field_comp, disp, pffmodel, matprop, inp, T_conn, area_T, trainedModel_path, figdir)
