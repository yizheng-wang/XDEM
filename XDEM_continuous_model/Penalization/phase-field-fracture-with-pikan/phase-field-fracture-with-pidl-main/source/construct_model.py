import torch
from pff_model import PFFModel
from material_properties import MaterialProperties
from network import KAN, init_xavier

def construct_model(PFF_model_dict, mat_prop_dict, network_dict, domain_extrema, device):
    # Phase field model
    pffmodel = PFFModel(PFF_model = PFF_model_dict["PFF_model"], 
                        se_split = PFF_model_dict["se_split"],
                        tol_ir = torch.tensor(PFF_model_dict["tol_ir"], device=device))

    # Material model
    matprop = MaterialProperties(mat_E = torch.tensor(mat_prop_dict["mat_E"], device=device), 
                                mat_nu = torch.tensor(mat_prop_dict["mat_nu"], device=device), 
                                w1 = torch.tensor(mat_prop_dict["w1"], device=device), 
                                l0 = torch.tensor(mat_prop_dict["l0"], device=device))

    # Neural network
    network = KAN([2, 15,15,15, 3], base_activation=torch.nn.SiLU, grid_size=15,  grid_range=[-0.5,  0.5], spline_order=3).cuda()
    torch.manual_seed(network_dict["seed"])
    #init_xavier(network)

    return pffmodel, matprop, network