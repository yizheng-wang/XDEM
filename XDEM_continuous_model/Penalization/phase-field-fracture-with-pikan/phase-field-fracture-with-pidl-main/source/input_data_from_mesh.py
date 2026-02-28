import torch
import numpy as np
from pathlib import Path
from utils import parse_mesh, hist_alpha_init

# Prepare input data
def prep_input_data(matprop, pffmodel, crack_dict, numr_dict, mesh_file, device):
    '''
    Input data is prepared from the .msh file.
    If gradient_type = numerical:  
        X, Y = nodal coordinates
        T_conn = connectivity
    If gradient_type = autodiff:   
        X, Y = coordinate of the Gauss point in one point Gauss quadrature
        T_conn = None
    area_T: area of elements

    hist_alpha = initial alpha field

    '''
    assert Path(mesh_file).suffix == '.msh', "Mesh file should be a .msh file"
    
    X, Y, T_conn, area_T = parse_mesh(filename = mesh_file, gradient_type=numr_dict["gradient_type"])

    inp = torch.from_numpy(np.column_stack((X, Y))).to(torch.float).to(device)
    T_conn = torch.from_numpy(T_conn).to(torch.long).to(device)
    area_T = torch.from_numpy(area_T).to(torch.float).to(device)
    if numr_dict["gradient_type"] == 'autodiff':
        T_conn = None

    hist_alpha = hist_alpha_init(inp, matprop, pffmodel, crack_dict)
    
    return inp, T_conn, area_T, hist_alpha