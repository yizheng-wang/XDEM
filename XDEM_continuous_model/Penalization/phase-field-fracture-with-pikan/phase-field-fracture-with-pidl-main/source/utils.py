import torch
import torch.nn as nn
import numpy as np
import gmshparser



class DistanceFunction:
    def __init__(self, x_init, y_init, theta, L, d0, order: int = 2):
        self.x_init = x_init
        self.y_init = y_init
        self.theta = theta
        self.L = L
        self.d0 = d0
        self.order = order

    def __call__(self, inp):
        '''
        This function computes distance function given a line with origin at (x_init, y_init),
        oriented at an angle theta from x-axis, and of length L. Value of the function is 1 at
        the line and goes to 0 at a distance of d0 from the line.

        '''

        L = torch.tensor([self.L], device=inp.device)
        d0 = torch.tensor([self.d0], device=inp.device)
        theta = torch.tensor([self.theta], device=inp.device)
        input_c = torch.clone(inp)

        # transform coordinate to shift origin to (x_init, y_init) and rotate axis by theta
        input_c[:, -2:] = input_c[:, -2:] - torch.tensor([self.x_init, self.y_init], device=inp.device)
        Rt = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=inp.device)
        input_c[:, -2:] = torch.matmul(input_c[:, -2:], Rt)
        x = input_c[:, -2]
        y = input_c[:, -1]

        if self.order == 1:
            dist_fn_p1 = nn.ReLU()(x*(L-x))/(abs(x*(L-x))+np.finfo(float).eps)* \
                            nn.ReLU()(d0-abs(y))/(abs(d0-abs(y))+np.finfo(float).eps)* \
                            (1-abs(y)/d0)
            
            dist_fn_p2 = nn.ReLU()(x-L)/(abs(x-L)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-((x-L)**2+y**2))/(abs(d0**2-((x-L)**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt((x-L)**2+y**2)/d0)
            
            dist_fn_p3 = nn.ReLU()(-x)/(abs(x)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-(x**2+y**2))/(abs(d0**2-(x**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt(x**2 + y**2)/d0)
            
            dist_fn = dist_fn_p1 + dist_fn_p2 + dist_fn_p3

            
        if self.order == 2:
            dist_fn_p1 = nn.ReLU()(x*(L-x))/(abs(x*(L-x))+np.finfo(float).eps)* \
                            nn.ReLU()(d0-abs(y))/(abs(d0-abs(y))+np.finfo(float).eps)* \
                            (1-abs(y)/d0)**2
            
            dist_fn_p2 = nn.ReLU()(x-L)/(abs(x-L)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-((x-L)**2+y**2))/(abs(d0**2-((x-L)**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt((x-L)**2+y**2)/d0)**2
            
            dist_fn_p3 = nn.ReLU()(-x)/(abs(x)+np.finfo(float).eps)* \
                            nn.ReLU()(d0**2-(x**2+y**2))/(abs(d0**2-(x**2+y**2))+np.finfo(float).eps)* \
                            (1-torch.sqrt(x**2 + y**2)/d0)**2
            
            dist_fn = dist_fn_p1 + dist_fn_p2 + dist_fn_p3

        return dist_fn
    

    

def hist_alpha_init(inp, matprop, pffmodel, crack_dict):
    '''
    This function computes the initial phase field for a sample with a crack.
    See the paper "Phase-field modeling of fracture with physics-informed deep learning" for details.

    '''
    hist_alpha = torch.zeros((inp.shape[0], ), device = inp.device)
    
    if crack_dict["L_crack"][0] > 0:
        l0 = matprop.l0
        for j, L_crack in enumerate(crack_dict["L_crack"]):
            Lc = torch.tensor([L_crack], device=inp.device)
            theta = torch.tensor([crack_dict["angle_crack"][j]], device=inp.device)
            input_c = torch.clone(inp)

            # transform coordinate to shift origin to (x_init, y_init) and rotate axis by theta
            input_c[:, -2:] = input_c[:, -2:] - torch.tensor([crack_dict["x_init"][j], crack_dict["y_init"][j]], device=inp.device)
            Rt = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=inp.device)
            input_c[:, -2:] = torch.matmul(input_c[:, -2:], Rt)
            x = input_c[:, -2]
            y = input_c[:, -1]

            if pffmodel.PFF_model == 'AT1':
                hist_alpha_p1 = nn.ReLU()(x*(Lc-x))/(abs(x*(Lc-x))+np.finfo(float).eps)* \
                                    nn.ReLU()(2*l0-abs(y))/(abs(2*l0-abs(y))+np.finfo(float).eps)* \
                                    (1-abs(y)/l0/2)**2

                hist_alpha_p2 = nn.ReLU()(x-Lc+np.finfo(float).eps)/(abs(x-Lc)+np.finfo(float).eps)* \
                                    nn.ReLU()(2*l0-torch.sqrt((x-Lc)**2+y**2)+np.finfo(float).eps)/(abs(2*l0-torch.sqrt((x-Lc)**2+y**2))+np.finfo(float).eps)* \
                                    (1-torch.sqrt((x-Lc)**2+y**2)/l0/2)**2

                hist_alpha_p3 = nn.ReLU()(-x+np.finfo(float).eps)/(abs(x)+np.finfo(float).eps)* \
                                    nn.ReLU()(2*l0-torch.sqrt(x**2+y**2)+np.finfo(float).eps)/(abs(2*l0-torch.sqrt(x**2+y**2))+np.finfo(float).eps)* \
                                    (1-torch.sqrt(x**2+y**2)/l0/2)**2
                
            elif pffmodel.PFF_model == 'AT2':
                hist_alpha_p1 = nn.ReLU()(x*(Lc-x))/(abs(x*(Lc-x))+np.finfo(float).eps)* \
                                    torch.exp(-abs(y)/l0)

                hist_alpha_p2 = nn.ReLU()(x-Lc+np.finfo(float).eps)/(abs(x-Lc)+np.finfo(float).eps)* \
                                    torch.exp(-torch.sqrt((x-Lc)**2+y**2)/l0)

                hist_alpha_p3 = nn.ReLU()(-x+np.finfo(float).eps)/(abs(x)+np.finfo(float).eps)* \
                                    torch.exp(-torch.sqrt(x**2+y**2)/l0)

            hist_alpha = hist_alpha + hist_alpha_p1 + hist_alpha_p2 + hist_alpha_p3

    return hist_alpha



def parse_mesh(filename="meshed_geom.msh", gradient_type = 'numerical'):
    '''
    Parses .msh file to obtain nodal coordinates and connectivity assuming triangular elements.
    If numr_dict["gradient_type"] = autodiff, then Gauss points of elements in a one point Gauss 
    quadrature are returned.

    '''

    mesh = gmshparser.parse(filename)

    X, Y, T = gmshparser.helpers.get_triangles(mesh)
    assert T != [], "Discretization must have only triangular elements"
    X, Y, T = np.asarray(X), np.asarray(Y), np.asarray(T)

    area = X[T[:, 0]]*(Y[T[:, 1]]-Y[T[:, 2]]) + X[T[:, 1]]*(Y[T[:, 2]]-Y[T[:, 0]]) + X[T[:, 2]]*(Y[T[:, 0]]-Y[T[:, 1]])
    area = 0.5*area

    if gradient_type == 'autodiff':
        X = (X[T[:, 0]] + X[T[:, 1]] + X[T[:, 2]])/3
        Y = (Y[T[:, 0]] + Y[T[:, 1]] + Y[T[:, 2]])/3

    return X, Y, T, area