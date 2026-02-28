import torch
import torch.nn as nn
from pathlib import Path
import sys

from utils import DistanceFunction

class FieldComputation:
    '''
    This class constructs the displacement and phase fields from the NN outputs by baking in the
    Dirichlet boundary conditions (BCs) and other constraints.

    net: neural network
    domain_extrema: tensor([[x_min, x_max], [y_min, y_max]])
    lmbda: prescribed displacement
    theta: Angle of the direction of loading from the x-axis (not used in all problems)
    alpha_ansatz: type of function to constrain alpha in {'smooth', 'nonsmooth'}
    
    DistanceFunction: using this object, distance function associated with a specified line
    to apply a BC can be constructed.
    (x_init, y_init): start of the line; theta: orientation of the line from the x-axis;
    L: length of the line; d0: support of the distance function;
    order: order of the polynomial to be used in the construction of the distance function

    fieldCalculation: applies BCs amd constraint on alpha (needs to be customized for each problem)

    update_hist_alpha: alpha_field for use in the next loading step to enforce irreversibility

    '''
    def __init__(self, net, domain_extrema, lmbda, theta, alpha_constraint = 'nonsmooth'):
        self.net = net
        self.domain_extrema = domain_extrema
        self.theta = theta
        self.lmbda = lmbda
        if alpha_constraint == 'smooth':
            self.alpha_constraint = torch.sigmoid
        else:
            self.alpha_constraint = NonsmoothSigmoid(2.0, 1e-3)

        self.load_dist = DistanceFunction(x_init=0.44, y_init=0, theta=0, L=0.06, d0=0.1, order = 2)
        self.fix_dist = DistanceFunction(x_init=-0.5, y_init=-0.5, theta=0, L=0.5, d0=0.1, order = 2)

    def fieldCalculation(self, inp):
        x0 = self.domain_extrema[0, 0]
        xL = self.domain_extrema[0, 1]
        y0 = self.domain_extrema[1, 0]
        yL = self.domain_extrema[1, 1]
        
        out = self.net(inp)
        out_disp = out[:, 0:2]
        
        alpha = self.alpha_constraint(out[:, 2])

        fix_window = 1-self.fix_dist(inp)
        load_window = self.load_dist(inp)

        u = fix_window*out_disp[:, 0]*self.lmbda
        v = fix_window*(1-load_window)*out_disp[:, 1]*self.lmbda + load_window*self.lmbda

        return u, v, alpha
    
    def update_hist_alpha(self, inp):
        _, _, pred_alpha = self.fieldCalculation(inp)
        pred_alpha = pred_alpha.detach()
        return pred_alpha
    

class NonsmoothSigmoid(nn.Module):
    '''
    Constructs a continuous piecewise linear increasing function with the
    central part valid in (-support, support) and its value going from 0 to 1. 
    Outside this region, the slope equals coeff.

    '''
    def __init__(self, support=2.0, coeff=1e-3):
        super(NonsmoothSigmoid, self).__init__()
        self.support = support
        self.coeff =  coeff
    def forward(self, x):
        a = x>self.support
        b = x<-self.support
        c = torch.logical_not(torch.logical_or(a, b))
        out = a*(self.coeff*(x-self.support)+1.0)+ \
                b*(self.coeff*(x+self.support))+ \
                c*(x/2.0/self.support+0.5)
        return out