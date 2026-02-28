import torch.optim as optim
import numpy as np

def get_optimizer(params, optimizer_type: str="LBFGS"):
    if optimizer_type == "LBFGS":
        optimizer = optim.LBFGS(params, lr=float(0.5), max_iter=20000, max_eval=20000000, history_size=250,
                             line_search_fn="strong_wolfe",
                             tolerance_change=1.0*np.finfo(float).eps, tolerance_grad=1.0*np.finfo(float).eps)           
    elif optimizer_type == "ADAM":
        optimizer = optim.Adam(params, lr=5e-4, betas=(0.9, 0.999), eps=1.0*np.finfo(float).eps, weight_decay=0)
    elif optimizer_type == "RPROP":
        optimizer = optim.Rprop(params, lr=1e-5, step_sizes=(1e-10, 50))
    else:
        raise ValueError("Optimizer type not recognized. Please choose from LBFGS, ADAM, RPROP.")
    return optimizer
