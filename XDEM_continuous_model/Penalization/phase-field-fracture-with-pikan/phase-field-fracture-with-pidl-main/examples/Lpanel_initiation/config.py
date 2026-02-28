import numpy as np
import torch
from pathlib import Path
import sys
from torch.utils.tensorboard import SummaryWriter




'''
## ############################################################################
Refer to the paper 
"Phase-field modeling of fracture with physics-informed deep learning"
for details of the model.
## ############################################################################

'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



## ############################################################################
## customized for each problem ################################################
## ############################################################################
'''
network_dict:
parameters to construct an MLP
seed: seed to initialize the network
activation: choose from {SteepTanh, SteepReLU, TrainableTanh, TrainableReLU}
init_coeff: initial coefficient in activation function 
setting init_coeff = 1 in SteepTanh/SteepReLU gives standard Tanh/ReLU activation
'''

network_dict = {"model_type": 'MLP',
                "hidden_layers": int(sys.argv[1]) if len(sys.argv) > 1 else 6,
                "neurons": int(sys.argv[2]) if len(sys.argv) > 2 else 100,
                "seed": int(sys.argv[3]) if len(sys.argv) > 3 else 1,
                "activation": str(sys.argv[4]) if len(sys.argv) > 4 else 'TrainableReLU',
                "init_coeff": float(sys.argv[5]) if len(sys.argv) > 5 else 1.0}

'''
optimizer_dict:
weight_decay: weighing of neural network weight regularization
optim_rel_tol_pretrain: relative tolerance of loss in pretraining as an stopping criteria
optim_rel_tol: relative tolerance of loss in main training as an stopping criteria
'''

optimizer_dict = {"weight_decay": 1e-5,
                  "n_epochs_RPROP": 10000,
                  "n_epochs_LBFGS": 0,
                  "optim_rel_tol_pretrain": 1e-6,
                  "optim_rel_tol": 5e-7}


# save intermediate model during training every "save_model_every_n" steps
training_dict = {"save_model_every_n": 100}

'''
numr_dict:
"alpha_constraint" in {'nonsmooth', 'smooth'}
"gradient_type" in {'numerical', 'autodiff'}

PFF_model_dict:
PFF_model in {'AT1', 'AT2'} 
se_split in {'volumetric', None}
tol_ir: irreversibility tolerance

mat_prop_dict:
w1: Gc/l0, where Gc is energy release rate.
In the normalized formulation, mat_E=1, w1=1, and only nu and l0 are the properties to be set.
'''
numr_dict = {"alpha_constraint": 'nonsmooth', "gradient_type": 'numerical'}
# numr_dict = {"alpha_constraint": 'smooth', "gradient_type": 'autodiff'}
PFF_model_dict = {"PFF_model" : 'AT2', "se_split" : 'volumetric', "tol_ir" : 5e-3}
mat_prop_dict = {"mat_E" : 1.0, "mat_nu" : 0.18, "w1" : 1.0, "l0" : 0.01}


# Domain definition
'''
domain_extrema: tensor([[x_min, x_max], [y_min, y_max]])
x_init: list of x-coordinates of one end of cracks
y_init: list of y-coordinates of one end of cracks
L_crack: list of crack lengths
angle_crack: list of angles of cracks from the x-axis with the origin shifted to (x_init[i], y_init[i])
'''
domain_extrema = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
crack_dict = {"x_init" : [0], "y_init" : [0], "L_crack" : [0], "angle_crack" : [0]}


# Prescribed incremental displacement
loading_angle = torch.tensor([np.pi/2])
disp = np.concatenate((np.linspace(0.0, 0.5, 11), np.linspace(0.55, 1.4, 86)), axis=0)
disp = disp[1:]

## ############################################################################
## ############################################################################



## ############################################################################
## Domain discretization ######################################################
## ############################################################################
'''
Current implementation only accepts discretization with triangular elements.
coarse_mesh_file: mesh is fine only where crack is initially present (for efficient pretraining)
fine_mesh_file: fine discretization also where crack is expected to propagate.
'''
coarse_mesh_file = "meshed_geom1.msh"
fine_mesh_file = "meshed_geom2.msh"

## #############################################################################
## #############################################################################




## ############################################################################
## Setting output directory ###################################################
## ############################################################################
PATH_ROOT = Path(__file__).parents[0]
model_path = PATH_ROOT/Path('hl_'+str(network_dict["hidden_layers"])+
                            '_Neurons_'+str(network_dict["neurons"])+
                            '_activation_'+network_dict["activation"]+
                            '_coeff_'+str(network_dict["init_coeff"])+
                            '_Seed_'+str(network_dict["seed"])+
                            '_PFFmodel_'+str(PFF_model_dict["PFF_model"])+
                            '_gradient_'+str(numr_dict["gradient_type"]))
model_path.mkdir(parents=True, exist_ok=True)
trainedModel_path = model_path/Path('best_models/')
trainedModel_path.mkdir(parents=True, exist_ok=True)
intermediateModel_path = model_path/Path('intermediate_models/')
intermediateModel_path.mkdir(parents=True, exist_ok=True)

with open(model_path/Path('model_settings.txt'), 'w') as file:
    file.write(f'hidden_layers: {network_dict["hidden_layers"]}')
    file.write(f'\nneurons: {network_dict["neurons"]}')
    file.write(f'\nseed: {network_dict["seed"]}')
    file.write(f'\nactivation: {network_dict["activation"]}')
    file.write(f'\ncoeff: {network_dict["init_coeff"]}')
    file.write(f'\nPFF_model: {PFF_model_dict["PFF_model"]}')
    file.write(f'\nse_split: {PFF_model_dict["se_split"]}')
    file.write(f'\ngradient_type: {numr_dict["gradient_type"]}')
    file.write(f'\ndevice: {device}')

## #############################################################################
## #############################################################################


# logging loss to tensorboard
writer = SummaryWriter(model_path/Path('TBruns'))
