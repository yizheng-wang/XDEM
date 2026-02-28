from config import *

PATH_SOURCE = Path(__file__).parents[2]
sys.path.insert(0, str(PATH_SOURCE/Path('source')))

from field_computation import FieldComputation
from construct_model import construct_model
from model_train import train



# run as: python .\main.py hidden_layers neurons seed activation init_coeff
# for example: python .\main.py 8 400 1 TrainableReLU 3.0


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

## #############################################################################
## #############################################################################



## #############################################################################
# Training #####################################################################
## #############################################################################
if __name__ == "__main__":
    train(field_comp, disp, pffmodel, matprop, crack_dict, numr_dict,
          optimizer_dict, training_dict, coarse_mesh_file, fine_mesh_file, 
          device, trainedModel_path, intermediateModel_path, writer, L_uniform = True)
