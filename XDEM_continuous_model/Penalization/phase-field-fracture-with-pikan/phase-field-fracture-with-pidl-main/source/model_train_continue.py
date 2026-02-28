import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path

from input_data_from_mesh import prep_input_data
from fit import fit, fit_with_early_stopping
from optim import *
from plotting import plot_field

def train_continue(disp_idx, field_comp, disp, pffmodel, matprop, crack_dict, numr_dict, optimizer_dict, training_dict, coarse_mesh_file, fine_mesh_file, device, trainedModel_path, intermediateModel_path, writer):
    '''
    Neural network training: pretraining with a coarser mesh in the first stage before the main training proceeds.
    
    Input is prepared from the .msh file.

    Network training to learn the solution of the BVP in step wise loading.
    Trained network from the previous load step is used for learning the solution
    in the current load step.

    Trained models and loss data are saved in the trainedModel_path directory.
    '''


    ## #############################################################################


    ## #############################################################################
    # Main training ################################################################

    # Prepare input data
    inp, T_conn, area_T, hist_alpha = prep_input_data(matprop, pffmodel, crack_dict, numr_dict, mesh_file=fine_mesh_file, device=device)
    outp = torch.zeros(inp.shape[0], 1).to(device)
    training_set = DataLoader(torch.utils.data.TensorDataset(inp, outp), batch_size=inp.shape[0], shuffle=False)

    # solve BVP by step wise loading.
    for j, disp_i in enumerate(disp[disp_idx:]):
        j = j + disp_idx + 1
        disp_i = disp[j]
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
