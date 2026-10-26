###### Imports ######
# Libraries
import numpy as np
import numpy.matlib as matlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_func_transforms = True

# Custom modules
from pymindy.dataset import create_synth_HMMINDy
from pymindy.dataset.create_synth_data import pop_layout # JAKE
from pymindy.training.hmmindy_core import HMMINDyCore
from pymindy.training.utils import train_hmmindy_data_MSE

####### Hyperparameters #######
# Distribution of models
n_models = 20
#savefilepath = ''

# Model parameters
nChan = 20 # number of data channels
nPop = 2 # number of populations per channel (typically even to balance E/I ratio)
nS = 3 # number of modulation states

# JAKE: channel-major E/I population indices ([e1 i1 e2 i2 ...] per channel)
pop_type, is_E, E_idx, I_idx, chan_of = pop_layout(nChan, nPop)

# Device
device = "cpu"
data_loader_worker_num = 10

# acceptable ranges for parameters (no range means no limitation on parameter values)
param_range = {"Se_min": 0.25, "Se_max": 4,
               "Si_min": 0.25, "Si_max": 4,
               "De_max": 1, "De_min": 0,
               "Di_max": 1, "Di_min": 0}
stack_interval = 15 # distance between timepoints in each minibatch stack
mini_batch_size = 64 # number of random timepoints fit in each batch
n_batch = 25 # number of batches (iterations through full fitting algorithm)

# Optimizer (NADAM) configuration
lr = 0.05 # learning rate
betas = (0.9, 0.999)
weight_decay = 2 * 1e-2
amsgrad = True
clipping_grad_norm = True
clipping_grad_norm_value = 1

# Kalman filter configuration
warm_up_step = 5 # steps to filter before recording error for backpropagation
kf_step = 35 # total steps through kalman filter
rec_step = 0 # free simulation steps
P0_scale = 5 # variance of initial P0 estimate

# Define dataloader function (for partitioning the data into batches)
def prepare_data_loader(dataset, device):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=mini_batch_size,
                                               shuffle=True, pin_memory=True,
                                               pin_memory_device=device, drop_last=True,
                                               num_workers=data_loader_worker_num,
                                               persistent_workers=True)
    return train_loader

###### Create distribution of HM-MINDy models ######
# Create population of synthetic data models
true_models = [create_synth_HMMINDy(nChan, nPop, nS) for model in range(n_models)]

###### Fit HM-MINDy model to each synthetic data model and save into a list #####
# Create empty list to hold fit models and dict to hold correlation stats
fit_models = n_models * [None]
pop_corrs_MINDy = dict()
pop_corrs_MINDy['W_corr'] = np.empty((n_models,))
pop_corrs_MINDy['Wee_corr'] = np.empty((n_models,))
pop_corrs_MINDy['Wei_corr'] = np.empty((n_models,))

for model_ind in range(n_models):
    # Extract necessary true model params (H, Q, R are used for fitting; W is used for fit evaluation)
    true_model = true_models[model_ind]
    W_truth = true_model['Params']['W']
    W_mask = np.sign(W_truth)
    H = true_model['H']
    R = true_model['R']
    Q = true_model['Q']
    data = true_model['data'].T
    P0 = P0_scale * np.eye(H.shape[1])

    # Initialize fit model
    init_model = HMMINDyCore.init(H, Q, R, nS, W_mask, nPop=nPop, fix_Se=False, fix_Ve=False,
                                         fix_Si=True, fix_Vi=True, Si_fixed_value=1,
                                         Vi_fixed_value=0, param_range=param_range,
                                         device=device, rand_scale=1e-4)
    
    # Initialize optimizer with fit model
    optimizer = torch.optim.AdamW(init_model.parameters(), lr=lr, betas=betas,
                                  weight_decay=weight_decay, fused=True, amsgrad=amsgrad)
    # Fit model
    tmp = create_synth_HMMINDy(nChan, nPop, nS)
    Gamma_ind_hist = tmp['mod_ind_hist']

    model, train_info, mod_ind_fit = train_hmmindy_data_MSE(data, init_model, n_batch, stack_interval,
                                       warm_up_step, kf_step, rec_step, optimizer,
                                       P0, Gamma_ind_hist=Gamma_ind_hist,
                                       clipping_grad_norm=clipping_grad_norm,
                                       clipping_grad_norm_value=clipping_grad_norm_value,
                                       device=device,
                                       prepare_data_loader_func=prepare_data_loader)
    # Extract W matrices and components
    Wee_fit = model.Wee.detach().cpu().numpy()
    Wei_fit = model.Wei.detach().cpu().numpy()
    Wie_fit = model.Wie.detach().cpu().numpy()
    Wii_fit = model.Wii.detach().cpu().numpy()
    W_fit = model.W.detach().cpu().numpy() # JAKE
    #W_fit = np.hstack([np.vstack([Wee_fit, Wei_fit]), -np.vstack([np.diag(Wie_fit), np.diag(Wii_fit)])])
    # Compute W correlations
    corr_W = np.corrcoef(W_fit.flatten(), W_truth.flatten())[0,1]
    corr_Wee = np.corrcoef(Wee_fit.flatten(), W_truth[np.ix_(E_idx, E_idx)].flatten())[0,1]
    corr_Wei = np.corrcoef(Wei_fit.flatten(), W_truth[np.ix_(I_idx, E_idx)].flatten())[0,1]
    #corr_Wee = np.corrcoef(Wee_fit.flatten(), W_truth[:Wee_fit.shape[0], :Wee_fit.shape[1]].flatten())[0,1]
    #corr_Wei = np.corrcoef(Wei_fit.flatten(), W_truth[Wei_fit.shape[0]:, :Wei_fit.shape[1]].flatten())[0,1]

    # Save fit model and stats into list
    fit_model = dict()
    fit_model['model'] = model
    fit_model['train_info'] = train_info
    fit_model['corrs'] = dict()
    fit_model['corrs']['W'] = corr_W
    fit_model['corrs']['Wee'] = corr_Wee
    fit_model['corrs']['Wei'] = corr_Wei
    fit_models[model_ind] = fit_model

    # Save correlation stat values into population dictionary
    pop_corrs_MINDy['W_corr'][model_ind] = corr_W
    pop_corrs_MINDy['Wee_corr'][model_ind] = corr_Wee
    pop_corrs_MINDy['Wei_corr'][model_ind] = corr_Wei

    # Display stats for current fit
    print(f"Model Number: {model_ind}")
    print(f"\tFinal Loss: {train_info['train_MSE_losses_record_mini_batch'][-1]}")
    print(f"\tW Correlation: {corr_W}")
    print(f"\tWee Correlation: {corr_Wee}")
    print(f"\tWei Correlation: {corr_Wei}")
