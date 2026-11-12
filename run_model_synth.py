# %% [markdown]
# # Imports

# %%
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime

import torch
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_func_transforms = True

# pop_layout() defines the channel-major population layout ([e1 i1 e2 i2 ...] per
# channel) used by create_synth_data.py and hmmindy_core.py
from pymindy.dataset.create_synth_data import create_synth_HMMINDy, pop_layout
from pymindy.pipeline import align_to_truth
from pymindy.training.hmmindy_core import HMMINDyCore
from pymindy.training.utils import train_hmmindy_data_MSE

# %% [markdown]
# # Hyperparameters

# %%
# Random generator
rng = np.random.default_rng()

# Distribution of models
n_models = 20 # Number of models in each distribution of fit models
savefilepath = ''

# Model parameters
nChan = 16 # number of data channels
nPop = 4 # number of populations per channel (typically even to balance E/I populations)
nS = 2 # number of modulation states

# Device
device="cpu"
data_loader_worker_num=10
# Acceptable ranges for parameters (no range means no limitation on parameter value)
param_range = {"Se_min": 0.25, "Se_max": 4, "Si_min": 0.25, "Si_max": 4, "De_max": 1, "De_min": 0, "Di_max": 1,
               "Di_min": 0} 
stack_interval = 15 # Distance between timepoints in each minibatch stack
mini_batch_size = 248 # number of random timepoints fit in each batch
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

# Channel-major population layout: population p of channel c is index nPop*c + p,
# alternating E/I within each channel ([e1 i1 e2 i2 ...])
pop_type, is_E, E_idx, I_idx, chan_of = pop_layout(nChan, nPop)

# H, Q, R, and the W mask are not set here: each synthetic model provides its own
# (H, Q, R directly, and the W mask from the nonzero entries of its true W)

# Initialize dataloader function (for partitioning data into batches)
def prepare_data_loader(dataset, device):
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=mini_batch_size,
            shuffle=True,
        pin_memory=False,
            drop_last=True,
        num_workers=data_loader_worker_num,
        persistent_workers=True
    )
    return train_loader

# %% [markdown]
# # Create HM-MINDy model distribution
# ## Create population of synthetic data models

# %%
true_models_mod = [create_synth_HMMINDy(nChan,nPop,nS,target_Gamma_corr=0) for i in range(n_models)]

# %% [markdown]
# ## Fit modulated MINDy model to each synthetic data model and save it in a list

# %%
#### Create empty list to hold fit models and dicts to hold stats and correlations
fit_models_mod = n_models * [None]

pop_stats_HMMINDy = dict()
pop_stats_HMMINDy['state_accuracy'] = np.empty((n_models,))
pop_stats_HMMINDy['rand_state_accuracy'] = np.empty((n_models,))
pop_stats_HMMINDy['kmeans_state_accuracy'] = np.empty((n_models,))
# Aligned confusion matrix for each fit, averaged in the mean confusion matrix cell
pop_stats_HMMINDy['conf_mat'] = np.empty((n_models,nS,nS))

pop_corrs_MINDy_mod = dict()
pop_corrs_MINDy_mod['W_corr'] = np.empty((n_models,))
pop_corrs_MINDy_mod['Wee_corr'] = np.empty((n_models,))
pop_corrs_MINDy_mod['Wei_corr'] = np.empty((n_models,))
pop_corrs_MINDy_mod['Gamma_corr'] = np.empty((n_models,nS))
pop_corrs_MINDy_mod['Gammaee_corr'] = np.empty((n_models,nS))
pop_corrs_MINDy_mod['Gammaei_corr'] = np.empty((n_models,nS))

for i in range(n_models):
    # Extract necessary true model params (H, Q, R are used for fitting; W and Gamma are used for fit evaluation and mask calculation)
    true_model = true_models_mod[i]
    W_truth = true_model['Params']['W']
    Gamma_truth = true_model['Params']['Gamma']
    W_mask = W_truth != 0
    H = true_model['H']
    Q = true_model['Q']
    R = true_model['R']
    mod_ind_true = true_model['mod_ind_hist'].astype(int)
    data = true_model['data'].T # (time x nChan)
    # The last column of the training data is the timepoint index
    data_ind = np.arange(data.shape[0])
    data = np.hstack((data,data_ind[:,None]))
    P0 = P0_scale * np.eye(H.shape[1])

    # Initialize fit model
    init_model_mod = HMMINDyCore.init(H, Q, R, nS, nPop=nPop, W_mask=W_mask, fixGamma=False, fix_Se=False, fix_Ve=False, fix_Si=True, fix_Vi=True, Si_fixed_value=1, Vi_fixed_value=0, param_range=param_range, device=device, 
                                      rand_scale=1e-4,stay_prob=0.999)

    # Initialize optimizer with fit model
    optimizer = torch.optim.AdamW(init_model_mod.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, fused=False, amsgrad=amsgrad)

    # K means baseline: cluster the data into nS states
    kmeans = KMeans(n_clusters=nS,n_init=5).fit(data[:,0:nChan])

    # Random HMM baseline: state history of an unrelated synthetic model
    tmp = create_synth_HMMINDy(nChan,nPop,nS,tRun=data.shape[0]+1)
    Gamma_ind_hist = tmp['mod_ind_hist']

    # Fit model
    mod_model, train_info, mod_ind_fit = train_hmmindy_data_MSE(data,
                                        init_model_mod, n_batch,
                                        stack_interval,
                                        warm_up_step,
                                        kf_step, rec_step,
                                        optimizer, P0,
                                        Gamma_ind_hist=Gamma_ind_hist,
                                        clipping_grad_norm=clipping_grad_norm,
                                        clipping_grad_norm_value=clipping_grad_norm_value,
                                        device=device,
                                        prepare_data_loader_func=prepare_data_loader)
    mod_ind_fit = mod_ind_fit.cpu().numpy()
    # Fit labels are not aligned to the true labels yet (that happens below)
    print(f"\tUnaligned State Accuracy: {np.sum((mod_ind_fit == mod_ind_true))/mod_ind_true.shape[0]}")

    # Hungarian algorithm: HMM state labels are arbitrary (fit state 0 may be true
    # state 1), so match fit labels to true labels before scoring. The same matching
    # is used to pair each true Gamma with its fit Gamma below.
    labels, mod_ind_fit = align_to_truth(mod_ind_true, mod_ind_fit, nS, nS)
    print(f"\tHungarian matches (true state, fit state): {labels}")

    # Extract W/Gamma matrices & components
    # The .W property assembles the full channel-major W from the fit blocks
    W_fit = mod_model.W.detach().cpu().numpy()
    Wee_fit = mod_model.Wee.detach().cpu().numpy()
    Wei_fit = mod_model.Wei.detach().cpu().numpy()
    # The .Gamma property is the full channel-major Gamma stack (kept unsigned,
    # matching create_synth_data.py's abs(Wmask)*...); reorder its states into true
    # state order, then slice its E/I blocks with E_idx/I_idx from pop_layout()
    Gamma_fit = mod_model.Gamma.detach().cpu().numpy()
    Gamma_fit = Gamma_fit[:,:,[fit_state for _, fit_state in labels]]
    Gammaee_fit = Gamma_fit[np.ix_(E_idx, E_idx)]
    Gammaei_fit = Gamma_fit[np.ix_(I_idx, E_idx)]

    # Compute W/Gamma correlations
    corr_W = np.corrcoef(W_fit.flatten(),W_truth.flatten())[0,1]
    corr_Wee = np.corrcoef(Wee_fit.flatten(),W_truth[np.ix_(E_idx,E_idx)].flatten())[0,1]
    corr_Wei = np.corrcoef(Wei_fit.flatten(),W_truth[np.ix_(I_idx,E_idx)].flatten())[0,1]
    corr_Gamma = np.zeros((nS,))
    corr_Gamma_ee = np.zeros((nS,))
    corr_Gamma_ei = np.zeros((nS,))
    for j in range(nS):
        corr_Gamma[j] = np.corrcoef(Gamma_fit[:,:,j].flatten(),Gamma_truth[:,:,j].flatten())[0,1]
        corr_Gamma_ee[j] = np.corrcoef(Gammaee_fit[:,:,j].flatten(),Gamma_truth[np.ix_(E_idx,E_idx)][:,:,j].flatten())[0,1]
        corr_Gamma_ei[j] = np.corrcoef(Gammaei_fit[:,:,j].flatten(),Gamma_truth[np.ix_(I_idx,E_idx)][:,:,j].flatten())[0,1]

    # Save fit model and stats into list
    fit_model = dict()
    fit_model['model'] = mod_model
    fit_model['train_info'] = train_info
    fit_model['mod_ind_fit'] = mod_ind_fit
    fit_model['labels'] = labels
    fit_model['corrs'] = dict()
    fit_model['corrs']['W'] = corr_W
    fit_model['corrs']['Wee'] = corr_Wee
    fit_model['corrs']['Wei'] = corr_Wei
    fit_model['corrs']['Gamma'] = corr_Gamma
    fit_model['corrs']['Gammaee'] = corr_Gamma_ee
    fit_model['corrs']['Gammaei'] = corr_Gamma_ei
    fit_models_mod[i] = fit_model
    # Save correlation values into population dictionary
    pop_corrs_MINDy_mod['W_corr'][i] = corr_W
    pop_corrs_MINDy_mod['Wee_corr'][i] = corr_Wee
    pop_corrs_MINDy_mod['Wei_corr'][i] = corr_Wei
    pop_corrs_MINDy_mod['Gamma_corr'][i,:] = corr_Gamma
    pop_corrs_MINDy_mod['Gammaee_corr'][i,:] = corr_Gamma_ee
    pop_corrs_MINDy_mod['Gammaei_corr'][i,:] = corr_Gamma_ei

    # Save stats only after alignment. The random and K-means baselines get the same
    # alignment so the comparison against HM-MINDy is fair.
    pop_stats_HMMINDy['state_accuracy'][i] = np.sum((mod_ind_fit == mod_ind_true))/mod_ind_true.shape[0]
    _, rand_ind_aligned = align_to_truth(mod_ind_true, Gamma_ind_hist, nS, nS)
    _, kmeans_ind_aligned = align_to_truth(mod_ind_true, kmeans.labels_, nS, nS)
    pop_stats_HMMINDy['rand_state_accuracy'][i] = np.sum((rand_ind_aligned == mod_ind_true))/mod_ind_true.shape[0]
    pop_stats_HMMINDy['kmeans_state_accuracy'][i] = np.sum((kmeans_ind_aligned == mod_ind_true))/mod_ind_true.shape[0]

    # Confusion matrix (aligned)
    conf_mat = np.zeros((nS,nS))
    for j in range(nS):
        for k in range(nS):
            conf_mat[j,k] = np.sum(np.logical_and(mod_ind_true==j, mod_ind_fit==k))/np.sum(mod_ind_true==j)
    print(conf_mat)
    pop_stats_HMMINDy['conf_mat'][i] = conf_mat
    
    # Display stats for current fit
    print(f"Model Number: {i}")
    print(f"\tFinal Loss: {train_info['train_MSE_losses_record_mini_batch'][-1]}")
    print(f"\tState Accuracy: {pop_stats_HMMINDy['state_accuracy'][i]}")
    # Baselines are aligned to the true labels the same way as the fit (Hungarian step above)
    print(f"\tRandom State Accuracy: {pop_stats_HMMINDy['rand_state_accuracy'][i]}")
    print(f"\tKmeans State Accuracy: {pop_stats_HMMINDy['kmeans_state_accuracy'][i]}")
    print(f"\tW Correlation: {corr_W}")
    print(f"\tWee Correlation: {corr_Wee}")
    print(f"\tWei Correlation: {corr_Wei}")
    for j in range(nS):
        print(f"\tGamma{j} Correlation: {corr_Gamma[j]}")
        print(f"\tGamma{j}ee Correlation: {corr_Gamma_ee[j]}")
        print(f"\tGamma{j}ei Correlation: {corr_Gamma_ei[j]}")

# %% [markdown]
# ## Plot results

# %%
# True vs. fit (aligned) state history for the last model
plt.figure()
plt.plot(true_models_mod[n_models-1]['mod_ind_hist'])
plt.plot(fit_models_mod[n_models-1]['mod_ind_fit'],'o',markevery=500,fillstyle='none')
ax = plt.gca()
ax.set_yticks(range(nS))
ax.set_xlabel('Time Points')
plt.savefig(f'images/synth_state_id_{n_models-1}.svg')
plt.show()

# %%
# Training loss for the last model
train_info = fit_models_mod[n_models-1]['train_info']
train_MSE=train_info['train_MSE'] # per batch includes total_size/mini_batch_size ,mini batches
train_MSE_mini_batch=train_info['train_MSE_losses_record_mini_batch']

## Plot the training loss

plt.figure(tight_layout=True)
plt.subplot(2,1,1)
plt.plot(train_MSE)
plt.xlabel("Batch")
plt.ylabel("MSE")

## Plot mini batch loss
plt.subplot(2,1,2)
plt.plot(train_MSE_mini_batch)
plt.xlabel("Mini Batch")
plt.ylabel("MSE")

# %%
# True vs. fit W and Gamma (EE blocks) for the last model
mod_model = fit_models_mod[n_models-1]['model']
W_truth = true_models_mod[n_models-1]['Params']['W']
Gamma_truth = true_models_mod[n_models-1]['Params']['Gamma']
W_fit = mod_model.W.detach().cpu().numpy()
# Reorder fit states into true state order using the saved Hungarian matches
labels = fit_models_mod[n_models-1]['labels']
Gamma_fit = mod_model.Gamma.detach().cpu().numpy()
Gamma_fit = Gamma_fit[:,:,[fit_state for _, fit_state in labels]]

fig,axes=plt.subplots(2,1+nS,figsize=(10*(1+nS),20))
sns.heatmap(W_truth[np.ix_(E_idx,E_idx)],ax=axes[0,0],square=True)
axes[0,0].set_title('True W (EE)')
sns.heatmap(W_fit[np.ix_(E_idx,E_idx)],ax=axes[1,0],square=True)
axes[1,0].set_title('Estimated W (EE)')
for j in range(nS):
    sns.heatmap(Gamma_truth[np.ix_(E_idx,E_idx)][:,:,j],ax=axes[0,1+j],square=True)
    axes[0,1+j].set_title(f'True Gamma{j} (EE)')
    sns.heatmap(Gamma_fit[np.ix_(E_idx,E_idx)][:,:,j],ax=axes[1,1+j],square=True)
    axes[1,1+j].set_title(f'Estimated Gamma{j} (EE)')
plt.show()

# %%
# Parameter recovery across all fits
plt.figure()
plt.boxplot([pop_corrs_MINDy_mod['W_corr'],pop_corrs_MINDy_mod['Wee_corr'],pop_corrs_MINDy_mod['Wei_corr'],
             pop_corrs_MINDy_mod['Gamma_corr'].flatten(),pop_corrs_MINDy_mod['Gammaee_corr'].flatten(),pop_corrs_MINDy_mod['Gammaei_corr'].flatten()])
ax = plt.gca()
ax.set_ylim([0, 1])
ax.set_xticklabels(['W','Wee','Wei','Gamma','Gammaee','Gammaei'])
ax.set_title('Fit vs. True Parameter Correlation')
plt.savefig('images/synth_param_corrs.svg')
plt.show()

# %%
# Does better Gamma recovery go with better state identification?
Gamma_corrs = np.mean(pop_corrs_MINDy_mod['Gamma_corr'],axis=1)
state_acc = pop_stats_HMMINDy['state_accuracy']

plt.figure()
plt.scatter(Gamma_corrs, state_acc)
plt.xlabel('Mean Gamma Correlation')
plt.ylabel('State Accuracy')
plt.show()

print('Correlation: ',np.corrcoef(Gamma_corrs,state_acc)[0,1])

# %%
# State accuracy vs. baselines (all aligned with the Hungarian algorithm in the fit cell)
state_acc = pop_stats_HMMINDy['state_accuracy']
random_state_acc = pop_stats_HMMINDy['rand_state_accuracy']
kmeans_state_acc = pop_stats_HMMINDy['kmeans_state_accuracy']
print(np.mean(state_acc))
print(np.median(state_acc))

plt.figure()
plt.boxplot([state_acc,random_state_acc,kmeans_state_acc])
ax = plt.gca()
ax.set_ylim([0, 1])
ax.set_xticklabels(['HM-MINDy','Random HMM','K-means'])
ax.set_title('State Accuracy')
plt.savefig('images/synth_state_accuracy.svg')
plt.show()

# %%
# Average the aligned per-fit confusion matrices saved in the fit cell
mean_conf_mat = np.mean(pop_stats_HMMINDy['conf_mat'], axis=0)

fig = plt.figure(figsize=(5,5))
sns.heatmap(mean_conf_mat,square=True,annot=True)
ax = plt.gca()
ax.set_xticklabels([f'State {j}' for j in range(nS)])
ax.set_yticklabels([f'State {j}' for j in range(nS)])
ax.set_ylabel('True State')
ax.set_xlabel('Predicted State')
plt.savefig(f'images/synth_conf_mat_mean.svg')
plt.show()

# %% [markdown]
# ## Save true and fit models to file

# %%
timestamp = datetime.datetime.now()
timestamp = timestamp.strftime("%m_%d_%Y_%H:%M:%S")
fit_file = 'HM-MINDy_synth_fit_'+timestamp+'.pkl'
with open(savefilepath + fit_file, 'wb') as file:
    pickle.dump({'true_models': true_models_mod, 'fit_models': fit_models_mod}, file)


