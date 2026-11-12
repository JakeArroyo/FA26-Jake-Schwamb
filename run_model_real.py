# %% [markdown]
# # Imports

# %%
import numpy as np
import numpy.matlib as matlib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#from plotly.subplots import make_subplots
import seaborn as sns
import pickle
import datetime

import os
import sys # JAKE: job array mode (see "Job array mode" below)
import argparse # JAKE

import torch
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_func_transforms = True
import scipy 

## from pymindy.dataset import PartitionDataset, create_synth_MINDy, create_synth_HMMINDy, simHMMINDy

# BEGIN NEW CODE - JAKE
# pop_layout() defines the channel-major population layout ([e1 i1 e2 i2 ...] per
# channel) used by create_synth_data.py and hmmindy_core.py; build_chan_mask() and
# build_W_mask() build the W mask directly in that layout
from pymindy.dataset.create_synth_data import (create_synth_MINDy, create_synth_HMMINDy, simHMMINDy,
                                               pop_layout, build_chan_mask, build_W_mask)
# Old (pre-general-nPop) fit pickles must be loaded through the legacy core
from pymindy.training.hmmindy_core_legacy import load_legacy_pickle
# END NEW CODE - JAKE
from pymindy.pipeline import Hungarian
# BEGIN JAKE CODE
from pymindy.pipeline import align_to_truth
# END JAKE CODE
#from pymindy.training.classic_core import pyMINDyCoreClassic
from pymindy.training.hmmindy_core import HMMINDyCore
from pymindy.training.utils import train_data_MSE, train_mod_data_MSE, train_hmmindy_data_MSE

# %% [markdown]
# # Hyperparameters

# %%
# BEGIN JAKE CODE
# Reproducibility / paper-matching settings (Schwamb et al., J Neurosci Methods 2026)
base_seed = 0 # every random draw below (W masks, init, minibatches, K-means, initial state sequence) derives from this
data_band = '0.5-12Hz' # folder in mat_files_alpha/: '8-12Hz' = paper's main analysis, '0.5-12Hz' = what earlier runs used
n_masks = 1 # Monte Carlo W masks (paper: 5 masks, each shared by all subjects, best mask reported). Set to 1 for a quick run
pStay_init = 0.99985 # stay probability of the random initial state sequence
# END JAKE CODE

# Random generator
rng = np.random.default_rng(base_seed) # JAKE: seeded

# Distribution of models
n_models = 20 # Number of models in each distribution of fit models
savefilepath = ''

# Model parameters
nChan = 16 # number of data channels
nPop = 2 # number of populations per channel (typically even to balance E/I populations)
nS = 2 # number of modulation states
# JAKE: True reproduces the original code's E->I modulation (broadcast einsum bug:
# every I target gets sum(I gains) * E source gain). False = per-entry product
legacy_gammaei = False # True = original code's E->I modulation (broadcast einsum bug). False = per-entry product

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

# BEGIN NEW CODE - JAKE
# Channel-major population layout: population p of channel c is index nPop*c + p,
# alternating E/I within each channel ([e1 i1 e2 i2 ...])
pop_type, is_E, E_idx, I_idx, chan_of = pop_layout(nChan, nPop)
nE = nPop // 2 # number of excitatory populations per channel

# W mask: top 25% of channel pairs (+ self) for E sources, local-only for I sources
# BEGIN JAKE CODE
# One seeded random mask per Monte Carlo draw; mask m is used for every subject in that draw.
# W_mask is reset to the best-performing mask after fitting (see the mask selection cell)
W_masks = [build_W_mask(nChan, nPop, build_chan_mask(nChan, 25, np.random.default_rng(base_seed + 10000 + m)))
           for m in range(n_masks)]
W_mask = W_masks[0]
# END JAKE CODE

# H, Q, and R initialization
# H observes each channel's E populations (averaged), mixed by L = eye - 0.05
L = np.eye(nChan) - 0.05*np.ones((nChan,nChan))
H = np.kron(L, is_E[:nPop][None,:] / nE)
Q = 0.25*np.eye(nPop*nChan)
R = 1.2*np.eye(nChan)
# END NEW CODE - JAKE

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

# %%
# BEGIN JAKE CODE
# Job array mode. Every (W mask, subject) fit is independent, so on the cluster each one
# runs as its own Slurm array task (lotus/submit_hmmindy.sh) instead of one long loop:
#   python run_model_real.py                       fit every run here, then summarize (original behavior)
#   python run_model_real.py --run R --run-dir D   fit only run R, save it in D, and stop
#   python run_model_real.py --aggregate D         skip fitting, load every run saved in D, then summarize
#   python run_model_real.py --count               print the number of runs (sizes the job array)
# Run R is W mask R // n_models, subject R % n_models. Its seeds depend only on R, so a run
# gives the same fit alone as inside the full loop.
n_runs = n_masks * n_models
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, help='index of the single run to fit (0 to n_runs-1)')
parser.add_argument('--run-dir', default='runs/local', help='where --run saves its result')
parser.add_argument('--aggregate', metavar='RUN_DIR', help='load saved runs from RUN_DIR instead of fitting')
parser.add_argument('--count', action='store_true', help='print the number of runs and exit')
args, _ = parser.parse_known_args() # known args only, so Jupyter kernel arguments are ignored
if args.count:
    print(n_runs)
    sys.exit(0)
if args.run is not None and args.aggregate is not None:
    sys.exit("Use --run or --aggregate, not both")
if args.run is not None and not 0 <= args.run < n_runs:
    sys.exit(f"--run must be between 0 and {n_runs-1}, got {args.run}")

if args.aggregate is not None:
    runs_to_fit = []
elif args.run is not None:
    runs_to_fit = [args.run]
else:
    runs_to_fit = range(n_runs)

# Settings that must match between the array tasks and the aggregate step; saved with
# every run and checked when the runs are loaded back
run_config = dict(base_seed=base_seed, data_band=data_band, n_masks=n_masks, n_models=n_models,
                  nChan=nChan, nPop=nPop, nS=nS, legacy_gammaei=legacy_gammaei, pStay_init=pStay_init,
                  n_batch=n_batch, mini_batch_size=mini_batch_size, lr=lr)

def run_file(run_dir, run):
    m, i = divmod(run, n_models)
    return os.path.join(run_dir, f'run_mask{m}_subject{i+1:02d}.pkl')
# END JAKE CODE

# %% [markdown]
# # Create HM-MINDy model distribution
# ## Create population of synthetic data models

# %%
#true_models_mod = [create_synth_HMMINDy(nChan,nPop,nS,target_Gamma_corr=0) for i in range(n_models)]
#scipy.io.savemat('ex_HMMINDy_3.mat',true_models_mod[0])
true_models_mod = n_models*[None]
#true_models_mod[0] = scipy.io.loadmat('mat_files_alpha/subject_02.mat')

# %% [markdown]
# ## Fit modulated MINDy model to each synthetic data model and save it in a list

# %%
#### Create empty lists to hold fit models and stats
# BEGIN JAKE CODE
# Every subject is fit once per W mask. Results for all (mask, subject) runs are kept in
# all_fit_models / all_stats; the mask selection cell below picks the best mask (paper
# procedure) and fills fit_models_mod / pop_stats_HMMINDy for the downstream cells.
all_fit_models = [n_models * [None] for _ in range(n_masks)]
all_stats = dict()
all_stats['state_accuracy'] = np.empty((n_masks, n_models))
all_stats['rand_state_accuracy'] = np.empty((n_masks, n_models))
all_stats['kmeans_state_accuracy'] = np.empty((n_masks, n_models))
# Aligned confusion matrix for each fit, averaged in the mean confusion matrix cell
all_stats['conf_mat'] = np.empty((n_masks, n_models, 2, nS))
for run in runs_to_fit: # JAKE: all runs, one run (--run), or none (--aggregate)
    m, i = divmod(run, n_models) # m = W mask index, i = subject index
    W_mask = W_masks[m]
    run_seed = base_seed + 100 * m + i
    torch.manual_seed(run_seed) # model init + minibatch shuffling
    np.random.seed(run_seed) # K-means re-labelling inside train_hmmindy_data_MSE
    run_rng = np.random.default_rng(run_seed) # initial state sequence
    print(f"===== W mask {m}, subject {i+1:02d} (seed {run_seed}) =====")
# END JAKE CODE
    true_models_mod[i] = scipy.io.loadmat(f'mat_files_alpha/{data_band}/subject_{i+1:02d}.mat') # JAKE: band set in hyperparameters
    # Extract necessary true model params (H, Q, R are used for fitting; W and Gamma are used for fit evaluation and mask calculation)
    true_model = true_models_mod[i]['data_str'] #[0]
    data = true_model['data'][0][0]
    #data = data[:60000,:]
    #true_model = scipy.io.loadmat('../MINDy_mod_paper/anesthesia-matformat/subj'+str(i+1)+'_full.mat')
    #data = true_model['full_data'].T #data_str['full_data']
    #mod_ind_true = np.squeeze(true_model['GammaIndHist'])
    #data = scipy.stats.zscore(data)
    data_ind = np.arange(data.shape[0])
    data = np.hstack((data,data_ind[:,None]))
    P0 = P0_scale * np.eye(H.shape[1])

    # Initialize fit model
    #init_model_mod = HMMINDyCore.init(H, Q, R, W_mask, n_HMM=nS, fix_Gamma=False, fix_Se=False, fix_Ve=False, fix_Si=True, fix_Vi=True, Si_fixed_value=1, Vi_fixed_value=0, param_range=param_range, device=device,
                                      #rand_scale=1e-4,stay_prob=0.999)
    init_model_mod = HMMINDyCore.init(H, Q, R, nS, nPop=nPop, W_mask=W_mask, fixGamma=False, fix_Se=False, fix_Ve=False, fix_Si=True, fix_Vi=True, Si_fixed_value=1, Vi_fixed_value=0, param_range=param_range, device=device, 
                                      rand_scale=1e-4,stay_prob=0.999, legacy_gammaei=legacy_gammaei)

    # Initialize optimizer with fit model
    optimizer = torch.optim.AdamW(init_model_mod.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, fused=False, amsgrad=amsgrad)

    # Plot data and true model index

    mod_ind_true = np.squeeze(true_model['mod_ind_hist'][0][0])
    #mod_ind_true = mod_ind_true[:60000]
    '''
    fig, axes = plt.subplots(int(nChan/2),2,figsize=(10,70))
    for ch in range(int(nChan/2)):
        axes[ch,0].plot(data[:,2*ch])
        axes[ch,0].plot((2*mod_ind_true))
        axes[ch,0].set_ylim([-10, 10])
        axes[ch,1].plot(data[:,2*ch+1])
        axes[ch,1].plot((2*mod_ind_true))
        axes[ch,1].set_ylim([-10, 10])
    '''
    #print(true_model['mod_ind_hist'][0][0].shape)
    #plt.figure()
    #plt.plot(data[:,0:nChan])
    #plt.show()
    #print(mod_ind_true.shape)
    #plt.plot((2*mod_ind_true))
    #plt.show()

    '''
    pca = PCA(n_components=4)
    pca.fit(data[:,0:nChan])
    data_red = pca.transform(data[:,0:nChan])
    data_red = np.hstack((data_red,data_ind[:,None]))
    print(pca.explained_variance_ratio_)
    print(data_red.shape)

    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes[0,0].plot(data_red[:,0])
    axes[0,0].plot(2*mod_ind_true)
    axes[0,1].plot(data_red[:,1])
    axes[0,1].plot(2*mod_ind_true)
    axes[1,0].plot(data_red[:,2])
    axes[1,0].plot(2*mod_ind_true)
    axes[1,1].plot(data_red[:,2])
    axes[1,1].plot(2*mod_ind_true)
    plt.show()
    '''
    # K means cluster data to get initial transitions estimate
    kmeans = KMeans(n_clusters=2,n_init=5,random_state=run_seed).fit(data[:,0:nChan]) # JAKE: seeded
    #print(kmeans.labels_)
    #plt.plot(kmeans.labels_)
    #plt.plot(np.std(data[:,0:nChan],1))
    #plt.show()
    #data_init_fit = np.hstack((data[:,0:20],kmeans.labels_[:,None]))

    # Fit initial model
    #mod_model, train_info = train_mod_data_MSE(data_init_fit,
    #                                    init_model_mod, 10,
    #                                    stack_interval,
    #                                    warm_up_step,
    #                                    kf_step, rec_step,
    #                                    optimizer, P0,
    #                                    clipping_grad_norm=clipping_grad_norm,
    #                                    clipping_grad_norm_value=clipping_grad_norm_value,
    #                                    device=device,
    #                                    prepare_data_loader_func=prepare_data_loader)
    
    # BEGIN JAKE CODE
    # Random initial state sequence. This used to come from create_synth_HMMINDy(), which
    # simulated a whole synthetic model just to keep its state history and could not be
    # seeded. Same distribution: uniform start, then stay with probability pStay_init.
    A_init = pStay_init * np.eye(nS) + ((1 - pStay_init)/(nS - 1)) * (np.ones((nS,nS)) - np.eye(nS))
    Gamma_ind_hist = np.empty((data.shape[0],))
    state = run_rng.choice(nS)
    for t in range(data.shape[0]):
        Gamma_ind_hist[t] = state
        state = run_rng.choice(nS, p=A_init[:, state])
    # END JAKE CODE
    #plt.plot(mod_ind_true)
    #plt.plot(Gamma_ind_hist)
    #plt.plot(kmeans.labels_)
    #plt.savefig("mod_ind_true.svg")
    #plt.show()

    
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
    # BEGIN JAKE CODE
    # Fit labels are not aligned to the true labels yet (that happens below), so this can be < 0.5
    print(f"\tUnaligned State Accuracy: {np.sum((mod_ind_fit == mod_ind_true))/mod_ind_true.shape[0]}")
    # END JAKE CODE
    
    
    '''
    # Extract W/Gamma matrices & components
    Wee_fit = mod_model.Wee.detach().cpu().numpy()
    Wei_fit = mod_model.Wei.detach().cpu().numpy()
    Wie_fit = mod_model.Wie.detach().cpu().numpy() #local blocks (diag vector for nPop=2)
    Wii_fit = mod_model.Wii.detach().cpu().numpy() #local blocks (diag vector for nPop=2)
    # JAKE: the .W property assembles the full channel-major W from the fit blocks
    W_fit = mod_model.W.detach().cpu().numpy()
    #W_fit = np.hstack([np.vstack([Wee_fit,Wei_fit]),-np.vstack([np.diag(Wie_fit),np.diag(Wii_fit)])])
    # JAKE: the .Gamma property is the full channel-major Gamma stack (kept unsigned,
    # matching create_synth_data.py's abs(Wmask)*...); slice its E/I blocks with
    # E_idx/I_idx from pop_layout()
    Gamma_fit = mod_model.Gamma.detach().cpu().numpy()
    Gammaee_fit = Gamma_fit[np.ix_(E_idx, E_idx)]
    Gammaei_fit = Gamma_fit[np.ix_(I_idx, E_idx)]
    #Gamma_fit = np.empty((nPop*nChan,nPop*nChan,nS))
    #for j in range(nS):
    #    Gamma_fit[:,:,j] = np.hstack([np.vstack([Gammaee_fit[:,:,j],Gammaei_fit[:,:,j]]),-np.vstack([np.diag(np.squeeze(Gammaie_fit[:,j])),np.diag(np.squeeze(Gammaii_fit[:,j]))])])

    # Compute W/Gamma correlations
    corr_W = np.corrcoef(W_fit.flatten(),W_truth.flatten())[0,1]
    corr_Wee = np.corrcoef(Wee_fit.flatten(),W_truth[np.ix_(E_idx,E_idx)].flatten())[0,1]
    corr_Wei = np.corrcoef(Wei_fit.flatten(),W_truth[np.ix_(I_idx,E_idx)].flatten())[0,1]
    #corr_Wee = np.corrcoef(Wee_fit.flatten(),W_truth[:Wee_fit.shape[0],:Wee_fit.shape[1]].flatten())[0,1]
    #corr_Wei = np.corrcoef(Wei_fit.flatten(),W_truth[Wee_fit.shape[0]:,:Wei_fit.shape[1]].flatten())[0,1]
    corr_Gamma = np.zeros((nS,))
    corr_Gamma_ee = np.zeros((nS,))
    corr_Gamma_ei = np.zeros((nS,))
    for j in range(nS):
        corr_Gamma[j] = np.corrcoef(Gamma_fit[:,:,j].flatten(),Gamma_truth[:,:,j].flatten())[0,1]
        corr_Gamma_ee[j] = np.corrcoef(Gammaee_fit[:,:,j].flatten(),Gamma_truth[np.ix_(E_idx,E_idx)][:,:,j].flatten())[0,1]
        corr_Gamma_ei[j] = np.corrcoef(Gammaei_fit[:,:,j].flatten(),Gamma_truth[np.ix_(I_idx,E_idx)][:,:,j].flatten())[0,1]
    '''

    
    # Save fit model and stats into list
    fit_model = dict()
    fit_model['model'] = mod_model
    fit_model['train_info'] = train_info
    fit_model['mod_ind_fit'] = mod_ind_fit
    all_fit_models[m][i] = fit_model
    
    '''
    fit_model['corrs'] = dict()
    fit_model['corrs']['W'] = corr_W
    fit_model['corrs']['Wee'] = corr_Wee
    fit_model['corrs']['Wei'] = corr_Wei
    fit_model['corrs']['Gamma'] = corr_Gamma
    fit_model['corrs']['Gammaee'] = corr_Gamma_ee
    fit_model['corrs']['Gammaei'] = corr_Gamma_ei
    all_fit_models[m][i] = fit_model
    # Save correlation values into population dictionary
    pop_corrs_MINDy_mod['W_corr'][i] = corr_W
    pop_corrs_MINDy_mod['Wee_corr'][i] = corr_Wee
    pop_corrs_MINDy_mod['Wei_corr'][i] = corr_Wei
    pop_corrs_MINDy_mod['Gamma_corr'][i,:] = corr_Gamma
    pop_corrs_MINDy_mod['Gammaee_corr'][i,:] = corr_Gamma_ee
    pop_corrs_MINDy_mod['Gammaei_corr'][i,:] = corr_Gamma_ei
    '''
    
    # BEGIN JAKE CODE
    # state_accuracy / rand_state_accuracy / kmeans_state_accuracy are now saved after
    # the Hungarian alignment below; saving them here recorded unaligned values
    # END JAKE CODE
    
    
    # Confusion matrix
    conf_mat = np.zeros((2,nS))
    for j in range(2):
        for k in range(nS):
            conf_mat[j,k] = np.sum(np.logical_and(mod_ind_true==j, mod_ind_fit==k))/np.sum(mod_ind_true==j)
    #conf_mat = np.vstack((conf_mat,np.ones((1,nS))))
    print(conf_mat)

    # BEGIN JAKE CODE
    # Hungarian algorithm: HMM state labels are arbitrary (fit state 0 may be true
    # state 1), so match fit labels to true labels before scoring. Matching on
    # timepoint counts maximizes raw accuracy, so with 2 states it is always >= 0.5.
    labels, mod_ind_fit = align_to_truth(mod_ind_true, mod_ind_fit, 2, nS)
    print(f"\tHungarian matches (true state, fit state): {labels}")

    # Save stats only after alignment. The random and K-means baselines get the same
    # alignment so the comparison against HM-MINDy is fair.
    all_stats['state_accuracy'][m, i] = np.sum((mod_ind_fit == mod_ind_true))/mod_ind_true.shape[0]
    _, rand_ind_aligned = align_to_truth(mod_ind_true, Gamma_ind_hist, 2, nS)
    _, kmeans_ind_aligned = align_to_truth(mod_ind_true, kmeans.labels_, 2, 2)
    all_stats['rand_state_accuracy'][m, i] = np.sum((rand_ind_aligned == mod_ind_true))/mod_ind_true.shape[0]
    all_stats['kmeans_state_accuracy'][m, i] = np.sum((kmeans_ind_aligned == mod_ind_true))/mod_ind_true.shape[0]
    # END JAKE CODE

    '''
    plt.figure()
    plt.plot(mod_ind_true)
    plt.plot(mod_ind_fit)
    ax = plt.gca()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Eyes Open','Eyes Closed'])
    ax.set_xlabel('Time Points')
    plt.savefig(f'images/state_id_mask{m}_{i}.svg')
    plt.show()
    '''
    
    # Confusion matrix
    conf_mat = np.zeros((2,nS))
    for j in range(2):
        for k in range(nS):
            conf_mat[j,k] = np.sum(np.logical_and(mod_ind_true==j, mod_ind_fit==k))/np.sum(mod_ind_true==j)
    print(conf_mat)
    # BEGIN JAKE CODE
    all_stats['conf_mat'][m, i] = conf_mat
    # Per-fit confusion matrix images are turned off (only conf_mat_mean.svg is needed).
    # Uncomment to save one image per (mask, subject) again. plt.figure() gives each
    # heatmap its own figure; without it, on the cluster (no plot window) every heatmap
    # was drawn on top of the previous ones. plt.close() frees the figure afterwards.
    '''
    plt.figure()
    # JAKE: labels passed to heatmap directly; set_xticklabels() crashed when seaborn drew fewer ticks
    #sns.heatmap(conf_mat,annot=True,xticklabels=['Eyes Open','Eyes Closed','State 3'],yticklabels=['Eyes Open','Eyes Closed'])
    sns.heatmap(conf_mat,annot=True,xticklabels=['Eyes Open','Eyes Closed'],yticklabels=['Eyes Open','Eyes Closed'])
    ax = plt.gca()
    ax.set_ylabel('True State')
    ax.set_xlabel('Predicted State')
    plt.savefig(f'images/conf_mat_mask{m}_{i}.svg') # JAKE: one file per (mask, subject)
    plt.show()
    plt.close()
    '''
    # END JAKE CODE
    
    # Display stats for current fit
    print(f"Model Number: {i} (W mask {m})")
    print(f"\tFinal Loss: {train_info['train_MSE_losses_record_mini_batch'][-1]}")
    print(f"\tState Accuracy: {np.sum((mod_ind_fit == mod_ind_true))/mod_ind_true.shape[0]}")
    print(f"\tAlternate State Accuracy: {0.5 * np.sum(np.logical_and(mod_ind_true==0, mod_ind_fit==0))/np.sum(mod_ind_true==0) + 0.5 * np.sum(np.logical_and(mod_ind_true==1, mod_ind_fit==1))/np.sum(mod_ind_true==1)}")
    
    # BEGIN JAKE CODE
    # Baselines are aligned to the true labels the same way as the fit (Hungarian step above)
    print(f"\tRandom State Accuracy: {all_stats['rand_state_accuracy'][m, i]}")
    print(f"\tKmeans State Accuracy: {all_stats['kmeans_state_accuracy'][m, i]}")
    # END JAKE CODE
    '''
    print(f"\tW Correlation: {corr_W}")
    print(f"\tWee Correlation: {corr_Wee}")
    print(f"\tWei Correlation: {corr_Wei}")
    for j in range(nS):
        print(f"\tGamma{j} Correlation: {corr_Gamma[j]}")
        print(f"\tGamma{j}ee Correlation: {corr_Gamma_ee[j]}")
        print(f"\tGamma{j}ei Correlation: {corr_Gamma_ei[j]}")
    '''

    # BEGIN JAKE CODE
    # Job array mode: save this run's fit and stats for the aggregate step. Written to a
    # temporary name first, so a job killed mid-write never leaves a half-written file.
    if args.run is not None:
        os.makedirs(args.run_dir, exist_ok=True)
        run_result = {'fit_model': fit_model,
                      'stats': {key: val[m, i] for key, val in all_stats.items()},
                      'config': run_config}
        out_file = run_file(args.run_dir, run)
        with open(out_file + '.tmp', 'wb') as file:
            pickle.dump(run_result, file)
        os.replace(out_file + '.tmp', out_file)
        print(f"Saved run {run} to {out_file}")
    # END JAKE CODE

# %%
# BEGIN JAKE CODE
# Job array mode: a single run stops here; the aggregate step makes the summary figures
if args.run is not None:
    sys.exit(0)

# Aggregate mode: fill all_fit_models / all_stats from the runs saved by the array tasks
if args.aggregate is not None:
    missing = [run_file(args.aggregate, r) for r in range(n_runs) if not os.path.exists(run_file(args.aggregate, r))]
    if missing:
        sys.exit(f"{len(missing)} of {n_runs} runs missing from {args.aggregate}:\n  " + "\n  ".join(missing))
    for run in range(n_runs):
        m, i = divmod(run, n_models)
        with open(run_file(args.aggregate, run), 'rb') as file:
            run_result = pickle.load(file)
        if run_result['config'] != run_config:
            sys.exit(f"{run_file(args.aggregate, run)} was fit with different settings:\n"
                     f"  saved:   {run_result['config']}\n  current: {run_config}")
        all_fit_models[m][i] = run_result['fit_model']
        for key, val in run_result['stats'].items():
            all_stats[key][m, i] = val
    print(f"Loaded {n_runs} runs from {args.aggregate}")
# END JAKE CODE

# %%
# BEGIN JAKE CODE
# W mask selection (paper procedure): keep the mask with the best mean state accuracy across
# subjects. Picking with the true labels inflates the score a bit, so the average over all
# masks (no selection) is printed too; that is the fairer number for comparing code versions.
mask_mean_acc = all_stats['state_accuracy'].mean(axis=1)
best_mask = int(np.argmax(mask_mean_acc))
for m in range(n_masks):
    acc = all_stats['state_accuracy'][m]
    print(f"W mask {m}: mean {acc.mean():.3f}, median {np.median(acc):.3f}, "
          f"IQR {np.percentile(acc, 25):.3f}-{np.percentile(acc, 75):.3f}" + ("  <- best" if m == best_mask else ""))
acc_all = all_stats['state_accuracy']
print(f"All masks (no selection): mean {acc_all.mean():.3f}, median {np.median(acc_all):.3f}")
print("Paper, 8-12 Hz: mean 0.615, median 0.62 (IQR 0.53-0.75)")

fit_models_mod = all_fit_models[best_mask]
pop_stats_HMMINDy = {key: val[best_mask] for key, val in all_stats.items()}
W_mask = W_masks[best_mask]
mod_model = fit_models_mod[-1]['model'] # last subject, as before (used by later cells)
train_info = fit_models_mod[-1]['train_info']
# END JAKE CODE


# %%
true_model = scipy.io.loadmat(f'mat_files_alpha/{data_band}/subject_{n_models:02d}.mat') # JAKE: same band/subject as the fit shown
true_model = true_model['data_str']
mod_ind_true = np.squeeze(true_model['mod_ind_hist'][0][0])
#mod_ind_true = mod_ind_true[:60000]
mod_ind_fit = fit_models_mod[n_models-1]['mod_ind_fit']
plt.figure()
plt.plot(mod_ind_true)
#plt.plot(mod_ind_fit)
plt.plot(mod_ind_fit,'o',markevery=500,fillstyle='none')
plt.savefig(f'images/state_id_{n_models-1}.svg')
plt.show()
plt.close()
'''
print(mod_model.HMM['A'])
print(np.sum(mod_ind_fit.numpy() == mod_ind_true)/mod_ind_true.shape[0])
print(f"\tW Correlation: {corr_W}")
print(f"\tWee Correlation: {corr_Wee}")
print(f"\tWei Correlation: {corr_Wei}")
for j in range(nS):
    print(f"\tGamma{j} Correlation: {corr_Gamma[j]}")
    print(f"\tGamma{j}ee Correlation: {corr_Gamma_ee[j]}")
    print(f"\tGamma{j}ei Correlation: {corr_Gamma_ei[j]}")
'''

# %%
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
# Extract W/Gamma matrices & components
# JAKE: E/I indices come from each fit model's own layout (not the nPop hyperparameter),
# so this also works on models loaded from a pickle fit with a different nPop.
# EE blocks are (nChan*nE x nChan*nE).
for i in range(n_models):
    mod_model = fit_models_mod[i]
    mod_model = mod_model['model']
    _, _, E_idx_fit, I_idx_fit, _ = pop_layout(mod_model.n_chan, mod_model.n_pop_per_chan)
    if i == 0:
        Gamma_fit_pop = np.zeros((E_idx_fit.size,E_idx_fit.size,nS,n_models))
        Gamma_fit_pop_diff = np.zeros((E_idx_fit.size,E_idx_fit.size,n_models))
    Wee_fit = mod_model.Wee.detach().cpu().numpy()
    Wei_fit = mod_model.Wei.detach().cpu().numpy()
    Wie_fit = mod_model.Wie.detach().cpu().numpy() #local blocks (diag vector for nPop=2)
    Wii_fit = mod_model.Wii.detach().cpu().numpy() #local blocks (diag vector for nPop=2)
    # JAKE: the .W property assembles the full channel-major W from the fit blocks
    W_fit = mod_model.W.detach().cpu().numpy()
    #W_fit = np.hstack([np.vstack([Wee_fit,Wei_fit]),-np.vstack([np.diag(Wie_fit),np.diag(Wii_fit)])])
    # JAKE: the .Gamma property is the full channel-major Gamma stack (kept unsigned,
    # matching create_synth_data.py's abs(Wmask)*...); slice its E/I blocks with
    # E_idx/I_idx from pop_layout()
    Gamma_fit = mod_model.Gamma.detach().cpu().numpy()
    Gammaee_fit = Gamma_fit[np.ix_(E_idx_fit, E_idx_fit)]
    Gammaei_fit = Gamma_fit[np.ix_(I_idx_fit, E_idx_fit)]
    #Gamma_fit = np.empty((nPop*nChan,nPop*nChan,nS))
    #for j in range(nS):
    #    Gamma_fit[:,:,j] = np.hstack([np.vstack([Gammaee_fit[:,:,j],Gammaei_fit[:,:,j]]),-np.vstack([np.diag(np.squeeze(Gammaie_fit[:,j])),np.diag(np.squeeze(Gammaii_fit[:,j]))])])
    Gamma_fit_pop[:,:,:,i] = Gammaee_fit
    Gamma_fit_pop_diff[:,:,i] = Gammaee_fit[:,:,1] - Gammaee_fit[:,:,0]

fig,axes=plt.subplots(1,3,figsize=(30,10))
sns.heatmap(W_fit[np.ix_(E_idx_fit,E_idx_fit)],ax=axes[0],square=True,vmin=0,vmax=0.75) # JAKE: EE block
axes[0].set_title('Estimated W')
sns.heatmap(np.mean(Gamma_fit_pop[:,:,0,:],axis=2),ax=axes[1],square=True,annot=True) # JAKE: already the EE block
axes[1].set_title('Estimated Gamma, Eyes Open')
sns.heatmap(np.mean(Gamma_fit_pop[:,:,1,:],axis=2),ax=axes[2],square=True,annot=True) # JAKE: already the EE block
axes[2].set_title('Estimated Gamma, Eyes Closed')


fig = plt.figure(figsize=(10,10))
sns.heatmap(np.mean(Gamma_fit_pop_diff,axis=2),square=True,annot=True)
plt.show()

# %%
Gamma_corrs = [99.64, 99.48, 98.46, 97.30, 98.81, 99.51, 96.59, 99.39, 99.04, 97.59, 94.21, 98.04, 99.06, 99.28, 99.39, 98.4, 98.98, 98.66, 98.37, 95.22]
state_acc = [57.94, 68.31, 62.72, 83.29, 71.7, 59.61, 72.68, 50.33, 60.84, 56.14, 74.88, 51.17, 81.21, 95.79, 66.6, 55.19, 81.7, 99.59, 74.22, 99.93]

plt.scatter(Gamma_corrs, state_acc)
plt.show()

print('Correlation: ',np.corrcoef(Gamma_corrs,state_acc)[0,1])

# %% [markdown]
# ## Save true and fit models to file

# %%
timestamp = datetime.datetime.now()
timestamp = timestamp.strftime("%m_%d_%Y_%H:%M:%S")
fit_file = 'HM-MINDy_alpha_fit_'+timestamp+'.pkl'
with open(savefilepath + fit_file, 'wb') as file:
    pickle.dump(fit_models_mod, file)

# %%
# BEGIN JAKE CODE
# Plot the results from this run (saved in cell 7 after Hungarian alignment)
# instead of hard-coded values from an earlier run
state_acc = pop_stats_HMMINDy['state_accuracy']
random_state_acc = pop_stats_HMMINDy['rand_state_accuracy']
kmeans_state_acc = pop_stats_HMMINDy['kmeans_state_accuracy']

# Hard-coded values from an earlier run, kept for reference:
#state_acc = np.array([0.57, 0.81, 0.66, 0.50, 0.51, 0.55, 0.71, 0.80, 0.55, 0.79, 0.62, 0.67, 0.53, 0.86, 0.73, 0.51, 0.52])
#random_state_acc = np.array([0.55,0.50,0.35,0.43,0.45,0.49,0.59,0.52,0.43,0.64,0.63,0.48,0.52,0.44,0.61,0.41,0.49,0.50,0.60,0.46])
#kmeans_state_acc = np.array([0.50,0.65,0.50,0.50,0.49,0.50,0.49,0.50,0.50,0.50,0.50,0.48,0.50,0.50,0.50,0.50,0.50,0.49,0.50,0.50])
# The two commented lines below flipped accuracies under 0.5 by hand; the Hungarian
# alignment in cell 7 now does this, so they are no longer needed
# END JAKE CODE
#pop_stats_HMMINDy['state_accuracy'][pop_stats_HMMINDy['state_accuracy']<0.5] = 1 - pop_stats_HMMINDy['state_accuracy'][pop_stats_HMMINDy['state_accuracy']<0.5]
#pop_stats_HMMINDy['kmeans_state_accuracy'][pop_stats_HMMINDy['kmeans_state_accuracy']<0.5] = 1 - pop_stats_HMMINDy['kmeans_state_accuracy'][pop_stats_HMMINDy['kmeans_state_accuracy']<0.5]
print(np.mean(state_acc))
print(np.median(state_acc))

# JAKE: new figure, otherwise the boxplot is drawn on top of the last open figure
# (the Gamma difference heatmap and scatter plot above) when run on the cluster
plt.figure()
plt.boxplot([state_acc,random_state_acc,kmeans_state_acc])
ax = plt.gca()
ax.set_ylim([0, 1])
ax.set_xticklabels(['HM-MINDy','Random HMM','K-means'])
ax.set_title('State Accuracy')
plt.savefig('images/alpha_state_accuracy.svg')
plt.show()
plt.close()

fig = go.Figure()
fig.add_trace(go.Violin(x=state_acc,name='HM-MINDy',box_visible=True,side='positive',points='all'))
fig.add_trace(go.Violin(x=random_state_acc,name='Random HMM',box_visible=True,side='positive',points='all'))
fig.add_trace(go.Violin(x=kmeans_state_acc,name='K Means',box_visible=True,side='positive',points='all'))
fig.update_xaxes(range=[0, 1])
fig.update_layout(title=dict(text='State Accuracy'),xaxis=dict(title=dict(text='Percent of Time Points Correctly Identified')))
#fig.write_image('alpha_state_accuracy.svg')

# %%
#with open('HM-MINDy_alpha_fit_07_13_2025_08:23:23.pkl','rb') as file:
    #fit_models_mod = pickle.load(file)

# JAKE: this file was saved with the old (nPop=2) HMMINDyCore, so load it through
# the legacy class; fits saved after the general-nPop rewrite use pickle.load()
with open('HM-MINDy_alpha_fit_10_19_2026_18:59:57.pkl','rb') as file:
    fit_models_mod = load_legacy_pickle(file)

# %%
# BEGIN JAKE CODE
# Average the aligned per-fit confusion matrices saved in cell 7
# instead of hard-coded values from an earlier run
mean_conf_mat = np.mean(pop_stats_HMMINDy['conf_mat'], axis=0)
# Hard-coded values from an earlier run, kept for reference:
#mean_conf_mat = [[0.56, 0.44],[0.31,0.69]]
# END JAKE CODE

fig = plt.figure(figsize=(5,5))
# JAKE: labels passed to heatmap directly; set_xticklabels() crashed when seaborn drew fewer ticks
sns.heatmap(mean_conf_mat,square=True,annot=True,xticklabels=['Eyes Open','Eyes Closed'],yticklabels=['Eyes Open','Eyes Closed'])
ax = plt.gca()
ax.set_ylabel('True State')
ax.set_xlabel('Predicted State')
plt.savefig(f'images/conf_mat_mean.svg')
plt.show()
plt.close()

# %%



