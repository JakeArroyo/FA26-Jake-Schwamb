import numpy as np
import scipy
from sklearn.cluster import KMeans
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pymindy.dataset import PartitionData
from pymindy.training.backprop_kalman_filter import MINDyBPKF
from pymindy.training.backprop_kalman_filter import HMMINDyBPKF
from pymindy.training.hmm import initialize_mod_ind, calculate_obs_prob, forward_back, baum_welch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from .. import logger


def prepare_dataset(data, stack_interval, kf_step, rec_step):
    """ This function converts the data to a dataset that can be used directly in the training step.

    Parameters:
    -----------
    data : np.array or list[np.array]
        The data to be converted. If it is a list, all the subsets will be merged to a ConcatDataset.
    stack_interval : int
        The interval for stacking the data.
    kf_step : int
        The number of steps for the Kalman filter stage.
    rec_step : int
        The number of steps for the forward simulation stage.

    Returns:
    --------
    torch.utils.data.Dataset
        The dataset that can be used directly in the training step.
    """

    def convert_batch_from_numpy_to_dataset(batch, stack_interval, kf_step, rec_step):
        return PartitionData(batch, stack_interval, kf_step, rec_step)
    if isinstance(data, list):
        return ConcatDataset(
            list(map(lambda x: convert_batch_from_numpy_to_dataset(x, stack_interval, kf_step, rec_step), data)))
    else:
        return convert_batch_from_numpy_to_dataset(data, stack_interval, kf_step, rec_step)





def train_data_MSE(data, mindy_core, n_batch, stack_interval,
                   warm_up_step,
                   kf_step, rec_step,
                   optimizer, P0_np, device="cpu", clipping_grad_norm=False, clipping_grad_norm_value=1, info=None,
                   prepare_data_loader_func=None):
    """ This function trains the model using the mean squared error loss.

    Parameters:
    -----------
    data : np.array or list[np.array]
        The data to be trained.
    mindy_core : pyMINDyCoreMini
        The model to be trained.
    n_batch : int
        The number of batches.
    stack_interval : int
        The interval for stacking the data.
    warm_up_step : int
        The number of steps for the warm-up stage.
    kf_step : int
        The number of steps for the Kalman filter stage.
    rec_step : int
        The number of steps for the forward simulation stage.
    optimizer : torch.optim.Optimizer
        The optimizer for training the model.
    P0_np : np.array
        The initial covariance matrix.
    device : str
        The device for pyTorch to use.
    clipping_grad_norm : bool
        Whether to clip the gradient norm.
    clipping_grad_norm_value : float
        The value to clip the gradient norm.
    info : str
        The information to be printed.
    prepare_data_loader_func : function
        The function to prepare the data loader.

    Returns:
    --------
    pyMINDyCoreMini
        The trained model.
    dict
        The information of the training.
    """


    train_MSE_losses_record = []
    train_MSE_losses_record_mini_batch=[]
    BPKF = MINDyBPKF(mindy_core, warm_up_step, kf_step, rec_step)

    if prepare_data_loader_func is None:
        raise Exception("Must use dataloader")

    train_set = prepare_dataset(data, stack_interval, kf_step, rec_step)
    train_loader = prepare_data_loader_func(train_set, device)
    #logger.info(f"{info} Training set size: {len(train_set)}")
    P0 = torch.from_numpy(P0_np).type(torch.float).to(device)
    mini_batch_i=0
    for epoch_i in range(n_batch):
        for idx, stack in enumerate(train_loader):
            optimizer.zero_grad()
            stack = stack.to(device, non_blocking=True)
            X0_est = torch.stack([torch.zeros(mindy_core.x_size, device=device) for _ in range(stack.shape[0])])
            P0s = torch.stack([P0.clone() for _ in range(stack.shape[0])])
            _train_MSE_loss = BPKF.batch_cul_error_MSE(stack, X0_est, P0s)
            _train_MSE_loss = torch.mean(_train_MSE_loss)

            with torch.no_grad():
                train_MSE_loss = _train_MSE_loss
                
                if idx==0:
                    
                 #   print_text = f"Batch {epoch_i}(Minibatch {mini_batch_i}): MSE loss:{train_MSE_loss.item():.2f}"

                 #   if info is None:
                 #       logger.info(print_text)
                 #   else:
                 #       logger.info(f"{info} " + print_text)
                    
                    train_MSE_losses_record.append(train_MSE_loss.item())
                train_MSE_losses_record_mini_batch.append(train_MSE_loss.item())
            loss=train_MSE_loss/mindy_core.H.shape[0]
            loss.backward()
            if clipping_grad_norm:
                clip_grad_norm_(mindy_core.parameters(), clipping_grad_norm_value)

            optimizer.step()

            with torch.no_grad():
                mindy_core.clamp_params()
            mini_batch_i+=1

    info = {
        "train_MSE": np.array(train_MSE_losses_record),
        "train_MSE_losses_record_mini_batch":np.array(train_MSE_losses_record_mini_batch)
    }
    return mindy_core, info

def train_mod_data_MSE(data, mindy_core, n_batch, stack_interval,
                   warm_up_step,
                   kf_step, rec_step,
                   optimizer, P0_np, device="cpu", clipping_grad_norm=False, clipping_grad_norm_value=1, info=None,
                   prepare_data_loader_func=None):
    """ This function trains the model using the mean squared error loss.

    Parameters:
    -----------
    data : np.array or list[np.array]
        The data to be trained. Note that the last column of data is the Gamma (modulation matrix) index
    mindy_core : pyMINDyCoreMini
        The model to be trained.
    n_batch : int
        The number of batches.
    stack_interval : int
        The interval for stacking the data.
    warm_up_step : int
        The number of steps for the warm-up stage.
    kf_step : int
        The number of steps for the Kalman filter stage.
    rec_step : int
        The number of steps for the forward simulation stage.
    optimizer : torch.optim.Optimizer
        The optimizer for training the model.
    P0_np : np.array
        The initial covariance matrix.
    device : str
        The device for pyTorch to use.
    clipping_grad_norm : bool
        Whether to clip the gradient norm.
    clipping_grad_norm_value : float
        The value to clip the gradient norm.
    info : str
        The information to be printed.
    prepare_data_loader_func : function
        The function to prepare the data loader.

    Returns:
    --------
    pyMINDyCoreMini
        The trained model.
    dict
        The information of the training.
    """


    train_MSE_losses_record = []
    train_MSE_losses_record_mini_batch=[]
    BPKF = HMMINDyBPKF(mindy_core, warm_up_step, kf_step, rec_step)

    if prepare_data_loader_func is None:
        raise Exception("Must use dataloader")

    train_set = prepare_dataset(data, stack_interval, kf_step, rec_step)
    train_loader = prepare_data_loader_func(train_set, device)
    logger.info(f"{info} Training set size: {len(train_set)}")
    P0 = torch.from_numpy(P0_np).type(torch.float).to(device)
    mini_batch_i=0
    for epoch_i in range(n_batch):
        for idx, stack in enumerate(train_loader):
            optimizer.zero_grad()
            Gamma_ind_stack = stack[:,:,-1]
            Gamma_ind_stack = Gamma_ind_stack[:,:,None].type(torch.int32)
            stack = stack[:,:,:-1].to(device, non_blocking=True)
            X0_est = torch.stack([torch.zeros(mindy_core.x_size, device=device) for _ in range(stack.shape[0])])
            P0s = torch.stack([P0.clone() for _ in range(stack.shape[0])])
            _train_MSE_loss = BPKF.batch_cul_error_MSE(stack, X0_est, P0s, Gamma_ind_stack)
            _train_MSE_loss = torch.mean(_train_MSE_loss)

            with torch.no_grad():
                train_MSE_loss = _train_MSE_loss
                if idx==0:
                    print_text = f"Batch {epoch_i}(Minibatch {mini_batch_i}): MSE loss:{train_MSE_loss.item():.2f}"

                    if info is None:
                        logger.info(print_text)
                    else:
                        logger.info(f"{info} " + print_text)
                    train_MSE_losses_record.append(train_MSE_loss.item())
                train_MSE_losses_record_mini_batch.append(train_MSE_loss.item())
            loss=train_MSE_loss/mindy_core.H.shape[0]
            loss.backward()
            if clipping_grad_norm:
                clip_grad_norm_(mindy_core.parameters(), clipping_grad_norm_value)

            optimizer.step()

            with torch.no_grad():
                mindy_core.clamp_params()
            mini_batch_i+=1

    info = {
        "train_MSE": np.array(train_MSE_losses_record),
        "train_MSE_losses_record_mini_batch":np.array(train_MSE_losses_record_mini_batch)
    }
    return mindy_core, info

def train_hmmindy_data_MSE(data, mindy_core, n_batch, stack_interval,
                   warm_up_step,
                   kf_step, rec_step,
                   optimizer, P0_np, Gamma_ind_hist=None, device="cpu", clipping_grad_norm=False, clipping_grad_norm_value=1, info=None,
                   prepare_data_loader_func=None):
    """ This function trains the model using the mean squared error loss.

    Parameters:
    -----------
    data : np.array or list[np.array]
        The data to be trained. Note that the last column of data is the Gamma (modulation matrix) index
    mindy_core : pyMINDyCoreMini
        The model to be trained.
    n_batch : int
        The number of batches.
    stack_interval : int
        The interval for stacking the data.
    warm_up_step : int
        The number of steps for the warm-up stage.
    kf_step : int
        The number of steps for the Kalman filter stage.
    rec_step : int
        The number of steps for the forward simulation stage.
    optimizer : torch.optim.Optimizer
        The optimizer for training the model.
    P0_np : np.array
        The initial covariance matrix.
    device : str
        The device for pyTorch to use.
    clipping_grad_norm : bool
        Whether to clip the gradient norm.
    clipping_grad_norm_value : float
        The value to clip the gradient norm.
    info : str
        The information to be printed.
    prepare_data_loader_func : function
        The function to prepare the data loader.

    Returns:
    --------
    pyMINDyCoreMini
        The trained model.
    dict
        The information of the training.
    """

    rng = np.random.default_rng()
    Kmeans_cent = None
    train_MSE_losses_record = []
    train_MSE_losses_record_mini_batch=[]
    BPKF = HMMINDyBPKF(mindy_core, warm_up_step, kf_step, rec_step)

    if prepare_data_loader_func is None:
        raise Exception("Must use dataloader")

    data_tensor = torch.from_numpy(data[:,:-1]).type(torch.float).to(device)
    #data = np.hstack((scipy.stats.zscore(data[:,:-1]),data[:,-1][:,None]))
    train_set = prepare_dataset(data, stack_interval, kf_step, rec_step)
    train_loader = prepare_data_loader_func(train_set, device)
    logger.info(f"{info} Training set size: {len(train_set)}")
    P0 = torch.from_numpy(P0_np).type(torch.float).to(device)
    X0_est = torch.zeros((mindy_core.x_size,), device=device)
    if Gamma_ind_hist is None:
        # Run KF for all states, returning a posteriori residuals to be used as observation probability
        with torch.no_grad():
            obs_prob = torch.zeros((mindy_core._nS,data.shape[0])).type(torch.float).to(device)
            for i in range(mindy_core._nS):
                _, _, post_resid = BPKF.kf_full_data(data_tensor,X0_est,P0,(i * torch.ones((data.shape[0],))).type(torch.int))
                obs_prob[i,:] = 1/post_resid
                print('iteration ', i, ' complete')
    
            # Calculate gamma & xi based on observation probability
            gamma, xi, _, _ = forward_back(obs_prob,mindy_core)
            Gamma_ind_hist = torch.argmax(gamma,0)
            std_gamma = torch.std(gamma,1)
            entropy = -torch.sum(gamma * torch.log(gamma),0)
            mean_entropy = torch.mean(entropy)
            alpha = None
            beta = None

        # Plot
        plt.plot(obs_prob.cpu().T)
        plt.show()

        plt.plot(Gamma_ind_hist.cpu())
        plt.plot(gamma.cpu().T)
        plt.plot(entropy.cpu())
        plt.show()
    else:
        Gamma_ind_hist = torch.from_numpy(Gamma_ind_hist).type(torch.int)
        gamma = 0
        xi = 0
        mean_entropy = 0
        std_gamma = torch.Tensor([0., 0.])
        #Gamma_ind_hist = Gamma_ind_hist#[:,None,None]
        
    mini_batch_i=0
    for epoch_i in range(n_batch):
        for idx, stack in enumerate(train_loader):
            optimizer.zero_grad()
            data_ind_stack = stack[:,:,-1].type(torch.int32)
            data_ind_stack = data_ind_stack[:,:,None].type(torch.int32)
            Gamma_ind_stack = Gamma_ind_hist[data_ind_stack]
            stack = stack[:,:,:-1].to(device, non_blocking=True)
            X0_stack = torch.stack([torch.zeros(mindy_core.x_size, device=device) for _ in range(stack.shape[0])])
            P0s = torch.stack([P0.clone() for _ in range(stack.shape[0])])
            _train_MSE_loss = BPKF.batch_cul_error_MSE(stack, X0_stack, P0s, Gamma_ind_stack)
            _train_MSE_loss = torch.mean(_train_MSE_loss)

            with torch.no_grad():
                train_MSE_loss = _train_MSE_loss
                if idx==0:
                    print_text = f"Batch {epoch_i}(Minibatch {mini_batch_i}): MSE loss:{train_MSE_loss.item():.2f}"

                    if info is None:
                        logger.info(print_text)
                    else:
                        logger.info(f"{info} " + print_text)
                    train_MSE_losses_record.append(train_MSE_loss.item())
                train_MSE_losses_record_mini_batch.append(train_MSE_loss.item())

                #print(X_est.shape)
                #print(stack.shape)
                #print(torch.squeeze(X_est[data_ind_stack,:]).shape)
                #obs_prob = calculate_obs_prob(stack, torch.squeeze(X_est[data_ind_stack,:]), mindy_core)
                #gamma, xi = forward_back(obs_prob,mindy_core)
                #Gamma_ind_hist[data_ind_stack] = torch.argmax(gamma,0)
                #new_HMM = baum_welch(gamma,xi,mindy_core)
                #mindy_core.set_HMM(new_HMM)
            loss=train_MSE_loss/mindy_core.H.shape[0]
            loss.backward()
            if clipping_grad_norm:
                clip_grad_norm_(mindy_core.parameters(), clipping_grad_norm_value)

            optimizer.step()

            with torch.no_grad():
                mindy_core.clamp_params()

            mini_batch_i+=1
        
        if (mean_entropy>0.15) or (std_gamma<0.25).all():
            '''
            fig, axes = plt.subplots(1,mindy_core._nS+1,figsize=(15,5))
            sns.heatmap(mindy_core.W.detach().cpu(),ax=axes[0],cmap='turbo',square=True)
            axes[0].set_title(f'W, iteration {epoch_i}')
            sns.heatmap(mindy_core.Gamma.detach().cpu()[:,:,0],ax=axes[1],cmap='turbo',square=True)
            axes[1].set_title(f'Gamma1, iteration {epoch_i}')
            sns.heatmap(mindy_core.Gamma.detach().cpu()[:,:,1],ax=axes[2],cmap='turbo',square=True)
            axes[2].set_title(f'Gamma2, iteration {epoch_i}')
            plt.show()
            '''
        #if (std_gamma<0.1).any():
            #if epoch_i ==0:
            eta = 1
            #else:
            #    eta = 0.8 #max(1/epoch_i,0.1)
            with torch.no_grad():
                
                # Get updated X_est
                #X_est, P_est = BPKF.warm_up_kalman_smooth(data_tensor,X_est,P_est,mod_ind_est) # Kalman smooth data to get X_est
                #X_est = X_est.type(torch.float).to(device)
                #P_est = P_est.type(torch.float).to(device)
                
                # Update Gamma index estimate
                # calculate observation probability given each Gamma based on the data, X_est, and the model
                # print(torch.cuda.memory_allocated())
                #print(obs_prob)
                #obs_prob = calculate_obs_prob(data_tensor,X_est,mindy_core)
                #print(obs_prob)
                # print(torch.cuda.memory_allocated())

                #plt.plot(obs_prob.T.cpu())
                #plt.show()

                obs_prob = torch.zeros((mindy_core._nS,data.shape[0])).type(torch.float).to(device)
                for i in range(mindy_core._nS):
                    _, _, post_resid = BPKF.kf_full_data(data_tensor,X0_est,P0,(i * torch.ones((data.shape[0],))).type(torch.int))
                    obs_prob[i,:] = 1/post_resid
                plt.plot(obs_prob.cpu().T)
                plt.show()
                
                # Forward back algorithm (gamma is probability of each state, xi is estimated transition probability)
                gamma_new, xi_new, alpha, beta = forward_back(obs_prob,mindy_core)
                gamma = (1 - eta) * gamma + eta * gamma_new
                entropy = -torch.sum(gamma * torch.log(gamma),0)
                mean_entropy = torch.mean(entropy)
                std_gamma = torch.std(gamma,1)

                for i in range(mindy_core._nS):
                    plt.plot(alpha[i,:].cpu())
                    plt.plot(beta[i,:].cpu())
                    ax = plt.gca()
                    ax.set_ylim([0, 1])
                    plt.title(f'Forward & Backward Likelihood Estimates (state {i+1})')
                    plt.show()
                
                '''                
                # Add in noise to alpha & beta encouraging higher entropy solutions
                (max_alpha, max_alpha_inds) = torch.max(alpha,0)
                max_alpha_inds = torch.vstack((max_alpha_inds.cpu(),torch.arange(max_alpha_inds.shape[0])))
                max_alpha_inds = list(max_alpha_inds)
                (min_alpha, min_alpha_inds) = torch.min(alpha,0)
                min_alpha_inds = torch.vstack((min_alpha_inds.cpu(),torch.arange(min_alpha_inds.shape[0])))
                min_alpha_inds = list(min_alpha_inds)
                (max_beta, max_beta_inds) = torch.max(beta,0)
                max_beta_inds = torch.vstack((max_beta_inds.cpu(),torch.arange(max_beta_inds.shape[0])))
                max_beta_inds = list(max_beta_inds)
                (min_beta, min_beta_inds) = torch.min(beta,0)
                min_beta_inds = torch.vstack((min_beta_inds.cpu(),torch.arange(min_beta_inds.shape[0])))
                min_beta_inds = list(min_beta_inds)
                alpha_max = alpha[max_alpha_inds]
                alpha_min = alpha[min_alpha_inds]
                beta_max = beta[max_beta_inds]
                beta_min = beta[min_beta_inds]
                alpha_max[entropy < mean_entropy] += abs(torch.normal(mean=torch.zeros((alpha_max[entropy < mean_entropy].shape)), std = 0.001 * torch.ones((alpha_max[entropy < mean_entropy].shape)))).to(device)
                alpha_min[entropy < mean_entropy] -= abs(torch.normal(mean=torch.zeros((alpha_min[entropy < mean_entropy].shape)), std = 0.001 * torch.ones((alpha_min[entropy < mean_entropy].shape)))).to(device)
                beta_max[entropy < mean_entropy] += abs(torch.normal(mean=torch.zeros((beta_max[entropy < mean_entropy].shape)), std = 0.001 * torch.ones((beta_max[entropy < mean_entropy].shape)))).to(device)
                beta_min[entropy < mean_entropy] -= abs(torch.normal(mean=torch.zeros((beta_min[entropy < mean_entropy].shape)), std = 0.001 * torch.ones((beta_min[entropy < mean_entropy].shape)))).to(device)
                alpha_max[entropy > mean_entropy] -= abs(torch.normal(mean=torch.zeros((alpha_max[entropy > mean_entropy].shape)), std = 0.001 * torch.ones((alpha_max[entropy > mean_entropy].shape)))).to(device)
                alpha_min[entropy > mean_entropy] += abs(torch.normal(mean=torch.zeros((alpha_min[entropy > mean_entropy].shape)), std = 0.001 * torch.ones((alpha_min[entropy > mean_entropy].shape)))).to(device)
                beta_max[entropy > mean_entropy] -= abs(torch.normal(mean=torch.zeros((beta_max[entropy > mean_entropy].shape)), std = 0.001 * torch.ones((beta_max[entropy > mean_entropy].shape)))).to(device)
                beta_min[entropy > mean_entropy] += abs(torch.normal(mean=torch.zeros((beta_min[entropy > mean_entropy].shape)), std = 0.001 * torch.ones((beta_min[entropy > mean_entropy].shape)))).to(device)
                alpha[max_alpha_inds] = alpha_max
                alpha[min_alpha_inds] = alpha_min
                beta[max_beta_inds] = beta_max
                beta[min_beta_inds] = beta_min
                
                (max_gamma, max_inds) = torch.max(gamma,0)
                max_inds = torch.vstack((max_inds.cpu(),torch.arange(max_inds.shape[0])))
                max_inds = list(max_inds)
                (min_gamma, min_inds) = torch.min(gamma,0)
                min_inds = torch.vstack((min_inds.cpu(),torch.arange(min_inds.shape[0])))
                min_inds = list(min_inds)
                print(mean_entropy)
                #print(max_inds[entropy < mean_entropy])
                gamma_max = gamma[max_inds]
                gamma_min = gamma[min_inds]
                print(gamma[max_inds])
                print(gamma_max[entropy < mean_entropy].shape)
                std_gamma = torch.std(gamma,1)
                print(torch.std(gamma,1))
                print(abs(torch.normal(mean=torch.zeros((len(gamma[:,entropy < mean_entropy]),)), std = 0.01 * torch.ones((len(gamma[:,entropy < mean_entropy]),)))).to(device).shape)
                if mean_entropy >0.25:
                    # make timepoints under the mean entropy further apart
                    gamma_max[entropy < mean_entropy] += abs(torch.normal(mean=torch.zeros((gamma_max[entropy < mean_entropy].shape)), std = 0.005 * torch.ones((gamma_max[entropy < mean_entropy].shape)))).to(device)
                    gamma_min[entropy < mean_entropy] -= abs(torch.normal(mean=torch.zeros((gamma_min[entropy < mean_entropy].shape)), std = 0.005 * torch.ones((gamma_min[entropy < mean_entropy].shape)))).to(device)
                    # make timepoints over the mean entropy closer together
                    gamma_max[entropy > mean_entropy] -= abs(torch.normal(mean=torch.zeros((gamma_max[entropy > mean_entropy].shape)), std = 0.005 * torch.ones((gamma_max[entropy > mean_entropy].shape)))).to(device)
                    gamma_min[entropy > mean_entropy] += abs(torch.normal(mean=torch.zeros((gamma_min[entropy > mean_entropy].shape)), std = 0.005 * torch.ones((gamma_min[entropy > mean_entropy].shape)))).to(device)
                else:
                    # make all timepoints further apart
                    gamma_max += abs(torch.normal(mean=torch.zeros((gamma_max.shape)), std = 0.005 * torch.ones((gamma_max.shape)))).to(device)
                    gamma_min -= abs(torch.normal(mean=torch.zeros((gamma_min.shape)), std = 0.005 * torch.ones((gamma_min.shape)))).to(device)

                gamma[max_inds] = gamma_max
                gamma[min_inds] = gamma_min
                gamma = torch.maximum(gamma,torch.Tensor([0]).to(device))
                gamma_sum = torch.sum(gamma,0)
                gamma[:,gamma_sum !=0] = gamma[:,gamma_sum !=0] / gamma_sum[gamma_sum != 0]
                #if (entropy < 0.68).any():
                #    # make gamma closer together
                #    gamma[max_inds] -= abs(torch.normal(mean=torch.zeros((len(max_inds[0]),)), std = 0.01 * torch.ones((len(max_inds[0]),)))).to(device)
                #    gamma[min_inds] += abs(torch.normal(mean=torch.zeros((len(max_inds[0]),)), std = 0.01 * torch.ones((len(max_inds[0]),)))).to(device)
                #else:
                    # make gamma further apart
                #    gamma[max_inds] += abs(torch.normal(mean=torch.zeros((len(max_inds[0]),)), std = 0.01 * torch.ones((len(max_inds[0]),)))).to(device)
                #    gamma[min_inds] -= abs(torch.normal(mean=torch.zeros((len(max_inds[0]),)), std = 0.01 * torch.ones((len(max_inds[0]),)))).to(device)
                '''
                xi = (1 - eta) * xi + eta * xi_new
                Gamma_ind_hist = torch.argmax(gamma,0)
                if (std_gamma < 0.1).all():
                    smooth_resid = torch.zeros(mindy_core._nS,len(Gamma_ind_hist))
                    for j in range(len(Gamma_ind_hist)):
                        smooth_resid[:,j] = torch.mean(obs_prob[:,np.maximum(0,j-200):np.minimum(len(Gamma_ind_hist),j+200)],1)

                    if Kmeans_cent is not None:
                        print('using pre-defined centroids')
                        kmeans = KMeans(n_clusters=mindy_core._nS,init=Kmeans_cent).fit(smooth_resid.T)
                    else:
                        kmeans = KMeans(n_clusters=mindy_core._nS,n_init=5).fit(smooth_resid.T)
                    print(kmeans.labels_)
                    Kmeans_cent = kmeans.cluster_centers_
                    plt.plot(smooth_resid.T)
                    plt.plot(kmeans.labels_)
                    plt.show()
                    Gamma_ind_hist = torch.from_numpy(kmeans.labels_)
                    
                    #print(torch.quantile(smooth_resid,torch.tensor([0.25,0.5,0.75,0.9])))
                    #Gamma_ind_hist[smooth_resid > torch.quantile(smooth_resid,torch.tensor([0.75]))] = 0
                    #Gamma_ind_hist[smooth_resid < torch.quantile(smooth_resid,torch.tensor([0.75]))] = 1
                
                # Baum Welch algorithm to update HMM parameters
                new_HMM = baum_welch(gamma,xi,mindy_core)
                mindy_core.set_HMM(new_HMM)
                print(mindy_core.HMM)

            
            # Plot Gamma index estimate
            plt.plot(gamma.T.cpu())
            plt.plot(entropy.cpu())
            plt.show()
            plt.plot(Gamma_ind_hist.cpu())
            plt.show()
            print(mean_entropy)
            print(torch.median(entropy))
            print(std_gamma)

    #plt.plot(alpha.T.cpu())
    #plt.plot(beta.T.cpu())
    #plt.show()
            

                #max_pchange = torch.max(abs(gamma[:,:-1]-gamma[:,1:]))
                #print(max_pchange)

    info = {
        "train_MSE": np.array(train_MSE_losses_record),
        "train_MSE_losses_record_mini_batch":np.array(train_MSE_losses_record_mini_batch)
    }
    return mindy_core, info, Gamma_ind_hist