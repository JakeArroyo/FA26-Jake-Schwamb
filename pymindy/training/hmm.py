import torch
import numpy as np
import math

def initialize_mod_ind(data_len,hmmindy_core):
    rng = np.random.default_rng()
    nS = hmmindy_core.Gamma.shape[2]
    mod_ind_np = rng.integers(nS,size=(data_len,))
    mod_ind = torch.from_numpy(mod_ind_np).type(torch.int32).to(hmmindy_core.device)
    return mod_ind

def calculate_obs_prob(data,X_est,hmmindy_core):
    nS = hmmindy_core.Gamma.shape[2]

    mod_ind = torch.outer(torch.ones(X_est.shape[0],),torch.arange(0,nS)).type(torch.int32)
    mod_ind = mod_ind.T[:,:,None]
    X_est = X_est.repeat(nS,1,1)
    X_est_next = torch.func.vmap(torch.func.vmap(hmmindy_core.func_F))(X_est,mod_ind)
    mu_est = torch.func.vmap(torch.func.vmap(hmmindy_core.func_H))(X_est_next)
    #print(torch.max(mu_est))

    log_likelihood = -(1/2)*(torch.log(2*torch.Tensor([math.pi]).to(hmmindy_core.device)) + torch.log(torch.diag(hmmindy_core.R))[None,None,:] + (1/torch.diag(hmmindy_core.R))[None,None,:] * (data[None,:,:] - mu_est)**2)
    log_likelihood = torch.sum(log_likelihood,2)/10000
    o_prob = torch.exp(log_likelihood)
    if torch.isnan(o_prob).any():
        o_prob[torch.isnan(o_prob)] = 0
    
    return o_prob

def forward_back(o_prob,hmmindy_core,alpha_prev=None,beta_prev=None):
    # Extract parameters
    nS, T = o_prob.shape
    pi = hmmindy_core.HMM['pi'] #.type(torch.float).to(hmmindy_core.device)
    A = hmmindy_core.HMM['A'] #.type(torch.float).to(hmmindy_core.device)
    eta = 1
    
    # Initialize matrices
    alpha = torch.zeros((nS,T)).type(torch.float).to(hmmindy_core.device)
    beta = torch.zeros((nS,T)).type(torch.float).to(hmmindy_core.device)
    c = torch.ones((T,)).type(torch.float).to(hmmindy_core.device)
    #print(o_prob)

    # Fill out alpha matrix, scaling by c
    alpha[:,0] = pi * o_prob[:,0]
    c[0] = torch.sum(alpha[:,0])
    #print(c[0])
    if c[0] != 0:
        alpha[:,0] = alpha[:,0]/c[0]
    #print(alpha[:,0])
    
    for t in range(1,T):
        alpha[:,t] = o_prob[:,t] * (A @ alpha[:,t-1])
        c[t] = torch.sum(alpha[:,t])
        if c[t] != 0:
            alpha[:,t] = alpha[:,t]/c[t]

    if torch.isnan(alpha).any():
        print(o_prob)
        print(A)
    #print(alpha)
    # Fill out beta matrix, scaling by c
    beta[:,T-1] = torch.ones((nS,))
    beta[:,T-1] /= torch.sum(beta[:,T-1])
    for t in range(T-2,-1,-1):
        beta[:,t] = (A.T @ (beta[:,t+1] * o_prob[:,t+1]))
        if (beta[:,t] < 0).any():
            print(beta[:,t+1])
            print(o_prob[:,t+1])
            print(A)
        c[t] = torch.sum(beta[:,t])
        if c[t] != 0:
            beta[:,t] = beta[:,t]/c[t]

    if alpha_prev is not None:
        alpha = (1-eta)*alpha_prev + eta*alpha
    if beta_prev is not None:
        beta = (1-eta)*beta_prev + eta*beta
    #print(alpha)
    #print(beta)
    # Compute gamma
    #print(torch.median(alpha))
    #print(torch.argmax(alpha,0))
    #print(torch.median(beta))
    #print(torch.argmax(beta[:,:-1],0))
    gamma = alpha * beta
    if nS < 3:
        gamma = torch.maximum(gamma,torch.Tensor([0]).to(hmmindy_core.device))
    else:
        gamma = torch.maximum(gamma,torch.Tensor([torch.finfo().eps]).to(hmmindy_core.device))
    gamma_sum = torch.sum(gamma,0)
    gamma[:,gamma_sum !=0] = gamma[:,gamma_sum !=0] / gamma_sum[gamma_sum != 0]
    #gamma /= torch.sum(gamma,0)
    #while (torch.sum(gamma,0) > 1).any():
    #    gamma -= (torch.sum(gamma,0) - 1) / gamma.shape[0]
    if (torch.sum(gamma,0) < 1).any():
        gamma += (1 - torch.sum(gamma,0)) / gamma.shape[0]
    #    gamma = torch.maximum(gamma,torch.Tensor([0]).to(hmmindy_core.device))

    # Compute xi
    alpha_3d = (alpha[:,:T-1].T)[:,:,None]
    beta_3d = (beta[:,1:].T)[:,None,:]
    o_prob_3d = (o_prob[:,1:].T)[:,None,:]
    xi = alpha_3d * o_prob_3d * A[None,:,:] * beta_3d
    c = np.squeeze((alpha.T)[:,None,:] @ ((A @ (o_prob * beta)).T)[:,:,None])
    c[c==0] = 1
    xi = xi / c[1:,None,None]

    del c
    torch.cuda.empty_cache()

    return gamma, xi, alpha, beta

def baum_welch(gamma,xi,hmmindy_core):
    pi = hmmindy_core.HMM['pi'] #.type(torch.float).to(hmmindy_core.device)
    A = hmmindy_core.HMM['A'] #.type(torch.float).to(hmmindy_core.device)
    eta = 0.01

    newpi = (1-eta)*pi + eta*gamma[:,0]
    if torch.sum(newpi) > 1:
        newpi -= (torch.sum(newpi) - 1)/len(newpi)
    if torch.sum(newpi) < 1:
        newpi += (1 - torch.sum(newpi))/len(newpi)

    newA = (1 - eta)*A + eta*(torch.sum(xi,0).T / torch.sum(gamma[:,:-1],1))
    newA = torch.maximum(newA,torch.Tensor([0]).to(hmmindy_core.device))
    if torch.isnan(newA).any():
        print(A)
    newA /= torch.sum(newA,1)
    if torch.isnan(newA).any():
        print(A)
        print(torch.sum(newA,1))
    while (torch.sum(newA,1) > 1.001).any() or (torch.sum(newA,1) < 0.999).any():
        if (torch.sum(newA,1) > 1.001).any():
            #print(torch.sum(newA,1))
            #print(newA)
            #print(((torch.sum(newA,1) - 1) / newA.shape[1]).shape)
            newA -= ((torch.sum(newA,1) - 1) / newA.shape[1])[:,None]
            #print(newA)
        if (torch.sum(newA,1) < 0.999).any():
            #print(torch.sum(newA,1))
            #print(newA)
            newA += ((1 - torch.sum(newA,1)) / newA.shape[1])[:,None]
            #print(newA)
        newA = torch.maximum(newA,torch.Tensor([0]).to(hmmindy_core.device))

    HMM = dict()
    HMM['pi'] = newpi
    HMM['A'] = newA
    return HMM