import numpy as np

# BEGIN NEW CODE - JAKE

###### channel_major_perm() ######
# channel_major_perm() returns the permutation of population-space indices that
# reorders a population-block-major state (all excitatory channels, then all
# inhibitory channels) into channel-major order (each channel's excitatory and
# inhibitory populations adjacent to one another).
#
# Parameters:
#   nChan: number of data channels
#   nPop: number of neural populations per channel
# Returns:
#   perm: permutation index array (length nChan*nPop) such that
#       X_channel_major = X_pop_block_major[perm] (for a vector), or
#       X_channel_major = X_pop_block_major[np.ix_(perm, perm)] (for a square matrix)

def channel_major_perm(nChan, nPop):
    return np.arange(nPop * nChan).reshape(nPop, nChan).T.flatten()

# END NEW CODE - JAKE

###### pop_layout() ######
# pop_layout() defines the channel-major population layout used throughout
# pymindy. Each channel owns nPop adjacent populations that alternate
# excitatory/inhibitory, e.g. for nPop=4: [xe1 xi1 xe2 xi2] per channel, so the
# full state is [ch0: E I E I, ch1: E I E I, ...]. Population p of channel c
# lives at index nPop*c + p.
#
# Parameters:
#   nChan: number of data channels
#   nPop: number of neural populations per channel (even, >= 2)
# Returns:
#   pop_type: +1 for excitatory, -1 for inhibitory (nChan*nPop,)
#   is_E: boolean mask of excitatory populations (nChan*nPop,)
#   E_idx: indices of excitatory populations (nChan*nPop/2,)
#   I_idx: indices of inhibitory populations (nChan*nPop/2,)
#   chan_of: channel each population belongs to (nChan*nPop,)

def pop_layout(nChan, nPop):
    if nPop < 2 or nPop % 2 != 0:
        raise ValueError(f"nPop must be an even integer >= 2, got nPop={nPop}")
    pop_type = np.tile([1, -1], nChan * nPop // 2)
    is_E = pop_type > 0
    E_idx = np.flatnonzero(is_E)
    I_idx = np.flatnonzero(~is_E)
    chan_of = np.repeat(np.arange(nChan), nPop)

    return pop_type, is_E, E_idx, I_idx, chan_of

###### build_chan_mask() ######
# build_chan_mask() generates a random symmetric channel-level connectivity
# mask: every channel connects to itself, plus the top pConn% of random
# off-diagonal channel pairs.
#
# Parameters:
#   nChan: number of data channels
#   pConn: percentage of off-diagonal channel pairs to connect (0-100)
#   rng: numpy random Generator
# Returns:
#   chan_mask: boolean channel connectivity mask (nChan x nChan)

def build_chan_mask(nChan, pConn, rng):
    # Generate random connections
    W0 = rng.random((nChan,nChan))
    W0 = W0 + W0.T

    # Take the top pConn (default: 25%) of off-diagonal connections + all diagonal connections
    tmp = np.percentile(OffDiag_inds(W0), 100-pConn)
    chan_mask = (OffDiag_mat(W0) >= tmp) | (np.eye(nChan) == 1)

    return chan_mask

###### build_W_mask() ######
# build_W_mask() expands a channel-level mask into the population-level W mask
# in channel-major order. Excitatory source populations (columns) project to
# every population of every channel allowed by chan_mask; inhibitory source
# populations only project locally, to the populations of their own channel.
#
# Parameters:
#   nChan: number of data channels
#   nPop: number of neural populations per channel (even, >= 2)
#   chan_mask: channel connectivity mask (nChan x nChan)
# Returns:
#   W_mask: boolean population connectivity mask (nChan*nPop x nChan*nPop)

def build_W_mask(nChan, nPop, chan_mask):
    pop_type = pop_layout(1, nPop)[0]
    # Within-channel pattern: every target population receives from the E (or I) sources
    E_cols = np.outer(np.ones(nPop), pop_type > 0)
    I_cols = np.outer(np.ones(nPop), pop_type < 0)
    chan_mask = (np.asarray(chan_mask) != 0) | (np.eye(nChan) == 1)
    W_mask = np.kron(chan_mask, E_cols) + np.kron(np.eye(nChan), I_cols)

    return W_mask > 0

###### create_synth_MINDy() ######
# create_synth_MINDy() creates a random MINDy model, then forward simulates it to
# generate synthetic data. All population-space quantities are built directly in
# the channel-major layout defined by pop_layout().
#
# Parameters:
#   nChan: number of data channels to simulate
#   nPop: number of neural populations per channel (even, >= 2; nPop/2 E and nPop/2 I)
#   pConn: probability of a long range excitatory connection existing (0-100) default: 25%
#   tRun: number of timesteps to simulate default: 20000
# Returns:
#   model: a dictionary with the following fields:
#       Params: dictionary containing the model parameters:
#           W: connectivity matrix (nChan*nPop x nChan*nPop)
#           D: decay (leak) for each neural population (nChan*nPop x 1)
#           C: baseline bias of each neural population (nChan*nPop x 1)
#           S: slope of neural activation sigmoid (nChan*nPop x 1)
#           V: offset of neural activation sigmoid (nChan*nPop x 1)
#       H: lead field matrix (nChan x nChan*nPop)
#       Q: process noise covariance matrix (nChan*nPop x nChan*nPop)
#       R: observation noise covariance matrix (nChan x nChan)
#       data: synthetic data generated by the model (nChan x tRun)
#
# Notes:
#   With nPairs = nPop/2 E/I pairs per channel, local and long-range input is
#   divided by nPairs so the total drive onto a population matches nPop=2.
def create_synth_MINDy(nChan,nPop,pConn=25,tRun=20000):
    rng = np.random.default_rng()
    pop_type, is_E, E_idx, I_idx, chan_of = pop_layout(nChan, nPop)
    nPairs = nPop // 2

    # Construct base mask
    chan_mask = build_chan_mask(nChan, pConn, rng)
    offdiag_mask = OffDiag_mat(chan_mask)

    # Generate long-range connectivity values: one low-rank (nChan x nChan) block
    # for every (target population, excitatory source population) combination
    nRank = np.ceil(nChan/4).astype(int)
    W_long = np.zeros((nPop*nChan, nPop*nChan))
    for p in range(nPop):
        for q in range(0, nPop, 2):
            tmp1 = rng.random((nChan,nRank))**3 + rng.random((nChan,nRank))/5
            tmp2 = rng.random((nRank,nChan))**3 + rng.random((nRank,nChan))/5
            # Normalize and multiply by channel mask
            Wblock = offdiag_mask * (tmp1 @ tmp2) / np.max(tmp1 @ tmp2) * 4/nChan
            Wblock = 4 * Wblock / 3
            Wblock = offdiag_mask * (Wblock + (16/nChan) * rng.random(nChan)**3)
            W_long[p::nPop, q::nPop] = Wblock / nPairs

    # Create base ratio of local/recurrent strengths (diagonal connections),
    # indexed [target type, source type] with 0 = E, 1 = I
    RecBase = np.array([[2, -5], [5, -1]])/5
    local_type = (pop_type[:nPop] < 0).astype(int)
    RecLocal = RecBase[np.ix_(local_type, local_type)] / nPairs
    # Randomly generate spatial pattern for recurrent connections
    recVec = np.abs(1 + 0.1 * rng.random((nChan,))**3)
    W_local = np.kron(np.diag(recVec), RecLocal)

    # Combine and restrict to the E-long-range / I-local structure
    Wfull = build_W_mask(nChan, nPop, chan_mask) * (W_long + W_local)

    # Generate other parameters
    C = 0
    V = 0
    De = np.maximum(0.5, np.minimum(0.9, 0.65 + rng.random()/50))
    Di = np.maximum(0.5, np.minimum(0.9, 0.8 + rng.random()/50))
    D = np.where(is_E, De, Di)
    S = np.where(is_E, 2.5, 1.0)

    # Generate lead field (H), measurement noise (R), and process noise (Q)
    # Only excitatory populations contribute to the measurements
    H = np.zeros((nChan, nPop*nChan))
    H[:, E_idx] = rng.normal(size=(nChan, E_idx.size))
    rtR = np.diag(0.2 + rng.random((nChan,))/10) + rng.random()/10
    rtQ = np.diag(0.5 * np.ones((nPop * nChan,)))

    # Compile into dictionary
    model = dict()
    model['Params'] = dict()
    model['Params']['W'] = Wfull
    model['Params']['D'] = D
    model['Params']['C'] = C
    model['Params']['S'] = S
    model['Params']['V'] = V
    model['H'] = H
    model['R'] = rtR @ rtR.T
    model['Q'] = rtQ @ rtQ.T

    # Forward simulate to get synthetic data
    y = simMINDy(model, tRun)
    model['data'] = y

    return model

###### create_synth_HMMINDy() ######
# create_synth_HMMINDy() creates a random HM-MINDy model, then forward simulates it to
# generate synthetic data.
#
# Parameters:
#   nChan: number of data channels to simulate
#   nPop: number of neural populations
#   nS: number of dynamic regime states
#   pConn: probability of a long range excitatory connection existing (0-100) default: 25%
#   pStay: probability of staying in any hidden markov state each timestep
#   tRun: number of timesteps to simulate default: 20000
#   Gamma_rank: rank of the Gamma (connection modulation) matrices default: 1
#   target_Gamma_corr: targeted correlation of generated Gamma matrices, if desired default: None
# Returns:
#   model: a dictionary with the following fields:
#       Params: dictionary containing the model parameters:
#           W: constant connectivity matrix (nChan*nPop x nChan*nPop)
#           Gamma: modulation connectivity matrix (nChan*nPop x nChan*nPop x nS)
#           D: decay (leak) for each neural population (nChan*nPop x 1)
#           C: baseline bias of each neural population (nChan*nPop x 1)
#           S: slope of neural activation sigmoid (nChan*nPop x 1)
#           V: offset of neural activation sigmoid (nChan*nPop x 1)
#       H: lead field matrix (nChan x nChan*nPop)
#       Q: process noise covariance matrix (nChan*nPop x nChan*nPop)
#       R: observation noise covariance matrix (nChan x nChan)
#       data: synthetic data generated by the model (nChan x tRun)
#       HMM: dictionary containing the following Hidden Markov Model parameters:
#           A: probability of any state transitioning to any other state (nS x nS)
#           pi: initial probability of any state (nS x 1)

def create_synth_HMMINDy(nChan, nPop, nS, pConn=25, pStay=0.9995, tRun=20000,
                         Gamma_rank=1, target_Gamma_corr=None):
        rng = np.random.default_rng()

        # Create base MINDy model
        model = create_synth_MINDy(nChan, nPop, pConn, tRun)
        Wmask = np.sign(model['Params']['W'])

        # Generate Gamma matrices
        mu0 = 1
        sigma0_u = 0.4
        sigma0_n = 0.05
        Gamma = np.empty((nPop*nChan,nPop*nChan,nS))

        if target_Gamma_corr is None:
            for i in range(nS):
                is_normal = rng.choice(range(2))
                mu_i = rng.normal(mu0, 0.1)
                if is_normal:
                    sigma_i = abs(rng.normal(sigma0_n, 0.01))
                    Gammak = rng.normal(mu_i, sigma_i, size=(nPop*nChan, Gamma_rank))
                else:
                    sigma_i = rng.normal(sigma0_u, 0.1)
                    Gammak = rng.uniform(mu_i - (sigma_i/2), mu_i + (sigma_i/2), size=(nPop*nChan, Gamma_rank))

                Gamma_full = Gammak @ Gammak.T
                Gamma[:,:,i] = np.abs(Wmask) * Gamma_full
        else:
            mu_i = rng.normal(mu0, 0.01, size=(nS,))
            sigma_i = abs(rng.normal(sigma0_n, 0.01, size=(nS,)))
            Sigma_i = np.diag(sigma_i) + target_Gamma_corr * (np.ones((nS,nS)) - np.eye(nS))
            Gamma_vecs = rng.multivariate_normal(mu_i, Sigma_i, size=(nPop*nChan,))
            for i in range(nS):
                Gammak = Gamma_vecs[:,i]
                Gamma[:,:,i] = np.abs(Wmask) * (Gammak[:,None] @ Gammak[None,:])

        # Save to model dictionary
        model['Params']['Gamma'] = Gamma

        # Generate HMM
        HMM = dict()
        HMM['A'] = pStay * np.eye(nS) + ((1 - pStay)/(nS - 1)) * OffDiag_mat(np.ones((nS,nS)))
        HMM['pi'] = (1/nS) * np.ones((nS,))

        # Simulate modulated data
        [y, mod_ind_hist] = simHMMINDy(model, HMM, tRun)

        model['data'] = y
        model['HMM'] = HMM
        model['mod_ind_hist'] = mod_ind_hist

        return model

###### OffDiag_inds() ######
# OffDiag_inds() takes in a square matrix and returns the values of that matrix
# which are not on the diagonal
#
# Parameters:
#   X: square matrix to obtain off-diagonal entries from
# Returns:
#   Out: vector containing the entries of X which are not on the diagonal

def OffDiag_inds(X):
    nX = X.shape
    if nX[0] != nX[1]:
        print('Error: X should be square')
    Out = X[np.eye(nX[0]) == 0]

    return Out

###### OffDiag_mat() ######
# OffDiag_mat() takes in a square matrix and returns that matrix with the diagonal
# entries set to 0.
#
# Parameters:
#   X: square matrix
# Returns:
#   Out: X with the diagonal entries set equal to 0

def OffDiag_mat(X):
    X = np.squeeze(X)
    nX = np.shape(X)
    Out = X * (np.ones(nX) - np.eye(nX[0]))

    return Out

###### simMINDy() ######
# simMINDy takes in a dictionary containing the parameters of a MINDy model, then
# forward simulates that model to generate a timeseries
#
# Parameters:
#   model: dictionary containing MINDy parameters with this structure:
#       Params: dictionary containing the model parameters:
#           W: connectivity matrix (nChan*nPop x nChan*nPop)
#           D: decay (leak) for each neural population (nChan*nPop x 1)
#           C: baseline bias of each neural population (nChan*nPop x 1)
#           S: slope of neural activation sigmoid (nChan*nPop x 1)
#           V: offset of neural activation sigmoid (nChan*nPop x 1)
#       H: lead field matrix (nChan x nChan*nPop)
#       Q: process noise covariance matrix (nChan*nPop x nChan*nPop)
#       R: observation noise covariance matrix (nChan x nChan)
#   tRun: number of timepoints to simulate model for
#   dropStart: number of timepoints to drop from the beginning of the simulation default=1
# Returns:
#   y: simulated data matrix (nChan x tRun)

def simMINDy(model, tRun, dropStart=1):
    W = model['Params']['W']
    D = model['Params']['D']
    C = model['Params']['C']
    S = model['Params']['S']
    V = model['Params']['V']
    H = model['H']
    R = model['R']
    Q = model['Q']
    [nY,nX] = H.shape

    rng = np.random.default_rng()
    x0 = rng.normal(size=(nX,))
    x = np.zeros((nX,tRun))
    x[:,0] = x0
    for i in range(1,tRun):
        x[:,i] = (W @ np.tanh(S * x[:,i-1] + V) + D * x[:,i-1] + C + Q @ rng.normal(size=(nX,)))
    x = x[:,dropStart:]
    y = H @ x + R @ rng.normal(size=(nY,tRun-dropStart))
    
    return y


###### simHMMINDy() ######
# simHMMINDy takes in a dictionary containing the parameters of a HM-MINDy model, then
# forward simulates that model to generate a timeseries

# Parameters:
#   model: a dictionary with the following fields:
#       Params: dictionary containing the model parameters:
#           W: constant connectivity matrix (nChan*nPop x nChan*nPop)
#           Gamma: modulation connectivity matrix (nChan*nPop x nChan*nPop x nS)
#           D: decay (leak) for each neural population (nChan*nPop x 1)
#           C: baseline bias of each neural population (nChan*nPop x 1)
#           S: slope of neural activation sigmoid (nChan*nPop x 1)
#           V: offset of neural activation sigmoid (nChan*nPop x 1)
#       H: lead field matrix (nChan x nChan*nPop)
#       Q: process noise covariance matrix (nChan*nPop x nChan*nPop)
#       R: observation noise covariance matrix (nChan x nChan)
#   HMM: dictionary containing the following Hidden Markov Model parameters:
#       A: probability of any state transitioning to any other state (nS x nS)
#       pi: initial probability of any state (nS x 1)
#   tRun: number of timepoints to simulate model for
#   dropStart: number of timepoints to drop from the beginning of the simulation default=1
# Returns:
#   y: simulated data matrix (nChan x tRun)
#   mod_ind_hist: timeseries of true latent dynamic states

def simHMMINDy(model, HMM, tRun, dropStart=1):
    W = model['Params']['W']
    Gamma = model['Params']['Gamma']
    D = model['Params']['D']
    C = model['Params']['C']
    S = model['Params']['S']
    V = model['Params']['V']
    H = model['H']
    R = model['R']
    Q = model['Q']
    [nY,nX] = H.shape
    nS = Gamma.shape[2]

    rng = np.random.default_rng()
    x0 = rng.normal(size=(nX,))
    x = np.zeros((nX,tRun))
    x[:,0] = x0
    mod_ind = rng.choice(nS, p=HMM['pi'])
    mod_ind_hist = np.zeros((tRun,))
    mod_ind_hist[0] = mod_ind

    for i in range(1,tRun):
        tmp_prob = np.zeros((nS,))
        tmp_prob[mod_ind] = 1
        hm_prob = HMM['A'] @ tmp_prob
        mod_ind = rng.choice(nS, p=hm_prob)
        mod_ind_hist[i] = mod_ind
        x[:,i] = (W @ np.tanh(S * x[:,i-1] + V) + D * x[:,i-1] + C + Q @ rng.normal(size=(nX,)))
    
    x = x[:,dropStart:]
    mod_ind_hist = mod_ind_hist[dropStart:]
    y = H @ x + R @ rng.normal(size=(nY,tRun-dropStart))
    
    return y, mod_ind_hist
