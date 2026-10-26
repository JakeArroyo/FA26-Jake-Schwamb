import numpy as np
import torch
from torch import nn
from torch.func import jacrev
from .abstract_core import AbstractMINDyCore
from ..dataset.create_synth_data import pop_layout, build_W_mask

class HMMINDyCore(AbstractMINDyCore):
    # The core pyMINDy model for neural dynamics, with HMM modulation.
    # Attributes:
    #   n_chan : int
    #       The number of data channels
    #   n_HMM : int
    #       The number of HMM states
    #   W : torch.tensor
    #       The full connectivity matrix (N x N, N = nChan*nPop), channel-major.
    #       Built as W_mask * |W_raw| * pop_type, so E columns are >= 0 and I columns <= 0
    #   Gamma : torch.tensor
    #       The modulation matrices (N x N x n_HMM), W_mask * |G1 @ G2| for each state
    #   D, C, S, V : torch.tensor
    #       Per-population decay, bias, activation slope and activation offset (N,)
    #   Wee, Wei, Wie, Wii, Gammaee, Gammaei, Gammaie, Gammaii : torch.tensor
    #       Read-only E/I slices of W and Gamma (see the properties for shapes)
    #   Se, Si, Ve, Vi, De, Di, Ce, Ci : torch.tensor
    #       Read-only E/I slices of S, V, D and C
    #   H : torch.tensor
    #       The observation (lead-field) matrix
    #   Q : torch.tensor
    #       The process noise covariance matrix
    #   R : torch.tensor
    #       The observation noise covariance matrix
    #   fix_Se : bool
    #       Whether Se is fixed or a trainable parameter
    #   fix_Si : bool
    #       Whether Si is fixed or a trainable parameter
    #   fix_Ve : bool
    #       Whether Ve is fixed or a trainable parameter
    #   fix_Vi : bool
    #       Whether Vi is fixed or a trainable parameter
    #   param_range : dict
    #       The range of acceptable values for each parameter
    #   device : str
    #       The device for pyTorch to use (typically either "cpu" or "cuda")
    #   Gamma_ind : int
    #       The index indicating the current Gamma modulation state, used in fitting
    #
    # Methods:
    #   init():
    #       Initializes the HM-MINDy object
    #   clamp_params():
    #       Clamps all parameters to the specified ranges
    #   func_F():
    #       Computes the next state (x)
    #   clamp_params():
    #       Clamps the params to the specified range
    #   get_params():
    #       Returns the parameters
    
    activation_function = torch.tanh

    @classmethod
    ###### init() ######
    # init() initializes the HM-MINDy object.
    #
    # Parameters:
    #   H : torch.tensor
    #       The observation matrix
    #   Q : torch.tensor
    #       The process noise covariance matrix
    #   R : torch.tensor
    #       The observation noise covariance matrix
    #   n_HMM : int
    #       The number of HMM states
    #   W_mask : np.ndarray
    #       Full (N x N) channel-major mask of allowed connections (nonzero = allowed),
    #       e.g. np.sign(W_truth). If 1, every channel pair is connected (E sources
    #       project everywhere, I sources stay local to their channel)
    #   Gamma_rank : int
    #       rank of the Gamma matrices
    #   fix_Wmask : bool
    #       Whether the connectivity mask is a fixed or trainable parameter
    #   fix_Se : bool
    #       Whether Se is a fixed or trainable parameter
    #   fix_Ve : bool
    #       Whether Ve is a fixed or trainable parameter
    #   fix_Si : bool
    #       Whether Si is a fixed or trainable parameter
    #   fix_Vi : bool
    #       Whether Vi is a fixed or trainable parameter
    #   Se_fixed_value : float
    #       The value of Se if it is fixed
    #   Ve_fixed_value : float
    #       The value of Ve if it is fixed
    #   Si_fixed_value : float
    #       The value of Si if it is fixed
    #   Vi_fixed_value : float
    #       The value of Vi if it is fixed
    #   param_range : dict
    #       The range of parameters. If None, the default range is used.
    #   stay_prob : float
    #       The probability of staying in a given hidden markov state each timestep
    #   device : str
    #       The device for pyTorch to use (typically either 'cpu' or 'cuda')
    #   rand_scale : float
    #       The scale for the random initialization of the parameters
    #   nPop : int
    #       The number of neural populations per channel (even, >= 2; see pop_layout())
    #
    # Returns:
    #   HMMINDyCore : AbstractMINDyCore
    #       The initialized HMMINDyCore object

    def init(cls, H, Q, R, n_HMM, W_mask=1, Gamma_rank=1, fix_Wmask=True, fixGamma=False,
             fix_Se=True, fix_Ve=True, fix_Si=True, fix_Vi=True, Gammaek1_fixed_value=1,
             Gammaik1_fixed_value=1, Gammaek2_fixed_value=1, Gammaik2_fixed_value=1,
             Se_fixed_value=1, Ve_fixed_value=0, Si_fixed_value=1, Vi_fixed_value=0,
             param_range=None, stay_prob=0.9999, device="cpu", rand_scale=0.5, nPop=2):
        if param_range is None:
            param_range = {"Se_min": 0.2, "Se_max": 4, "Si_min": 0.2, "Si_max": 4,
                           "De_max": 1, "De_min": 0, "Di_max": 1, "Di_min": 0,
                           "Gammaee_min": 0, "Gammaei_min": 0, "Gammaie_min": 0,
                           "Gammaii_min": 0}

        # BEGIN JAKE CODE
        # nPop is passed in from the caller (e.g. the notebook's hyperparameters).
        # pop_layout() validates it and defines the channel-major E/I layout.
        n_chan = H.shape[1] // nPop # number of channels (H's population axis is channel-major)
        if H.shape[1] != nPop * n_chan or H.shape[0] != n_chan:
            raise ValueError(f"H must have shape (nChan, nPop*nChan); got {H.shape} with nPop={nPop}")
        pop_type, is_E, E_idx, I_idx, chan_of = pop_layout(n_chan, nPop)
        N = nPop * n_chan

        # Full N x N mask of allowed connections (W_mask=1 means all channel pairs connected)
        if np.isscalar(W_mask):
            W_mask = build_W_mask(n_chan, nPop, np.ones((n_chan, n_chan)))
        if W_mask.shape != (N, N):
            raise ValueError(f"W_mask must have shape ({N}, {N}); got {W_mask.shape}")
        W_mask = torch.from_numpy(np.asarray(W_mask) != 0).type(torch.float).to(device)

        # Generate random W magnitudes (sign comes from pop_type in the W property)
        W_raw = W_mask * torch.rand(N, N, dtype=torch.float, device=device) * rand_scale

        # Generate per-population parameters D, C, S, & V
        S = torch.where(torch.from_numpy(is_E), Se_fixed_value, Si_fixed_value).type(torch.float).to(device)
        V = torch.where(torch.from_numpy(is_E), Ve_fixed_value, Vi_fixed_value).type(torch.float).to(device)
        D = torch.rand(N, dtype=torch.float, device=device) * rand_scale
        C = torch.rand(N, dtype=torch.float, device=device) * rand_scale
        # END JAKE CODE
        
        # Convert Q, R, and H to torch
        _Q = torch.from_numpy(Q).type(torch.float).to(device)
        _R = torch.from_numpy(R).type(torch.float).to(device)
        _H = torch.from_numpy(H).type(torch.float).to(device)

        # Generate Gamma matrices
        mu0 = torch.tensor([0.95])
        sigma0_u = torch.tensor([0.45])
        sigma0_n = torch.tensor([0.05])

        # BEGIN JAKE CODE
        # Gamma[:,:,k] = W_mask * |G1[:,:,k] @ G2[:,:,k]| (G1: target factors, G2: source factors)
        G1 = torch.empty((N,Gamma_rank,n_HMM))
        G2 = torch.empty((Gamma_rank,N,n_HMM))
        # END JAKE CODE

        # Generate one Gamma for each HMM state
        for i in range(n_HMM):
            p = torch.tensor([0.5, 0.5])
            is_normal = p.multinomial(num_samples=1)
            mu_i = torch.normal(mu0, 0.1).item()
            if is_normal:
                sigma_i = abs(torch.normal(sigma0_n, torch.tensor([0.01])))
                Gammak = torch.normal(mu0 * torch.ones((N, Gamma_rank)), sigma0_n * torch.ones(N, Gamma_rank))
            else:
                sigma_i = abs(torch.normal(sigma0_u).item())
                Gammak = torch.FloatTensor(size=(N,Gamma_rank)).uniform_(mu_i-(sigma_i/2), mu_i + (sigma_i/2))

            # BEGIN JAKE CODE
            G1[:,:,i] = Gammak
            G2[:,:,i] = Gammak.T
            # END JAKE CODE

        G1 = G1.type(torch.float).to(device)
        G2 = G2.type(torch.float).to(device)

        # Generate Hidden Markov Model
        HMM = dict()
        HMM['A'] = (stay_prob * torch.eye(n_HMM) + ((1-stay_prob)/(n_HMM-1)) *
                    (torch.ones((n_HMM,n_HMM)) - torch.eye(n_HMM))).T
        HMM['A'] = HMM['A'].type(torch.float).to(device)
        HMM['pi'] = ((1/n_HMM) * torch.ones((n_HMM,))).type(torch.float).to(device)

        return HMMINDyCore(n_chan, n_HMM, nPop, W_raw, G1, G2, S, V, D, C, _H, _Q, _R,
                           fix_Wmask, fix_Se, fix_Si, fix_Ve, fix_Vi, W_mask,
                           param_range, HMM, device)

    def __init__(self, n_chan, n_HMM, nPop, W_raw, G1, G2, S, V, D, C, H, Q, R, fix_Wmask,
                 fix_Se, fix_Si, fix_Ve, fix_Vi, W_mask, param_range, HMM, device):
        super(HMMINDyCore, self).__init__()

        # BEGIN JAKE CODE
        self._device = device
        pop_type, is_E, E_idx, I_idx, chan_of = pop_layout(n_chan, nPop)
        self.register_buffer('_pop_type', torch.from_numpy(pop_type).type(torch.float).to(device))
        self.register_buffer('_is_E', torch.from_numpy(is_E).to(device))
        self.register_buffer('_W_mask', W_mask)
        self._E_idx = torch.from_numpy(E_idx).to(device)
        self._I_idx = torch.from_numpy(I_idx).to(device)

        self._W_raw = nn.Parameter(W_raw)
        self._G1 = nn.Parameter(G1)
        self._G2 = nn.Parameter(G2)
        self._D = nn.Parameter(D)
        self._C = nn.Parameter(C)
        self._Gamma_ind = 0
        self._H = H
        self._Q = Q
        self._R = R
    
        self.fix_Wmask = fix_Wmask
        self.fix_Se = fix_Se
        self.fix_Si = fix_Si
        self.fix_Ve = fix_Ve
        self.fix_Vi = fix_Vi

        # S and V are single per-population vectors; when only one of the E/I
        # halves is fixed, the fixed half is read from the initial values instead
        # of the parameter (so it receives no gradient)
        self.register_buffer('_S_fix', torch.where(self._is_E, fix_Se, fix_Si))
        self.register_buffer('_V_fix', torch.where(self._is_E, fix_Ve, fix_Vi))
        self.register_buffer('_S0', S.clone())
        self.register_buffer('_V0', V.clone())
        if fix_Se and fix_Si:
            self._S = S
        else:
            self._S = nn.Parameter(S)
        if fix_Ve and fix_Vi:
            self._V = V
        else:
            self._V = nn.Parameter(V)

        self.param_range = param_range
        # Per-population ranges for D (always) and S (only used when S is trainable)
        self.register_buffer('_D_min', torch.where(self._is_E, param_range['De_min'], param_range['Di_min']).type(torch.float))
        self.register_buffer('_D_max', torch.where(self._is_E, param_range['De_max'], param_range['Di_max']).type(torch.float))
        if not (fix_Se and fix_Si):
            self.register_buffer('_S_min', torch.where(self._is_E, param_range.get('Se_min', -torch.inf), param_range.get('Si_min', -torch.inf)).type(torch.float))
            self.register_buffer('_S_max', torch.where(self._is_E, param_range.get('Se_max', torch.inf), param_range.get('Si_max', torch.inf)).type(torch.float))
        # END JAKE CODE

        with torch.no_grad():
            self.clamp_params()

        self._HMM = HMM
        self._nS = n_HMM

    # Methods
    ###### clamp_params() ######
    # clamp_params() clamps all parameters to the specified ranges. Also applies the W mask.
    #
    # Parameters:
    #    None
    #
    # Returns:
    #   None
    def clamp_params(self):
        # BEGIN JAKE CODE
        if not (self.fix_Se and self.fix_Si):
            self._S.data = torch.clamp(self._S.data, self._S_min, self._S_max)
        if not self.fix_Wmask:
            # Re-pick the channel mask from the current cross-channel E->any strengths:
            # keep the top 25% of off-diagonal channel pairs (symmetrized, as in
            # build_chan_mask()), plus every channel's own local block
            n_chan = self.n_chan
            nPop = self.n_pop_per_chan
            W_E = torch.abs(self._W_raw.data) * self._is_E[None,:]
            chan_strength = W_E.reshape(n_chan, nPop, n_chan, nPop).sum(dim=(1,3))
            chan_strength = chan_strength + chan_strength.T
            off_diag = ~torch.eye(n_chan, dtype=torch.bool, device=self.device)
            tmp = torch.quantile(chan_strength[off_diag], 0.75)
            chan_mask = ((chan_strength >= tmp) & off_diag).cpu().numpy()
            W_mask = build_W_mask(n_chan, nPop, chan_mask)
            self._W_mask = torch.from_numpy(W_mask).type(torch.float).to(self.device)
        self._W_raw.data = self._W_mask * self._W_raw.data
        # END JAKE CODE

    ###### func_F() ######
    # func_F() computes the next state (x)
    #
    # Parameters:
    #   x : torch.tensor
    #       The current state
    #   mod_ind : int
    #       The index of the current modulation state/Gamma
    #
    # Returns:
    #   x_next : torch.tensor
    #       The next state
    def func_F(self, x, mod_ind):
        # BEGIN JAKE CODE
        # x_next = (W * Gamma_k) @ tanh(S*x + V) + D*x + C, all in channel-major order
        # (index_select rather than [:,:,mod_ind] so this also works under vmap)
        N = self.n_pop
        mod_ind = torch.as_tensor(mod_ind, device=self.device).reshape(1)
        G1k = torch.index_select(self.G1, 2, mod_ind).reshape(N, -1)
        G2k = torch.index_select(self.G2, 2, mod_ind).reshape(-1, N)
        Gammak = self._W_mask * torch.abs(G1k @ G2k)

        activation = self.activation_function(self.S * x + self.V)
        x_next = (self.W * Gammak) @ activation + self.D * x + self.C
        # END JAKE CODE

        return x_next

    ###### func_H() ######
    # func_H() computes the EEG measurement (y) from current state (x)
    #
    # Parameters:
    #   x : torch.tensor
    #       The current state
    #
    # Returns:
    #   y : torch.tensor
    #       The current EEG measurement
    def func_H(self, x):
        return self.H @ x

    ###### func_dFdx() ######
    # func_dFdx() overrides the default gradient function to pass in the correct
    # Gamma index as well
    #
    # Parameters:
    #   x : torch.tensor
    #       The current state
    #   Gamma_ind : int
    #       The index of the current modulation state
    #
    # Returns:
    #   dFdx : the jacobian of func_F
    def func_dFdx(self, x, Gamma_ind):
        return jacrev(self.func_F)(x, Gamma_ind)

    ###### func_dHdx() ######
    # func_dHdx() overrides the default gradient function since y is linear in x
    #
    # Parameters:
    #    x : torch.tensor
    #       The current state (not necessary except for override purposes)
    #
    # Returns:
    #   H : torch.tensor
    #       The lead-field matrix
    def func_dHdx(self, x):
        return self.H

    ###### get_params() ######
    # get_params() returns the model parameters (W, Gamma, D, C, S, V, Q, H, R)
    # as a dictionary
    #
    # Parameters:
    #   None
    #
    # Returns:
    #   params : dict
    #       A dictionary of the model parameters
    def get_params(self):
        params = {'W': self.W, 'Gamma': self.Gamma, 'D': self.D, 'C': self.C,
                  'S': self.S, 'V': self.V, 'Q': self.Q, 'H': self.H, 'R': self.R}
        return params

    # Properties
    @property
    def n_chan(self):
        return self.H.shape[0]

    @property
    def n_pop(self):
        return self.H.shape[1]

    # Number of populations per channel (nPop), derived from H's shape (nChan x nPop*nChan)
    @property
    def n_pop_per_chan(self):
        return self.H.shape[1] // self.H.shape[0]

    # STUB: referenced by training.utils (X0_est sizing) but never defined.
    # The state vector x spans every population of every channel.
    @property
    def x_size(self):
        return self.H.shape[1]

    # STUB: referenced by func_F's old Gamma block; one entry per channel,
    # matching mindy_core's n_pair (the number of excitatory populations).
    @property
    def n_pair(self):
        return self.n_chan

    # STUB: training.utils calls set_HMM()/reads .HMM after each Baum-Welch
    # update; hmm.forward_back and hmm.baum_welch read .HMM['pi'] / .HMM['A'].
    @property
    def HMM(self):
        return self._HMM

    def set_HMM(self, HMM):
        self._HMM = HMM

    @property
    def device(self):
        return self._device

    @property
    def H(self):
        return self._H

    @property
    def Q(self):
        return self._Q

    @property
    def R(self):
        return self._R

    # BEGIN JAKE CODE
    # Full state-space parameters, in the same channel-major order as x
    # (see pop_layout()). Useful for comparing against W_truth/Gamma_truth.
    @property
    def W(self):
        return self._W_mask * torch.abs(self._W_raw) * self._pop_type[None,:]

    @property
    def G1(self):
        return torch.abs(self._G1)

    @property
    def G2(self):
        return torch.abs(self._G2)

    @property
    def Gamma(self):
        return self._W_mask[:,:,None] * torch.einsum('irk,rjk->ijk', self.G1, self.G2)

    @property
    def D(self):
        return torch.nn.Sigmoid()(self._D) * (self._D_max - self._D_min) + self._D_min

    @property
    def C(self):
        return self._C

    @property
    def S(self):
        return torch.where(self._S_fix, self._S0, self._S)

    @property
    def V(self):
        return torch.where(self._V_fix, self._V0, self._V)

    # Read-only E/I slices, kept so analysis code written for nPop=2 keeps working.
    # Rows are targets and columns are sources, so Wei is E->I (I rows, E columns).
    # Wee/Wei (and Gammaee/Gammaei) are (nChan*nPop/2 x nChan*nPop/2[ x n_HMM]).
    # Wie/Wii (and Gammaie/Gammaii) are local-only, so they are returned as one
    # (nPop/2 x nPop/2) block per channel: (nChan x nPop/2 x nPop/2[ x n_HMM]).
    # For nPop=2 those blocks are 1x1 and get squeezed to the old (nChan[ x n_HMM])
    # vectors. Wie/Wii are magnitudes (the W property carries the minus sign).
    def _local_blocks(self, M):
        # M: (nChan*nPairs x nChan*nPairs[ x n_HMM]), block-diagonal by channel
        n_chan = self.n_chan
        nPairs = self.n_pop_per_chan // 2
        blocks = torch.diagonal(M.reshape(n_chan, nPairs, n_chan, nPairs, *M.shape[2:]), dim1=0, dim2=2)
        blocks = torch.movedim(blocks, -1, 0)
        if nPairs == 1:
            blocks = blocks.reshape(n_chan, *M.shape[2:])
        return blocks

    @property
    def Wee(self):
        return self.W[self._E_idx][:, self._E_idx]

    @property
    def Wei(self):
        return self.W[self._I_idx][:, self._E_idx]

    @property
    def Wie(self):
        return self._local_blocks(-self.W[self._E_idx][:, self._I_idx])

    @property
    def Wii(self):
        return self._local_blocks(-self.W[self._I_idx][:, self._I_idx])

    @property
    def Gammaee(self):
        return self.Gamma[self._E_idx][:, self._E_idx]

    @property
    def Gammaei(self):
        return self.Gamma[self._I_idx][:, self._E_idx]

    @property
    def Gammaie(self):
        return self._local_blocks(self.Gamma[self._E_idx][:, self._I_idx])

    @property
    def Gammaii(self):
        return self._local_blocks(self.Gamma[self._I_idx][:, self._I_idx])

    # Old rank factors: target (k1) and source (k2) factors of the E/I populations.
    # Gammaik1 keeps its old (rank x nI x n_HMM) orientation.
    @property
    def Gammaek1(self):
        return self.G1[self._E_idx]

    @property
    def Gammaik1(self):
        return self.G1[self._I_idx].permute(1, 0, 2)

    @property
    def Gammaek2(self):
        return self.G2[:, self._E_idx]

    @property
    def Gammaik2(self):
        return self.G2[:, self._I_idx]

    @property
    def Se(self):
        return self.S[self._E_idx]

    @property
    def Si(self):
        return self.S[self._I_idx]

    @property
    def De(self):
        return self.D[self._E_idx]

    @property
    def Di(self):
        return self.D[self._I_idx]

    @property
    def Ve(self):
        return self.V[self._E_idx]

    @property
    def Vi(self):
        return self.V[self._I_idx]

    @property
    def Ce(self):
        return self.C[self._E_idx]

    @property
    def Ci(self):
        return self.C[self._I_idx]
    # END JAKE CODE
