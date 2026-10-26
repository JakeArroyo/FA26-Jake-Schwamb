# BEGIN JAKE CODE
# Legacy (nPop=2, separate Wee/Wei/Wie/Wii parameters) HMMINDyCore, frozen as of
# commit 5f906f9. Used only to unpickle fit results saved before hmmindy_core.py
# switched to the general-nPop (full W_raw/G1/G2) parameterization. Old pickles
# still reference pymindy.training.hmmindy_core.HMMINDyCore, so load them with
# load_legacy_pickle() below rather than pickle.load().
import pickle
# END JAKE CODE
import torch
from torch import nn
from torch.func import jacrev
from .abstract_core import AbstractMINDyCore

class HMMINDyCoreLegacy(AbstractMINDyCore):
    # The core pyMINDy model for neural dynamics, with HMM modulation.
    # Attributes:
    #   n_chan : int
    #       The number of data channels
    #   n_HMM : int
    #       The number of HMM states
    #   Wee : torch.tensor
    #       The excitatory-to-excitatory connectivity weight matrix
    #   Wei : torch.tensor
    #       The excitatory-to-inhibitory connectivity weight matrix
    #   Wie : torch.tensor
    #       The inhibitory-to-excitatory connectivity weight matrix
    #   Wii : torch.tensor
    #       The inhibitory-to-inhibitory connectivity weight matrix
    #   Gammaee : torch.tensor
    #       The excitatory-to-excitatory modulation connectivity weight matrix
    #   Gammaei : torch.tensor
    #       The excitatory-to-inhibitory modulation connectivity weight matrix
    #   Gammaie : torch.tensor
    #       The inhibitory-to-excitatory modulation connectivity weight matrix
    #   Gammaii : torch.tensor
    #       The inhibitory-to-inhibitory modulation connectivity weight matrix
    #   Se : torch.tensor
    #       The scaling factor for the excitatory firing function
    #   Si : torch.tensor
    #       The scaling factor for the inhibitory firing function
    #   Ve : torch.tensor
    #       The bias term for the excitatory firing function
    #   Vi : torch.tensor
    #       The bias term for the inhibitory firing function
    #   De : torch.tensor
    #       The decay rate for the excitatory neurons
    #   Di : torch.tensor
    #       The decay rate for the inhibitory neurons
    #   Ce : torch.tensor
    #       The bias term for the excitatory neurons
    #   Ci : torch.tensor
    #       The bias term for the inhibitory neurons
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
    #   W_mask : torch.tensor
    #       The mask specifying which sort of connection each element of W is; if 1, no mask on excitatory parameters 
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
    #       The number of neural populations per channel (E + I, so must be 2)
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
        # nPop is passed in from the caller (e.g. the notebook's hyperparameters)
        # The model is parameterized as E/I pairs (Wee/Wei/Wie/Wii), so only nPop=2 is supported
        if nPop != 2:
            raise ValueError(f"HMMINDyCore only supports nPop=2 (E/I populations), got nPop={nPop}")
        n_chan = H.shape[1] // nPop # number of channels (H's population axis is channel-major)
        if H.shape[1] != nPop * n_chan or H.shape[0] != n_chan:
            raise ValueError(f"H must have shape (nChan, nPop*nChan); got {H.shape} with nPop={nPop}")

        # Convert the W mask to torch (if mask is specified)
        # W_mask is channel-major: [chan0_E, chan0_I, chan1_E, chan1_I, ...], so the
        # E and I populations of a given block are the even/odd indices, not the
        # first/second half.
        # (use strided indexing so that Gamma pulls EE/II diagonal sub-blocks instead of half slicing)
        if (W_mask != 1).any():
            W_mask_ee = torch.from_numpy(W_mask[0::nPop,0::nPop])
            W_mask_ei = torch.from_numpy(W_mask[1::nPop,0::nPop])
            W_mask_ee = W_mask_ee.type(torch.int32).to(device)
            W_mask_ei = W_mask_ei.type(torch.int32).to(device)
        else:
            W_mask_ee = 1
            W_mask_ei = 1

        # END JAKE CODE

        # Generate random W matrix, section by section
        Wee = W_mask_ee * torch.rand(n_chan, n_chan, dtype=torch.float, device=device) * rand_scale
        Wei = W_mask_ei * torch.rand(n_chan, n_chan, dtype=torch.float, device=device) * rand_scale
        Wie = torch.rand(n_chan, dtype=torch.float, device=device) * rand_scale
        Wii = torch.rand(n_chan, dtype=torch.float, device=device) * rand_scale

        # Generate excitatory and inhibitory components of parameters D, C, S, & V
        Se = Se_fixed_value * torch.ones(n_chan, dtype=torch.float, device=device)
        Si = Si_fixed_value * torch.ones(n_chan, dtype=torch.float, device=device)
        Ve = Ve_fixed_value * torch.ones(n_chan, dtype=torch.float, device=device)
        Vi = Vi_fixed_value * torch.ones(n_chan, dtype=torch.float, device=device)
        De = torch.rand(n_chan, dtype=torch.float, device=device) * rand_scale
        Di = torch.rand(n_chan, dtype=torch.float, device=device) * rand_scale
        Ce = torch.rand(n_chan, dtype=torch.float, device=device) * rand_scale
        Ci = torch.rand(n_chan, dtype=torch.float, device=device) * rand_scale
        
        # Convert Q, R, and H to torch
        _Q = torch.from_numpy(Q).type(torch.float).to(device)
        _R = torch.from_numpy(R).type(torch.float).to(device)
        _H = torch.from_numpy(H).type(torch.float).to(device)

        # Generate Gamma matrices
        mu0 = torch.tensor([0.95])
        sigma0_u = torch.tensor([0.45])
        sigma0_n = torch.tensor([0.05])
        
        Gamma = torch.empty((nPop*n_chan,nPop*n_chan,n_HMM))
        Gammaek1 = torch.empty((n_chan,Gamma_rank,n_HMM))
        Gammaek2 = torch.empty((Gamma_rank,n_chan,n_HMM))
        Gammaik1 = torch.empty((Gamma_rank,n_chan,n_HMM))
        Gammaik2 = torch.empty((Gamma_rank,n_chan,n_HMM))

        # Generate one Gamma for each HMM state
        for i in range(n_HMM):
            p = torch.tensor([0.5, 0.5])
            is_normal = p.multinomial(num_samples=1)
            mu_i = torch.normal(mu0, 0.1).item()
            if is_normal:
                sigma_i = abs(torch.normal(sigma0_n, torch.tensor([0.01])))
                Gammak = torch.normal(mu0 * torch.ones((nPop*n_chan, Gamma_rank)), sigma0_n * torch.ones(nPop*n_chan, Gamma_rank))
            else:
                sigma_i = abs(torch.normal(sigma0_u).item())
                Gammak = torch.FloatTensor(size=(nPop*n_chan,Gamma_rank)).uniform_(mu_i-(sigma_i/2), mu_i + (sigma_i/2))
            Gamma_full = Gammak @ Gammak.T
            
            Gamma[:,:,i] = torch.from_numpy(abs(W_mask)).type(torch.float) * Gamma_full

            # BEGIN JAKE CODE
            # Gamma is channel-major like W_mask, so pull the EE/II sub-blocks via
            # strided indexing rather than the first/second half.
            Gammaek1[:,:,i] = torch.sqrt(torch.diag(Gamma[0::nPop,0::nPop,i]))[:,None]
            Gammaek2[:,:,i] = torch.sqrt(torch.diag(Gamma[0::nPop,0::nPop,i]))[None,:]
            Gammaik1[:,:,i] = torch.sqrt(torch.diag(Gamma[1::nPop,1::nPop,i]))[None,:]
            Gammaik2[:,:,i] = torch.sqrt(torch.diag(Gamma[1::nPop,1::nPop,i]))[None,:]
            # END JAKE CODE

        Gammaek1 = Gammaek1.type(torch.float).to(device)
        Gammaek2 = Gammaek2.type(torch.float).to(device)
        Gammaik1 = Gammaik1.type(torch.float).to(device)
        Gammaik2 = Gammaik2.type(torch.float).to(device)

        # Generate Hidden Markov Model
        HMM = dict()
        HMM['A'] = (stay_prob * torch.eye(n_HMM) + ((1-stay_prob)/(n_HMM-1)) *
                    (torch.ones((n_HMM,n_HMM)) - torch.eye(n_HMM))).T
        HMM['A'] = HMM['A'].type(torch.float).to(device)
        HMM['pi'] = ((1/n_HMM) * torch.ones((n_HMM,))).type(torch.float).to(device)

        return HMMINDyCoreLegacy(n_chan, n_HMM, Wee, Wei, Wie, Wii, Gammaek1, Gammaik1, 
                           Gammaek2, Gammaik2, Se, Si, Ve, Vi, De, Di, Ce, Ci, _H,
                           _Q, _R, fix_Wmask, fix_Se, fix_Si, fix_Ve, fix_Vi,
                           W_mask_ee, W_mask_ei, param_range, HMM, device)

    def __init__(self, n_chan, n_HMM, Wee, Wei, Wie, Wii, Gammaek1, Gammaik1, Gammaek2,
                 Gammaik2, Se, Si, Ve, Vi, De, Di, Ce, Ci, H, Q, R, fix_Wmask, fix_Se,
                 fix_Si, fix_Ve, fix_Vi, W_mask_ee, W_mask_ei, param_range, HMM, device):
        super(HMMINDyCoreLegacy, self).__init__()

        self._device = device
        self._Wee = nn.Parameter(Wee)
        self._Wei = nn.Parameter(Wei)
        self._Wie = nn.Parameter(Wie)
        self._Wii = nn.Parameter(Wii)
        self._W_mask_ee = W_mask_ee
        self._W_mask_ei = W_mask_ei
        self._Gammaek1 = nn.Parameter(Gammaek1)
        self._Gammaik1 = nn.Parameter(Gammaik1)
        self._Gammaek2 = nn.Parameter(Gammaek2)
        self._Gammaik2 = nn.Parameter(Gammaik2)
        self._De = nn.Parameter(De)
        self._Di = nn.Parameter(Di)
        self._Ce = nn.Parameter(Ce)
        self._Ci = nn.Parameter(Ci)
        self._Gamma_ind = 0
        self._H = H
        self._Q = Q
        self._R = R
    
        self.fix_Wmask = fix_Wmask
        self.fix_Se = fix_Se
        self.fix_Si = fix_Si
        self.fix_Ve = fix_Ve
        self.fix_Vi = fix_Vi

        if fix_Se:
            self._Se = Se
        else:
            self._Se = nn.Parameter(Se)
        if fix_Si:
            self._Si = Si
        else:
            self._Si = nn.Parameter(Si)

        if fix_Ve:
            self._Ve = Ve
        else:
            self._Ve = nn.Parameter(Ve)
        if fix_Vi:
            self._Vi = Vi
        else:
            self._Vi = nn.Parameter(Vi)

        self.param_range = param_range
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
        if not self.fix_Se:
            self._Se.data.clamp_(self.param_range['Se_min'], self.param_range['Se_max'])
        if not self.fix_Si:
            self._Si.data.clamp_(self.param_range['Si_min'], self.param_range['Si_max'])
        if not self.fix_Wmask:
            off_diag = torch.ones((self.n_pop, self.n_pop), dtype=torch.float, device=self.device) - torch.eye(self.n_pop)
            tmp = torch.quantile(off_diag * (self.W), 0.75)
            Wmask = torch.eye(self.n_pop) + off_diag * self.W>=tmp
            # BEGIN JAKE CODE
            nPop = self.n_pop_per_chan
            self._W_mask_ee = Wmask[0::nPop, 0::nPop]
            self._W_mask_ei = Wmask[1::nPop, 0::nPop]
            # END JAKE CODE
        self._Wee.data = self._W_mask_ee * self._Wee.data
        self._Wei.data = self._W_mask_ei * self._Wei.data

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
        # Get current state. x is channel-major ([chan0_E, chan0_I, chan1_E, chan1_I, ...]),
        # so the E/I populations are the even/odd indices, not the first/second half.
        nPop = self.n_pop_per_chan
        xe = x[0::nPop]
        xi = x[1::nPop]
        # END JAKE CODE

        # Apply activation function, tanh(Sx + V)
        activation_e = self.activation_function(self.Se * xe + self.Ve)
        activation_i = self.activation_function(self.Si * xi + self.Vi)

        # Get Gamma matrix (the properties build all modulation states at once)
        Gammaee = self.Gammaee
        Gammaei = self.Gammaei
        Gammaie = self.Gammaie
        Gammaii = self.Gammaii

        # Calculate next x
        xe_next = (self.Wee * torch.squeeze(Gammaee[:,:,mod_ind])) @ activation_e - (self.Wie * torch.squeeze(Gammaie[:,mod_ind])) * activation_i + self.De * xe + self.Ce
        xi_next = (self.Wei * torch.squeeze(Gammaei[:,:,mod_ind])) @ activation_e - (self.Wii * torch.squeeze(Gammaii[:,mod_ind])) * activation_i + self.Di * xi + self.Ci

        # BEGIN JAKE CODE
        # Interleave back into channel-major order ([chan0_E, chan0_I, chan1_E, chan1_I, ...])
        # to match x's layout, rather than concatenating into blocks.
        x_next = torch.stack([xe_next, xi_next], dim=-1).flatten()
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

    @property
    def Wee(self):
        return torch.abs(self._Wee)

    @property
    def Wei(self):
        return torch.abs(self._Wei)

    @property
    def Wie(self):
        return torch.abs(self._Wie)

    @property
    def Wii(self):
        return torch.abs(self._Wii)

    @property
    def Gammaek1(self):
        return torch.abs(self._Gammaek1)

    @property
    def Gammaik1(self):
        return torch.abs(self._Gammaik1)

    @property
    def Gammaek2(self):
        return torch.abs(self._Gammaek2)

    @property
    def Gammaik2(self):
        return torch.abs(self._Gammaik2)

    @property
    def Gammaee(self):
        return torch.abs(self._W_mask_ee[:,:,None] * torch.einsum('ijk,jlk->ilk', \
                self.Gammaek1, self.Gammaek2))

    @property
    def Gammaei(self):
        return torch.abs(self._W_mask_ei[:,:,None] * torch.einsum('jik,jlk->ilk', \
                self.Gammaik1, self.Gammaek2))

    @property
    def Gammaie(self):
        return torch.abs(torch.squeeze(self.Gammaek1) * torch.squeeze(self.Gammaik2))

    @property
    def Gammaii(self):
        return torch.abs(torch.squeeze(self.Gammaik1) * torch.squeeze(self.Gammaik2))

    @property
    def Se(self):
        return self._Se

    @property
    def Si(self):
        return self._Si

    @property
    def De(self):
        return torch.nn.Sigmoid()(self._De) * (self.param_range['De_max'] - \
                self.param_range['De_min']) + self.param_range['De_min']

    @property
    def Di(self):
        return torch.nn.Sigmoid()(self._Di) * (self.param_range['Di_max'] - \
                self.param_range['Di_min']) + self.param_range['Di_min']

    @property
    def Ve(self):
        return self._Ve

    @property
    def Vi(self):
        return self._Vi

    @property
    def Ce(self):
        return self._Ce

    @property
    def Ci(self):
        return self._Ci

    
    # BEGIN JAKE CODE
    # Reconstructed full state-space parameters, assembled in the same
    # channel-major interleaved order as x ([chan0_E, chan0_I, chan1_E, chan1_I, ...]).
    # D/C/S/V use the same stack+flatten interleave as func_F()'s x_next; W/Gamma
    # extend that to both matrix axes (interleave rows, then interleave columns).
    # Useful for comparing the fitted model against W_truth/Gamma_truth.
    @property
    def W(self):
        n = self.n_chan
        nPop = self.n_pop_per_chan
        # Diagonal-only local connections (Wie: E<-I, Wii: I<-I) carry the inhibitory
        # sign explicitly here, matching the -Wie/-Wii terms used in func_F().
        col_E = torch.stack([self.Wee, self.Wei], dim=1).reshape(nPop * n, n)
        col_I = torch.stack([-torch.diag(self.Wie), -torch.diag(self.Wii)], dim=1).reshape(nPop * n, n)
        return torch.stack([col_E, col_I], dim=2).reshape(nPop * n, nPop * n)

    @property
    def Gamma(self):
        n = self.n_chan
        nPop = self.n_pop_per_chan
        nS = self._nS
        Gamma = torch.zeros(nPop * n, nPop * n, nS, dtype=self.Gammaee.dtype, device=self.device)
        for i in range(nS):
            col_E = torch.stack([self.Gammaee[:, :, i], self.Gammaei[:, :, i]], dim=1).reshape(nPop * n, n)
            col_I = torch.stack([torch.diag(self.Gammaie[:, i]), torch.diag(self.Gammaii[:, i])], dim=1).reshape(nPop * n, n)
            Gamma[:, :, i] = torch.stack([col_E, col_I], dim=2).reshape(nPop * n, nPop * n)
        return Gamma

    @property
    def D(self):
        return torch.stack([self.De, self.Di], dim=-1).flatten()

    @property
    def C(self):
        return torch.stack([self.Ce, self.Ci], dim=-1).flatten()

    @property
    def S(self):
        return torch.stack([self.Se, self.Si], dim=-1).flatten()

    @property
    def V(self):
        return torch.stack([self.Ve, self.Vi], dim=-1).flatten()
    # END JAKE CODE

# BEGIN JAKE CODE
###### load_legacy_pickle() ######
# load_legacy_pickle() loads a pickle saved with the old HMMINDyCore, mapping
# that class onto HMMINDyCoreLegacy so its properties (W, Gamma, Wee, ...) work.
#
# Parameters:
#   file : file object
#       An open binary file (e.g. open('HM-MINDy_alpha_fit_....pkl', 'rb'))
#
# Returns:
#   The unpickled object (e.g. the fit_models_mod list)
class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pymindy.training.hmmindy_core' and name == 'HMMINDyCore':
            return HMMINDyCoreLegacy
        return super().find_class(module, name)

def load_legacy_pickle(file):
    return _LegacyUnpickler(file).load()
# END JAKE CODE
