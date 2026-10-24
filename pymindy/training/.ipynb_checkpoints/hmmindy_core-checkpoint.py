import torch
from torch import nn
from .abstract_core import AbstractMINDyCore

class HMMINDyCore(AbstractMINDyCore):
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
    #       Initializes the pyMINDyCoreClassic object
    #   func_F():
    #       Computes the next state (x)
    #   clamp_params():
    #       Clamps the params to the specified range
    #   get_params():
    #       Returns the parameters
    
    activation_function = torch.tanh

    @classmethod
    ###### init() ######
    # init() initializes the pyMINDyCoreClassic object.
    #
    # Parameters:
    #   H : np.array
    #       The observation matrix
    #   Q : np.array
    #       The process noise covariance matrix
    #   R : np.array
    #       The observation noise covariance matrix
    #   n_HMM : int
    #       The number of HMM states
    #   W_mask : np.array
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
    #
    # Returns:
    #   HMMINDyCore : AbstractMINDyCore
    #       The initialized HMMINDyCore object

    def init(cls, H, Q, R, n_HMM, W_mask=1, Gamma_rank=1, fix_Wmask=True, fixGamma=False,
             fix_Se=True, fix_Ve=True, fix_Si=True, fix_Vi=True, Gammaek1_fixed_value=1,
             Gammaik1_fixed_value=1, Gammaek2_fixed_value=1, Gammaik2_fixed_value=1,
             Se_fixed_value=1, Ve_fixed_value=0, Si_fixed_value=1, Vi_fixed_value=0,
             param_range=None, stay_prob=0.9999, device="cpu", rand_scale=0.5):
        if param_range is None:
            param_range = {"Se_min": 0.2, "Se_max": 4, "Si_min": 0.2, "Si_max": 4,
                           "De_max": 1, "De_min": 0, "Di_max": 1, "Di_min": 0,
                           "Gammaee_min": 0, "Gammaei_min": 0, "Gammaie_min": 0,
                           "Gammaii_min": 0}
        n_chan = H.shape[1] // 2 # number of excitatory populations

        # Convert the W mask to torch (if mask is specified)
        if (W_mask != 1).any():
            W_mask_ee = torch.from_numpy(W_mask[0:n_chan,0:n_chan])
            W_mask_ei = torch.from_numpy(W_mask[n_chan:,0:n_chan])
            W_mask_ee = W_mask_ee.type(torch.int32).to(device)
            W_mask_ei = W_mask_ei.type(torch.int32).to(device)
        else:
            W_mask_ee = 1
            W_mask_ei = 1

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
        
        Gamma = torch.empty((2*n_chan,2*n_chan,n_HMM))
        Gammaek1 = torch.empty((n_chan,Gamma_rank,n_HMM))
        Gammaek2 = torch.empty((Gamma_rank,n_chan,n_HMM))
        Gammaik1 = torch.empty((Gamma_rank,n_chan,n_HMM))
        Gammaik2 = torch.empty((Gamma_rank,n_chan,n_HMM))

        for i in range(n_HMM):
            p = torch.tensor([0.5, 0.5])
            is_normal = p.multinomial(num_samples=1)
            mu_i = torch.normal(mu0, torch.tensor([0.1]))
            if is_normal:
                sigma_i = abs(torch.normal(sigma0_n, torch.tensor([0.01])))
                Gammak = torch.normal(mu0 * torch.ones((2*n_chan, Gamma_rank)), sigma0_n * torch.ones(2*n_chan, Gamma_rank))
            else:
                sigma_i = torch.normal(sigma0_u, torch.tensor([0.1]))
                Gammak = torch.FloatTensor(size=(2*n_chan,Gamma_rank)).uniform_(mu_i-(sigma_i/2), mu_i + (sigma_i/2))
            Gamma_full = Gammak @ Gammak.T
            
            Gamma[:,:,i] = abs(W_mask) * Gamma_full
            Gammaek1[:,:,i] = torch.sqrt(torch.diag(Gamma[:n_chan,:n_chan,i]))[:,None]
            Gammaek2[:,:,i] = torch.sqrt(torch.diag(Gamma[:n_chan,:n_chan,i]))[None,:]
            Gammaik1[:,:,i] = torch.sqrt(torch.diag(Gamma[n_chan:,n_chan:,i]))[None,:]
            Gammaik2[:,:,i] = torch.sqrt(torch.diag(Gamma[n_chan:,n_chan:,i]))[None,:]
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

        return HMMINDyCore(n_chan, n_HMM, Wee, Wei, Wie, Wii, Gammaek1, Gammaek2,
                           Gammaik1, Gammaik2, Se, Si, Ve, Vi, De, Di, Ce, Ci, _H,
                           _Q, _R, fix_Wmask, fix_Se, fix_Si, fix_Ve, fix_Vi,
                           W_mask_ee, W_mask_ei, param_range, HMM, device)


