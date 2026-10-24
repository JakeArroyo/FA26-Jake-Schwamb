import torch
from torch import nn
from .abstract_core import AbstractMINDyCore

class pyMINDyCoreClassic(AbstractMINDyCore):
    # The core pyMINDy model for neural dynamics
    # Attributes:
    #   n_pair : int
    #       The number of data channels
    #   Wee : torch.tensor
    #       The excitatory-to-excitatory connectivity weight matrix
    #   Wei : torch.tensor
    #       The excitatory-to-inhibitory connectivity weight matrix
    #   Wie : torch.tensor
    #       The inhibitory-to-excitatory connectivity weight matrix
    #   Wii : torch.tensor
    #       The inhibitory-to-inhibitory connectivity weight matrix
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
    #
    # Methods:
    #   init(cls, H, Q, R, fix_Se=True, fix_Ve=True, fix_Si=True, fix_Vi=True,
    #         Se_fixed_value=1, Ve_fixed_value=0, Si_fixed_value=1, Vi_fixed_value=0,
    #         param_range=None, device="cpu", rand_scale=0.5):
    #       Initializes the pyMINDyCoreClassic object
    #   func_F(self, x):
    #       Computes the next state (x)
    #   clamp_params(self):
    #       Clamps the params to the specified range
    #   get_params(self):
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
    #   W_mask : np.array
    #       The mask specifying which sort of connection each element of W is 
    #   fix_Se : bool
    #       Whether Se is a fixed or trainable parameter
    #   fix_Ve : bool
    #       Whether Ve is a fixed or trainable parameter
    #   fix_Si : bool
    #       Whether Si is a fixed or trainable parameter
    #   fix Vi : bool
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
    #   device : str
    #       The device for pyTorch to use (typically either 'cpu' or 'cuda')
    #   rand_scale : float
    #       The scale for the random initialization of the parameters
    #
    # Returns:
    #   pyMINDyCoreClassic : AbstractMINDyCore
    #       The initialized pyMINDyCoreClassic object

    def init(cls, H, Q, R, W_mask, fix_Se=True, fix_Ve=True, fix_Si=True, fix_Vi=True,
             Se_fixed_value=1, Ve_fixed_value=0, Si_fixed_value=1, Vi_fixed_value=0,
             param_range=None, device="cpu", rand_scale=0.5):
        if param_range is None:
            param_range = {"Se_min": 0.2, "Se_max": 4, "Si_min": 0.2, "Si_max": 4, "De_max": 1, "De_min": 0,
                           "Di_max": 1, "Di_min": 0}

        n_pair = H.shape[1] // 2 # number of excitatory populations

        # Convert the W_mask to torch
        W_mask_ee = torch.from_numpy(W_mask[0:n_pair,0:n_pair])
        W_mask_ei = torch.from_numpy(W_mask[n_pair:,0:n_pair])
        W_mask_ee = W_mask_ee.type(torch.int32).to(device)
        W_mask_ei = W_mask_ei.type(torch.int32).to(device)

        # Generate random W matrix, section by section
        Wee = W_mask_ee * torch.rand(n_pair, n_pair, dtype=torch.float, device=device) * rand_scale
        Wei = W_mask_ei * torch.rand(n_pair, n_pair, dtype=torch.float, device=device) * rand_scale
        Wie = torch.rand(n_pair, dtype=torch.float, device=device) * rand_scale
        Wii = torch.rand(n_pair, dtype=torch.float, device=device) * rand_scale

        # Generate excitatory and inhibitory components of parameters D, C, S, & V
        #Se = Se_fixed_value * torch.ones(
