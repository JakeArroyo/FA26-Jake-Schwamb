import abc
from torch import nn
from torch.func import jacrev

# AbstractMINDyCore defines an abstract torch neural network which gives us the
# built-in method infrastructure to train with torch autograd
class AbstractMINDyCore(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    # Use autograd to calculate the jacobian of the functions by default
    def func_dFdx(self, x):
        return jacrev(self.func_F)(x)

    def func_dHdx(self, x):
        return jacrev(self.func_H)(x)

    # Abstract properties (these are defined in the more specific MINDy core classes)
    @property
    @abc.abstractmethod
    def n_chan(self):
        pass

    @property
    @abc.abstractmethod
    def x_size(self):
        pass

    @property
    @abc.abstractmethod
    def y_size(self):
        pass

    @property
    @abc.abstractmethod
    def device(self):
        pass

    @property
    @abc.abstractmethod
    def Q(self):
        pass

    @property
    @abc.abstractmethod
    def R(self):
        pass

    @property
    @abc.abstractmethod
    def func_F(self, x):
        pass

    @property
    @abc.abstractmethod
    def func_H(self, x):
        pass

    @property
    @abc.abstractmethod
    def get_params(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def clamp_params(self) -> None:
        pass
