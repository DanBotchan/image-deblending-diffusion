from abc import ABC, abstractmethod
from typing import (Optional,
                    Union,
                    List,
                    )
import torch
import torch.nn.functional as F

class AlphaMap(ABC):
    """Base class for all alpha maps."""

    def __init__(self,
                 map_type: str,
                 size: int = 256,
                 device = 'cuda',
                 alpha_min: float = 0.0,
                 alpha_max: float = 1.0,
                 init_alpha: Optional[Union[List[float], float]] = None,
                 requires_grad: bool = False,
                 ):
        self.map_type = map_type
        self.size = (size, size)
        self.device = device
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.min_slope = alpha_max - alpha_min
        self.init_alpha = init_alpha
        self.requires_grad = requires_grad
        self.init_params = None
        self.initialized = False
        self._initialize_parameters()

    def tanh(self, img):
        return 0.5 * F.tanh(img) + 0.5

    @abstractmethod
    def _initialize_parameters(self):
        """Initialize map-specific parameters."""
        pass

    @abstractmethod
    def forward(self):
        """Generate the alpha map from current parameters."""
        pass

    @abstractmethod
    def get_parameters(self):
        """Return list of parameters that can be optimized."""
        pass

    def get_smoothness_loss(self, weight=1e-3):
        """
        Default smoothness loss for non-spatial alpha maps.
        Returns zero loss to maintain compatibility.

        Args:
            weight: Regularization weight (ignored for non-spatial maps)

        Returns:
            Zero tensor for compatibility
        """
        # Return zero loss for polynomial-based alpha maps
        # Only spatial maps will override this with actual smoothness computation
        return torch.tensor(0.0, device=self.device, requires_grad=False)

    def __call__(self):
        """Allow using instance as a callable."""
        return self.forward()

    def sum(self):
        return self.forward().sum()

    def mean(self):
        return self.forward().mean()

    # Basic arithmetic operations
    def __add__(self, other): return self.forward() + other

    def __radd__(self, other): return other + self.forward()

    def __sub__(self, other): return self.forward() - other

    def __rsub__(self, other): return other - self.forward()

    def __mul__(self, other): return self.forward() * other

    def __rmul__(self, other): return other * self.forward()

    def __truediv__(self, other): return self.forward() / other

    def __rtruediv__(self, other): return other / self.forward()

    # Comparison operations
    def __lt__(self, other): return self.forward() < other

    def __le__(self, other): return self.forward() <= other

    def __gt__(self, other): return self.forward() > other

    def __ge__(self, other): return self.forward() >= other

    def __eq__(self, other): return self.forward() == other

    def __ne__(self, other): return self.forward() != other

    # Power operations
    def __pow__(self, other): return self.forward() ** other

    def __rpow__(self, other): return other ** self.forward()

    # Absolute value
    def __abs__(self): return abs(self.forward())

    # Negation
    def __neg__(self): return -self.forward()
