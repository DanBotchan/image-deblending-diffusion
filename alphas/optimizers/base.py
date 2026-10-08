from abc import ABC, abstractmethod

from utils.image import show_tensor  # noqa: F401
from .registry import register_alpha_optimizer




class AlphaOptimizer(ABC):
    """Base class for all alpha map optimizers."""
    def __init__(self, **kwargs):
        self.optimization_states = []

    @abstractmethod
    def optimize_step(self,
                      alpha_map,
                      measurement,
                      x_0_hat,
                      operator,
                      alpha_scale,
                      iterations: int = 1):
        """
        Perform one step of optimization.

        Args:
            alpha_map: The alpha map to optimize
            measurement: Target image
            x_0_hat: Current estimate of source images
            operator: Blending operator
            alpha_scale: Scale factor for updates
            iterations: Number of optimization iterations to perform

        Returns:
            float: Loss value
        """
        raise NotImplementedError

    def record_state(self, alpha_tensor):
        """Record the current state of alpha map for analysis."""
        self.optimization_states.append(alpha_tensor.detach().clone())

    def get_optimization_history(self):
        """Get the history of alpha map states."""
        return self.optimization_states

    def reset_history(self):
        self.optimization_states = []

@register_alpha_optimizer('none')
class NoneOptimizer(AlphaOptimizer):
    def optimize_step(self,
                      alpha_map,
                      measurement,
                      x_0_hat,
                      operator,
                      alpha_scale,
                      iterations: int = 1):
        """Perform one optimization step."""
        return 0.0