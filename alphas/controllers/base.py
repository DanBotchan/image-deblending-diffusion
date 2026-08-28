import torch

from .registry import register_alpha_controller
from alphas.alpha_maps import AlphaMap
from alphas.schedulers import AlphaScheduler
from alphas.optimizers import AlphaOptimizer
from operators.base import LinearOperator


@register_alpha_controller('base')
class AlphaController:
    """Controls the alpha map optimization process."""

    def __init__(self,
                 optimizer: AlphaOptimizer,
                 scheduler: AlphaScheduler,
                 iterations: int = 1):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iterations = iterations

    def get_new_alpha(self,
                      x_0_hat: torch.Tensor,
                      measurement: torch.Tensor,
                      alpha_map: AlphaMap,
                      curr_t: int,
                      operator: LinearOperator,
                      **kwargs):
        schedule_params = self.scheduler.get_schedule_params(curr_t)
        self.optimizer.optimize_step(alpha_map,
                                     measurement,
                                     x_0_hat,
                                     operator,
                                     alpha_scale=schedule_params['alpha_scale'],
                                     iterations=self.iterations)
        return alpha_map

    def get_optimizer_states(self):
        return self.optimizer.optimization_states

    def reset_optimizer(self):
        self.optimizer.reset_history()


# Registered separately so configs using "multi_step" keep working
@register_alpha_controller('multi_step')
class MultiStepAlphaController(AlphaController):
    def __init__(self,
                 optimizer: AlphaOptimizer,
                 scheduler: AlphaScheduler,
                 iterations: int = 150):
        super().__init__(optimizer, scheduler, iterations)
