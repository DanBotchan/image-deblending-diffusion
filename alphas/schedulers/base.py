from abc import ABC, abstractmethod
from typing import Dict

import torch

from .registry import register_alpha_scheduler


class AlphaScheduler(ABC):
    def __init__(self,
                 chaotic_stage_alpha,
                 critical_stage_alpha,
                 refinement_stage_alpha,
                 critical_stage_start,
                 critical_stage_stop,
                 **kwargs):
        self.chaotic_stage_alpha = chaotic_stage_alpha
        self.critical_stage_alpha = critical_stage_alpha
        self.refinement_stage_alpha = refinement_stage_alpha
        self.critical_stage_start = critical_stage_start
        self.critical_stage_stop = critical_stage_stop

    @abstractmethod
    def get_alpha_scale(self,
                        curr_timestep: torch.Tensor,
                        ) -> float:
        """Get the alpha scale for the current timestep."""
        pass

    def get_schedule_params(self,
                            curr_timestep: torch.Tensor,
                            ) -> Dict[str, float]:
        return {'alpha_scale':self.get_alpha_scale(curr_timestep),
                }


@register_alpha_scheduler('constant')
class ConstScheduler(AlphaScheduler):
    """Constant alpha class for alpha scale schedulers."""

    def get_alpha_scale(self,
                        curr_timestep: torch.Tensor,
                        ) -> float:
        return 0.0
