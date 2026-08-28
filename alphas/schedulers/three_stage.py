from .base import AlphaScheduler
from .registry import register_alpha_scheduler

@register_alpha_scheduler('three_stage')
class ThreeStageScheduler(AlphaScheduler):
    def get_alpha_scale(self,
                        curr_timestep: int
                        ) -> float:
        if curr_timestep > self.critical_stage_start:
            return self.chaotic_stage_alpha
        elif self.critical_stage_start >= curr_timestep >= self.critical_stage_stop:
            return self.critical_stage_alpha
        else:
            return self.refinement_stage_alpha