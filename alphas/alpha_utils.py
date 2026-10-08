from typing import TYPE_CHECKING

from .controllers import (get_alpha_controller,
                          AlphaController,
                          )
from .optimizers import get_optimizer
from .schedulers import get_scheduler

if TYPE_CHECKING:
    from utils.config import ExpConfig


def build_controller(exp_config: "ExpConfig",
                     ) -> AlphaController:
    return get_alpha_controller(name=exp_config['alphas']['controller'],
        scheduler=get_scheduler(
            name=exp_config['alphas']['scheduler'],
            chaotic_stage_alpha=0.1,
            critical_stage_alpha=0.1,
            refinement_stage_alpha=0.1,
            critical_stage_start=exp_config['general']['critical_start'],
            critical_stage_stop=exp_config['general']['critical_stop'],
        ),
        optimizer=get_optimizer(name=exp_config['alphas']['optimizer']),
    )
