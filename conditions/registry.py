from core.registry import Registry

from alphas.controllers import AlphaController
from operators.base import LinearOperator
from models.latent_diffusion.scale_factor_methods import BaseScheduler
from utils.exp_ops import (ExpConfig, )

CATEGORY = 'conditioning_method'


def register_conditioning_method(name: str):
    return Registry.register(CATEGORY, name)


def get_cond_method(exp_config: ExpConfig,
                    operator: LinearOperator,
                    guidance_scheduler: BaseScheduler,
                    alpha_controller: AlphaController,
                    ):
    name = exp_config['general']['conditioning_method']
    cls = Registry.get(CATEGORY, name)

    # Build kwargs for conditioning method
    kwargs = {
        'operator': operator,
        'guidance_scheduler': guidance_scheduler,
        'alpha_controller': alpha_controller,
        'device': exp_config.get_device(),
        'lpips': exp_config['general']['lpips'],
        'exc_scale': exp_config['general']['exc_scale'],
        'lpips_freq': exp_config['general'].get('lpips_freq', 1),
    }

    # Add CLIP-specific parameters if available
    if 'clip_scale' in exp_config['general']:
        kwargs['clip_scale'] = exp_config['general']['clip_scale']
    if 'clip_model' in exp_config['general']:
        kwargs['clip_model_name'] = exp_config['general']['clip_model']

    return cls(**kwargs)
