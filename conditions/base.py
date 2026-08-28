from abc import ABC, abstractmethod
from typing import Optional

import torch

from alphas.alpha_maps import (AlphaMap,
                               get_alpha_map,
                               )
from alphas.controllers import AlphaController
from helpers.exclusion_loss import ExclusionLoss
from utils import Device


class ConditionBase(ABC):
    def __init__(self,
                 operator,
                 guidance_scheduler,
                 alpha_controller: AlphaController,
                 exc_scale: float,
                 device: Device,
                 *args,
                 **kwargs):
        self.device = device
        self.operator = operator
        self.guidance_scaler = guidance_scheduler
        self.alpha_controller = alpha_controller
        self._lpips = None  # Lazy-initialized to save ~80MB GPU when unused
        self.exc_loss = ExclusionLoss()
        self.exc_scale = exc_scale
        self.lpips_freq = kwargs.get('lpips_freq', 1)

    @property
    def lpips(self):
        if self._lpips is None:
            import lpips as lpips_lib
            self._lpips = lpips_lib.LPIPS(net='vgg').to(self.device)
        return self._lpips

    @abstractmethod
    def conditioning(self,
                     x_t: torch.Tensor,
                     x_t_minus: torch.Tensor,
                     x_0: torch.Tensor,
                     measurement: torch.Tensor,
                     alpha: AlphaMap,
                     curr_t: int,
                     lpips_scale: float = 0.0,
                     scale: float = 1.0,
                     **kwargs
                     ):
        pass

    @abstractmethod
    def grad_and_value(self,
                       x_t: torch.Tensor,
                       x_0: torch.Tensor,
                       measurement: torch.Tensor,
                       alpha: torch.Tensor,
                       lpips_scale: float,
                       **kwargs):
        pass

    def norm_grad(self,
                  grads: torch.Tensor,
                  alpha: AlphaMap,
                  ) -> torch.Tensor:
        size = grads.shape[-1]
        cache_key = (size,)
        cached = getattr(alpha, '_norm_grad_cache', None)
        if cached is None or cached[0] != cache_key:
            alpha_resize = get_alpha_map(name=alpha.map_type,
                                         init_alpha=alpha.get_scaled_parameters(),
                                         device=alpha.device,
                                         requires_grad=False,
                                         size=size,
                                         )
            alpha_temp = alpha_resize.forward().detach()
            alpha._norm_grad_cache = (cache_key, alpha_temp)
        else:
            alpha_temp = cached[1]
        grads[0] *= alpha_temp
        grads[1] *= (1 - alpha_temp)
        return grads


    def get_exc_loss(self,
                     x_0: torch.Tensor = None,
                     x_0_hat: torch.Tensor = None,
                     ) -> Optional[torch.Tensor]:
        x = x_0 if x_0 is not None else x_0_hat
        if self.exc_scale:
            return self.exc_scale * self.exc_loss(x[0].unsqueeze(0),
                                                  x[1].unsqueeze(0))
        return None

    def get_guidance_scale(self,
                           curr_t: int,
                           ) -> float:
        return self.guidance_scaler.get_scale(curr_t)

    def get_new_alpha(self,
                      x_0_recon: torch.Tensor = None,
                      measurement: torch.Tensor = None,
                      alpha: AlphaMap = None,
                      curr_t: int = None,
                      x_0: torch.Tensor = None,
                      prev_alpha: 'AlphaMap' = None,
                      **kwargs
                      ):
        x_0_recon = x_0_recon if x_0_recon is not None else x_0
        alpha = alpha if alpha is not None else prev_alpha
        return self.alpha_controller.get_new_alpha(x_0_hat=x_0_recon,
                                                   measurement=measurement,
                                                   curr_t=curr_t,
                                                   alpha_map=alpha,
                                                   operator=self.operator,
                                                   **kwargs
                                                   )
