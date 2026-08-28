from typing import (Dict,
                    Any,
                    )

import torch

from models.pixel_diffusion.samplers.sampler_utils import extract_and_expand
from models.pixel_diffusion.samplers.registry import register_sampler_method
from models.pixel_diffusion.samplers.guided_diffusion import SpacedDiffusion

class DDIM(SpacedDiffusion):
    def p_sample(self,
                 model,
                 x: torch.Tensor,
                 t: torch.Tensor,
                 clip_denoised: bool = True,
                 with_grad: bool = False,
                 eta: float = 0.0,
                 **kwargs,
                 ) -> Dict[str, Any]:
        if with_grad:
            x = x.requires_grad_()
            out = self.p_mean_variance(model, x, t)

        else:
            with torch.no_grad():
                out = self.p_mean_variance(model, x, t)

        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])

        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev))

        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
        sample = mean_pred
        if t != 0:
            sample += sigma * noise

        return {"sample": sample,
                "pred_xstart": out["pred_xstart"],
                "eps": eps,
                }

    def p_sample_loop(self, **kwargs):
        raise NotImplementedError("Not implemented")

    def  predict_eps_from_x_start(self,
                                  x_t: torch.Tensor,
                                  t: torch.Tensor,
                                  pred_xstart: torch.Tensor) -> torch.Tensor:
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2