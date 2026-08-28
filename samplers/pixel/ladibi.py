from typing import Union, Optional, List

import torch
import numpy as np
from tqdm import tqdm

from utils import (flush,
                   Device
                   )
from conditions.base import ConditionBase
from alphas.alpha_maps.base import AlphaMap
from models.pixel_diffusion.samplers.sampler_utils import _extract_into_tensor
from models.pixel_diffusion.samplers.registry import register_sampler_method
from models.pixel_diffusion.samplers.ddim import DDIM


@register_sampler_method(name='ladibi')
class DDIMLADiBI(DDIM):
    def p_sample_loop(self,
                      model,
                      ref_img: torch.Tensor,
                      init_smp: torch.Tensor,
                      measurement: torch.Tensor,
                      alpha: AlphaMap,
                      start_idx: int,
                      num_steps: int,
                      num_recurrence: int,
                      lpips_scale: Union[float, int],
                      eta: Union[float, int],
                      cond_method: ConditionBase,
                      device: Device,
                      prompts: Optional[List[str]] = None):

        x_t = self.q_sample(x_start=init_smp, t=start_idx - 1)
        indices = list(range(self.num_timesteps))[start_idx - 1::-1]

        # Precompute recurrence coefficients (constant across all steps)
        sqrt_one_minus_beta = np.sqrt(1 - self.betas)
        sqrt_beta = np.sqrt(self.betas)

        pbar = tqdm(indices)
        for i, idx in enumerate(pbar):
            for ii in range(num_recurrence):
                time = torch.tensor([idx] * x_t.shape[0], device=device)
                with_grad = ii % 2 == 0
                out = self.p_sample(model=model,
                                    x=x_t,
                                    t=time[0].unsqueeze(0),
                                    with_grad=with_grad,
                                    eta=eta)
                if with_grad:
                    cond_kwargs = dict(
                        x_t=x_t,
                        x_t_minus=out['sample'],
                        x_0=out['pred_xstart'],
                        alpha=alpha,
                        curr_t=self._scale_timesteps(time[0]),
                        measurement=measurement,
                        lpips_scale=lpips_scale,
                        e_t=out['eps'],
                        step_idx=i,
                    )
                    if prompts is not None:
                        cond_kwargs['prompts'] = prompts
                    dc_outputs = cond_method.conditioning(**cond_kwargs)
                    out['sample'] = out['sample'].detach()
                    x_t_minus = dc_outputs['x_t_minus']
                    del dc_outputs

                else:
                    out = {k: v.detach() for k, v in out.items()}
                    x_t_minus = out['sample']

                if ii != num_recurrence - 1:
                    coeff1 = _extract_into_tensor(sqrt_one_minus_beta, time, x_t.shape)
                    coeff2 = _extract_into_tensor(sqrt_beta, time, x_t.shape)
                    x_t = x_t_minus * coeff1 + torch.randn_like(x_t) * coeff2
                else:
                    x_t = x_t_minus

                with torch.no_grad():
                    distance = self.get_distance(img=out['pred_xstart'],
                                                 measurement=measurement,
                                                 alpha=alpha,
                                                 operator=cond_method.operator)

                if ii == num_recurrence - 1:
                    self.save_progress(idx=idx,
                                       time=time,
                                       alpha=alpha,
                                       init_smp=init_smp,
                                       ref_img=ref_img,
                                       start_idx=start_idx,
                                       measurement=measurement,
                                       out=out['pred_xstart'],
                                       distance=distance,
                                       num_rec=ii,
                                       operator=cond_method.operator)

                pbar.set_postfix({'distance': distance.item(), 'alpha': alpha.sum().item()}, refresh=False)
                pbar.refresh()

            if (idx + 1) % 5 == 0:
                with torch.enable_grad():
                    alpha = cond_method.get_new_alpha(x_0_recon=out['pred_xstart'],
                                                      measurement=measurement,
                                                      alpha=alpha,
                                                      curr_t=time[0])

            flush()

        return {'sample': out['sample'],
                'alpha': alpha,
                }
