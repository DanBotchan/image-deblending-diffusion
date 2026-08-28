import contextlib
from typing import Dict, Union

import numpy as np
import torch
from tqdm import tqdm

from alphas.alpha_maps import AlphaMap
from conditions import ConditionBase
from models.latent_diffusion.modules.diffusionmodules.util import noise_like
from utils import flush
from .base import DDIMSamplerBase
from .registry import register_sampler_method


@register_sampler_method('ladibi')
class LADiBI(DDIMSamplerBase):
    def one_denoise_step(self,
                         img: torch.Tensor,
                         time: int,
                         pos_text_emb: torch.Tensor,
                         neg_text_emb: torch.Tensor,
                         cfg_scale: float,
                         ) -> torch.Tensor:
        exp_config = getattr(self, 'exp_config', None)
        use_amp = exp_config['general'].get('use_amp', False) if exp_config is not None else False
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast('cuda', dtype=amp_dtype) if use_amp else contextlib.nullcontext()
        if cfg_scale:
            x_in = torch.cat([img] * 2)
            t_in = torch.cat([time] * 2)
            c_in = torch.cat([pos_text_emb, neg_text_emb])
            with ctx:
                raw = self.model.module.apply_model(x_in, t_in, c_in)
            e_t_pos, e_t_uncond = raw.float().chunk(2)
            e_t = e_t_uncond + cfg_scale * (e_t_pos - e_t_uncond)
            del e_t_uncond, c_in, t_in, x_in, e_t_pos, raw

        else:
            x_in = img
            t_in = time
            c_in = pos_text_emb
            with ctx:
                e_t = self.model.module.apply_model(x_in, t_in, c_in).float()
            del c_in, t_in, x_in

        return e_t

    def dc_step(self,
                e_t: torch.Tensor,
                x_0: torch.Tensor,
                measurement: torch.Tensor,
                img: torch.Tensor,
                time: torch.Tensor,
                recons_image: torch.Tensor,
                sqrt_one_minus_at: torch.Tensor,
                a_t: torch.Tensor,
                alpha: AlphaMap,
                cond_method: ConditionBase,
                lpips_scale: float,
                scale: float = 1.0,
                step_idx: int = 0,
                ) -> Dict[str, Union[torch.Tensor, AlphaMap]]:
        """ Data Consistency step """
        out = cond_method.conditioning(e_t=e_t,
                                       x_prev=img,
                                       alpha=alpha,
                                       curr_t=time[0],
                                       x_0=x_0,
                                       x_0_recon=recons_image,
                                       measurement=measurement,
                                       lpips_scale=lpips_scale,
                                       scale=scale,
                                       step_idx=step_idx,
                                       )
        if cond_method.type == "PDS":
            pred_x0 = (img - sqrt_one_minus_at * out['e_t']) / a_t.sqrt()
            e_t = out['e_t']

        elif cond_method.type == "MPGD":
            pred_x0 = out['x_0']
            e_t = (img - a_t.sqrt() * pred_x0) / sqrt_one_minus_at

        else:
            raise ValueError('Condition method type should be PDS or MPGD')

        return {'pred_x0': pred_x0,
                'e_t': e_t,
                }

    def p_sample_loop(self,
                      ref_img: torch.Tensor,
                      measurement: torch.Tensor,
                      pos_text_emb: torch.Tensor,
                      neg_text_emb: torch.Tensor,
                      alpha: AlphaMap,
                      cond_method: ConditionBase,
                      start_idx: int,
                      num_steps: int,
                      num_recurrence: int,
                      eta: float,
                      cfg_scale: float,
                      lpips_scale: float,
                      temperature: float = 1.0,
                      **kwargs,
                      ):
        if start_idx == num_steps:
            start_idx -= 1

        self.make_schedule(ddim_num_steps=num_steps,
                           ddim_eta=eta,
                           verbose=False)
        x_0_start = self.encode_fn(measurement.unsqueeze(0))
        x_0_start = torch.vstack([x_0_start, x_0_start])
        img = self.q_sample(x_start=x_0_start,
                            t=self.ddim_timesteps[start_idx])
        del x_0_start
        timesteps = self.ddim_timesteps[:start_idx + 1]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False

        alphas_prev = torch.tensor(self.ddim_alphas_prev, device='cuda')
        sigmas_temp = torch.sqrt((1 - alphas_prev) /
                                 (1 - self.ddim_alphas) *
                                 (1 - self.ddim_alphas / alphas_prev)
                                 ).tolist()
        if cond_method.type == 'PDS':
            sigmas_temp = sigmas_temp[::-1]

        lean_decode = (self.exp_config['general'].get('lean_decode', True)
                       if getattr(self, 'exp_config', None) is not None
                       else True)

        pbar = tqdm(time_range,
                    desc='DDIM Sampler',
                    total=total_steps,
                    position=1,
                    leave=False)
        for i, idx in enumerate(pbar):
            index = total_steps - i - 1
            b, *_, device = *img.shape, img.device
            shp = (b, 1, 1, 1)
            time = torch.full((b,), idx, device=device, dtype=torch.long)
            a_t = torch.full(shp, self.ddim_alphas[index], device=device)
            sigma_t = torch.full(shp, self.ddim_sigmas[index], device=device)
            a_prev = torch.full(shp, self.ddim_alphas_prev[index],
                                device=device)
            sqrt_one_minus_at = torch.full(shp,
                                           self.ddim_sqrt_one_minus_alphas[
                                               index], device=device)

            beta_t = a_t / a_prev
            for num_rec in range(num_recurrence):
                with_grad = num_rec % 2 == 0
                is_last_rec = num_rec == num_recurrence - 1
                img = img.detach()
                if cond_method.type == 'PDS':
                    torch.set_grad_enabled(with_grad)
                    img = img.requires_grad_(with_grad)

                e_t = self.one_denoise_step(
                    img=img,
                    time=time,
                    pos_text_emb=pos_text_emb,
                    neg_text_emb=neg_text_emb,
                    cfg_scale=cfg_scale,
                )

                if with_grad:
                    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    if cond_method.type == 'MPGD':
                        torch.set_grad_enabled(True)
                        pred_x0 = pred_x0.detach().requires_grad_(True)

                    recons_image = self.decode_grad(pred_x0)
                    dc_outputs = self.dc_step(
                        e_t=e_t,
                        measurement=measurement,
                        img=img,
                        alpha=alpha,
                        time=time,
                        x_0=pred_x0,
                        recons_image=recons_image,
                        sqrt_one_minus_at=sqrt_one_minus_at,
                        a_t=a_t,
                        scale=1.0, #(torch.sqrt(a_t) / sigmas_temp[i]),
                        lpips_scale=lpips_scale,
                        cond_method=cond_method,
                        step_idx=i,
                    )
                    del recons_image, pred_x0
                    pred_x0 = dc_outputs['pred_x0']
                    e_t = dc_outputs['e_t']
                    del dc_outputs
                else:
                    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

                with torch.no_grad():
                    # lean_decode: this non-grad VAE decode only feeds
                    # get_new_alpha / save_progress / the distance below, all of
                    # which read the LAST recurrence's value. Decoding only on
                    # the final recurrence is numerically identical (the
                    # img/x_prev update is unchanged, the decode has no RNG side
                    # effect) and removes up to (num_recurrence-1) VAE decodes
                    # per step (~1.2x). Disable with general.lean_decode=false.
                    if is_last_rec or not lean_decode:
                        recons_image = self.decode_wo_grad(pred_x0.detach())
                    dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
                    noise = sigma_t * noise_like(img.shape, device,
                                                 False) * temperature
                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                    img = beta_t.sqrt() * x_prev +\
                          (1 - beta_t).sqrt() * noise_like(img.shape, device,
                                                           False)
                    del dir_xt, noise

                if is_last_rec or not lean_decode:
                    distance = self.get_distance(img=recons_image,
                                                 measurement=measurement,
                                                 alpha=alpha,
                                                 operator=cond_method.operator)

            if (i + 1) % 5 == 0:
                with torch.enable_grad():
                    alpha = cond_method.get_new_alpha(x_0_recon=recons_image,
                                                      measurement=measurement,
                                                      alpha=alpha,
                                                      curr_t=time[0])

            self.save_progress(idx=index,
                               time=time[0].item(),
                               alpha=alpha,
                               out=recons_image,
                               distance=distance,
                               num_rec=num_rec,
                               start_idx=start_idx,
                               operator=cond_method.operator,
                               ref_img=ref_img)

            del recons_image
            flush()
            img = x_prev

        return {'sample': pred_x0,
                'alpha': alpha,
                }

    def initialize_operator(self,
                            measurement: torch.Tensor,
                            pos_text_emb: torch.Tensor,
                            neg_text_emb: torch.Tensor,
                            alpha: AlphaMap,
                            cond_method: ConditionBase,
                            num_steps: int,
                            eta: float,
                            cfg_scale: float,
                            lpips_scale: float,
                            num_M: int = 10,
                            temperature: float = 1.0,
                            **kwargs,
                            ):
        self.make_schedule(ddim_num_steps=num_steps,
                           ddim_eta=eta,
                           verbose=False)

        encoded_measurement = self.encode_fn(measurement.unsqueeze(0))
        x_0_start = encoded_measurement.repeat(2, 1, 1, 1)
        del encoded_measurement
        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False

        for j in range(num_M):
            img = self.q_sample(x_start=x_0_start,
                                t=self.ddim_timesteps[-1])
            pbar = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
            for i, idx in enumerate(pbar):
                index = total_steps - i - 1
                b, *_, device = *img.shape, img.device
                shp = (b, 1, 1, 1)
                time = torch.full((b,), idx, device=device, dtype=torch.long)
                a_t = torch.full(shp, alphas[index], device=device)
                a_prev = torch.full(shp, alphas_prev[index], device=device)
                sigma_t = torch.full(shp, sigmas[index], device=device)
                sqrt_one_minus_at = torch.full(shp,
                                               sqrt_one_minus_alphas[index],
                                               device=device)
                with_grad = j != num_M - 1
                torch.set_grad_enabled(with_grad)
                img = img.detach().requires_grad_(with_grad)
                e_t = self.one_denoise_step(img=img,
                                            time=time,
                                            pos_text_emb=pos_text_emb,
                                            neg_text_emb=neg_text_emb,
                                            cfg_scale=cfg_scale)

                if with_grad:
                    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    recons_image = self.decode_grad(pred_x0)
                    dc_outputs = self.dc_step(
                        e_t=e_t,
                        measurement=measurement,
                        img=img,
                        alpha=alpha,
                        time=time,
                        x_0=pred_x0,
                        recons_image=recons_image,
                        sqrt_one_minus_at=sqrt_one_minus_at,
                        a_t=a_t,
                        lpips_scale=lpips_scale,
                        cond_method=cond_method,
                        step_idx=i,
                    )
                    del recons_image, pred_x0
                    pred_x0 = dc_outputs['pred_x0']
                    del dc_outputs
                else:
                    img = img.detach().requires_grad_(False)
                    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

                torch.set_grad_enabled(False)
                with torch.no_grad():
                    recons_image = self.decode_wo_grad(pred_x0)
                    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                    noise = sigma_t * noise_like(img.shape, device,
                                                 False) * temperature
                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                    del dir_xt, noise, pred_x0, e_t

                flush()
                img = x_prev

            with torch.enable_grad():
                alpha = cond_method.get_new_alpha(x_0_recon=recons_image,
                                                  measurement=measurement,
                                                  alpha=alpha,
                                                  curr_t=time[0])
            del recons_image

        return alpha
