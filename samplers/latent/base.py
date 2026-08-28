import numpy as np
import torch
from tqdm import tqdm

from alphas.alpha_maps import AlphaMap
from conditions import ConditionBase
from models.latent_diffusion.models.diffusion.BASE import BaseDeblendDiffusion
from models.latent_diffusion.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor
)
from models.latent_diffusion.util import default
from .registry import register_sampler_method


class DDIMSamplerBase(BaseDeblendDiffusion):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.encode_fn = lambda x: self.model.module.get_first_stage_encoding(
            self.model.module.encode_first_stage(x))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform",
                      ddim_eta=0., verbose=True):

        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=1000,
            verbose=verbose)

        alphas_cumprod = self.model.module.alphas_cumprod
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(
            self.model.module.device)

        self.register_buffer('betas', to_torch(self.model.module.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(self.model.module.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = \
            make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas',
                             np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps',
                             sigmas_for_original_sampling_steps)

    def ddim_sampling(self, cond, shape, operated_image=None, operation=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None,
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None, ):
        device = self.model.module.betas.device
        b = shape[0]

        img = torch.randn(shape, device=device)

        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            b, *_, device = *img.shape, img.device

            # select parameters corresponding to the currently considered
            # timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index],
                                device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1),
                                           sqrt_one_minus_alphas[index],
                                           device=device)

            operation_func = operation.operation_func
            other_guidance_func = operation.other_guidance_func
            criterion = operation.loss_func
            other_criterion = operation.other_criterion
            max_iters = operation.max_iters
            loss_cutoff = operation.loss_cutoff

            if operation.guidance_3:

                print(GPUtil.showUtilization())

                torch.set_grad_enabled(True)
                img_in = img.detach().requires_grad_(True)
                #
                # optimizer = torch.optim.SGD([img_in], lr=1)
                # optimizer.zero_grad()

                if operation.original_guidance:
                    x_in = torch.cat([img_in] * 2)
                    t_in = torch.cat([ts] * 2)
                    c_in = torch.cat([unconditional_conditioning, cond])
                    e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in,
                                                                    c_in).chunk(
                        2)
                    e_t = e_t_uncond + unconditional_guidance_scale * (
                            e_t - e_t_uncond)
                    # del x_in
                else:
                    e_t = self.model.module.apply_model(img_in, ts, cond)

                pred_x0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
                recons_image = self.model.module.decode_first_stage_with_grad(
                    pred_x0)

                if other_guidance_func != None:
                    op_im = other_guidance_func(recons_image)
                elif operation_func != None:
                    op_im = operation_func(recons_image)
                else:
                    op_im = recons_image

                if other_criterion != None:
                    selected = -1 * other_criterion(op_im, operated_image)
                else:
                    selected = -1 * criterion(op_im, operated_image)

                print(ts)
                print(selected)

                grad = torch.autograd.grad(selected.sum(), img_in)[0]
                grad = grad * operation.optim_guidance_3_wt

                # selected.sum().backward()
                # optimizer.step()
                # optimizer.zero_grad()
                # grad = (img - img_in) * operation.optim_guidance_3_wt

                e_t = e_t - sqrt_one_minus_at * grad.detach()

                img_in = img_in.requires_grad_(False)

                del img_in, pred_x0, recons_image, op_im, selected, grad
                if operation.original_guidance:
                    del x_in
                torch.set_grad_enabled(False)
                print("Done ?")

                print(GPUtil.showUtilization())

            else:
                if operation.original_guidance:
                    x_in = torch.cat([img] * 2)
                    t_in = torch.cat([ts] * 2)
                    c_in = torch.cat([unconditional_conditioning, cond])
                    e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in,
                                                                    c_in).chunk(
                        2)
                    e_t = e_t_uncond + unconditional_guidance_scale * (
                            e_t - e_t_uncond)
                else:
                    e_t = self.model.module.apply_model(img, ts, cond)

            with torch.no_grad():
                # current prediction for x_0
                pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

                # direction pointing to x_t
                dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                noise = sigma_t * noise_like(img.shape, device,
                                             False) * temperature

                x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                img = x_prev

                del pred_x0, dir_xt, noise, x_prev

        return img

    def p_sample_ddim(self, x, c, t, index, repeat_noise=False,
                      use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None,
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or \
            unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (
                    e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c,
                                               **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else \
            self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps \
            else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if \
            use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if \
            use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1),
                                       sqrt_one_minus_alphas[index],
                                       device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device,
                                     repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t,
                                    x0.shape) * noise)

    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0,
               unconditional_conditioning=None,
               use_original_steps=False):
        timesteps = np.arange(
            self.ddpm_num_timesteps) if use_original_steps else \
            self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device,
                            dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                          use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec

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

        temperature = 1.0
        img = kwargs.get('z_start', None)

        self.make_schedule(ddim_num_steps=num_steps, ddim_eta=self.eta,
                           verbose=False)
        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        # sampling
        cond = text_emb

        betas = self.betas[self.ddim_timesteps]
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False

        pbar = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, idx in enumerate(pbar):
            index = total_steps - i - 1
            b, *_, device = *img.shape, img.device
            time = torch.full((b,), idx, device=device, dtype=torch.long)

            # select parameters corresponding to the currently considered
            # timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index],
                                device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1),
                                           sqrt_one_minus_alphas[index],
                                           device=device)
            beta_t = a_t / a_prev
            actual_beta_t = torch.full((b, 1, 1, 1), betas[index],
                                       device=device)

            num_rec = self.num_recurrence  # if self.critical_stage_stop <=
            # time[0] <= self.critical_stage_start else 1
            for ii in range(num_rec):
                torch.set_grad_enabled(True)
                img_in = img.detach().requires_grad_(True)

                x_in = torch.cat([img_in] * 2)
                t_in = torch.cat([time] * 2)
                c_in = torch.cat([self.null_text_emb, cond])
                e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in,
                                                                c_in).chunk(2)

                cfg_scale = self.cfg_scale  # if i <= 10 else 1.0
                e_t = e_t_uncond + cfg_scale * (e_t - e_t_uncond)

                if ii < num_rec - 1:
                    pred_x0 = (img_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    recons_image = \
                        self.model.module.decode_first_stage_with_grad(
                        pred_x0)
                    e_t, alpha, distance = self.measurement_cond_fn(x_t=e_t,
                                                                    x_prev=img_in,
                                                                    alpha=alpha,
                                                                    with_grad=True,
                                                                    curr_t=
                                                                    time[0],
                                                                    exc_scale=self.exc_scale,
                                                                    x_0_hat=recons_image,
                                                                    measurement=self.measurement,
                                                                    sqrt_one_minus_at=sqrt_one_minus_at,
                                                                    )
                    img_in = img_in.detach()
                    del img_in, pred_x0, recons_image
                torch.set_grad_enabled(False)

                with torch.no_grad():
                    # current prediction for x_0
                    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    recons_image = \
                        self.model.module.decode_first_stage_with_grad(
                        pred_x0)

                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                    noise = sigma_t * noise_like(img.shape, device,
                                                 False) * temperature
                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                    img = beta_t.sqrt() * x_prev + (
                            1 - beta_t).sqrt() * noise_like(img.shape, device,
                                                            False)
                    del pred_x0, dir_xt, noise

                if ii == num_rec - 1:
                    self.save_progress(idx=index, alpha=alpha,
                                       out=recons_image, distance=distance)
                pbar.set_postfix(
                    {'distance': distance.item(), 'alpha': alpha.sum().item()},
                    refresh=False)

            img = x_prev

        return img, alpha

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=x_start.device)
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t,
                                    x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t,
                                    x_start.shape) * noise)


@register_sampler_method('no_constraint')
class NoConstraint(DDIMSamplerBase):
    def p_sample_loop(self, z_start, num_steps, text_emb, alpha=None,
                      **kwargs):
        temperature = 1.0
        img = z_start

        self.make_schedule(ddim_num_steps=num_steps, ddim_eta=self.eta,
                           verbose=False)
        num_rec = self.num_recurrence
        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        pseudo_scales = np.linspace(1.0, 0.0, num_rec)

        # sampling
        cond = text_emb

        betas = self.betas[self.ddim_timesteps]
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        for param in self.model.module.first_stage_model.parameters():
            param.requires_grad = False

        distance = torch.tensor([0], device=self.device)
        pbar = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, idx in enumerate(pbar):
            index = total_steps - i - 1
            b, *_, device = *img.shape, img.device
            time = torch.full((b,), idx, device=device, dtype=torch.long)
            critical_stage = (
                    idx <= self.critical_stage_start and idx >=
                    self.critical_stage_stop)

            # select parameters corresponding to the currently considered
            # timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index],
                                device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1),
                                           sqrt_one_minus_alphas[index],
                                           device=device)
            beta_t = a_t / a_prev
            for ii in range(num_rec):
                torch.set_grad_enabled(False)
                img_in = img.detach().requires_grad_(False)

                with torch.no_grad():
                    x_in = torch.cat([img_in] * 2)
                    t_in = torch.cat([time] * 2)
                    c_in = torch.cat([self.null_text_emb, cond])
                    e_t_uncond, e_t = self.model.module.apply_model(x_in, t_in,
                                                                    c_in).chunk(
                        2)
                    e_t = e_t_uncond + self.cfg_scale * (e_t - e_t_uncond)

                    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    recons_image = self.model.module.decode_first_stage_with_grad(
                        pred_x0)

                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                    noise = sigma_t * noise_like(img.shape, device,
                                                 False) * temperature
                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
                    img = beta_t.sqrt() * x_prev + (
                            1 - beta_t).sqrt() * noise_like(img.shape, device,
                                                            False)
                    del pred_x0, dir_xt, noise

                if True:  #ii == num_rec - 1:
                    self.save_progress(idx=index, alpha=alpha,
                                       out=recons_image, distance=distance,
                                       num_rec=ii)
                pbar.set_postfix(
                    {'distance': distance.item(), 'alpha': alpha.sum().item()},
                    refresh=False)
                pbar.refresh()
            img = x_prev

        return img, alpha
