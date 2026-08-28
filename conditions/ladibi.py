import torch

from alphas.alpha_maps import AlphaMap
from .base import ConditionBase
from .loss_utils import MultiScaleLossMixin, compute_measurement_loss
from .registry import register_conditioning_method


@register_conditioning_method(name='ladibi_pds')
class LADiBIPDS(ConditionBase):
    type = "PDS"

    def conditioning(self,
                     x_prev: torch.Tensor,
                     e_t: torch.Tensor,
                     x_0: torch.Tensor,
                     x_0_recon: torch.Tensor,
                     measurement: torch.Tensor,
                     alpha: AlphaMap,
                     curr_t: int,
                     lpips_scale: float = 0.0,
                     scale: float = 1.0,
                     step_idx: int = 0,
                     **kwargs
                     ):
        grads = self.grad_and_value(x_prev=x_prev,
                                    x_0_recon=x_0_recon,
                                    alpha=alpha,
                                    measurement=measurement,
                                    lpips_scale=lpips_scale,
                                    curr_t=curr_t,
                                    step_idx=step_idx,
                                    **kwargs)

        e_t -= grads * self.get_guidance_scale(curr_t) * scale

        return {'e_t': e_t,
                'alpha': alpha,
                }

    def grad_and_value(self,
                       x_prev: torch.Tensor,
                       x_0_recon: torch.Tensor,
                       measurement: torch.Tensor,
                       alpha: torch.Tensor,
                       lpips_scale: float,
                       step_idx: int = 0,
                       **kwargs):
        blended = self.operator.forward(x_0_recon, alpha=alpha)
        loss = compute_measurement_loss(blended, measurement,
                                        lpips_fn=self.lpips,
                                        lpips_scale=lpips_scale,
                                        lpips_freq=self.lpips_freq,
                                        step_idx=step_idx)
        exc_loss = self.get_exc_loss(x_0=x_0_recon)
        if exc_loss:
            loss = loss + exc_loss

        loss = loss * (-1)

        return torch.autograd.grad(outputs=loss, inputs=x_prev)[0]


@register_conditioning_method(name='ladibi_pds_ms')
class LADiBIPDSMS(MultiScaleLossMixin, LADiBIPDS):

    def grad_and_value(self,
                       x_prev: torch.Tensor,
                       x_0_recon: torch.Tensor,
                       measurement: torch.Tensor,
                       alpha: torch.Tensor,
                       lpips_scale: float,
                       step_idx: int = 0,
                       **kwargs):
        blended = self.operator.forward(x_0_recon, alpha=alpha, **kwargs)
        loss = self.compute_multiscale_loss(blended, measurement,
                                            lpips_scale=lpips_scale,
                                            step_idx=step_idx)
        exc_loss = self.get_exc_loss(x_0=x_0_recon)
        if exc_loss:
            loss = loss + exc_loss

        loss = loss * (-1)

        return torch.autograd.grad(outputs=loss, inputs=x_prev)[0]


@register_conditioning_method(name='ladibi_mpgd')
class LADiBIMPGD(ConditionBase):
    type = "MPGD"

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
        grads = self.grad_and_value(x_t=x_t,
                                    x_0=x_0,
                                    alpha=alpha,
                                    measurement=measurement,
                                    lpips_scale=lpips_scale,
                                    curr_t=curr_t,
                                    **kwargs)

        x_t_minus -= grads * self.get_guidance_scale(curr_t) * scale

        return {'x_t_minus': x_t_minus,
                'alpha': alpha,
                }

    def grad_and_value(self,
                       x_t: torch.Tensor,
                       x_0: torch.Tensor,
                       measurement: torch.Tensor,
                       alpha: torch.Tensor,
                       lpips_scale: float,
                       step_idx: int = 0,
                       **kwargs):
        blended = self.operator.forward(x_0, alpha=alpha)
        loss = compute_measurement_loss(blended, measurement,
                                        lpips_fn=self.lpips,
                                        lpips_scale=lpips_scale,
                                        lpips_freq=self.lpips_freq,
                                        step_idx=step_idx)
        exc_loss = self.get_exc_loss(x_0=x_0)
        if exc_loss:
            loss = loss + exc_loss

        return torch.autograd.grad(outputs=loss, inputs=x_t)[0]


@register_conditioning_method(name='ladibi_mpgd_ms')
class LADiBIMPGDMS(MultiScaleLossMixin, LADiBIMPGD):

    def grad_and_value(self,
                       x_t: torch.Tensor,
                       x_0: torch.Tensor,
                       measurement: torch.Tensor,
                       alpha: torch.Tensor,
                       lpips_scale: float,
                       step_idx: int = 0,
                       **kwargs):
        blended = self.operator.forward(x_0, alpha=alpha, **kwargs)
        loss = self.compute_multiscale_loss(blended, measurement,
                                            lpips_scale=lpips_scale,
                                            step_idx=step_idx)
        exc_loss = self.get_exc_loss(x_0=x_0)
        if exc_loss:
            loss += exc_loss

        return torch.autograd.grad(outputs=loss, inputs=x_t)[0]
