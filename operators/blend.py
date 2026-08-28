"""
Alpha blending operator for the deblend package.
"""
from typing import Optional, TYPE_CHECKING

import torch

from core.registry import Registry
from .base import LinearOperator

if TYPE_CHECKING:
    from alphas.alpha_maps import AlphaMap

CATEGORY = 'operator'


def register_operator(name: str):
    return Registry.register(CATEGORY, name)


def get_operator(name: str = 'blend_am', **kwargs) -> LinearOperator:
    return Registry.get(CATEGORY, name)(**kwargs)


@register_operator(name='blend_am')
class BlendOperator(LinearOperator):
    """
    Alpha blending operator.

    Blends two images using an alpha map:
    blended = img1 * alpha + img2 * (1 - alpha)
    """

    def __init__(self, alpha: "AlphaMap"):
        self.alpha_class = alpha

    def forward(self,
                data: torch.Tensor,
                alpha: Optional["AlphaMap"] = None,
                **kwargs) -> torch.Tensor:
        alpha_val = self.alpha if alpha is None else alpha.forward()
        img1 = data[0]
        img2 = data[1]
        blended = (img1 * alpha_val + img2 * (1 - alpha_val))
        if len(blended.shape) == 3:
            blended = blended.unsqueeze(0)
        return blended

    @property
    def alpha(self) -> torch.Tensor:
        """Get the stored (ground truth) alpha map value."""
        return self.alpha_class.forward().detach()
