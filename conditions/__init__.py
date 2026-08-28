"""
Conditioning methods for the deblend package.

Contains only the conditioning method used by the paper pipeline.
"""
from .registry import get_cond_method, register_conditioning_method
from .base import ConditionBase

# Trigger registration of the paper method.
from . import ladibi  # noqa: F401

__all__ = [
    'get_cond_method',
    'register_conditioning_method',
    'ConditionBase',
]
