"""Paper samplers for latent Stage 1 and pixel Stage 2."""

from .latent.registry import get_sampler as get_latent_sampler
from .latent.registry import register_sampler_method as register_latent_sampler

# Import both packages to register LADiBI implementations.
from . import latent  # noqa: F401
from . import pixel  # noqa: F401

__all__ = ["get_latent_sampler", "register_latent_sampler"]
