"""
Common utility functions for the deblend package.
"""
import gc
import random
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .config import ExpConfig


def flush():
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()


def set_seed(config: "ExpConfig",
             seed: Optional[int] = None,
             ) -> int:
    """
    Set random seed for reproducibility.

    Args:
        config: Experiment configuration containing general.seed
        seed: Optional explicit seed (overrides config)

    Returns:
        The seed that was set
    """
    if seed is None:
        seed = config['general']['seed']

    if isinstance(seed, torch.Tensor):
        seed = seed.item()

    if seed is not None and seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed
