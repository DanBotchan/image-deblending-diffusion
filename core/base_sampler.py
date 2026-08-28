"""
Base sampler interface for the deblend package.

Provides abstract base class for all samplers (both latent and pixel space).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from alphas.alpha_maps import AlphaMap
    from conditions.base import ConditionBase
    from utils.config import ExpConfig


class BaseSampler(ABC):
    """
    Abstract base class for deblending samplers.

    Provides common interface and utilities for both latent-space
    (Stage 1) and pixel-space (Stage 2) samplers.
    """

    def __init__(self):
        self.exp_config: Optional["ExpConfig"] = None
        self.eta: float = 0.0

    def init_experiment(self,
                        exp_config: "ExpConfig",
                        **kwargs) -> None:
        """
        Initialize sampler for a new experiment.

        Args:
            exp_config: Experiment configuration
            **kwargs: Additional initialization parameters
        """
        self.exp_config = exp_config
        self.eta = exp_config['general'].get('eta', 0.0)

    @abstractmethod
    def p_sample_loop(self,
                      measurement: torch.Tensor,
                      alpha: "AlphaMap",
                      cond_method: "ConditionBase",
                      **kwargs) -> Dict[str, Any]:
        """
        Run the main sampling loop for deblending.

        Args:
            measurement: Blended input image
            alpha: Alpha map for blending
            cond_method: Conditioning method
            **kwargs: Additional parameters

        Returns:
            Dictionary containing 'sample' and 'alpha' keys
        """
        pass

    @abstractmethod
    def q_sample(self,
                 x_start: torch.Tensor,
                 t: int) -> torch.Tensor:
        """
        Forward diffusion process - add noise to clean sample.

        Args:
            x_start: Clean sample
            t: Timestep

        Returns:
            Noised sample at timestep t
        """
        pass

    def get_distance(self,
                     img: torch.Tensor,
                     measurement: torch.Tensor,
                     alpha: "AlphaMap",
                     operator: "BlendOperator") -> torch.Tensor:
        """
        Compute distance between blended prediction and measurement.

        Args:
            img: Predicted images
            measurement: Ground truth blended image
            alpha: Alpha map
            operator: Blend operator

        Returns:
            L2 distance between blended prediction and measurement
        """
        blended = operator.forward(img, alpha=alpha)
        return torch.linalg.norm(blended - measurement)

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        """
        Scale timesteps for display/logging.

        Override in subclasses if needed for specific scaling.
        """
        return t

    def save_progress(self,
                      idx: int,
                      time: torch.Tensor,
                      alpha: "AlphaMap",
                      out: torch.Tensor,
                      distance: torch.Tensor,
                      measurement: torch.Tensor,
                      num_rec: int,
                      start_idx: int,
                      operator: "BlendOperator",
                      ref_img: torch.Tensor = None,
                      init_smp: torch.Tensor = None) -> None:
        """
        Save progress visualization during sampling.

        Args:
            idx: Current iteration index
            time: Current timestep
            alpha: Current alpha map
            out: Current output prediction
            distance: Current distance to measurement
            measurement: Target blended image
            num_rec: Recurrence number
            start_idx: Starting index
            operator: Blend operator
            ref_img: Reference ground truth images (optional)
            init_smp: Initial samples from Stage 1 (Stage 2 only)
        """
        from utils.image import clear_color

        general_conf = self.exp_config['general']
        save_interval = general_conf.get('save_each_steps', 10)

        if (idx % save_interval != 0 and idx and idx != start_idx - 1):
            return

        # Determine layout based on available data
        num_rows = 2
        if ref_img is not None:
            num_rows += 1
        if init_smp is not None:
            num_rows += 1

        fig, ax = plt.subplots(num_rows, 4, figsize=(12, 4 * num_rows))

        def add_alpha_colorbar(ax_subplot, img):
            cbar = plt.colorbar(img, ax=ax_subplot, cmap='gray',
                                fraction=0.046, pad=0.04)
            cbar.set_label('Alpha')
            cbar.set_ticks(np.arange(0, 1.1, 0.2))
            img.set_cmap('gray')
            return cbar

        def plot_image(ax_subplot, img, title: str,
                       add_colorbar: bool = False,
                       normalize: bool = True,
                       vmin=None, vmax=None):
            img_plot = ax_subplot.imshow(clear_color(img, normalize=normalize),
                                         vmin=vmin, vmax=vmax)
            ax_subplot.set_title(title)
            if add_colorbar and 'Alpha' in title:
                add_alpha_colorbar(ax_subplot, img_plot)
            return img_plot

        current_row = 0

        # Reference images row (if available)
        if ref_img is not None:
            plot_image(ax[current_row, 0], ref_img[0], 'GT IMG 1')
            plot_image(ax[current_row, 1], ref_img[1], 'GT IMG 2')
            plot_image(ax[current_row, 2], measurement, 'Blended GT')
            plot_image(ax[current_row, 3],
                       operator.alpha.repeat(1, 3, 1, 1),
                       f'GT Alpha:\n{operator.alpha_class}',
                       add_colorbar=True, vmin=0, vmax=1, normalize=False)
            current_row += 1

        # Initial samples row (Stage 2 only)
        if init_smp is not None:
            init_alpha = getattr(self, 'init_alpha_map', alpha)
            im_blended = operator.forward(data=init_smp, alpha=init_alpha)
            plot_image(ax[current_row, 0], init_smp[0], 'Init SMP 1')
            plot_image(ax[current_row, 1], init_smp[1], 'Init SMP 2')
            plot_image(ax[current_row, 2], im_blended, 'Init Blended')
            plot_image(ax[current_row, 3],
                       init_alpha.forward().repeat(1, 3, 1, 1),
                       f'Init Alpha:\n{init_alpha}',
                       add_colorbar=True, vmin=0, vmax=1, normalize=False)
            current_row += 1

        # Current prediction row
        blended = operator.forward(data=out, alpha=alpha)
        plot_image(ax[current_row, 0], out[0], 'Pred X0 1')
        plot_image(ax[current_row, 1], out[1], 'Pred X0 2')
        plot_image(ax[current_row, 2], blended, 'Pred Blended')
        plot_image(ax[current_row, 3],
                   alpha.forward().repeat(1, 3, 1, 1),
                   f'Pred Alpha:\n{alpha}',
                   add_colorbar=True, vmin=0, vmax=1, normalize=False)

        # Title and save
        scaled_t = self._scale_timesteps(torch.tensor(idx))
        fig.suptitle(f'Progress (t={scaled_t}, rec={num_rec})\n'
                     f'Distance: {distance.item():.4f}, '
                     f'Alpha mean: {alpha.mean().item():.4f}')
        plt.tight_layout()

        save_to = self.exp_config.get_path('progress', include_smp=True)
        save_to.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to / f"x_{str(idx).zfill(4)}.png", bbox_inches='tight')
        plt.close(fig)
