from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.exp_ops import ExpConfig
from utils.image import clear_color
from alphas.alpha_maps import AlphaMap

class BaseDeblendDiffusion():
    def init_experiment(self,
                        exp_config: ExpConfig,
                        **kwargs,
                        ):
        self.exp_config = exp_config
        self.eta = self.exp_config['general']['eta']

    def _amp_ctx(self):
        exp_config = getattr(self, 'exp_config', None)
        use_amp = exp_config['general'].get('use_amp', False) if exp_config is not None else False
        if use_amp:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast('cuda', dtype=dtype)
        import contextlib
        return contextlib.nullcontext()

    def decode_grad(self,
                    latent: torch.Tensor,
                    ) -> torch.Tensor:
        with self._amp_ctx():
            return self.model.module.decode_first_stage_with_grad(latent).float()

    def decode_wo_grad(self,
                       latent: torch.Tensor,
                       ) -> torch.Tensor:
        with self._amp_ctx():
            return self.model.module.decode_first_stage(latent).float()

    def get_distance(self,
                     img: torch.Tensor,
                     alpha: AlphaMap,
                     measurement: torch.Tensor,
                     operator,
                     ) -> torch.Tensor:
        blended = operator.forward(img,
                                   alpha=alpha)
        return torch.linalg.norm(blended - measurement)


    def save_progress(self,
                      idx: int,
                      time: int,
                      ref_img: torch.Tensor,
                      alpha: AlphaMap,
                      out: torch.Tensor,
                      distance: torch.Tensor,
                      num_rec: int,
                      start_idx: int,
                      operator,
                      ) -> None:

        general_conf = self.exp_config['general']
        if (idx % general_conf['save_each_steps'] != 0 and
            idx and
            idx != start_idx):
            return

        # Setup figure dimensions based on samples
        fig, ax = plt.subplots(2, 4, figsize=(12, 4 * 2))

        def add_alpha_colorbar(ax_subplot: plt.Figure,
                               img: np.ndarray):
            """Helper function to add small, black and white colorbars to
            alpha plots"""
            # Make colorbar smaller and black/white
            cbar = plt.colorbar(img,
                                ax=ax_subplot,
                                cmap='gray',
                                fraction=0.046,
                                pad=0.04)
            cbar.set_label('Alpha')
            cbar.set_ticks(np.arange(0, 1.1, 0.2))
            # Make the colorbar black and white
            img.set_cmap('gray')
            return cbar

        def plot_image(ax_subplot: plt.Figure,
                       img: np.ndarray,
                       title: str,
                       add_colorbar: bool = False,
                       normalize: bool = True,
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None, ):
            """Helper function to plot images with consistent formatting"""
            img_plot = ax_subplot.imshow(clear_color(img, normalize=normalize),
                                         vmin=vmin,
                                         vmax=vmax)
            ax_subplot.set_title(title)
            if add_colorbar and 'Alpha' in title:
                add_alpha_colorbar(ax_subplot, img_plot)

            return img_plot

        blended = operator(data=out, alpha=alpha)
        measurement = operator(data=ref_img)
        plot_image(ax[0, 0], ref_img[0], 'IMG 1')
        plot_image(ax[0, 1], ref_img[1], 'IMG 2')
        plot_image(ax[0, 2], measurement, 'Blended GT')
        plot_image(ax[0, 3],
                   operator.alpha.repeat(1, 3, 1, 1),
                   f'GT Alpha:\n{operator.alpha_class}',
                   add_colorbar=True, vmin=0, vmax=1, normalize=False)
        plot_image(ax[1, 0], out[0], 'Pred Xstart 1')
        plot_image(ax[1, 1], out[1], 'Pred Xstart 2')
        plot_image(ax[1, 2], blended, 'Blended IMG')
        plot_image(ax[1, 3], alpha.forward().repeat(1, 3, 1, 1),
                   f'SMP Alpha:\n{alpha}',
                   add_colorbar=True, vmin=0, vmax=1, normalize=False)
        fig.suptitle(f'Mid results (timestep: {time}, II: {num_rec})\n'
                     f'Distance: {distance.item():.4f}, Alpha mean: '
                     f'{alpha.mean().item():.4f}')
        plt.tight_layout()
        save_to = self.exp_config.get_path('progress', include_smp=True) / \
                  f"x_{str(idx).zfill(4)}.png"
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches='tight')
        plt.close(fig)