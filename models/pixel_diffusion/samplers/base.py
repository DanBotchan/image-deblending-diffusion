import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import show_tensor  # noqa: F401
from utils.image import clear_color


class BaseDeblendDiffusion:
    def init_experiment(self,
                        exp_config,
                        init_alpha_map,
                        **kwargs,
                        ):
        self.exp_config = exp_config
        self.init_alpha_map = init_alpha_map
        self.eta = self.exp_config['general']['eta']

    def get_distance(self,
                     img: torch.Tensor,
                     measurement: torch.Tensor,
                     alpha,
                     operator,
                     ):
        blended = operator.forward(img, alpha=alpha)
        return torch.linalg.norm(blended - measurement)

    def save_progress(self,
                      idx: int,
                      time: int,
                      init_smp: torch.Tensor,
                      alpha: torch.Tensor,
                      out: torch.Tensor,
                      distance: torch.Tensor,
                      measurement: torch.Tensor,
                      num_rec: int,
                      start_idx: int,
                      operator,
                      ref_img: torch.Tensor = None,
                      ) -> None:

        general_conf = self.exp_config['general']
        if (idx % general_conf['save_each_steps'] != 0 and
            idx and
            idx != start_idx - 1):
            return

        num_rows = 3 if ref_img is not None else 2
        fig, ax = plt.subplots(num_rows, 4, figsize=(12, 4 * num_rows))

        def add_alpha_colorbar(ax_subplot, img):
            """Helper function to add small, black and white colorbars to alpha plots"""
            # Make colorbar smaller and black/white
            cbar = plt.colorbar(img, ax=ax_subplot, cmap='gray', fraction=0.046, pad=0.04)
            cbar.set_label('Alpha')
            cbar.set_ticks(np.arange(0, 1.1, 0.2))
            # Make the colorbar black and white
            img.set_cmap('gray')
            return cbar

        def plot_image(ax_subplot, img, title: str, add_colorbar: bool = False, normalize=True, vmin=None, vmax=None):
            """Helper function to plot images with consistent formatting"""
            img_plot = ax_subplot.imshow(clear_color(img, normalize=normalize), vmin=vmin, vmax=vmax)
            ax_subplot.set_title(title)
            if add_colorbar and 'Alpha' in title:
                add_alpha_colorbar(ax_subplot, img_plot)
            return img_plot

        # Generate blended image
        blended = operator.forward(data=out, alpha=alpha)

        # Plot original/reference images (top row)
        plot_image(ax[0, 0], ref_img[0], 'IMG 1')
        plot_image(ax[0, 1], ref_img[1], 'IMG 2')
        plot_image(ax[0, 2], measurement, 'Blended GT')
        plot_image(ax[0, 3],
                   operator.alpha.repeat(1, 3, 1, 1),
                   f'GT Alpha:\n{operator.alpha_class}',
                   add_colorbar=True, vmin=0, vmax=1, normalize=False)

        # Plot samples if available (middle row)
        if init_smp is not None:
            im_blended = operator.forward(data=init_smp, alpha=self.init_alpha_map)
            plot_image(ax[1, 0], init_smp[0], 'Orig SMP 1')
            plot_image(ax[1, 1], init_smp[1], 'Orig SMP 2')
            plot_image(ax[1, 2], im_blended, 'Blended IMG')
            plot_image(ax[1, 3], self.init_alpha_map.forward().repeat(1, 3, 1, 1), f'Init Sample Alpha:\n{self.init_alpha_map}',
                       add_colorbar=True, vmin=0, vmax=1,
                       normalize=False)
            result_row = 2
        else:
            result_row = 1

        # Plot results (bottom row)
        plot_image(ax[result_row, 0], out[0], 'Pred Xstart 1')
        plot_image(ax[result_row, 1], out[1], 'Pred Xstart 2')
        plot_image(ax[result_row, 2], blended, 'Blended IMG')

        # Handle alpha visualization
        alpha_img = alpha.forward().repeat(1, 3, 1, 1)
        plot_image(ax[result_row, 3], alpha_img, f'SMP Alpha:\n{alpha}', add_colorbar=True, vmin=0, vmax=1,
                   normalize=False)

        # Add title and adjust layout
        fig.suptitle(f'Mid results (timestep: {self._scale_timesteps(torch.tensor(idx))}, II: {num_rec})\n'
                     f'Distance: {distance.item():.4f}, Alpha mean: {alpha.mean().item():.4f}')
        plt.tight_layout()
        save_to = self.exp_config.get_path('progress', include_smp=True) / \
                  f"x_{str(idx).zfill(4)}.png"
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches='tight')
        plt.close(fig)
