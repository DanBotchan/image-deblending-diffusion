"""
Image utility functions for the deblend package.
"""
from pathlib import Path
from typing import Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

if TYPE_CHECKING:
    from alphas.alpha_maps import AlphaMap
    from operators.blend import BlendOperator


def normalize_np(img: np.ndarray) -> np.ndarray:
    """Normalize img in arbitrary range to [0, 1]."""
    img = img - np.min(img)
    if not np.all(img == img[0]):
        img = img / np.max(img)
    return img


def clear(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to normalized numpy array."""
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(x)


def clear_color(x: torch.Tensor, normalize: bool = True) -> np.ndarray:
    """
    Convert color tensor to numpy array for visualization.

    Args:
        x: Input tensor of shape (C, H, W) or (B, C, H, W)
        normalize: Whether to normalize output to [0, 1]

    Returns:
        Numpy array of shape (H, W, C)
    """
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    x = np.transpose(x, (1, 2, 0))
    if normalize:
        return normalize_np(x)
    else:
        return x


def img_to_save(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to uint8 numpy array suitable for saving."""
    x = clear_color(x)
    x = x * 255
    x = x.astype(np.uint8)
    return x


def show_tensor(img: torch.Tensor) -> None:
    """Display a tensor as an image using matplotlib."""
    img = make_grid(img)
    if len(img.shape) < 4:
        img = img.unsqueeze(0)
    plt.imshow(normalize_np(img.permute(0, 2, 3, 1).cpu().detach().numpy()[0]))
    plt.show()


def save_results(sample: torch.Tensor,
                 operator: "BlendOperator",
                 alpha: "AlphaMap",
                 measurement: torch.Tensor,
                 results_path: Union[str, Path]) -> None:
    """
    Save deblending results to disk.

    Args:
        sample: Deblended samples tensor of shape (2, C, H, W)
        operator: Blend operator for creating blended image
        alpha: Alpha map used for blending
        measurement: Ground truth blended image
        results_path: Directory to save results
    """
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    sample1 = sample[0]
    if sample.shape[0] > 1:
        sample2 = sample[1]
    else:
        sample2 = np.zeros_like(clear_color(sample[0]))

    img1_save = Image.fromarray(img_to_save(sample1))
    if sample.shape[0] > 1:
        img2_save = Image.fromarray(img_to_save(sample2))

    blend_save = Image.fromarray(
        img_to_save(operator.forward(sample, alpha=alpha)))
    measure_save = Image.fromarray(img_to_save(measurement))

    img1_path = results_path / 'sample1.png'
    if sample.shape[0] > 1:
        img2_path = results_path / 'sample2.png'
    blend_path = results_path / 'blended_sample.png'
    measure_path = results_path / 'blended_gt.png'

    img1_save.save(img1_path)
    if sample.shape[0] > 1:
        img2_save.save(img2_path)
    blend_save.save(blend_path)
    measure_save.save(measure_path)
    print(f'Saved to: {results_path}')


def prepare_im(load_dir: str,
               image_size: int,
               device: torch.device) -> torch.Tensor:
    """Load and prepare an image for the pipeline."""
    ref_img = torch.from_numpy(
        normalize_np(plt.imread(load_dir)[:, :, :3].astype(np.float32))).to(device)
    ref_img = ref_img.permute(2, 0, 1)
    ref_img = ref_img.view(1, 3, image_size, image_size)
    ref_img = ref_img * 2 - 1
    return ref_img


def unnormalize(img: torch.Tensor, s: float = 0.95) -> torch.Tensor:
    """Unnormalize image tensor."""
    scaling = torch.quantile(img.abs(), s)
    return img / scaling


def normalize(img: torch.Tensor, s: float = 0.95) -> torch.Tensor:
    """Normalize image tensor."""
    scaling = torch.quantile(img.abs(), s)
    return img * scaling


def dynamic_thresholding(img: torch.Tensor, s: float = 0.95) -> torch.Tensor:
    """Apply dynamic thresholding to image tensor."""
    img = normalize(img, s=s)
    return torch.clip(img, -1., 1.)


def total_variation_loss(img: torch.Tensor, weight: float) -> torch.Tensor:
    """Compute total variation loss for an image."""
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2)).mean()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2)).mean()
    return weight * (tv_h + tv_w)
