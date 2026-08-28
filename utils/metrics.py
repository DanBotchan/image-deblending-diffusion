"""
Evaluation metrics for deblending results.

Computes PSNR, SSIM, LPIPS, and MSE between deblended samples and ground truth,
with automatic sample-to-GT assignment (tries both orderings, picks best by MSE).
"""
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips as lpips_lib

_lpips_model = None


def _get_lpips_model(device: torch.device):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips_lib.LPIPS(net='vgg').to(device)
    return _lpips_model


def _to_numpy_01(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] tensor of shape (C, H, W) to numpy (H, W, C) in [0, 1]."""
    img = tensor.detach().cpu().clamp(-1, 1).float()
    img = (img + 1.0) / 2.0
    return img.permute(1, 2, 0).numpy()


def _compute_pairwise_metrics(sample: torch.Tensor,
                               gt: torch.Tensor,
                               device: torch.device) -> dict:
    """Compute metrics between a single sample-GT pair (both shape (C, H, W), range [-1, 1])."""
    smp_np = _to_numpy_01(sample)
    gt_np = _to_numpy_01(gt)

    psnr = float(peak_signal_noise_ratio(gt_np, smp_np, data_range=1.0))
    ssim = float(structural_similarity(gt_np, smp_np, data_range=1.0, channel_axis=2))
    mse = float(np.mean((gt_np - smp_np) ** 2))

    lpips_model = _get_lpips_model(device)
    with torch.no_grad():
        lpips_val = float(lpips_model(
            sample.unsqueeze(0).to(device),
            gt.unsqueeze(0).to(device)
        ).item())

    return {'psnr': psnr, 'ssim': ssim, 'lpips': lpips_val, 'mse': mse}


def compute_metrics(sample: torch.Tensor,
                    ref_img: torch.Tensor,
                    device: torch.device) -> dict:
    """
    Compute evaluation metrics with automatic sample-to-GT assignment.

    Tries both orderings (sample1->gt1, sample2->gt2) and the swap
    (sample1->gt2, sample2->gt1), picks the assignment with lower total MSE.

    Args:
        sample: Deblended output, shape (2, C, H, W), range [-1, 1]
        ref_img: Ground truth images, shape (2, C, H, W), range [-1, 1]
        device: Torch device for LPIPS computation

    Returns:
        Dict with per-image and mean metrics, plus assignment info.
    """
    s1, s2 = sample[0], sample[1]
    g1, g2 = ref_img[0], ref_img[1]

    # Try both assignments
    order_a = {
        'img1': _compute_pairwise_metrics(s1, g1, device),
        'img2': _compute_pairwise_metrics(s2, g2, device),
    }
    order_b = {
        'img1': _compute_pairwise_metrics(s1, g2, device),
        'img2': _compute_pairwise_metrics(s2, g1, device),
    }

    mse_a = order_a['img1']['mse'] + order_a['img2']['mse']
    mse_b = order_b['img1']['mse'] + order_b['img2']['mse']

    if mse_a <= mse_b:
        best = order_a
        swapped = False
    else:
        best = order_b
        swapped = True

    # Compute means
    mean_metrics = {}
    for key in ('psnr', 'ssim', 'lpips', 'mse'):
        mean_metrics[key] = (best['img1'][key] + best['img2'][key]) / 2.0

    return {
        'img1': best['img1'],
        'img2': best['img2'],
        'mean': mean_metrics,
        'swapped': swapped,
    }
