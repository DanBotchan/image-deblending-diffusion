"""
Shared loss computation utilities for conditioning methods.

Centralizes multi-scale loss, measurement loss, and gradient computation
to eliminate duplication across LADiBI, PSLD, and CLIP conditioning classes.
"""
import torch
import torch.nn.functional as F


class MultiScaleLossMixin:
    """Mixin for multi-scale measurement loss computation.

    Computes blending loss at multiple resolutions with optional LPIPS.
    Downsampling targets are computed without gradient tracking for efficiency.

    Class attributes (override in subclass if needed):
        scales: list of resolution scales (1.0 = full resolution)
        scale_weights: per-scale loss weights
    """
    scales = [1.0, 0.5, 0.25]
    scale_weights = [0.25, 0.5, 1.0]

    def compute_multiscale_loss(self,
                                pred: torch.Tensor,
                                target: torch.Tensor,
                                lpips_scale: float = 0.0,
                                step_idx: int = 0,
                                ) -> torch.Tensor:
        """Compute multi-scale L2 + optional LPIPS loss.

        Args:
            pred: Predicted blended image.
            target: Target measurement image.
            lpips_scale: Weight for LPIPS. Skipped entirely when 0.
            step_idx: Current step index (for LPIPS frequency gating).

        Returns:
            Scalar loss tensor with gradients.
        """
        lpips_freq = getattr(self, 'lpips_freq', 1)
        use_lpips = lpips_scale > 0 and (step_idx % lpips_freq == 0)

        loss = torch.tensor(0.0, device=pred.device)
        current_pred = pred
        current_target = target

        for scale, weight in zip(self.scales, self.scale_weights):
            if scale != 1.0:
                with torch.no_grad():
                    current_target = F.avg_pool2d(current_target, 2)
                current_pred = F.avg_pool2d(current_pred, 2)

            norm_diff = torch.linalg.norm(current_pred - current_target)
            if use_lpips:
                norm_diff = norm_diff + lpips_scale * self.lpips(current_pred, current_target)
            loss = loss + weight * norm_diff

        return loss


def compute_measurement_loss(pred: torch.Tensor,
                             target: torch.Tensor,
                             lpips_fn=None,
                             lpips_scale: float = 0.0,
                             lpips_freq: int = 1,
                             step_idx: int = 0,
                             ) -> torch.Tensor:
    """Compute single-scale measurement loss (L2 + optional LPIPS).

    Args:
        pred: Predicted blended image.
        target: Target measurement image.
        lpips_fn: LPIPS network callable. Ignored when lpips_scale <= 0.
        lpips_scale: Weight for LPIPS term.
        lpips_freq: Compute LPIPS every N steps.
        step_idx: Current step index.

    Returns:
        Scalar loss tensor.
    """
    loss = torch.linalg.norm(target - pred)
    if lpips_scale > 0 and (step_idx % lpips_freq == 0) and lpips_fn is not None:
        loss = loss + lpips_scale * lpips_fn(pred, target)
    return loss
