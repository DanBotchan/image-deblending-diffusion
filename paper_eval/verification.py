from __future__ import annotations

import math
from typing import Any

from .io import resolve_artifact_path
from .models import ExperimentSpec, Issue


DEFAULT_TOLERANCE = {
    "psnr": 1e-5,
    "ssim": 1e-8,
    "lpips": 1e-5,
}
_LPIPS_CACHE: tuple[Any, Any, Any, Any] | None = None


def _lpips_components():
    global _LPIPS_CACHE
    if _LPIPS_CACHE is None:
        import lpips as lpips_lib
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = lpips_lib.LPIPS(net="alex").to(device).eval()
        _LPIPS_CACHE = model, lpips_lib, torch, device
    return _LPIPS_CACHE


def verify_stored_metrics(
    spec: ExperimentSpec,
    rows: list[dict[str, str]],
    include_lpips: bool = False,
    tolerance: dict[str, float] | None = None,
) -> tuple[dict[str, Any], list[Issue]]:
    """Recompute canonical image metrics independently from stored CSV values.

    Only rows admitted by the protocol evaluator are verified. Prediction order
    follows the historical order_reassigned flag.
    """
    import cv2
    import numpy as np
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    limits = {**DEFAULT_TOLERANCE, **(tolerance or {})}
    issues: list[Issue] = []
    deltas: dict[str, list[float]] = {"psnr": [], "ssim": [], "lpips": []}
    checked_rows = 0
    lpips_model = None
    lpips_lib = None
    torch = None
    device = None
    if include_lpips:
        lpips_model, lpips_lib, torch, device = _lpips_components()

    def load(raw: str, resolution: int):
        path = resolve_artifact_path(spec.project_root, raw)
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            target = (resolution, resolution)
            if image.size != target:
                image = image.resize(target, Image.Resampling.LANCZOS)
            return np.asarray(image)

    def compare(pair: str, source: str, metric: str, actual: float, stored_raw: str) -> None:
        try:
            stored = float(stored_raw)
        except (TypeError, ValueError):
            issues.append(
                Issue("error", "missing_metric", spec.name, f"{source}_{metric} is not numeric", pair)
            )
            return
        delta = abs(actual - stored)
        deltas[metric].append(delta)
        if not math.isfinite(stored) or delta > limits[metric]:
            issues.append(
                Issue(
                    "error",
                    "metric_recompute_mismatch",
                    spec.name,
                    f"{source}_{metric}: stored={stored:.12g}, recomputed={actual:.12g}, delta={delta:.3g}",
                    pair,
                )
            )

    for row in rows:
        pair = row["pair"]
        try:
            resolution = int(float(row.get("target_resolution") or 256))
            prediction = [
                load(row["sample1_path"], resolution),
                load(row["sample2_path"], resolution),
            ]
            target = [
                load(row["gt1_path"], resolution),
                load(row["gt2_path"], resolution),
            ]
        except (KeyError, OSError, ValueError) as exc:
            issues.append(Issue("error", "metric_image_load_failed", spec.name, str(exc), pair))
            continue
        for index, (pred, gt) in enumerate(zip(prediction, target), start=1):
            source = f"sample{index}"
            psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
            pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
            gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
            ssim = float(structural_similarity(gt_gray, pred_gray, data_range=255))
            compare(pair, source, "psnr", psnr, row.get(f"{source}_psnr", ""))
            compare(pair, source, "ssim", ssim, row.get(f"{source}_ssim", ""))
            if include_lpips and lpips_model is not None and lpips_lib is not None and torch is not None:
                pred_tensor = lpips_lib.im2tensor(pred).to(device)
                gt_tensor = lpips_lib.im2tensor(gt).to(device)
                with torch.no_grad():
                    value = float(lpips_model(pred_tensor, gt_tensor).item())
                compare(pair, source, "lpips", value, row.get(f"{source}_lpips", ""))
        checked_rows += 1

    metrics = ("psnr", "ssim", "lpips") if include_lpips else ("psnr", "ssim")
    return {
        "status": "verified" if not issues else "failed",
        "checked_rows": checked_rows,
        "checked_metrics": list(metrics),
        "max_absolute_delta": {
            metric: max(deltas[metric], default=None) for metric in metrics
        },
        "tolerance": {metric: limits[metric] for metric in metrics},
    }, issues
