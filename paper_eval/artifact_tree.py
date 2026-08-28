"""Evaluate completed pair/sample artifact trees without rerunning sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import resolve_artifact_path
from .models import Issue


_LPIPS_COMPONENTS: tuple[Any, Any, Any, Any] | None = None


def _lpips_components() -> tuple[Any, Any, Any, Any]:
    global _LPIPS_COMPONENTS
    if _LPIPS_COMPONENTS is None:
        import lpips
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = lpips.LPIPS(net="alex").to(device).eval()
        _LPIPS_COMPONENTS = model, lpips, torch, device
    return _LPIPS_COMPONENTS


def discover_candidates(root: Path) -> tuple[dict[tuple[str, str], Path], list[Issue]]:
    """Find pair/sample leaves in either root/pair_* or root/results/pair_*."""
    found: dict[tuple[str, str], Path] = {}
    issues: list[Issue] = []
    for base in (root, root / "results"):
        for candidate in sorted(base.glob("pair_*/smp_*")):
            if not candidate.is_dir():
                continue
            key = (candidate.parent.name, candidate.name)
            if key in found:
                issues.append(
                    Issue(
                        "error",
                        "duplicate_artifact_candidate",
                        root.name,
                        f"Candidate exists at both {found[key]} and {candidate}",
                        key[0],
                    )
                )
            else:
                found[key] = candidate
    return found, issues


def _scalar_path(value: Any) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected one path, found {len(value)}")
        return str(value[0])
    return str(value)


def evaluate_artifact_tree(
    root: Path,
    project_root: Path,
    experiment: str,
    resolution: int = 256,
    include_lpips: bool = True,
) -> tuple[list[dict[str, Any]], list[Issue]]:
    """Compute historical reassigned PSNR/SSIM/LPIPS rows from raw artifacts."""
    candidates, issues = discover_candidates(root)
    rows, evaluation_issues = evaluate_artifact_candidates(
        candidates,
        project_root,
        experiment,
        resolution=resolution,
        include_lpips=include_lpips,
    )
    return rows, [*issues, *evaluation_issues]


def evaluate_artifact_candidates(
    candidates: dict[tuple[str, str], Path],
    project_root: Path,
    experiment: str,
    resolution: int = 256,
    include_lpips: bool = True,
) -> tuple[list[dict[str, Any]], list[Issue]]:
    """Evaluate an explicitly pinned set of pair/sample artifact directories."""
    import cv2
    import numpy as np
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    issues: list[Issue] = []
    rows: list[dict[str, Any]] = []

    def load(path: Path) -> Any:
        with Image.open(path) as image:
            image = image.convert("RGB")
            target = (resolution, resolution)
            if image.size != target:
                image = image.resize(target, Image.Resampling.LANCZOS)
            return np.asarray(image)

    def metrics(prediction: Any, target: Any) -> dict[str, float]:
        result = {
            "psnr": float(peak_signal_noise_ratio(target, prediction, data_range=255)),
            "ssim": float(
                structural_similarity(
                    cv2.cvtColor(target, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY),
                    data_range=255,
                )
            ),
        }
        if include_lpips:
            model, lpips, torch, device = _lpips_components()
            with torch.no_grad():
                result["lpips"] = float(
                    model(
                        lpips.im2tensor(prediction).to(device),
                        lpips.im2tensor(target).to(device),
                    ).item()
                )
        return result

    for (pair, sample), candidate in sorted(candidates.items()):
        required = ("details.json", "sample1.png", "sample2.png")
        missing = [name for name in required if not (candidate / name).is_file()]
        if missing:
            issues.append(
                Issue("error", "missing_artifact_file", experiment, f"Missing {missing} in {candidate}", pair)
            )
            continue
        try:
            details = json.loads((candidate / "details.json").read_text(encoding="utf-8"))
            gt_paths = [
                resolve_artifact_path(project_root, _scalar_path(details["image1_path"])),
                resolve_artifact_path(project_root, _scalar_path(details["image2_path"])),
            ]
            prediction_paths = [candidate / "sample1.png", candidate / "sample2.png"]
            predictions = [load(path) for path in prediction_paths]
            targets = [load(path) for path in gt_paths]
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(Issue("error", "artifact_load_failed", experiment, str(exc), pair))
            continue

        direct_mse = float(np.mean((predictions[0].astype(float) - targets[0]) ** 2)) + float(
            np.mean((predictions[1].astype(float) - targets[1]) ** 2)
        )
        swapped_mse = float(np.mean((predictions[0].astype(float) - targets[1]) ** 2)) + float(
            np.mean((predictions[1].astype(float) - targets[0]) ** 2)
        )
        swapped = swapped_mse < direct_mse
        if swapped:
            predictions.reverse()
            prediction_paths.reverse()

        first = metrics(predictions[0], targets[0])
        second = metrics(predictions[1], targets[1])
        row: dict[str, Any] = {
            "model": experiment,
            "pair": pair,
            "sample": sample,
            "target_resolution": resolution,
            "order_reassignment_enabled": True,
            "order_reassigned": swapped,
            "sample1_path": str(prediction_paths[0].resolve()),
            "sample2_path": str(prediction_paths[1].resolve()),
            "gt1_path": str(gt_paths[0].resolve()),
            "gt2_path": str(gt_paths[1].resolve()),
            "sample_time": details.get("sample_time", ""),
            "smp_seed": details.get("smp_seed", ""),
        }
        for source, values in (("sample1", first), ("sample2", second)):
            for metric, value in values.items():
                row[f"{source}_{metric}"] = value
        for metric in ("psnr", "ssim", "lpips"):
            if metric in first and metric in second:
                row[f"avg_{metric}"] = (first[metric] + second[metric]) / 2.0
        rows.append(row)
    return rows, issues
