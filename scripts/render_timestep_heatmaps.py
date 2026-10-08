#!/usr/bin/env python3
"""Render the supplementary starting-timestep heatmaps from a timestep summary.

Input is the ``summary.csv`` written by ``scripts/reproduce.py timestep``
(columns stage1, stage2, psnr, ssim, lpips; "stage1 only" rows are ignored).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

METRICS = ("psnr", "ssim", "lpips")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stage2"] != "stage1 only"]
    values = {(int(r["stage1"]), int(r["stage2"])): {m: float(r[m]) for m in METRICS} for r in rows}
    stage1 = sorted({s1 for s1, _ in values})
    stage2 = sorted({s2 for _, s2 in values})
    missing = [(a, b) for a in stage1 for b in stage2 if (a, b) not in values]
    if missing:
        raise SystemExit(f"Incomplete grid, missing {missing}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        matrix = np.array([[values[a, b][metric] for b in stage2] for a in stage1])
        best = np.unravel_index(matrix.argmin() if metric == "lpips" else matrix.argmax(), matrix.shape)
        title = {"psnr": "PSNR ↑", "ssim": "SSIM ↑", "lpips": "LPIPS ↓"}[metric]
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(matrix, annot=True, fmt=".2f" if metric == "psnr" else ".4f",
                    cmap="RdYlBu" if metric == "lpips" else "RdYlBu_r",
                    xticklabels=stage2, yticklabels=stage1,
                    cbar_kws={"label": metric.upper()}, linewidths=0.5, linecolor="gray",
                    ax=ax, annot_kws={"size": 11})
        ax.add_patch(plt.Rectangle((best[1], best[0]), 1, 1, fill=False, edgecolor="black", lw=4))
        ax.text(best[1] + 0.5, best[0] + 0.15, "★", ha="center", va="center", fontsize=20,
                color="black", weight="bold")
        ax.set_title(title, fontsize=16, weight="bold", pad=15)
        ax.set_xlabel("Stage 2: Resampling (steps/600)", fontsize=14, weight="bold")
        ax.set_ylabel("Stage 1: Noising (steps/100)", fontsize=14, weight="bold")
        ax.tick_params(axis="both", labelrotation=0)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"heatmap_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote heatmaps to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
