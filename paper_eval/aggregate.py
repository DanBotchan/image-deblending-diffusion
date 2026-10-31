from __future__ import annotations

import statistics
from typing import Any

from .io import finite_number
from .models import Issue


METRICS = ("psnr", "ssim", "lpips")


def aggregate_metrics(
    experiment: str, rows: list[dict[str, str]]
) -> tuple[dict[str, Any], list[Issue]]:
    summary: dict[str, Any] = {"canonical_row_count": len(rows)}
    issues: list[Issue] = []
    for source in ("sample1", "sample2"):
        for metric in METRICS:
            field = f"{source}_{metric}"
            values = [value for row in rows if (value := finite_number(row, field)) is not None]
            if len(values) != len(rows):
                issues.append(
                    Issue(
                        "error",
                        "missing_metric",
                        experiment,
                        f"{field}: {len(values)}/{len(rows)} finite values",
                    )
                )
                continue
            summary[f"{field}_mean"] = statistics.fmean(values)
            summary[f"{field}_stdev"] = statistics.pstdev(values)
    for metric in METRICS:
        first = summary.get(f"sample1_{metric}_mean")
        second = summary.get(f"sample2_{metric}_mean")
        if first is not None and second is not None:
            summary[f"both_sources_{metric}_mean"] = (first + second) / 2.0
    return summary, issues
