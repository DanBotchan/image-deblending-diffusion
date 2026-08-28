from __future__ import annotations

import math
import statistics
from typing import Any

from .models import Issue


METRICS = ("psnr", "ssim", "lpips")


def _values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def aggregate_metrics(
    experiment: str, rows: list[dict[str, str]]
) -> tuple[dict[str, Any], list[Issue]]:
    summary: dict[str, Any] = {"canonical_row_count": len(rows)}
    issues: list[Issue] = []
    for source in ("sample1", "sample2"):
        for metric in METRICS:
            field = f"{source}_{metric}"
            values = _values(rows, field)
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
