from __future__ import annotations

import statistics
from typing import Any

from .io import finite_number, group_by_pair
from .models import Issue, ProtocolResult


def _time(row: dict[str, str]) -> float | None:
    value = finite_number(row, "sample_time")
    return value if value is not None and value > 0 else None


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_seconds": statistics.fmean(ordered),
        "median_seconds": statistics.median(ordered),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
    }


def summarize_timing(result: ProtocolResult) -> tuple[dict[str, Any], list[Issue]]:
    spec = result.spec
    issues: list[Issue] = []
    if spec.protocol == "single":
        values = [_time(row) for row in result.canonical_rows]
        valid = [value for value in values if value is not None]
        if not valid:
            issues.append(
                Issue(
                    "warning",
                    "timing_unavailable",
                    spec.name,
                    "No positive sample_time values; zeros are treated as missing placeholders",
                )
            )
            return {"status": "unavailable", "valid_rows": 0}, issues
        if len(valid) != len(values):
            issues.append(
                Issue(
                    "warning",
                    "timing_partial",
                    spec.name,
                    f"Positive timing available for {len(valid)}/{len(values)} rows",
                )
            )
        status = "available"
        if statistics.median(valid) < 1.0:
            status = "suspect"
            issues.append(
                Issue(
                    "warning",
                    "timing_implausibly_low",
                    spec.name,
                    "Median sample_time is below one second and is not treated as paper-ready runtime",
                )
            )
        return {"status": status, "valid_rows": len(valid), **_stats(valid)}, issues

    first = group_by_pair(result.stage1_rows)
    second = group_by_pair(result.stage2_rows)
    canonical: list[float] = []
    diagnostic: list[float] = []
    stage1_totals: list[float] = []
    selected_stage2: list[float] = []
    for pair, selected_id in result.selected_samples.items():
        stage1_times = [_time(row) for row in first.get(pair, [])]
        stage2_times = [_time(row) for row in second.get(pair, [])]
        selected_rows = [row for row in second.get(pair, []) if row["sample"] == selected_id]
        selected_time = _time(selected_rows[0]) if selected_rows else None
        if any(value is None for value in stage1_times) or selected_time is None:
            continue
        stage1_total = sum(value for value in stage1_times if value is not None)
        canonical.append(stage1_total + selected_time)
        stage1_totals.append(stage1_total)
        selected_stage2.append(selected_time)
        if (
            len(stage2_times) == spec.expected_stage1_candidates
            and all(value is not None for value in stage2_times)
        ):
            diagnostic.append(stage1_total + sum(value for value in stage2_times if value is not None))
    if not canonical:
        issues.append(Issue("warning", "timing_unavailable", spec.name, "No complete canonical timing lineage"))
        return {"status": "unavailable", "valid_pairs": 0}, issues
    if len(canonical) != len(result.selected_samples):
        issues.append(
            Issue(
                "warning",
                "timing_partial",
                spec.name,
                f"Complete canonical timing for {len(canonical)}/{len(result.selected_samples)} pairs",
            )
        )
    issues.append(
        Issue(
            "warning",
            "selection_timing_unavailable",
            spec.name,
            "Canonical sampling time excludes CLIP selection overhead, which is not recorded",
        )
    )
    summary: dict[str, Any] = {
        "status": "available",
        "valid_pairs": len(canonical),
        "canonical_sampling": _stats(canonical),
        "selection_overhead": {"status": "unavailable"},
        "stage1_three_candidates": _stats(stage1_totals),
        "selected_stage2": _stats(selected_stage2),
    }
    if diagnostic:
        summary["diagnostic_all_stage2_candidates"] = {
            "valid_pairs": len(diagnostic),
            **_stats(diagnostic),
        }
    return summary, issues
