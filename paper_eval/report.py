from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_csv, write_json
from .models import AuditResult


def _flat_experiment(item: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in item.items() if key not in ("metrics", "timing", "verification")}
    row.update(item["metrics"])
    verification = item.get("verification")
    if verification:
        row["verification_status"] = verification["status"]
        row["verified_row_count"] = verification["checked_rows"]
        for metric, delta in verification["max_absolute_delta"].items():
            row[f"{metric}_verification_max_delta"] = delta
    timing = item["timing"]
    if timing.get("status") == "available":
        if "mean_seconds" in timing:
            row["runtime_sampling_mean_seconds"] = timing["mean_seconds"]
            row["runtime_sampling_median_seconds"] = timing["median_seconds"]
        elif "canonical_sampling" in timing:
            row["runtime_sampling_mean_seconds"] = timing["canonical_sampling"]["mean_seconds"]
            row["runtime_sampling_median_seconds"] = timing["canonical_sampling"]["median_seconds"]
            row["diagnostic_runtime_mean_seconds"] = timing.get(
                "diagnostic_all_stage2_candidates", {}
            ).get("mean_seconds")
    return row


def write_reports(result: AuditResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "experiment_count": len(result.experiments),
            "error_count": result.error_count,
            "warning_count": result.warning_count,
        },
        "provenance": result.provenance,
        "experiments": result.experiments,
        "issues": [issue.as_dict() for issue in result.issues],
    }
    write_json(output_dir / "audit.json", payload)

    rows = [_flat_experiment(item) for item in result.experiments]
    fields = sorted({key for row in rows for key in row})
    write_csv(output_dir / "paper_metrics.csv", rows, fields)
    issue_rows = [issue.as_dict() for issue in result.issues]
    write_csv(
        output_dir / "issues.csv",
        issue_rows,
        ["severity", "code", "experiment", "pair", "message"],
    )

    lines = [
        "# Paper result audit",
        "",
        f"- Experiments: {len(result.experiments)}",
        f"- Errors: {result.error_count}",
        f"- Warnings: {result.warning_count}",
        "",
        "## Experiment status",
        "",
        "| Experiment | Protocol | Rows | Errors |",
        "|---|---|---:|---:|",
    ]
    for item in result.experiments:
        lines.append(
            f"| {item['name']} | {item['protocol']} | "
            f"{item['metrics']['canonical_row_count']} | {item['error_count']} |"
        )
    lines.extend(["", "## Issues", ""])
    if not result.issues:
        lines.append("No issues found.")
    else:
        for issue in result.issues:
            pair = f" ({issue.pair})" if issue.pair else ""
            lines.append(
                f"- **{issue.severity.upper()} `{issue.code}`** {issue.experiment}{pair}: {issue.message}"
            )
    (output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
