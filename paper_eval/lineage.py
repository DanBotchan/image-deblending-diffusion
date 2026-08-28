"""Raw-artifact lineage checks for selected two-stage samples."""

from __future__ import annotations

import json
from pathlib import Path

from .io import resolve_artifact_path
from .models import Issue


def validate_stage_lineage(
    project_root: Path,
    stage1_candidate: Path,
    stage2_candidate: Path,
    experiment: str,
    pair: str,
    sample: str,
) -> list[Issue]:
    """Require same identity, seed, and exact Stage-1 inputs for Stage 2."""
    issues: list[Issue] = []
    for stage, candidate in (("stage1", stage1_candidate), ("stage2", stage2_candidate)):
        if candidate.parent.name != pair or candidate.name != sample:
            issues.append(
                Issue(
                    "error", "stage_identity_mismatch", experiment,
                    f"{stage} path {candidate} does not encode {pair}/{sample}", pair,
                )
            )
        if not candidate.is_dir():
            issues.append(
                Issue("error", "stage_candidate_missing", experiment, f"Missing {stage} directory: {candidate}", pair)
            )
    if issues:
        return issues
    try:
        first = json.loads((stage1_candidate / "details.json").read_text(encoding="utf-8"))
        second = json.loads((stage2_candidate / "details.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [Issue("error", "lineage_metadata_invalid", experiment, str(exc), pair)]
    if str(first.get("smp_seed")) != str(second.get("smp_seed")):
        issues.append(
            Issue("error", "seed_lineage_mismatch", experiment, f"{pair}/{sample} seed differs between stages", pair)
        )
    for source in (1, 2):
        field = f"sample{source}_path"
        if field not in second:
            issues.append(Issue("error", "stage2_input_path_missing", experiment, f"Missing {field}", pair))
            continue
        declared = resolve_artifact_path(project_root, str(second[field])).resolve()
        expected = (stage1_candidate / f"sample{source}.png").resolve()
        if declared != expected:
            issues.append(
                Issue(
                    "error", "stage2_input_lineage_mismatch", experiment,
                    f"{field} is {declared}, expected exact selected Stage-1 output {expected}", pair,
                )
            )
    return issues
