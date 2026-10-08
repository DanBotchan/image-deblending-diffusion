from __future__ import annotations

import math

from .io import group_by_pair, read_rows, resolve_artifact_path
from .models import ExperimentSpec, Issue, ProtocolResult


def _number(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _selection_score(row: dict[str, str], field: str) -> float | None:
    direct = _number(row, field)
    if direct is not None:
        return direct
    if field == "avg_clip_score":
        first = _number(row, "sample1_clip_score")
        second = _number(row, "sample2_clip_score")
        if first is not None and second is not None:
            return (first + second) / 2.0
    return None


def _validate_pair_count(spec: ExperimentSpec, pairs: set[str], issues: list[Issue]) -> None:
    if len(pairs) != spec.expected_pairs:
        issues.append(
            Issue(
                "error",
                "pair_count_mismatch",
                spec.name,
                f"Expected {spec.expected_pairs} pairs, found {len(pairs)}",
            )
        )


def _validate_paths(spec: ExperimentSpec, rows: list[dict[str, str]], issues: list[Issue]) -> None:
    path_fields = ("sample1_path", "sample2_path", "gt1_path", "gt2_path")
    for row in rows:
        for field in path_fields:
            raw = row.get(field, "")
            if not raw:
                issues.append(Issue("error", "missing_path", spec.name, f"Missing {field}", row["pair"]))
                continue
            path = resolve_artifact_path(spec.project_root, raw)
            if not path.exists():
                issues.append(
                    Issue("error", "image_not_found", spec.name, f"{field} does not exist: {path}", row["pair"])
                )


def evaluate_protocol(spec: ExperimentSpec, check_paths: bool = False) -> ProtocolResult:
    if spec.protocol == "single":
        return _single(spec, check_paths)
    if spec.protocol == "select_then_refine":
        return _select_then_refine(spec, check_paths)
    raise ValueError(f"Unsupported protocol: {spec.protocol}")


def _single(spec: ExperimentSpec, check_paths: bool) -> ProtocolResult:
    result = ProtocolResult(spec=spec)
    if spec.evaluation_csv is None:
        result.issues.append(Issue("error", "manifest_error", spec.name, "single requires evaluation_csv"))
        return result
    rows, issues = read_rows(spec.evaluation_csv, spec.name)
    result.input_files.append(spec.evaluation_csv)
    result.issues.extend(issues)
    grouped = group_by_pair(rows)
    _validate_pair_count(spec, set(grouped), result.issues)
    for pair, candidates in grouped.items():
        if len(candidates) != 1:
            result.issues.append(
                Issue("error", "unexpected_candidate_count", spec.name, f"Expected 1 row, found {len(candidates)}", pair)
            )
        else:
            result.canonical_rows.append(candidates[0])
    if check_paths:
        _validate_paths(spec, result.canonical_rows, result.issues)
    return result


def _select_then_refine(spec: ExperimentSpec, check_paths: bool) -> ProtocolResult:
    result = ProtocolResult(spec=spec)
    if spec.stage1_csv is None or spec.stage2_csv is None:
        result.issues.append(
            Issue("error", "manifest_error", spec.name, "select_then_refine requires stage1_csv and stage2_csv")
        )
        return result

    stage1, issues1 = read_rows(spec.stage1_csv, spec.name)
    stage2, issues2 = read_rows(spec.stage2_csv, spec.name)
    result.stage1_rows = stage1
    result.stage2_rows = stage2
    result.input_files.extend((spec.stage1_csv, spec.stage2_csv))
    result.issues.extend(issues1 + issues2)
    first = group_by_pair(stage1)
    second = group_by_pair(stage2)
    _validate_pair_count(spec, set(first), result.issues)
    _validate_pair_count(spec, set(second), result.issues)
    if set(first) != set(second):
        missing_stage2 = sorted(set(first) - set(second))
        extra_stage2 = sorted(set(second) - set(first))
        result.issues.append(
            Issue(
                "error",
                "stage_pair_mismatch",
                spec.name,
                f"Missing in Stage 2: {missing_stage2[:5]}; extra in Stage 2: {extra_stage2[:5]}",
            )
        )

    for pair in sorted(set(first) & set(second)):
        candidates = first[pair]
        if len(candidates) != spec.expected_stage1_candidates:
            result.issues.append(
                Issue(
                    "error",
                    "stage1_candidate_count",
                    spec.name,
                    f"Expected {spec.expected_stage1_candidates}, found {len(candidates)}",
                    pair,
                )
            )
            continue
        scored = [(row, _selection_score(row, spec.selection_score)) for row in candidates]
        if any(score is None for _, score in scored):
            result.issues.append(
                Issue("error", "missing_selection_score", spec.name, f"Missing {spec.selection_score}", pair)
            )
            continue
        best_score = max(score for _, score in scored if score is not None)
        winners = [row for row, score in scored if score == best_score]
        if len(winners) != 1:
            if spec.tie_break == "error":
                result.issues.append(
                    Issue(
                        "error",
                        "selection_tie",
                        spec.name,
                        f"{len(winners)} candidates tie at {best_score}; selection is not auditable",
                        pair,
                    )
                )
                continue
            result.issues.append(
                Issue(
                    "warning",
                    "selection_tie_resolved",
                    spec.name,
                    f"{len(winners)} candidates tie at {best_score}; chose first CSV row",
                    pair,
                )
            )
        selected = winners[0]
        selected_id = selected["sample"]
        result.selected_samples[pair] = selected_id

        refined = {row["sample"]: row for row in second[pair]}
        if len(refined) not in (1, spec.expected_stage1_candidates):
            result.issues.append(
                Issue(
                    "error",
                    "stage2_candidate_count",
                    spec.name,
                    f"Expected 1 or {spec.expected_stage1_candidates}, found {len(refined)}",
                    pair,
                )
            )
        if selected_id not in refined:
            result.issues.append(
                Issue(
                    "error",
                    "selected_refinement_missing",
                    spec.name,
                    f"Stage 1 selected {selected_id}, but Stage 2 contains {sorted(refined)}",
                    pair,
                )
            )
            continue
        chosen = refined[selected_id]
        if selected.get("smp_seed") and chosen.get("smp_seed") and selected["smp_seed"] != chosen["smp_seed"]:
            result.issues.append(
                Issue(
                    "error",
                    "seed_lineage_mismatch",
                    spec.name,
                    f"{selected_id} seed differs between stages",
                    pair,
                )
            )
        result.canonical_rows.append(chosen)

    if check_paths:
        _validate_paths(spec, result.canonical_rows, result.issues)
    return result
