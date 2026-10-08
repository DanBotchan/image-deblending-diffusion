#!/usr/bin/env python3
"""Build a minimal, auditable evaluation CSV from completed artifact directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.aggregate import aggregate_metrics
from paper_eval.artifact_tree import discover_candidates, evaluate_artifact_candidates
from paper_eval.io import write_csv, write_json
from paper_eval.models import Issue


def tree_fingerprint(candidates: dict[tuple[str, str], Path]) -> str:
    digest = hashlib.sha256()
    for key, candidate in sorted(candidates.items()):
        for name in ("details.json", "sample1.png", "sample2.png"):
            path = candidate / name
            digest.update("/".join((*key, name)).encode())
            if not path.is_file():
                digest.update(b"<missing>")
                continue
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def pinned_candidates(manifest: Path) -> tuple[dict[tuple[str, str], Path], list[Issue]]:
    """Candidate directories listed in a selection manifest, keyed by pair/sample."""
    found: dict[tuple[str, str], Path] = {}
    issues: list[Issue] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate = Path(row["sample1_path"]).parent
            key = (row["pair"], row["sample"])
            if (candidate.parent.name, candidate.name) != key:
                issues.append(Issue("error", "manifest_identity_mismatch", manifest.name,
                                    f"{candidate} does not encode {key}", key[0]))
            found[key] = candidate
    return found, issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--candidates-csv", type=Path,
                        help="Evaluate only the candidate directories named by this "
                             "selection manifest (e.g. the selected Stage-1 outputs).")
    parser.add_argument("--candidates-per-pair", type=int, default=1,
                        help="Required candidates per pair; 0 accepts any number.")
    args = parser.parse_args()

    if args.candidates_csv:
        candidates, discovery_issues = pinned_candidates(args.candidates_csv)
    else:
        candidates, discovery_issues = discover_candidates(args.artifact_root)
    rows, issues = evaluate_artifact_candidates(
        candidates,
        args.project_root,
        args.experiment,
        resolution=args.resolution,
        include_lpips=not args.no_lpips,
    )
    issues = discovery_issues + issues
    pairs = {row["pair"] for row in rows}
    if len(pairs) != args.expected_pairs:
        issues.append(
            Issue(
                "error",
                "pair_count_mismatch",
                args.experiment,
                f"Expected {args.expected_pairs} pairs, found {len(pairs)}",
            )
        )
    counts = {}
    for row in rows:
        counts[row["pair"]] = counts.get(row["pair"], 0) + 1
    for pair, count in counts.items():
        if args.candidates_per_pair and count != args.candidates_per_pair:
            issues.append(
                Issue("error", "unexpected_candidate_count", args.experiment,
                      f"Expected {args.candidates_per_pair}, found {count}", pair)
            )

    metrics, metric_issues = aggregate_metrics(args.experiment, rows)
    issues.extend(metric_issues)
    fields = [
        "model", "pair", "sample", "target_resolution",
        "order_reassignment_enabled", "order_reassigned",
        "sample1_path", "sample2_path", "gt1_path", "gt2_path",
        "sample_time", "smp_seed",
        "sample1_psnr", "sample1_ssim", "sample1_lpips",
        "sample2_psnr", "sample2_ssim", "sample2_lpips",
        "avg_psnr", "avg_ssim", "avg_lpips",
    ]
    write_csv(args.output_csv, rows, fields)
    write_json(
        args.output_json,
        {
            "experiment": args.experiment,
            "artifact_root": str(args.artifact_root.resolve()),
            "artifact_tree_sha256": tree_fingerprint(candidates),
            "metrics": metrics,
            "issues": [issue.as_dict() for issue in issues],
        },
    )
    errors = sum(issue.severity == "error" for issue in issues)
    print(f"Evaluated {len(rows)} rows from {len(pairs)} pairs: {errors} errors")
    print(f"CSV: {args.output_csv.resolve()}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
