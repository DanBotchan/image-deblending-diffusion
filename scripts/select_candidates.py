#!/usr/bin/env python3
"""Select Stage-1 candidates without ground truth and write the Stage-2 manifest.

Default: the paper's frozen permutation-invariant PickScore protocol.

Outputs next to ``--output-csv``:
  <name>.csv               Stage-2 manifest (one selected candidate per pair)
  <name>.scores.json       scores for every candidate
  <name>.provenance.json   selector revisions and file hashes
  <name>.budgets.json      (with --budgets) winner per pair at every budget
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_repro.selection import (  # noqa: E402
    SELECTORS, discover, score_candidates, select, selector_provenance,
    sha256, write_manifest, write_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage1_root", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--selector", choices=SELECTORS, default="pickscore")
    parser.add_argument("--expected-candidates", type=int, default=3)
    parser.add_argument("--budgets", type=lambda s: [int(v) for v in s.split(",")],
                        help="Cumulative candidate budgets, e.g. 1,2,3,4,6,8,12,16. The "
                             "manifest then holds every distinct winner across budgets.")
    parser.add_argument("--all", action="store_true",
                        help="Write every candidate to the manifest (selector comparison).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    root = args.stage1_root.resolve()
    grouped = discover(root, args.expected_candidates)
    score_candidates(grouped, args.selector, args.device, args.batch_size, args.offline)

    output = args.output_csv
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.all:
        rows = [item for _, candidates in sorted(grouped.items()) for item in candidates]
    elif args.budgets:
        if max(args.budgets) > args.expected_candidates:
            raise SystemExit("A budget exceeds --expected-candidates")
        by_budget = {
            pair: {str(b): select(candidates, b)["sample"] for b in args.budgets}
            for pair, candidates in sorted(grouped.items())
        }
        output.with_suffix(".budgets.json").write_text(
            json.dumps(by_budget, indent=2) + "\n", encoding="utf-8")
        chosen = {(pair, sample) for pair, choice in by_budget.items() for sample in choice.values()}
        rows = [item for _, candidates in sorted(grouped.items()) for item in candidates
                if (item["pair"], item["sample"]) in chosen]
    else:
        rows = [select(candidates) for _, candidates in sorted(grouped.items())]

    write_manifest(output, rows)
    score_path = output.with_suffix(".scores.json")
    write_scores(score_path, grouped)
    provenance = {
        "stage1_root": str(root),
        "pairs": len(grouped),
        "candidates": sum(len(c) for c in grouped.values()),
        "expected_candidates_per_pair": args.expected_candidates,
        "manifest_rows": len(rows),
        "budgets": args.budgets,
        **selector_provenance(args.selector),
        "ground_truth_used": False,
        "output_csv_sha256": sha256(output),
        "scores_json_sha256": sha256(score_path),
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
