#!/usr/bin/env python3
"""Reproduce a paper experiment end to end: sample, select, refine, evaluate, tabulate.

Examples:
  # Main result + two-stage ablation for one scenario, on four GPUs
  python scripts/reproduce.py main --scenario spatial --gpus 0,1,2,3

  # Held-out Vehicles/Flowers domains
  python scripts/reproduce.py main --scenario spatial --split generalization \\
      --data-root /path/to/BID_generalization

  # Print every command without running anything
  python scripts/reproduce.py timestep --dry-run

  # Check the plumbing in a few minutes (1 pair, ~10x fewer steps; numbers are meaningless)
  python scripts/reproduce.py text-guidance --scenario spatial --smoke

Outputs go to <output-root>/<split>/<experiment>/<scenario>/; the final table
is written there as summary.md and summary.csv. Rerunning the same command
resumes an interrupted run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_repro.experiments import EXPERIMENTS, Context, write_protocol  # noqa: E402
from paper_repro.runner import Runner  # noqa: E402
from paper_repro.tables import TABLES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--scenario", help="known_scalar, unknown_scalar or spatial "
                                           "(default: the only/first scenario the experiment supports)")
    parser.add_argument("--split", choices=("comparison", "generalization"), default="comparison")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("DEBLEND_DATA_ROOT"),
                        help="Image root for the chosen split (default: $DEBLEND_DATA_ROOT)")
    parser.add_argument("--output-root", type=Path,
                        default=Path(os.environ.get("DEBLEND_OUTPUT_ROOT", "outputs")))
    parser.add_argument("--gpus", default=os.environ.get("DEBLEND_GPUS", "0"),
                        help="Comma-separated GPU ids; one job runs per id (repeat an id to share a GPU)")
    parser.add_argument("--pairs", type=int, help="Use only the first N pairs of each range")
    parser.add_argument("--smoke", action="store_true",
                        help="1 pair and ~10x fewer steps: verifies the pipeline, not the numbers")
    parser.add_argument("--stage1-root", type=Path,
                        help="selector: existing 3-candidate Stage-1 tree to reuse")
    parser.add_argument("--generate-prompts", action="store_true",
                        help="auto-prompts: rerun Qwen2-VL instead of using the frozen prompts")
    blends = parser.add_mutually_exclusive_group()
    blends.add_argument("--matched-blends", action="store_true", help="Use matched spatial blends (default)")
    blends.add_argument("--legacy-blends", action="store_true", help="Use the older mixed-blend protocol")
    parser.add_argument("--offline", action="store_true", help="Load selector/VLM weights from cache only")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands and exit")
    parser.add_argument("--no-table", action="store_true")
    args = parser.parse_args()

    builder, scenarios = EXPERIMENTS[args.experiment]
    scenario = args.scenario or scenarios[0]
    if scenario not in scenarios:
        parser.error(f"{args.experiment} supports --scenario {', '.join(scenarios)}")
    if args.data_root is None:
        parser.error("Set --data-root or DEBLEND_DATA_ROOT to the dataset image root")
    if args.split == "generalization" and args.experiment != "main":
        parser.error("Only the main experiment is evaluated on the generalization split")

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    tag = "smoke" if args.smoke else args.split
    out = (args.output_root / tag / args.experiment / scenario).resolve()
    ctx = Context(out=out, data_root=args.data_root.resolve(), split=args.split, slots=len(gpus),
                  smoke=args.smoke, pairs=1 if args.smoke else args.pairs, offline=args.offline,
                  frozen_prompts=not args.generate_prompts,
                  extra={"stage1_root": args.stage1_root.resolve() if args.stage1_root else None,
                         "legacy_blends": args.legacy_blends})
    identity = {"experiment": args.experiment, "scenario": scenario, "split": args.split,
                "data_root": str(ctx.data_root), "pairs": ctx.pairs, "smoke": ctx.smoke,
                "stage1_root": str(ctx.extra["stage1_root"]) if ctx.extra["stage1_root"] else None,
                "generate_prompts": args.generate_prompts, "legacy_blends": args.legacy_blends}
    identity_path = out / "run_settings.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            parser.error("Run settings differ from the existing output; use a new --output-root")
    elif out.exists() and any(out.iterdir()):
        parser.error("Existing output has no run settings; use a new --output-root")
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    phases = builder(ctx, scenario)
    if not args.dry_run:
        write_protocol(ctx, args.experiment, scenario, phases)
    print(f"{args.experiment} / {scenario} -> {out}")

    runner = Runner(out, gpus, dry_run=args.dry_run)
    for title, jobs in phases:
        runner.run_phase(title, jobs)
    if not (args.dry_run or args.no_table):
        print()
        TABLES[args.experiment](out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
