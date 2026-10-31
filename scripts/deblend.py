#!/usr/bin/env python3
"""Deblend a real image using two prompts, without source images or ground truth."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--prompt1", required=True)
    parser.add_argument("--prompt2", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/custom"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="Reduced steps for checking setup; poor image quality")
    parser.add_argument("--offline", action="store_true", help="Use cached selector weights only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if args.candidates < 1:
        parser.error("--candidates must be positive")
    from PIL import Image
    with Image.open(args.image) as image:
        image.verify()
    image_path = args.image.resolve()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory is not empty; choose a new --output directory")
    if args.dry_run:
        print(f"Input: {image_path}\nOutput: {out}\nStage 1 -> PickScore selection -> Stage 2")
        return
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "input.csv"
    # The existing loaders require two image paths. These are used only for
    # shape/metadata; both stages read the measurement directly and skip metrics.
    row = {"img1_path": str(image_path), "img2_path": str(image_path),
           "img1_class": "custom", "img2_class": "custom",
           "img1_prompt": args.prompt1, "img2_prompt": args.prompt2,
           "a": 0.5, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0,
           "z_seeds": str([args.seed + i for i in range(args.candidates)])}
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (out / "input.json").write_text(json.dumps({"image": str(image_path),
        "prompt1": args.prompt1, "prompt2": args.prompt2, "seed": args.seed,
        "candidates": args.candidates, "smoke": args.smoke,
        "ground_truth_available": False}, indent=2) + "\n")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, USE_TF="0", USE_FLAX="0")
    selected = out / "selected.csv"
    def override(key: str, value: Path) -> str:
        return f"{key}={json.dumps(str(value))}"

    common = [override("+general.input_image", image_path), "general.N_pairs=1", "datasets.dataset_path=/"]
    stages = [
        ["scripts/first_stage.py", "+paper=spatial_stage1", *common,
         override("general.output_dir", out / "stage1"), override("datasets.annotations_path", manifest),
         f"general.N_samples_for_pair={args.candidates}"],
        ["scripts/select_candidates.py", str(out / "stage1/results"), "--output-csv", str(selected),
         "--expected-candidates", str(args.candidates), *(["--offline"] if args.offline else [])],
        ["scripts/second_stage.py", "+paper=spatial_stage2", *common,
         override("general.output_dir", out / "stage2"), override("datasets.annotations_path", selected)],
    ]
    if args.smoke:
        from paper_repro.runner import smoke_overrides
        for index in (0, 2):
            steps, start = (100, 80) if index == 0 else (600, 350)
            stages[index].extend(smoke_overrides([f"sampler.timestep_respacing={steps}",
                f"sampler.start_idx={start}", "general.n_recurrence=8"]))
    for command in stages:
        subprocess.run([sys.executable, *command], cwd=REPO, env=env, check=True)
    print(f"Recovered images and alpha map: {out / 'stage2/results/pair_000'}")


if __name__ == "__main__":
    main()
