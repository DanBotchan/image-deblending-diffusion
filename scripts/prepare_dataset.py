#!/usr/bin/env python3
"""Copy the exact source pairs needed by the release from a prepared BID dataset."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Prepared dataset containing pairs/pair_XXX/img1.png and img2.png")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("comparison", "generalization"), default="comparison")
    parser.add_argument("--check-only", action="store_true", help="Validate the source without copying")
    args = parser.parse_args()
    from PIL import Image
    manifests = sorted((REPO / "manifests" / args.split).glob("*.csv"))
    paths = set()
    for manifest in manifests:
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                paths.update((row["img1_path"], row["img2_path"]))
    missing = [path for path in sorted(paths) if not (args.source_root / path).is_file()]
    if missing:
        parser.error(f"Missing {len(missing)} source images; first missing: {args.source_root / missing[0]}")
    for relative in sorted(paths):
        source = args.source_root / relative
        with Image.open(source) as image:
            image.verify()
        if not args.check_only:
            target = args.output_root / relative
            if source.resolve() != target.resolve():
                if target.exists():
                    if source.read_bytes() != target.read_bytes():
                        parser.error(f"Refusing to replace a different image: {target}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    print(f"Validated {len(paths)} source images for {args.split}.")
    if not args.check_only:
        print(f"Dataset ready: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
