#!/usr/bin/env python3
"""Select Stage-1 candidates with the paper's frozen PickScore protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

MODEL_ID = "yuvalkirstain/PickScore_v1"
MODEL_REVISION = "a4e4367c6dfa7288a00c550414478f865b875800"
PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PROCESSOR_REVISION = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"

OUTPUT_FIELDS = [
    "sample1_path", "sample2_path",
    "image1_path", "image1_class", "image1_prompt",
    "image2_path", "image2_class", "image2_prompt",
    "smp_seed", "pair", "sample", "assignment_swapped",
    "pickscore_mean",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(root: Path, expected_candidates: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    identities: set[tuple[str, str]] = set()
    for details_path in sorted(root.glob("**/pair_*/smp_*/details.json")):
        candidate = details_path.parent
        pair, sample = candidate.parent.name, candidate.name
        identity = (pair, sample)
        if identity in identities:
            raise RuntimeError(f"Duplicate candidate identity: {pair}/{sample}")
        identities.add(identity)
        for name in ("sample1.png", "sample2.png", "alphas.json"):
            if not (candidate / name).is_file():
                raise FileNotFoundError(candidate / name)
        details = json.loads(details_path.read_text(encoding="utf-8"))
        grouped.setdefault(pair, []).append(
            {"pair": pair, "sample": sample, "path": candidate, "details": details}
        )
    if not grouped:
        raise RuntimeError(f"No Stage-1 candidates found below {root}")
    for pair, candidates in grouped.items():
        candidates.sort(key=lambda item: item["sample"])
        if len(candidates) != expected_candidates:
            raise RuntimeError(
                f"{pair}: expected {expected_candidates} candidates, found {len(candidates)}"
            )
    return grouped


def score_candidates(grouped, device: str, batch_size: int, offline: bool):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        PROCESSOR_ID, revision=PROCESSOR_REVISION, local_files_only=offline
    )
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=offline
    ).eval().to(device)

    pending = [item for candidates in grouped.values() for item in candidates]
    scale = model.logit_scale.exp().detach()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        images = []
        prompts = []
        for item in batch:
            with Image.open(item["path"] / "sample1.png") as image:
                images.append(image.convert("RGB").copy())
            with Image.open(item["path"] / "sample2.png") as image:
                images.append(image.convert("RGB").copy())
            prompts.extend([
                str(item["details"]["image1_prompt"]),
                str(item["details"]["image2_prompt"]),
            ])
        image_inputs = processor(images=images, return_tensors="pt")
        text_inputs = processor(
            text=prompts, padding=True, truncation=True, return_tensors="pt"
        )
        image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.inference_mode():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            matrices = scale * torch.einsum(
                "btd,bid->bti",
                text_features.reshape(-1, 2, text_features.shape[-1]),
                image_features.reshape(-1, 2, image_features.shape[-1]),
            )
        for item, matrix in zip(batch, matrices.detach().cpu().tolist()):
            direct = matrix[0][0] + matrix[1][1]
            swapped = matrix[0][1] + matrix[1][0]
            item["assignment_swapped"] = swapped > direct
            item["pickscore_mean"] = max(direct, swapped) / 2.0
            item["score_matrix"] = matrix
        print(f"Scored {min(offset + len(batch), len(pending))}/{len(pending)}", flush=True)
    return pending


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage1_root", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-candidates", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    root = args.stage1_root.resolve()
    grouped = discover(root, args.expected_candidates)
    scored = score_candidates(grouped, args.device, args.batch_size, args.offline)

    winners = []
    for pair, candidates in sorted(grouped.items()):
        # candidates are sorted by sample ID, so max() preserves the first row on ties.
        winner = max(candidates, key=lambda item: item["pickscore_mean"])
        details, candidate = winner["details"], winner["path"]
        winners.append({
            "sample1_path": str((candidate / "sample1.png").resolve()),
            "sample2_path": str((candidate / "sample2.png").resolve()),
            "image1_path": details["image1_path"],
            "image1_class": details["image1_class"],
            "image1_prompt": details["image1_prompt"],
            "image2_path": details["image2_path"],
            "image2_class": details["image2_class"],
            "image2_prompt": details["image2_prompt"],
            "smp_seed": details["smp_seed"],
            "pair": pair,
            "sample": winner["sample"],
            "assignment_swapped": winner["assignment_swapped"],
            "pickscore_mean": winner["pickscore_mean"],
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(winners)

    score_path = args.output_csv.with_suffix(".scores.json")
    score_rows = [{
        "pair": item["pair"], "sample": item["sample"],
        "candidate_path": str(item["path"].resolve()),
        "assignment_swapped": item["assignment_swapped"],
        "pickscore_mean": item["pickscore_mean"],
        "score_matrix": item["score_matrix"],
    } for item in scored]
    score_path.write_text(json.dumps(score_rows, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "stage1_root": str(root),
        "pairs": len(winners),
        "candidates": len(scored),
        "expected_candidates_per_pair": args.expected_candidates,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "processor_id": PROCESSOR_ID,
        "processor_revision": PROCESSOR_REVISION,
        "selection": "maximum mean logit over both prompt-image assignments",
        "tie_break": "first sample ID",
        "ground_truth_used": False,
        "output_csv_sha256": sha256(args.output_csv),
        "scores_json_sha256": sha256(score_path),
    }
    provenance_path = args.output_csv.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
