"""Ground-truth-free selection among Stage-1 candidates.

Every selector sees only the two Stage-1 output images of a candidate and the
two input prompts. Because the two diffusion branches are unordered, the
text-image selectors score both prompt-image assignments and keep the better
one before comparing candidates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

PICKSCORE = {
    "model_id": "yuvalkirstain/PickScore_v1",
    "model_revision": "a4e4367c6dfa7288a00c550414478f865b875800",
    "processor_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "processor_revision": "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
    "score": "logit-scaled cosine",
}
CLIP = {
    "model_id": "openai/clip-vit-base-patch32",
    "model_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "processor_id": "openai/clip-vit-base-patch32",
    "processor_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "score": "cosine",
}
SELECTORS = ("pickscore", "clip", "first")

MANIFEST_FIELDS = [
    "sample1_path", "sample2_path",
    "image1_path", "image1_class", "image1_prompt",
    "image2_path", "image2_class", "image2_prompt",
    "smp_seed", "pair", "sample", "assignment_swapped", "score",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(root: Path, expected_candidates: int) -> dict[str, list[dict[str, Any]]]:
    """Group complete Stage-1 candidates by pair, sorted by sample ID."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    identities: set[tuple[str, str]] = set()
    for details_path in sorted(root.glob("**/pair_*/smp_*/details.json")):
        candidate = details_path.parent
        pair, sample = candidate.parent.name, candidate.name
        if (pair, sample) in identities:
            raise RuntimeError(f"Duplicate candidate identity: {pair}/{sample}")
        identities.add((pair, sample))
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


def _load_scorer(selector: str, device: str, offline: bool):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    from transformers import AutoModel, AutoProcessor

    spec = PICKSCORE if selector == "pickscore" else CLIP
    processor = AutoProcessor.from_pretrained(
        spec["processor_id"], revision=spec["processor_revision"], local_files_only=offline
    )
    model = AutoModel.from_pretrained(
        spec["model_id"], revision=spec["model_revision"], local_files_only=offline
    ).eval().to(device)
    return model, processor


def score_candidates(
    grouped: dict[str, list[dict[str, Any]]],
    selector: str,
    device: str = "cuda",
    batch_size: int = 8,
    offline: bool = False,
) -> None:
    """Attach ``score``, ``assignment_swapped`` and ``score_matrix`` to every candidate."""
    pending = [item for candidates in grouped.values() for item in candidates]
    if selector == "first":
        for item in pending:
            item.update(score=0.0, assignment_swapped=False, score_matrix=None)
        return
    if selector not in ("pickscore", "clip"):
        raise ValueError(f"Unknown selector {selector!r}; expected one of {SELECTORS}")

    import torch
    from PIL import Image

    model, processor = _load_scorer(selector, device, offline)
    scale = model.logit_scale.exp().detach() if selector == "pickscore" else 1.0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        images, prompts = [], []
        for item in batch:
            for name in ("sample1.png", "sample2.png"):
                with Image.open(item["path"] / name) as image:
                    images.append(image.convert("RGB").copy())
            prompts.extend([
                str(item["details"]["image1_prompt"]),
                str(item["details"]["image2_prompt"]),
            ])
        image_inputs = processor(images=images, return_tensors="pt")
        text_inputs = processor(
            text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.inference_mode():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # matrices[b, t, i]: prompt t against output image i of candidate b.
            matrices = scale * torch.einsum(
                "btd,bid->bti",
                text_features.reshape(-1, 2, text_features.shape[-1]),
                image_features.reshape(-1, 2, image_features.shape[-1]),
            )
        for item, matrix in zip(batch, matrices.detach().cpu().tolist()):
            direct = matrix[0][0] + matrix[1][1]
            swapped = matrix[0][1] + matrix[1][0]
            item["assignment_swapped"] = swapped > direct
            item["score"] = max(direct, swapped) / 2.0
            item["score_matrix"] = matrix
        print(f"[{selector}] scored {min(offset + len(batch), len(pending))}/{len(pending)}", flush=True)


def select(candidates: Sequence[dict[str, Any]], budget: int | None = None) -> dict[str, Any]:
    """Highest-scoring candidate among the first ``budget``; ties keep the lower sample ID."""
    pool = list(candidates if budget is None else candidates[:budget])
    if not pool:
        raise ValueError("Cannot select from an empty candidate pool")
    return max(pool, key=lambda item: item["score"])


def manifest_row(item: dict[str, Any]) -> dict[str, Any]:
    details, candidate = item["details"], item["path"]
    return {
        "sample1_path": str((candidate / "sample1.png").resolve()),
        "sample2_path": str((candidate / "sample2.png").resolve()),
        "image1_path": details["image1_path"],
        "image1_class": details["image1_class"],
        "image1_prompt": details["image1_prompt"],
        "image2_path": details["image2_path"],
        "image2_class": details["image2_class"],
        "image2_prompt": details["image2_prompt"],
        "smp_seed": details["smp_seed"],
        "pair": item["pair"],
        "sample": item["sample"],
        "assignment_swapped": item.get("assignment_swapped", False),
        "score": item.get("score", ""),
    }


def write_manifest(path: Path, items: Iterable[dict[str, Any]]) -> None:
    """Write a Stage-2 input manifest (one row per Stage-1 candidate to refine)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_row(item) for item in items)


def write_scores(path: Path, grouped: dict[str, list[dict[str, Any]]]) -> None:
    rows = [{
        "pair": item["pair"],
        "sample": item["sample"],
        "candidate_path": str(item["path"].resolve()),
        "assignment_swapped": item["assignment_swapped"],
        "score": item["score"],
        "score_matrix": item["score_matrix"],
    } for candidates in grouped.values() for item in candidates]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def selector_provenance(selector: str) -> dict[str, Any]:
    if selector == "first":
        return {"selector": "first", "rule": "first sample ID (no scoring)"}
    spec = PICKSCORE if selector == "pickscore" else CLIP
    return {
        "selector": selector,
        **spec,
        "rule": "maximum mean score over the better of the two prompt-image assignments",
        "tie_break": "first sample ID",
    }
