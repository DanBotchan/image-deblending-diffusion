#!/usr/bin/env python3
"""Generate the paper's automated prompts from blended observations only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_REVISION = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
INSTRUCTION = (
    "This image is an alpha-blend (like a double exposure) of two different "
    "photographs overlaid on each other. Identify the two underlying scenes. "
    "Answer with exactly two lines:\n"
    "1. <one sentence describing only the first photo's content>\n"
    "2. <one sentence describing only the second photo's content>\n"
    "Do not mention the blend, overlay, or the other photo in either sentence."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_two_lines(text: str) -> tuple[str | None, str | None]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    parsed = []
    for line in lines:
        for prefix in ("1.", "2.", "1)", "2)", "-", "*"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        parsed.append(line)
    if len(parsed) >= 2:
        return parsed[0], parsed[1]
    if len(parsed) == 1:
        # This matches the frozen experiment parser and is recorded in the log.
        return parsed[0], parsed[0]
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--blended-column", default="blended_polynomial_path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    frame = pd.read_csv(args.input_csv)
    if args.blended_column not in frame.columns:
        raise SystemExit(
            f"Missing {args.blended_column!r}; automated prompts must receive a "
            "pre-rendered blended observation, never source images."
        )
    end = len(frame) if args.end is None else args.end
    if not 0 <= args.start < end <= len(frame):
        raise SystemExit(f"Invalid row interval [{args.start}, {end}) for {len(frame)} rows")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=args.offline,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.offline
    )

    raw_log = []
    for ordinal, (row_index, row) in enumerate(frame.iloc[args.start:end].iterrows(), 1):
        blend_path = (args.data_root / str(row[args.blended_column])).resolve()
        with Image.open(blend_path) as image:
            blend = image.convert("RGB").copy()
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}],
        }]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=[blend], return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        output_ids = generated[:, inputs["input_ids"].shape[1]:]
        reply = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        first, second = parse_two_lines(reply)
        if first and second:
            frame.loc[row_index, "img1_prompt"] = first
            frame.loc[row_index, "img2_prompt"] = second
        raw_log.append({
            "row_index": int(row_index),
            "pair_id": str(row.get("pair_id", row_index)),
            "blended_path": str(blend_path),
            "raw_response": reply,
            "img1_prompt": first,
            "img2_prompt": second,
            "parse_success": bool(first and second),
        })
        print(f"[{ordinal}/{end - args.start}] {first!r} | {second!r}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    raw_path = args.output_csv.with_suffix(".raw.json")
    raw_path.write_text(json.dumps(raw_log, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "instruction": INSTRUCTION,
        "decoding": {"do_sample": False, "max_new_tokens": 128},
        "input_csv": str(args.input_csv.resolve()),
        "input_csv_sha256": sha256(args.input_csv),
        "row_interval": [args.start, end],
        "input_to_vlm": "blended observation only",
        "output_csv_sha256": sha256(args.output_csv),
        "raw_log_sha256": sha256(raw_path),
    }
    args.output_csv.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
