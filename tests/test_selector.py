import json
from pathlib import Path

import pytest

from scripts.select_candidates import discover


def make_candidate(root: Path, pair: int, sample: int):
    candidate = root / f"pair_{pair:03d}" / f"smp_{sample:04d}"
    candidate.mkdir(parents=True)
    for name in ("sample1.png", "sample2.png", "alphas.json"):
        (candidate / name).write_bytes(b"test")
    (candidate / "details.json").write_text(
        json.dumps({
            "image1_prompt": "fruit", "image2_prompt": "animal",
            "image1_path": "one.png", "image2_path": "two.png",
            "image1_class": "Fruit", "image2_class": "Animal",
            "smp_seed": sample,
        }),
        encoding="utf-8",
    )


def test_discover_requires_exact_candidate_budget(tmp_path):
    for sample in range(3):
        make_candidate(tmp_path, 0, sample)
    grouped = discover(tmp_path, expected_candidates=3)
    assert [item["sample"] for item in grouped["pair_000"]] == [
        "smp_0000", "smp_0001", "smp_0002"
    ]

    with pytest.raises(RuntimeError, match="expected 4 candidates"):
        discover(tmp_path, expected_candidates=4)
