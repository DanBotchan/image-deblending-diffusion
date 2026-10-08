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


def scored(samples_and_scores):
    return [{"pair": "pair_000", "sample": s, "score": v} for s, v in samples_and_scores]


def test_select_respects_budget_and_ties():
    from paper_repro.selection import select
    pool = scored([("smp_0000", 1.0), ("smp_0001", 3.0), ("smp_0002", 3.0), ("smp_0003", 9.0)])
    assert select(pool)["sample"] == "smp_0003"
    assert select(pool, budget=1)["sample"] == "smp_0000"
    # Ties keep the lower sample ID.
    assert select(pool, budget=3)["sample"] == "smp_0001"


def test_first_selector_needs_no_model(tmp_path):
    from paper_repro.selection import score_candidates, select
    for sample in range(2):
        make_candidate(tmp_path, 0, sample)
    grouped = discover(tmp_path, expected_candidates=2)
    score_candidates(grouped, "first")
    assert select(grouped["pair_000"])["sample"] == "smp_0000"
