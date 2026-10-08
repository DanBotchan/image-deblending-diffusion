import ast
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = {"comparison": 300, "generalization": 100}


def test_manifest_hashes_and_protocol_fields():
    expected_hashes = {}
    for line in (ROOT / "manifests" / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        expected_hashes[relative] = digest

    for family, expected_rows in EXPECTED_ROWS.items():
        for name in (
            "dataset.csv",
            "dataset_random_alpha_scalar.csv",
            "dataset_random_alpha_map.csv",
        ):
            relative = f"{family}/{name}"
            path = ROOT / "manifests" / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hashes[relative]
            frame = pd.read_csv(path)
            assert len(frame) == expected_rows
            assert frame["pair_id"].is_unique
            assert all(len(ast.literal_eval(value)) >= 3 for value in frame["z_seeds"])
            assert (
                {"a", "b", "c", "d", "e", "f"} <= set(frame.columns)
                or "alpha" in frame.columns
            )
            assert "img1_source" not in frame.columns
            assert "img2_source" not in frame.columns


def test_auto_prompt_manifest_only_replaces_prompts():
    """Auto prompts must be evaluated on exactly the manual arm's spatial blends."""
    manual = pd.read_csv(ROOT / "manifests/comparison/dataset_random_alpha_map.csv")
    auto = pd.read_csv(ROOT / "manifests/auto_prompts/dataset_random_alpha_map_auto.csv")
    prompts = ["img1_prompt", "img2_prompt"]
    others = [column for column in manual.columns if column not in prompts]
    assert list(auto.columns) == list(manual.columns)
    assert auto[others].equals(manual[others])
    changed = (auto[prompts] != manual[prompts]).any(axis=1)
    assert changed[:100].all() and not changed[100:].any()


def test_as_run_auto_manifest_is_known_scalar_with_auto_prompts():
    """The paper's Auto arm: known-scalar rows (alpha 0.5) carrying the VLM prompts."""
    known = pd.read_csv(ROOT / "manifests/comparison/dataset.csv")
    matched = pd.read_csv(ROOT / "manifests/auto_prompts/dataset_random_alpha_map_auto.csv")
    as_run = pd.read_csv(ROOT / "manifests/auto_prompts/dataset_auto_as_run.csv")
    prompts = ["img1_prompt", "img2_prompt"]
    others = [column for column in known.columns if column not in prompts]
    assert as_run[others].equals(known[others])
    assert as_run[prompts].equals(matched[prompts])
