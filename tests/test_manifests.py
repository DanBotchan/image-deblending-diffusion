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
