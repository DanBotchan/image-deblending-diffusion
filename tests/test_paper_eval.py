from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from paper_eval.aggregate import aggregate_metrics
from paper_eval.artifact_tree import discover_candidates, evaluate_artifact_candidates, evaluate_artifact_tree
from paper_eval.lineage import validate_stage_lineage
from paper_eval.models import ExperimentSpec
from paper_eval.protocols import evaluate_protocol
from paper_eval.timing import summarize_timing


METRIC_FIELDS = [
    "sample1_psnr", "sample1_ssim", "sample1_lpips",
    "sample2_psnr", "sample2_ssim", "sample2_lpips",
]


def row(pair: str, sample: str, score: float, value: float, time: float, seed: str = "7"):
    result = {
        "pair": pair,
        "sample": sample,
        "avg_clip_score": str(score),
        "sample_time": str(time),
        "smp_seed": seed,
    }
    for field in METRIC_FIELDS:
        result[field] = str(value)
    return result


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, name: str, rows: list[dict[str, str]]) -> Path:
        path = self.root / name
        fields = list(dict.fromkeys(key for item in rows for key in item))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def spec(self, stage1: Path, stage2: Path, **kwargs) -> ExperimentSpec:
        return ExperimentSpec(
            name="test.ours",
            family="test",
            scenario="test",
            method="Ours",
            protocol="select_then_refine",
            expected_pairs=1,
            project_root=self.root,
            stage1_csv=stage1,
            stage2_csv=stage2,
            **kwargs,
        )

    def test_selects_only_clip_winner_and_keeps_diagnostic_timing_separate(self):
        stage1 = self.write_csv(
            "stage1.csv",
            [
                row("pair_000", "smp_0000", 0.1, 1, 10),
                row("pair_000", "smp_0001", 0.9, 2, 11),
                row("pair_000", "smp_0002", 0.2, 3, 12),
            ],
        )
        stage2 = self.write_csv(
            "stage2.csv",
            [
                row("pair_000", "smp_0000", 0.1, 1, 20),
                row("pair_000", "smp_0001", 0.9, 20, 21),
                row("pair_000", "smp_0002", 0.2, 100, 22),
            ],
        )

        result = evaluate_protocol(self.spec(stage1, stage2))
        metrics, metric_issues = aggregate_metrics(result.spec.name, result.canonical_rows)
        timing, timing_issues = summarize_timing(result)

        self.assertEqual(result.selected_samples, {"pair_000": "smp_0001"})
        self.assertEqual(metrics["sample1_psnr_mean"], 20.0)
        self.assertNotEqual(metrics["sample1_psnr_mean"], (1 + 20 + 100) / 3)
        self.assertEqual(timing["canonical_sampling"]["mean_seconds"], 54.0)
        self.assertEqual(
            timing["diagnostic_all_stage2_candidates"]["mean_seconds"], 96.0
        )
        self.assertFalse(result.issues + metric_issues)
        self.assertEqual(
            {issue.code for issue in timing_issues}, {"selection_timing_unavailable"}
        )

    def test_missing_selected_refinement_is_an_error(self):
        stage1 = self.write_csv(
            "stage1.csv",
            [
                row("pair_000", "smp_0000", 0.1, 1, 1),
                row("pair_000", "smp_0001", 0.9, 2, 1),
                row("pair_000", "smp_0002", 0.2, 3, 1),
            ],
        )
        stage2 = self.write_csv(
            "stage2.csv", [row("pair_000", "smp_0000", 0.1, 1, 1)]
        )

        result = evaluate_protocol(self.spec(stage1, stage2))
        self.assertIn("selected_refinement_missing", {issue.code for issue in result.issues})
        self.assertEqual(result.canonical_rows, [])

    def test_tie_requires_explicit_predeclared_rule(self):
        candidates = [
            row("pair_000", "smp_0001", 0.9, 2, 1),
            row("pair_000", "smp_0000", 0.9, 1, 1),
            row("pair_000", "smp_0002", 0.2, 3, 1),
        ]
        stage1 = self.write_csv("stage1.csv", candidates)
        stage2 = self.write_csv("stage2.csv", candidates)

        strict = evaluate_protocol(self.spec(stage1, stage2))
        self.assertIn("selection_tie", {issue.code for issue in strict.issues})

        resolved = evaluate_protocol(
            self.spec(stage1, stage2, tie_break="first_csv_row")
        )
        self.assertEqual(resolved.selected_samples["pair_000"], "smp_0001")
        self.assertIn(
            "selection_tie_resolved", {issue.code for issue in resolved.issues}
        )

    def test_seed_lineage_mismatch_is_an_error(self):
        stage1_rows = [
            row("pair_000", "smp_0000", 0.1, 1, 1),
            row("pair_000", "smp_0001", 0.9, 2, 1),
            row("pair_000", "smp_0002", 0.2, 3, 1),
        ]
        stage2_rows = [row("pair_000", "smp_0001", 0.9, 2, 1, seed="99")]
        result = evaluate_protocol(
            self.spec(
                self.write_csv("stage1.csv", stage1_rows),
                self.write_csv("stage2.csv", stage2_rows),
            )
        )
        self.assertIn("seed_lineage_mismatch", {issue.code for issue in result.issues})


class ArtifactTreeTest(unittest.TestCase):
    def test_discovers_nested_results_and_reassigns_swapped_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "run" / "results" / "pair_000" / "smp_0000"
            candidate.mkdir(parents=True)
            gt1, gt2 = root / "gt1.png", root / "gt2.png"
            Image.new("RGB", (8, 8), (255, 0, 0)).save(gt1)
            Image.new("RGB", (8, 8), (0, 0, 255)).save(gt2)
            Image.new("RGB", (8, 8), (0, 0, 254)).save(candidate / "sample1.png")
            Image.new("RGB", (8, 8), (254, 0, 0)).save(candidate / "sample2.png")
            (candidate / "details.json").write_text(
                json.dumps(
                    {
                        "image1_path": str(gt1),
                        "image2_path": str(gt2),
                        "sample_time": 1.5,
                        "smp_seed": 7,
                    }
                ),
                encoding="utf-8",
            )

            candidates, discovery_issues = discover_candidates(root / "run")
            rows, issues = evaluate_artifact_tree(
                root / "run", root, "test.raw", resolution=8, include_lpips=False
            )

            self.assertEqual(list(candidates), [("pair_000", "smp_0000")])
            self.assertFalse(discovery_issues + issues)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["order_reassigned"])
            self.assertGreater(rows[0]["sample1_ssim"], 0.99)
            self.assertGreater(rows[0]["sample2_ssim"], 0.99)

    def test_explicit_candidate_set_does_not_expand_to_larger_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = {}
            gt1, gt2 = root / "gt1.png", root / "gt2.png"
            Image.new("RGB", (8, 8), (255, 0, 0)).save(gt1)
            Image.new("RGB", (8, 8), (0, 0, 255)).save(gt2)
            for number in range(2):
                candidate = root / f"pair_{number:03d}" / "smp_0000"
                candidate.mkdir(parents=True)
                Image.new("RGB", (8, 8), (255, 0, 0)).save(candidate / "sample1.png")
                Image.new("RGB", (8, 8), (0, 0, 255)).save(candidate / "sample2.png")
                (candidate / "details.json").write_text(
                    json.dumps({"image1_path": str(gt1), "image2_path": str(gt2)}), encoding="utf-8"
                )
                candidates[(candidate.parent.name, candidate.name)] = candidate
            rows, issues = evaluate_artifact_candidates(
                {("pair_001", "smp_0000"): candidates[("pair_001", "smp_0000")]},
                root, "subset", resolution=8, include_lpips=False,
            )
            self.assertFalse(issues)
            self.assertEqual([(row["pair"], row["sample"]) for row in rows], [("pair_001", "smp_0000")])


class RawLineageTest(unittest.TestCase):
    def make_candidate(self, root: Path, stage: str, sample: str, seed: int) -> Path:
        candidate = root / stage / "pair_000" / sample
        candidate.mkdir(parents=True)
        for source in (1, 2):
            Image.new("RGB", (4, 4), (source * 50, 0, 0)).save(candidate / f"sample{source}.png")
        (candidate / "details.json").write_text(json.dumps({"smp_seed": seed}), encoding="utf-8")
        return candidate

    def test_requires_exact_same_sample_stage1_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = self.make_candidate(root, "stage1", "smp_0001", 7)
            wrong = self.make_candidate(root, "stage1", "smp_0002", 7)
            refined = self.make_candidate(root, "stage2", "smp_0001", 7)
            details = {"smp_seed": 7}
            for source in (1, 2):
                details[f"sample{source}_path"] = str(wrong / f"sample{source}.png")
            (refined / "details.json").write_text(json.dumps(details), encoding="utf-8")
            issues = validate_stage_lineage(root, selected, refined, "test", "pair_000", "smp_0001")
            self.assertEqual(
                [issue.code for issue in issues],
                ["stage2_input_lineage_mismatch", "stage2_input_lineage_mismatch"],
            )

    def test_accepts_exact_same_identity_seed_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = self.make_candidate(root, "stage1", "smp_0001", 7)
            refined = self.make_candidate(root, "stage2", "smp_0001", 7)
            details = {"smp_seed": 7}
            for source in (1, 2):
                details[f"sample{source}_path"] = str(selected / f"sample{source}.png")
            (refined / "details.json").write_text(json.dumps(details), encoding="utf-8")
            self.assertFalse(validate_stage_lineage(root, selected, refined, "test", "pair_000", "smp_0001"))


if __name__ == "__main__":
    unittest.main()
