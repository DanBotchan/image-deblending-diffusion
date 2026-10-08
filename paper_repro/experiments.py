"""Paper experiments expressed as phases of independent command-line jobs.

Each builder receives a :class:`Context` and returns ``[(title, [Job, ...]), ...]``.
All method settings come from ``configs/paper/<scenario>_stage{1,2}.yaml``;
the overrides below are exactly what differs for an experiment, and match
the runs reported in the paper. Where a supplementary run deviated from the
canonical method settings, the override is marked ``AS RUN`` and documented
in ``docs/EXPERIMENTS.md``.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .runner import REPO, Job, python, shards, smoke_overrides

SCENARIOS = ("known_scalar", "unknown_scalar", "spatial")
MANIFEST_FILES = {
    "known_scalar": "dataset.csv",
    "unknown_scalar": "dataset_random_alpha_scalar.csv",
    "spatial": "dataset_random_alpha_map.csv",
}
STAGE1_SAMPLER = {"known_scalar": (80, 64), "unknown_scalar": (100, 80), "spatial": (100, 80)}
STAGE2_SAMPLER = (600, 350)
# Stage-1 sampler (steps, start) used by the candidate-budget and
# automated-prompt runs for every scenario. AS RUN; see docs/EXPERIMENTS.md.
LEGACY_STAGE1_SAMPLER = (80, 64)

# A phase's jobs may be a callable, resolved when the phase starts, so that
# Stage-2 shards are sized from the selection manifest written just before.
Phases = list[tuple[str, "list[Job] | Callable[[], list[Job]]"]]


@dataclass
class Context:
    out: Path                      # experiment output root
    data_root: Path                # directory containing pairs/...
    split: str = "comparison"      # manifests/<split>/
    slots: int = 1                 # concurrent jobs (GPUs)
    smoke: bool = False            # tiny run that only checks the plumbing
    pairs: Optional[int] = None    # limit pairs (debugging / smoke)
    offline: bool = False
    frozen_prompts: bool = True    # auto-prompts: use the shipped VLM output
    extra: dict = field(default_factory=dict)

    def manifest(self, scenario: str) -> Path:
        return REPO / "manifests" / self.split / MANIFEST_FILES[scenario]

    def pair_range(self, start: int, end: int) -> tuple[int, int]:
        if self.pairs is not None:
            end = min(end, start + self.pairs)
        return start, end


def manifest_length(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _sampler(steps: int, start: int) -> list[str]:
    return [f"sampler.timestep_respacing={steps}", f"sampler.start_idx={start}"]


def _finished(root: Path, pairs: range, samples: int) -> Callable[[], bool]:
    def check() -> bool:
        return all((root / "results" / f"pair_{p:03d}" / f"smp_{s:04d}" / "sample1.png").is_file()
                   for p in pairs for s in range(samples))
    return check


def _rows_finished(root: Path, manifest: Path) -> Callable[[], bool]:
    def check() -> bool:
        if not manifest.is_file():
            return False
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return bool(rows) and all(
            (root / "results" / row["pair"] / row["sample"] / "sample1.png").is_file() for row in rows)
    return check


def _exists(path: Path) -> Callable[[], bool]:
    return path.is_file


def stage1_jobs(ctx: Context, name: str, out: Path, config: str, manifest: Path,
                start: int, end: int, candidates: int, overrides: list[str]) -> list[Job]:
    """Shard pairs [start, end) of ``manifest`` over the available slots."""
    overrides = smoke_overrides(overrides) if ctx.smoke else overrides
    jobs = []
    for lo, hi in shards(start, end, ctx.slots):
        argv = python("scripts/first_stage.py", f"+paper={config}",
                      f"general.output_dir={out}",
                      f"datasets.dataset_path={ctx.data_root}",
                      f"datasets.annotations_path={manifest}",
                      f"general.dataset_subset=[{lo},{hi}]",
                      f"general.N_samples_for_pair={candidates}",
                      "general.skip_finished=true", *overrides)
        jobs.append(Job(f"{name}.stage1.{lo:03d}-{hi:03d}", argv,
                        _finished(out, range(lo, hi), candidates)))
    return jobs


def stage2_jobs(ctx: Context, name: str, out: Path, config: str, manifest: Path,
                rows: int, overrides: list[str]) -> list[Job]:
    """Refine every row of a selection manifest.

    ``rows`` is an upper bound used only while the manifest does not exist yet
    (``--dry-run``); otherwise the shards cover the manifest's actual rows.
    """
    overrides = smoke_overrides(overrides) if ctx.smoke else overrides
    if manifest.is_file():
        rows = manifest_length(manifest)
    jobs = []
    for lo, hi in shards(0, rows, ctx.slots):
        argv = python("scripts/second_stage.py", f"+paper={config}",
                      f"general.output_dir={out}",
                      f"datasets.dataset_path={ctx.data_root}",
                      f"datasets.annotations_path={manifest}",
                      f"general.dataset_subset=[{lo},{hi}]",
                      "general.skip_finished=true", *overrides)
        jobs.append(Job(f"{name}.stage2.{lo:03d}-{hi:03d}", argv, _rows_finished(out, manifest)))
    return jobs


def select_job(ctx: Context, name: str, stage1: Path, output: Path, selector: str,
               candidates: int, *flags: str) -> Job:
    argv = python("scripts/select_candidates.py", stage1, "--output-csv", output,
                  "--selector", selector, "--expected-candidates", candidates, *flags)
    if ctx.offline:
        argv.append("--offline")
    return Job(f"{name}.select-{selector}", argv, _exists(output))


def eval_job(name: str, root: Path, output: Path, pairs: int, *flags: str) -> Job:
    argv = python("scripts/evaluate_artifact_tree.py", root,
                  "--output-csv", output.with_suffix(".csv"),
                  "--output-json", output.with_suffix(".json"),
                  "--experiment", name, "--project-root", REPO,
                  "--expected-pairs", pairs, *flags)
    return Job(f"{name}.eval", argv, _exists(output.with_suffix(".json")))


# ---------------------------------------------------------------------------
# Main comparison (Table 1) and the two-stage ablation
# ---------------------------------------------------------------------------

def main_experiment(ctx: Context, scenario: str) -> Phases:
    root = ctx.out
    n = manifest_length(ctx.manifest(scenario))
    start, end = ctx.pair_range(0, n)
    stage1, stage2 = root / "stage1", root / "stage2"
    selected = root / "selected.csv"
    s1_steps, s1_start = STAGE1_SAMPLER[scenario]
    return [
        ("Stage 1: 3 candidates per pair", stage1_jobs(
            ctx, "main", stage1, f"{scenario}_stage1", ctx.manifest(scenario), start, end, 3,
            _sampler(s1_steps, s1_start))),
        ("PickScore selection", [select_job(ctx, "main", stage1, selected, "pickscore", 3)]),
        ("Stage 2: refine the selected candidate", lambda: stage2_jobs(
            ctx, "main", stage2, f"{scenario}_stage2", selected, end - start,
            _sampler(*STAGE2_SAMPLER))),
        ("Evaluation", [
            eval_job("final", stage2, root / "metrics" / "stage2", end - start),
            eval_job("stage1_selected", stage1, root / "metrics" / "stage1_selected", end - start,
                     "--candidates-csv", selected),
        ]),
    ]


# ---------------------------------------------------------------------------
# Text-guidance ablation (100 pairs, one candidate)
# ---------------------------------------------------------------------------

PROMPT_MODES = {
    # Cross-negative prompting: each branch negates the other branch's prompt.
    "ours": ["prompts.positive_template='{prompt}'", "prompts.negative_template='{neg_prompt}'"],
    # Standard classifier-free guidance with an empty negative prompt.
    "standard_cfg": ["prompts.positive_template='{prompt}'", "prompts.negative_template=''"],
    # Cross-negative prompting with class labels ("fruits", "animals") as prompts.
    "class_level": ["prompts.positive_template='{cls}'", "prompts.negative_template='{neg_cls}'"],
}
TEXT_GUIDANCE_CELLS = [("baseline", 0.0)] + [
    (mode, cfg) for mode in PROMPT_MODES for cfg in (1.0, 2.0, 3.0)]


def text_guidance(ctx: Context, scenario: str) -> Phases:
    if scenario not in ("known_scalar", "spatial"):
        raise SystemExit("The paper's text-guidance ablation covers known_scalar and spatial")
    start, end = ctx.pair_range(0, 100)
    s1_steps, s1_start = STAGE1_SAMPLER[scenario]
    phases: Phases = [("Stage 1", []), ("Selection", []), ("Stage 2", []), ("Evaluation", [])]
    for mode, cfg in TEXT_GUIDANCE_CELLS:
        cell = f"{mode}_cfg{cfg:g}"
        root = ctx.out / cell
        prompts = PROMPT_MODES["ours" if mode == "baseline" else mode]
        phases[0][1].extend(stage1_jobs(
            ctx, cell, root / "stage1", f"{scenario}_stage1", ctx.manifest(scenario), start, end, 1,
            [*_sampler(s1_steps, s1_start), f"general.cfg={cfg}", *prompts]))
        phases[1][1].append(select_job(ctx, cell, root / "stage1", root / "selected.csv", "first", 1))
        phases[2][1].extend(stage2_jobs(
            ctx, cell, root / "stage2", f"{scenario}_stage2", root / "selected.csv", end - start,
            _sampler(*STAGE2_SAMPLER)))
        phases[3][1].append(eval_job(cell, root / "stage2", root / "metrics", end - start))
    return phases


# ---------------------------------------------------------------------------
# Timestep and recurrence grids (20 spatial pairs, one candidate)
# ---------------------------------------------------------------------------

TIMESTEP_STAGE1 = (20, 40, 60, 80, 100)
TIMESTEP_STAGE2 = (50, 200, 350, 500)
RECURRENCE_STAGE1 = (4, 6, 8, 10)
RECURRENCE_STAGE2 = (2, 4, 6, 8, 10)
# Both grids kept the Stage-2 alpha map fixed at its Stage-1 estimate. AS RUN.
FIXED_STAGE2_ALPHA = ["alphas=constant"]


def _grid(ctx: Context, stage1_cells: dict[str, list[str]],
          stage2_cells: dict[str, list[str]]) -> Phases:
    start, end = ctx.pair_range(0, 20)
    phases: Phases = [("Stage 1", []), ("Selection", []), ("Stage 2", []), ("Evaluation", [])]
    for s1_name, s1_overrides in stage1_cells.items():
        root = ctx.out / s1_name
        phases[0][1].extend(stage1_jobs(
            ctx, s1_name, root / "stage1", "spatial_stage1", ctx.manifest("spatial"),
            start, end, 1, s1_overrides))
        phases[1][1].append(select_job(ctx, s1_name, root / "stage1", root / "selected.csv", "first", 1))
        phases[3][1].append(eval_job(f"{s1_name}.stage1", root / "stage1",
                                     root / "metrics_stage1", end - start))
        for s2_name, s2_overrides in stage2_cells.items():
            cell = f"{s1_name}.{s2_name}"
            phases[2][1].extend(stage2_jobs(
                ctx, cell, root / s2_name, "spatial_stage2", root / "selected.csv", end - start,
                [*FIXED_STAGE2_ALPHA, *s2_overrides]))
            phases[3][1].append(eval_job(cell, root / s2_name, root / f"metrics_{s2_name}", end - start))
    return phases


def timestep(ctx: Context, scenario: str = "spatial") -> Phases:
    return _grid(
        ctx,
        {f"s1_{t}": _sampler(100, t) for t in TIMESTEP_STAGE1},
        {f"s2_{t}": _sampler(STAGE2_SAMPLER[0], t) for t in TIMESTEP_STAGE2},
    )


def recurrence(ctx: Context, scenario: str = "spatial") -> Phases:
    return _grid(
        ctx,
        {f"n1_{n}": [*_sampler(*STAGE1_SAMPLER["spatial"]), f"general.n_recurrence={n}"]
         for n in RECURRENCE_STAGE1},
        {f"n2_{n}": [*_sampler(*STAGE2_SAMPLER), f"general.n_recurrence={n}"]
         for n in RECURRENCE_STAGE2},
    )


# ---------------------------------------------------------------------------
# PickScore vs. CLIP selection (same candidates, all refined)
# ---------------------------------------------------------------------------

def selector_comparison(ctx: Context, scenario: str) -> Phases:
    """Refine all three Stage-1 candidates of the main run, then compare selectors.

    Reuses ``<out>/../main/<scenario>/stage1`` when present (pass the main
    experiment's directory with --stage1-root otherwise).
    """
    stage1 = Path(ctx.extra.get("stage1_root") or ctx.out.parent.parent / "main" / ctx.out.name / "stage1")
    n = manifest_length(ctx.manifest(scenario))
    start, end = ctx.pair_range(0, n)
    s1_steps, s1_start = STAGE1_SAMPLER[scenario]
    root = ctx.out
    phases: Phases = []
    if not ctx.extra.get("stage1_root"):
        phases.append(("Stage 1 (shared with the main experiment)", stage1_jobs(
            ctx, "main", stage1, f"{scenario}_stage1", ctx.manifest(scenario), start, end, 3,
            _sampler(s1_steps, s1_start))))
    phases += [
        ("Selection (manifest of all candidates, PickScore, CLIP)", [
            select_job(ctx, "all", stage1, root / "all_candidates.csv", "first", 3, "--all"),
            select_job(ctx, "pickscore", stage1, root / "pickscore.csv", "pickscore", 3),
            select_job(ctx, "clip", stage1, root / "clip.csv", "clip", 3),
        ]),
        ("Stage 2: refine every candidate", lambda: stage2_jobs(
            ctx, "all", root / "stage2", f"{scenario}_stage2", root / "all_candidates.csv",
            3 * (end - start), _sampler(*STAGE2_SAMPLER))),
        ("Evaluation", [eval_job("all", root / "stage2", root / "metrics", end - start,
                                 "--candidates-per-pair", "3")]),
    ]
    return phases


# ---------------------------------------------------------------------------
# Candidate budget (10 pairs per scenario, 16 candidates)
# ---------------------------------------------------------------------------

BUDGETS = (1, 2, 3, 4, 6, 8, 12, 16)


def budget_seed(pair_id: str, index: int) -> int:
    """Deterministic extra seeds beyond the 10 stored per pair (frozen protocol)."""
    digest = hashlib.sha256(f"{pair_id}:{index}:pickscore-saturation-v1".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7fffffff


def write_budget_manifest(source: Path, target: Path, pairs: int, candidates: int) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)[:pairs]
    for row in rows:
        seeds = list(ast.literal_eval(row["z_seeds"]))
        while len(seeds) < candidates:
            seed = budget_seed(row["pair_id"], len(seeds))
            if seed not in seeds:
                seeds.append(seed)
        row["z_seeds"] = str(seeds)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_budget(ctx: Context, scenario: str) -> Phases:
    """AS RUN: every scenario used the 80-step/64-start Stage-1 sampler and a
    fixed Stage-2 alpha; "unknown scalar" was sampled with the known-scalar
    configuration (alpha given to the method) on the unknown-scalar pairs."""
    start, end = ctx.pair_range(0, 10)
    candidates = 4 if ctx.smoke else max(BUDGETS)
    budgets = [b for b in BUDGETS if b <= candidates]
    root = ctx.out
    manifest = root / "annotations.csv"
    write_budget_manifest(ctx.manifest(scenario), manifest, end, candidates)
    stage1_config = "spatial_stage1" if scenario == "spatial" else "known_scalar_stage1"
    winners = root / "winners.csv"
    return [
        ("Stage 1: 16 seeded candidates per pair", stage1_jobs(
            ctx, "budget", root / "stage1", stage1_config, manifest, start, end, candidates,
            _sampler(*LEGACY_STAGE1_SAMPLER))),
        ("PickScore selection at every cumulative budget", [select_job(
            ctx, "budget", root / "stage1", winners, "pickscore", candidates,
            "--budgets", ",".join(map(str, budgets)))]),
        ("Stage 2: refine each distinct winner once", lambda: stage2_jobs(
            ctx, "budget", root / "stage2", "known_scalar_stage2", winners,
            len(budgets) * (end - start), _sampler(*STAGE2_SAMPLER))),
        ("Evaluation", [eval_job("winners", root / "stage2", root / "metrics", end - start,
                                 "--candidates-per-pair", "0")]),
    ]


# ---------------------------------------------------------------------------
# Automated prompts (Qwen2-VL from the blended observation only)
# ---------------------------------------------------------------------------

# Frozen Qwen2-VL prompts (pairs 0-99), written into two manifests:
#   AS RUN (paper tables): the known-scalar manifest, so the Auto arm is sampled
#     on alpha = 0.5 blends while None/Manual use the spatial blends;
#   matched (--matched-blends): the spatial manifest, so all arms share blends.
FROZEN_AUTO_PROMPTS = {
    "as_run": REPO / "manifests" / "auto_prompts" / "dataset_auto_as_run.csv",
    "matched": REPO / "manifests" / "auto_prompts" / "dataset_random_alpha_map_auto.csv",
}
AUTO_ARMS = ("none", "auto", "manual")


def automated_prompts(ctx: Context, scenario: str = "spatial") -> Phases:
    """Preliminary Stage-1 study on pairs 0-19 and end-to-end confirmation on 20-99.

    AS RUN: Stage 1 used the 80-step/64-start sampler, Stage 2 a fixed alpha, and
    the Auto arm the alpha = 0.5 blends (prompts still come from the spatial
    blend). ``matched_blends`` samples the Auto arm on the spatial blends instead.
    """
    root = ctx.out
    manual = ctx.manifest("spatial")
    variant = "matched" if ctx.extra.get("matched_blends") else "as_run"
    auto = FROZEN_AUTO_PROMPTS[variant] if ctx.frozen_prompts else root / f"auto_prompts_{variant}.csv"
    phases: Phases = []
    if not ctx.frozen_prompts:
        # The VLM always sees the spatial blend; the manifest it writes into
        # decides which blend the Auto arm is sampled on.
        source = manual if variant == "matched" else ctx.manifest("known_scalar")
        argv = python("scripts/generate_auto_prompts.py", source, "--data-root", ctx.data_root,
                      "--blended-column", "blended_polynomial_path",
                      "--output-csv", auto, "--start", 0, "--end", 100)
        if ctx.offline:
            argv.append("--offline")
        phases.append(("Generate prompts with Qwen2-VL", [Job("auto-prompts", argv, _exists(auto))]))

    arms = {"none": (manual, ["general.cfg=0.0"]), "auto": (auto, []), "manual": (manual, [])}
    sampler = _sampler(*LEGACY_STAGE1_SAMPLER)
    pre_start, pre_end = ctx.pair_range(0, 20)
    con_start, con_end = ctx.pair_range(20, 100)
    stage1, select, stage2, evaluate = [], [], [], []
    for arm, (manifest, overrides) in arms.items():
        pre = root / "preliminary" / arm
        stage1 += stage1_jobs(ctx, f"pre.{arm}", pre / "stage1", "spatial_stage1", manifest,
                              pre_start, pre_end, 1, [*sampler, *overrides])
        evaluate.append(eval_job(f"pre.{arm}", pre / "stage1", pre / "metrics", pre_end - pre_start))

        con = root / "confirmation" / arm
        stage1 += stage1_jobs(ctx, f"con.{arm}", con / "stage1", "spatial_stage1", manifest,
                              con_start, con_end, 3, [*sampler, *overrides])
        # Prompt-free sampling has nothing for a text-image selector to score.
        selector = "first" if arm == "none" else "pickscore"
        select.append(select_job(ctx, f"con.{arm}", con / "stage1", con / "selected.csv", selector, 3))
        stage2 += stage2_jobs(ctx, f"con.{arm}", con / "stage2", "spatial_stage2", con / "selected.csv",
                              con_end - con_start, [*FIXED_STAGE2_ALPHA, *_sampler(*STAGE2_SAMPLER)])
        evaluate.append(eval_job(f"con.{arm}", con / "stage2", con / "metrics", con_end - con_start))
    return phases + [("Stage 1", stage1), ("PickScore selection", select),
                     ("Stage 2", stage2), ("Evaluation", evaluate)]


EXPERIMENTS: dict[str, tuple[Callable[[Context, str], Phases], tuple[str, ...]]] = {
    "main": (main_experiment, SCENARIOS),
    "text-guidance": (text_guidance, ("known_scalar", "spatial")),
    "timestep": (timestep, ("spatial",)),
    "recurrence": (recurrence, ("spatial",)),
    "selector": (selector_comparison, SCENARIOS),
    "candidate-budget": (candidate_budget, SCENARIOS),
    "auto-prompts": (automated_prompts, ("spatial",)),
}


def write_protocol(ctx: Context, experiment: str, scenario: str, phases: Phases) -> None:
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "protocol.json").write_text(json.dumps({
        "experiment": experiment, "scenario": scenario, "split": ctx.split,
        "data_root": str(ctx.data_root), "smoke": ctx.smoke, "pairs_limit": ctx.pairs,
        "phases": [{"title": title,
                    "commands": [job.command() for job in (jobs() if callable(jobs) else jobs)]}
                   for title, jobs in phases],
    }, indent=2) + "\n", encoding="utf-8")
