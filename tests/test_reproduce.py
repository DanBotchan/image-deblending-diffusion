"""Every command the reproduction drivers emit must compose to the intended config."""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from paper_repro.experiments import (EXPERIMENTS, Context, budget_seed,
                                     write_budget_manifest)
from paper_repro.runner import shards, smoke_overrides

REPO = Path(__file__).resolve().parents[1]
CONFIG_ROOT = str(REPO / "configs")


def jobs_of(experiment, scenario, tmp_path, **kwargs):
    builder, _ = EXPERIMENTS[experiment]
    ctx = Context(out=tmp_path / experiment / scenario, data_root=tmp_path / "data", slots=2, **kwargs)
    for _, jobs in builder(ctx, scenario):
        yield from (jobs() if callable(jobs) else jobs)


def resolve(argv):
    overrides = [arg for arg in argv[2:] if "=" in arg]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_ROOT, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def sampling_configs(experiment, scenario, tmp_path, **kwargs):
    for job in jobs_of(experiment, scenario, tmp_path, **kwargs):
        script = Path(job.argv[1]).name
        if script in ("first_stage.py", "second_stage.py"):
            yield job.name, script, resolve(job.argv)


@pytest.mark.parametrize("experiment", sorted(EXPERIMENTS))
def test_every_sampling_command_composes(experiment, tmp_path):
    _, scenarios = EXPERIMENTS[experiment]
    for scenario in scenarios:
        configs = list(sampling_configs(experiment, scenario, tmp_path))
        assert configs, f"{experiment}/{scenario} emits no sampling jobs"
        for name, _, cfg in configs:
            assert cfg["general"]["skip_finished"] is True, name
            assert cfg["datasets"]["dataset_path"] == str(tmp_path / "data"), name


def test_main_matches_paper_configs(tmp_path):
    configs = {name: cfg for name, _, cfg in sampling_configs("main", "unknown_scalar", tmp_path)}
    stage1 = next(cfg for name, cfg in configs.items() if ".stage1." in name)
    stage2 = next(cfg for name, cfg in configs.items() if ".stage2." in name)
    assert (stage1["sampler"]["timestep_respacing"], stage1["sampler"]["start_idx"]) == (100, 80)
    assert stage1["general"]["N_samples_for_pair"] == 3
    assert stage1["alphas"]["pred_map_type"] == "param_scalar"
    assert (stage2["sampler"]["timestep_respacing"], stage2["sampler"]["start_idx"]) == (600, 350)
    assert stage2["alphas"]["optimizer"] == "param_adam_ms"


def test_text_guidance_prompt_modes(tmp_path):
    seen = {}
    for name, script, cfg in sampling_configs("text-guidance", "spatial", tmp_path):
        if script == "first_stage.py":
            cell = name.split(".")[0]
            seen[cell] = (cfg["general"]["cfg"], cfg["prompts"]["positive_template"],
                          cfg["prompts"]["negative_template"], cfg["general"]["N_samples_for_pair"])
    assert seen["baseline_cfg0"][0] == 0.0
    assert seen["ours_cfg2"] == (2.0, "{prompt}", "{neg_prompt}", 1)
    assert seen["standard_cfg_cfg3"] == (3.0, "{prompt}", "", 1)
    assert seen["class_level_cfg1"] == (1.0, "{cls}", "{neg_cls}", 1)
    assert len(seen) == 10


def test_grids_fix_stage2_alpha_and_sweep_values(tmp_path):
    starts, recurrences = set(), set()
    for name, script, cfg in sampling_configs("timestep", "spatial", tmp_path):
        if script == "second_stage.py":
            assert cfg["alphas"]["optimizer"] == "none"
            starts.add(cfg["sampler"]["start_idx"])
    assert starts == {50, 200, 350, 500}
    for name, script, cfg in sampling_configs("recurrence", "spatial", tmp_path):
        recurrences.add((script, cfg["general"]["n_recurrence"]))
    assert {n for s, n in recurrences if s == "first_stage.py"} == {4, 6, 8, 10}
    assert {n for s, n in recurrences if s == "second_stage.py"} == {2, 4, 6, 8, 10}


def test_candidate_budget_as_run_protocol(tmp_path):
    configs = list(sampling_configs("candidate-budget", "unknown_scalar", tmp_path))
    stage1 = next(cfg for _, script, cfg in configs if script == "first_stage.py")
    assert stage1["general"]["N_samples_for_pair"] == 16
    assert stage1["alphas"]["pred_map_type"] == "same"          # alpha given, as run
    assert stage1["datasets"]["annotations_path"].endswith("annotations.csv")
    assert (stage1["sampler"]["timestep_respacing"], stage1["sampler"]["start_idx"]) == (80, 64)


def test_smoke_shrinks_steps_but_keeps_grid(tmp_path):
    assert smoke_overrides(["sampler.timestep_respacing=600", "sampler.start_idx=350",
                            "general.n_recurrence=8", "general.cfg=2.0"]) == [
        "sampler.timestep_respacing=60", "sampler.start_idx=35",
        "general.n_recurrence=2", "general.cfg=2.0"]
    starts = {cfg["sampler"]["start_idx"]
              for _, script, cfg in sampling_configs("timestep", "spatial", tmp_path, smoke=True, pairs=1)
              if script == "first_stage.py"}
    assert starts == {2, 4, 6, 8, 10}


def test_shards_cover_range_exactly():
    assert shards(0, 300, 4) == [(0, 75), (75, 150), (150, 225), (225, 300)]
    assert shards(20, 23, 8) == [(20, 21), (21, 22), (22, 23)]
    assert shards(0, 5, 1) == [(0, 5)]


def test_budget_seeds_extend_stored_seeds(tmp_path):
    target = tmp_path / "budget.csv"
    write_budget_manifest(REPO / "manifests/comparison/dataset.csv", target, pairs=10, candidates=16)
    import ast
    import csv
    with target.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    seeds = ast.literal_eval(rows[0]["z_seeds"])
    assert len(seeds) == 16 == len(set(seeds))
    assert seeds[10] == budget_seed(rows[0]["pair_id"], 10)


def test_auto_prompt_arms_as_run_and_matched(tmp_path):
    def manifests(**extra):
        found = {}
        for name, script, cfg in sampling_configs("auto-prompts", "spatial", tmp_path, extra=extra):
            if script == "first_stage.py":
                found[name.split(".")[1]] = Path(cfg["datasets"]["annotations_path"]).name
        return found
    assert manifests(legacy_blends=True) == {"none": "dataset_random_alpha_map.csv", "auto": "dataset_auto_as_run.csv",
                           "manual": "dataset_random_alpha_map.csv"}
    assert manifests()["auto"] == "dataset_random_alpha_map_auto.csv"


def test_completion_requires_marker(tmp_path):
    from paper_repro.experiments import _finished
    sample = tmp_path / "results/pair_000/smp_0000"
    sample.mkdir(parents=True)
    (sample / "sample1.png").touch()
    assert not _finished(tmp_path, range(1), 1)()
    (sample / ".complete").touch()
    assert _finished(tmp_path, range(1), 1)()


def test_run_settings_reject_changed_inputs(tmp_path):
    import json
    import subprocess
    import sys
    out = tmp_path / "comparison/main/known_scalar"
    out.mkdir(parents=True)
    (out / "run_settings.json").write_text(json.dumps({"pairs": 1}))
    result = subprocess.run([sys.executable, str(REPO / "scripts/reproduce.py"), "main",
                             "--data-root", str(tmp_path), "--output-root", str(tmp_path),
                             "--pairs", "2", "--dry-run"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Run settings differ" in result.stderr
