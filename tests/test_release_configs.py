from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

CONFIG_ROOT = str((Path(__file__).resolve().parents[1] / "configs").resolve())

EXPECTED = {
    "known_scalar_stage1": (80, 64, 3, "constant"),
    "unknown_scalar_stage1": (100, 80, 3, "three_stage"),
    "spatial_stage1": (100, 80, 3, "three_stage"),
    "known_scalar_stage2": (600, 350, 3, "constant"),
    "unknown_scalar_stage2": (600, 350, 3, "three_stage"),
    "spatial_stage2": (600, 350, 3, "three_stage"),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_paper_config_matches_recorded_protocol(name, monkeypatch):
    monkeypatch.setenv("DEBLEND_STAGE2_MANIFEST", "/tmp/selected.csv")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_ROOT, version_base=None):
        cfg = compose(config_name="config", overrides=[f"+paper={name}"])
        resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    steps, start, candidates, alpha_scheduler = EXPECTED[name]
    assert resolved["sampler"]["timestep_respacing"] == steps
    assert resolved["sampler"]["start_idx"] == start
    assert resolved["general"]["N_samples_for_pair"] == candidates
    assert resolved["general"]["n_recurrence"] == 8
    assert resolved["alphas"]["scheduler"] == alpha_scheduler
    assert "/mnt/" not in str(resolved)

    if name.endswith("stage1"):
        assert resolved["general"]["cfg"] == 2.0
        assert resolved["general"]["lpips"] == 100
        assert resolved["prompts"]["negative_template"] == "{neg_prompt}"
    else:
        assert resolved["general"]["lpips"] == 20
        assert resolved["guidance"] == {
            "guidance_method": "three_stage_linear",
            "chaotic": 0.25,
            "critical": 5.0,
            "refine": 12.0,
        }
