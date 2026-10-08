# Release verification

Checks run on this release (one NVIDIA RTX 3090 per job, Python 3.9, PyTorch 2.5.1+cu121).

## Unit tests

`pytest` passes without a GPU. Besides the evaluation and selection logic, the tests compose
every Stage-1/Stage-2 command that `scripts/reproduce.py` emits for every experiment through
Hydra and check the resolved settings (sampler steps and start, candidates, prompt templates,
alpha optimizer, recurrence counts, manifests).

## End-to-end smoke runs

Every driver ran end to end with `--smoke` (one pair, ~10x fewer steps): `main` for all three
scenarios, `text-guidance` (known scalar and spatial), `timestep`, `recurrence`, `selector`,
`candidate-budget` and `auto-prompts`. The DoubleDIP adapter was run for one pair and passed
through `select_candidates.py --selector clip` and `evaluate_artifact_tree.py`.

## Numerical faithfulness

Spatial scenario, pair 0, canonical settings, seed 1220058128:

| Comparison | Stage-1 output |
|---|---|
| Release, run twice | bit-identical images and alpha |
| Release vs. development code (`deblend` @ 96e0591) | bit-identical images and alpha |
| Release vs. the paper's original artifact (Nov 2025, legacy codebase) | not identical: 28.0 / 20.2 dB PSNR between the two runs' outputs; 19.0 vs. 17.4 dB PSNR to ground truth |

Stage 2 refining the paper's exact Stage-1 candidate with the same seed gives 19.48 dB PSNR to
ground truth vs. 19.41 dB for the paper's artifact (26.8 / 22.5 dB between the two runs' outputs).

The release therefore preserves the development code exactly and is deterministic on a given
GPU/software stack. The paper's artifacts were produced by an older codebase and environment;
the long guided sampling chain amplifies small numerical differences, so reruns reproduce the
paper statistically rather than pixel-for-pixel.

## Automated prompts

Rerunning `scripts/generate_auto_prompts.py` with the pinned Qwen2-VL revision reproduced all 80
raw VLM replies used for pairs 20-99 byte-for-byte; pairs 0-19 in
`manifests/auto_prompts/` were regenerated the same way.

## Baselines

For every Table 1 cell of BIDeN, DAD, GIP and LCD, the evaluation command in
`docs/BASELINES.md`, run from the stated checkpoint on the first three pairs, reproduces the
stored paper outputs: BIDeN and DAD from a fresh upstream checkout with the patch applied (max
difference one intensity level), GIP and LCD from `baselines/` (bit-identical). The stored
outputs in turn reproduce the published Table 1 values. Each training entry point (BIDeN
`alpha_train.py`, DAD `alpha_train.py`/`my_train.py`, GIP, LCD) completed a short run on an
8-image subset of the training set. The three cross-split inconsistencies found this way are
listed in `docs/BASELINES.md`.
