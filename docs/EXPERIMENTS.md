# Paper experiment map

The pipeline experiments listed below run through one driver:

```bash
python scripts/reproduce.py <experiment> --scenario <scenario> --gpus 0,1,2,3
```

The driver runs Stage 1, ground-truth-free selection, Stage 2 and evaluation as plain
command lines (`--dry-run` prints them), shards pairs over the listed GPUs, resumes an
interrupted run, and writes the table as `summary.md`/`summary.csv` under
`$DEBLEND_OUTPUT_ROOT/<split>/<experiment>/<scenario>/`. `--smoke` runs one pair with ~10x
fewer steps to check an installation in minutes (its numbers are meaningless). Use a new output root when run settings change. Existing runs created before run-settings tracking need a new output root; only samples with a completion marker are resumed.

| Paper result | Command | Protocol |
|---|---|---|
| Main comparison, Fruits/Animals (Table 1, left) | `main --scenario {known_scalar,unknown_scalar,spatial}` | 300 pairs, 3 Stage-1 candidates, PickScore, one same-ID Stage-2 refinement |
| Main comparison, Vehicles/Flowers (Table 1, right) | `main --split generalization --data-root <BID_generalization> --scenario ...` | 100 held-out pairs, same settings |
| Two-stage ablation | produced by `main` (`summary.md` rows "Stage 1 (selected)" and "Stage 2") | PickScore-selected Stage-1 candidate (resized to 256) vs. its same-ID Stage-2 refinement |
| Text-guidance ablation | `text-guidance --scenario {known_scalar,spatial}` | pairs 0-99, 1 candidate; prompt-free (CFG 0), and Ours / standard CFG / class-level at CFG 1, 2, 3 |
| Starting-timestep grid | `timestep` | spatial pairs 0-19, 1 candidate; Stage-1 start 20/40/60/80/100 (of 100) x Stage-2 start 50/200/350/500 (of 600), plus Stage-1-only rows |
| Recurrence grid | `recurrence` | spatial pairs 0-19, 1 candidate; Stage-1 n 4/6/8/10 x Stage-2 n 2/4/6/8/10 |
| PickScore vs. CLIP | `selector --scenario ...` | the main run's 3 Stage-1 candidates, all refined in Stage 2; each selector's choice is evaluated on its matching refinement |
| Candidate budget | `candidate-budget --scenario ...` | pairs 0-9, 16 seeded candidates; PickScore over cumulative prefixes 1/2/3/4/6/8/12/16; each distinct winner refined once |
| Automated prompts | `auto-prompts` | Qwen2-VL prompts from the blend only; preliminary Stage-1 study on spatial pairs 0-19 (1 candidate), end-to-end confirmation on pairs 20-99 (3 candidates, PickScore, Stage 2) |
| DoubleDIP best-of-three | `baselines/doubledip/run_doubledip.py`, see `docs/BASELINES.md` | 3 seeds, 8,000 iterations, permutation-invariant PickScore/CLIP selection |

`selector` reuses the Stage-1 tree of `main` for the same scenario (and runs it if missing).
Candidate-budget seeds beyond the ten stored per pair are derived deterministically
(`paper_repro.experiments.budget_seed`). `auto-prompts` uses the frozen VLM output in
`manifests/auto_prompts/`; pass `--generate-prompts` to rerun Qwen2-VL instead.
`scripts/render_timestep_heatmaps.py <timestep summary.csv>` draws the supplementary heatmaps.

## Held-out Vehicles and Flowers

The held-out split uses the main experiment configuration and the 100-pair manifests in
`manifests/generalization/`. Point `--data-root` at the prepared held-out images.

```bash
for scenario in known_scalar unknown_scalar spatial; do
  python scripts/reproduce.py main --scenario "$scenario" --split generalization \
    --data-root /absolute/path/to/BID_generalization
done
```

Add `--gpus 0,1,2,3` to distribute pairs across four GPUs. Each run writes `summary.md`
and `summary.csv` under `outputs/generalization/main/<scenario>/` by default.

## Release coverage and metrics

The alpha-map accuracy and out-of-family matte experiments are outside the driver's
coverage. The candidate-budget table uses sampling-time RGB SSIM and VGG LPIPS. The
driver reports PSNR, grayscale SSIM, and AlexNet LPIPS after ordering each pair to
minimize total mean squared error against ground truth. The release applies LPIPS guidance
every fourth guided timestep. The original main runs applied it at every guided timestep.

## Canonical method settings

The main experiment uses `configs/paper/<scenario>_stage{1,2}.yaml`.

| Setting | Known scalar | Unknown scalar | Spatial |
|---|---:|---:|---:|
| Stage-1 steps / start | 80 / 64 | 100 / 80 | 100 / 80 |
| Stage-1 candidates | 3 | 3 | 3 |
| Stage-1 CFG / LPIPS weight / recurrences | 2.0 / 100 / 8 | 2.0 / 100 / 8 | 2.0 / 100 / 8 |
| Stage-2 steps / start | 600 / 350 | 600 / 350 | 600 / 350 |
| Stage-2 LPIPS weight / recurrences | 20 / 8 | 20 / 8 | 20 / 8 |
| Alpha | fixed 0.5 | Adam every 5 steps, lr 0.01 | Adam every 5 steps, lr 0.01 |

## Supplementary runs that deviated from the canonical settings

These experiments use the settings below.

- **Timestep and recurrence grids**: Stage 2 keeps the alpha map fixed at its Stage-1
  estimate (`alphas=constant`) instead of continuing to optimize it.
- **Candidate budget**: every scenario uses the 80-step/64-start Stage-1 sampler and a fixed
  Stage-2 alpha. The *unknown-scalar* rows were sampled with the known-scalar configuration on
  the unknown-scalar pairs, so the true alpha was given to the method.
- **Automated prompts**: Stage 1 uses the 80-step/64-start sampler and Stage 2 a fixed alpha.
  As described in the paper, the VLM prompts come from the spatially varying blend, but the
  Auto arm is sampled on the alpha = 0.5 blend of the same pair, while None and Manual use the
  spatial blends (`manifests/auto_prompts/dataset_auto_as_run.csv`). The default samples all three arms on the same spatial blends; pass `--legacy-blends`
  for the older variant.
- **DoubleDIP best-of-three**: the paper's selection paired outputs with prompts in
  ground-truth source order; see `docs/BASELINES.md` for the flags that reproduce it.
