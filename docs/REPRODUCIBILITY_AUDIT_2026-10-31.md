# Reproducibility audit — 2026-10-31

**Verdict: the release does not yet reproduce every experiment in the current local paper through its documented commands.** Main-table evaluation is consistent with the stored results. Missing supplementary implementations, a stale automated-prompt default, inconsistent metric definitions, and changed guidance frequency prevent a blanket reproduction claim. Small fresh diffusion runs also differ from the original generations.

This audit targets `../PaperOverleaf/` and the original artifacts in `../Experiments/`, `../image_gallery/paper/`, and `../deblend/`. It does not use the older ICML PDF as the current manuscript. Production sampling code and original artifacts were not modified.

## Cost and interpretation

- Re-score three fixed pairs per saved experiment cell with the release evaluator. This avoids regenerating the expensive grids.
- Generate one pair/seed at full release sampler settings in representative main cells, plus one controlled legacy-frequency comparison. These are numerical spot checks, not reduced-step smoke tests. Once the objective mismatch was found, queued ablation sampling was cancelled before launch to avoid paying for comparisons with an already known protocol difference. The four active refinements were allowed to finish. The one-seed checks do **not** reproduce the main experiment's three-candidate selection budget.
- Re-score all three candidates on three pairs in each of the six main scenario/split settings to check frozen selection. No diffusion generation is needed for this check.
- Aggregate existing full-run metric files when checking published table arithmetic; reading those files is cheap and is distinct from regenerating the full experiment.
- A few pairs cannot establish that a rerun matches a population mean, confidence interval, ablation trend, or ranking. Compare fresh and old scores on the **same pair and seed**, not the subset mean against a 100/300-pair paper mean.

All scripts, exact subprocess commands, logs, manifests, environment details, scores, and generated samples are retained in [`outputs/repro_audit_20261031/`](../outputs/repro_audit_20261031/). Baseline checkpoints are deserialized with `weights_only=True` and an explicit whitelist of NumPy numeric metadata types.

## Experiment coverage

| Paper experiment | Release support | Audit outcome |
|---|---|---|
| Main comparison: 3 scenarios × 2 splits | `main` | Stored subset scores and frozen selection checked in all six cells; representative fresh Stage 1 and Stage 2 checks recorded below. |
| Two-stage ablation | `main` intermediate/final metrics | Exact Stage 1/Stage 2 roots and selected IDs checked; all 18 current table entries match stored aggregates. |
| Text guidance | `text-guidance` | Three saved pairs for each of 20 guidance/scenario cells; fresh ablation generation omitted after the objective mismatch was found. |
| Starting timestep | `timestep` | Three saved pairs in every pinned grid cell; no new grid generation. |
| Recurrence count | `recurrence` | Three saved pairs in all 20 displayed grid cells; no new grid generation. |
| PickScore vs. CLIP | `selector` | Frozen PickScore choices reproduced on all 18 sampled inputs, with the same three original Stage 1 candidates. CLIP also reproduces all 18 historical choices. |
| Candidate budget | `candidate-budget` | Three saved pairs at all 8 budgets in all 3 scenarios; **driver metric definition differs from the paper table**. No new budget-pool generation. |
| Automated prompts | `auto-prompts` | Saved original arms and newer release reruns inspected; **current paper needs `--matched-blends`**. Three matched Auto outputs re-scored; no new Auto sampling. Frozen prompts used; VLM generation not rerun. |
| DoubleDIP best-of-three | standalone adapter | All 3 candidates on 3 saved pairs in each of 6 scenario/split cells re-scored. Fresh 8,000-iteration seed 0 executes but differs by +2.02 dB PSNR from the original; it does not validate best-of-three generation across all cells. |
| Supervised baselines | patches/adapters, external weights | Saved subset scores checked across table cells; fresh GIP and LCD inference checked below. No retraining, and no fresh BIDeN/DAD inference in this audit. |
| Alpha-map accuracy | **no release table/fit driver** | Reconstructed optimized alpha on three selected inputs in each of 4 Ours scenario/split cells; no shipped least-squares baseline fitting or complete alpha table. |
| Out-of-family mattes | **not implemented in release** | Original six experiment cells located and sampled outputs inspected; Gaussian/reflection GT constructors and `exp_grid4` live only in development experiments. |

## Findings requiring changes

### 1. Out-of-family matte experiment is absent

The current supplement includes a 3-matte-family × 2-estimator experiment in `appendix/3_4_4_alpha_out_of_family.tex`. The release contains no driver or registration for `exp_gt_gaussian`, `exp_gt_reflection`, or `exp_grid4`. The implementation exists only in `../deblend/experiments/alpha_generalization/{variants.py,run_variant.py,full_pipeline_screen.py}`.

The original paper confirmation uses manifest rows **100–119**, one candidate, Stage 1 `80/64`, 8 recurrences, and LPIPS guidance weight 100. The six original aggregate rows match the numbers printed in that supplementary table. To make this reproducible from the release, ship the matte definitions, estimator, exact subset, runner, and matching metric protocol. Re-running the standard spatial experiment cannot substitute for this experiment.

### 2. Alpha-map accuracy table lacks its evaluation implementation

`paper_eval/artifact_tree.py` emits image metrics, not the supplementary alpha MAE table. That table requires optimized-alpha reconstruction for Ours, least-squares scalar/polynomial fitting from each supervised baseline's recovered sources and observation, clipping, and complementing alpha after a source swap. These operations are not exposed by the release drivers. `alpha_checks.json` verifies that existing selected Ours alpha parameters can be read and reconstructed; it does not certify the full table or baseline fits.

### 3. Automated-prompt default is stale relative to the current paper

`paper_repro/experiments.py` defaults to `dataset_auto_as_run.csv`, whose Auto arm uses alpha=0.5 blends. The current paper describes the same **spatial** blends for all arms. Its Auto row (15.63 / .658 / .296) matches the newer release run in `outputs/auto_rerun/confirmation/auto/metrics.json`: **15.627902 / .657580 / .296329**, using `dataset_random_alpha_map_auto.csv`.

Use:

```bash
python scripts/reproduce.py auto-prompts --matched-blends --pairs 3 \
  --data-root ../Datasets/BID_eval_v2 --output-root outputs/auto_audit
```

The experiment map should make the current-paper protocol the default or explicitly label the required flag. The historical as-run variant can remain separately available. This audit does not attribute the separate `given05` run's scores to the default as-run variant: given-alpha and estimated-alpha runs are different protocols.

### 4. Candidate-budget scores use different metric definitions

The original budget table is generated by `../deblend/experiments/pickscore_saturation/analyze.py` from `details.json["metrics"]["mean"]`. Those are sampling-time metrics from `utils/metrics.py`: RGB SSIM and **VGG** LPIPS on floating-point tensors. The release driver evaluates PNGs through `paper_eval/artifact_tree.py`: grayscale SSIM and **AlexNet** LPIPS at 256 pixels. Resizing and quantization also affect PSNR.

For the **same saved images**, known-scalar pair 0 at budget 1:

| Evaluation | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Original budget table protocol | 21.313200 | .640466 | .304297 |
| Release driver evaluator | 21.317267 | .665596 | .202982 |

Across the three-pair budget checks, absolute deltas reach .8804 dB PSNR, .05961 SSIM, and .11724 LPIPS. This is an evaluation mismatch, not evidence of sampling drift. To reproduce the existing table, retain its sampling-time metric profile and summarize those logs. Alternatively, re-evaluate and update the paper consistently. The out-of-family table also reads sampling-time metrics and requires the corresponding profile.

### 5. LPIPS guidance frequency changes the legacy sampling objective

The release's `configs/general/default.yaml` sets `lpips_freq: 4` and none of the
paper scenario configs overrides it. `conditions/registry.py` passes this into
`conditions/loss_utils.py`, which omits the LPIPS term unless
`step_idx % lpips_freq == 0`. The legacy samplers in both
`../stable-diffusion-guided/conditions/ladibi.py` and
`../debleding/conditions/ladibi.py` compute that term on every guided timestep.
The saved original main-run configs have no frequency override.

This is a change to the objective, not merely a faster equivalent implementation.
The existing unit tests assert the LPIPS **weight** but do not assert its
**frequency**. The controlled full-setting run with `general.lpips_freq=1` is reported below.
It does **not** recover the original score: 19.8548 versus 17.3855 dB.
This audit leaves production defaults unchanged; other code and stack
differences may remain. For example, the release replaces the legacy explicit
attention softmax/matrix products with PyTorch scaled dot-product attention;
this is mathematically equivalent but can change floating-point results. The fresh default-setting runs below use frequency 4. Newer development experiments
(candidate budget, automated prompts, and the out-of-family confirmation) have
`lpips_freq: 4` explicitly saved in their own configs. Preserve those as-run
settings separately; a global frequency change without per-experiment overrides
would not reproduce all experiments either.

### 6. Fresh diffusion samples are not exact regenerations

The full-setting same-seed comparisons below use the release evaluator on both new and original images. They establish execution and quantify numerical differences; they do not establish statistical reproduction of the paper's aggregate results. Existing `docs/VERIFICATION.md` already notes legacy-code/environment differences, but its statistical-faithfulness wording should not be inferred from a one-pair check. The guidance-frequency mismatch above must also be resolved.

All rows below use pair 0, candidate 0 and its fixed manifest seed, evaluated at 256 pixels. They compare the same candidate ID, including candidates that were not the paper selector’s winner. Stage 2 starts from the fresh Stage 1 output, so its difference includes propagated Stage 1 changes.

| Cell | Old PSNR | New PSNR | Δ PSNR | Old → new SSIM | Old → new LPIPS |
|---|---:|---:|---:|---:|---:|
| comparison/known_scalar/stage1 | 19.8007 | 19.4270 | -0.3737 | 0.4972 → 0.4907 | 0.4594 → 0.4777 |
| comparison/known_scalar/stage2 | 19.9248 | 21.3173 | +1.3924 | 0.5979 → 0.6656 | 0.3143 → 0.2030 |
| comparison/spatial/stage1 | 17.3855 | 19.0004 | +1.6149 | 0.4695 → 0.5086 | 0.4657 → 0.4727 |
| comparison/spatial/stage2 | 19.4073 | 21.4655 | +2.0583 | 0.6404 → 0.7046 | 0.2999 → 0.2290 |
| comparison/unknown_scalar/stage1 | 18.6650 | 17.3738 | -1.2911 | 0.4872 → 0.4904 | 0.4599 → 0.4336 |
| comparison/unknown_scalar/stage2 | 20.0800 | 19.1579 | -0.9220 | 0.6258 → 0.6957 | 0.2900 → 0.1786 |
| generalization/spatial/stage1 | 14.6343 | 15.0883 | +0.4541 | 0.5964 → 0.6294 | 0.3087 → 0.3426 |
| generalization/spatial/stage2 | 17.2956 | 18.3991 | +1.1035 | 0.8216 → 0.8674 | 0.1177 → 0.0746 |
| legacy_frequency/stage1 | 17.3855 | 19.8548 | +2.4693 | 0.4695 → 0.5254 | 0.4657 → 0.4127 |

### 7. DoubleDIP also differs on fresh same-seed generation

The adapter completes 8,000 iterations on known-scalar pair 0, seed 0, with the
paper's output-order flag. Using the same evaluator, the new result scores
**18.946124 / .508672 / .568788**, versus the original's
**16.922573 / .504844 / .605861** (PSNR / SSIM / LPIPS). Both original and new
runs use learning rate .001 and the same image size. Thus the original saved
three-candidate pools and their selection scores can be reproduced in evaluation,
but this check does **not** certify regeneration of those pools. The adapter
uses cuDNN benchmarking, and this audit has not isolated the cause of the
trajectory difference; it should not be attributed to a specific cause without
further controlled checks. See `doubledip_comparison.json`.

## Other constraints and successful checks

- Pinned raw-image checks cover **330 records across 110 cells** (three pairs each). PSNR and SSIM match the stored scores exactly; the maximum LPIPS difference is **2.91e-5**, consistent with the CPU/GPU evaluation difference. Supplementary checks add 159 records; repeated budget winners are reused, not independent samples.
- `pytest -q`: **38 passed** in 46.26 seconds; one expected perfect-image PSNR divide-by-zero warning.
- All 186 current main-table and two-stage numeric entries match the stored full-run aggregates at four decimals. These aggregates were not all freshly regenerated in this audit. Main-table DoubleDIP uses the best-of-three selected outputs, not the older single-shot results.
- Re-scoring 18 sampled inputs from the pinned Stage 1 roots reproduces **18/18** frozen PickScore choices and **18/18** historical CLIP choices. Historical `Ours_three_samples/results` links can target Stage 2; use the explicit roots in `../deblend/configs/paper_audit/pickscore_selector.json` when checking original lineage.
- Fresh GIP CPU inference on three pairs per split differs by at most **one intensity level** from the original GPU outputs; maximum score differences are .00492 dB PSNR, .000183 SSIM and .000122 LPIPS. Fresh LCD outputs are **pixel-identical** for one pair per split, using the as-run sampler for each split; see `baseline_checks.json`.
- BIDeN/DAD need upstream code plus the release patches; trained weights are external. DAD's complete original training arguments and GIP's original iteration budget were not recorded (`docs/BASELINES.md`), so exact retraining is not established. This audit deliberately avoids retraining.
- The three documented held-out baseline setting differences (BIDeN checkpoint, DAD clipping, LCD sampler) matter when reproducing Table 1. Follow the as-run exceptions in `docs/BASELINES.md`.
- No new pretrained/VLM weights were downloaded. Missing release experiments were not silently executed through development-only code and declared release successes.

## Evidence and reruns

See the audit directory for:

- `environment.json`: release commit, package versions, sample policy, test outcome.
- `input_hashes.json`: SHA-256 hashes of the manuscript, release configs, and manifests used for this audit.
- `pinned_artifact_checks.{json,csv}`: pinned original subset image scores versus stored scores.
- `supplement_checks.json`: budgets, automatic prompts, DoubleDIP candidates, and out-of-family subsets; metric-profile differences are retained explicitly.
- `selection_checks.json`: PickScore and CLIP scores/winners for the sampled original candidate pools.
- `current_table_checks.json`: 186 current manuscript values versus stored aggregates.
- `alpha_checks.json`: sampled optimized-alpha reconstruction and registry coverage.
- `fresh_comparison.json`: same-pair/seed fresh-versus-original diffusion comparisons.
- `baseline_checks.json` and `baseline_score_checks.json`: pixel and score differences for fresh baseline inference.
- `doubledip_comparison.json`: fresh/original DoubleDIP sample scores.
- `guidance_frequency_check.json`: a small loss-function probe demonstrating that frequency 4 omits the LPIPS term on three out of four timesteps.
- `run_sampling.py`, `run_followups.py`, `run_legacy_frequency.py`: full-setting, bounded generation commands; `logs/` contains each exact invocation. `followups.log` ends with an intentional `KeyboardInterrupt` to cancel future phases; the four already-active Stage 2 subprocesses were allowed to finish and their complete artifacts were checked separately.
- `check_pinned_artifacts.py`, `check_supplement.py`, `check_selection.py`, `check_alpha.py`, `check_tables.py`, `compare_fresh.py`: read-only scoring/inspection programs. They rely on the original project's local artifacts and are not standalone public-release reproductions.

Do not increase `--pairs` on an existing run directory and assume evaluation is refreshed: the release runner skips existing selection CSVs and metric JSONs. Use a new output root when changing sample limits or settings.

## Controlled guidance-frequency result

On spatial pair 0, seed 1220058128, changing only `general.lpips_freq` from 4 to 1 changes Stage 1 PSNR from **19.0004** to **19.8548 dB**; the original scores **17.3855 dB**. The control has SSIM **0.5254** and LPIPS **0.4127**, versus original **0.4695 / 0.4657**. Restoring the legacy frequency alone therefore does not restore the original score. Further differences remain to be isolated. This is one controlled sample, not a validation of every experiment or aggregate score. See `frequency_comparison.json`.
