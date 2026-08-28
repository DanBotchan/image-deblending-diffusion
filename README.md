# Image Deblending with Diffusion Models

Official code release for **Image Deblending with Diffusion Models**. The method is training-free: it jointly estimates two source images and the alpha-blending parameters using pretrained diffusion priors.

This repository is a paper-only extraction. It intentionally excludes exploratory DeepFloyd, FLUX, SD3.5, and speed-search code, raw experiment galleries, private datasets, and baseline source trees.

## Method

1. Stage 1 uses Stable Diffusion v1.4 at 512 x 512 for text-guided semantic separation.
2. Three stochastic candidates are generated with fixed per-pair seeds.
3. A frozen, permutation-invariant PickScore policy selects one candidate without ground truth.
4. Stage 2 uses OpenAI's unconditional 256 x 256 ImageNet diffusion model for pixel-space refinement.
5. PSNR, grayscale SSIM, and LPIPS are evaluated after resolving the two-source ordering by minimum total MSE.

The canonical pipeline takes approximately 50.6--71.0 minutes per input on one NVIDIA RTX 3090, excluding selector overhead.

## Installation

The recorded environment uses Python 3.9, PyTorch 2.5.1+cu121, and CUDA 12.1.

```bash
conda create -n deblend-paper python=3.9 -y
conda activate deblend-paper
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

See `requirements-lock-cu121.txt` for the recorded package versions. The Stable Diffusion checkpoint is license-gated and must be obtained separately; see `docs/MODELS.md`.

## Data layout

Set the dataset root before running an experiment:

```bash
export DEBLEND_DATA_ROOT=/absolute/path/to/BID_eval_v2
export DEBLEND_OUTPUT_ROOT=$PWD/outputs
```

The expected annotation files are:

```text
$DEBLEND_DATA_ROOT/
├── dataset.csv
├── dataset_random_alpha_scalar.csv
└── dataset_random_alpha_map.csv
```

See `docs/DATA.md` for the CSV schema and dataset preparation policy.

## Reproduce the main method

Choose one scenario: `known_scalar`, `unknown_scalar`, or `spatial`.

```bash
python scripts/first_stage.py +paper=known_scalar_stage1

python scripts/select_candidates.py \
  outputs/known_scalar/stage1/results \
  --output-csv outputs/known_scalar/selected.csv

export DEBLEND_STAGE2_MANIFEST=$PWD/outputs/known_scalar/selected.csv
python scripts/second_stage.py +paper=known_scalar_stage2

python scripts/evaluate_artifact_tree.py \
  outputs/known_scalar/stage2/results \
  --output-csv outputs/known_scalar/metrics.csv \
  --output-json outputs/known_scalar/metrics.json \
  --experiment known_scalar \
  --project-root "$PWD" \
  --expected-pairs 300
```

The optimized-alpha scenarios use `100` Stage-1 steps starting at `80`; known scalar uses `80` steps starting at `64`. All Stage-2 scenarios use `600` steps starting at `350` and eight recurrence iterations.

## Tests

```bash
pytest
```

The unit tests cover artifact discovery, source-order reassignment, same-candidate Stage-1/Stage-2 lineage, metric aggregation, and selector protocol behavior.

## Baselines and supplementary experiments

Baseline implementations are not vendored. `docs/BASELINES.md` records the upstream methods and the paper adaptations. `docs/EXPERIMENTS.md` maps every main-paper and supplementary experiment to its release command or remaining packaging task.

## License status

The project-owned code is currently all-rights-reserved until the authors choose an open-source license. Upstream-derived files and pretrained weights retain their own terms. See `LICENSE` and `THIRD_PARTY_NOTICES.md` before redistribution.
