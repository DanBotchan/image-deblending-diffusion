<div align="center">

# Image Deblending with Diffusion Models

**Dan Botchan · [Ohad Fried](https://www.ohadf.com/) · [Yacov Hel-Or](https://faculty.runi.ac.il/toky/)**<br>
Reichman University, Herzliya, Israel

**WACV 2027**

<!-- Add Paper and arXiv badges here once the links exist. -->
[![Project page](https://img.shields.io/badge/Project-page-2f56c9)](https://danbotchan.github.io/image-deblending-diffusion/)
[![WACV 2027](https://img.shields.io/badge/WACV-2027-a8650a)](#citation)
[![Python 3.9](https://img.shields.io/badge/python-3.9-4e5564)](#installation)

<img src="assets/teaser.png" alt="A blended image of lemons and a cat is separated into the two recovered source images and the recovered alpha map." width="100%">

</div>

Official code for **Image Deblending with Diffusion Models**. Given a single alpha-blended
image and one short text prompt per source, the method recovers **both source images and the
alpha map**. It is training-free: it uses only pretrained diffusion priors and selects among
candidates without ground truth.

- **Two images from one prior.** Deblending is posed as a joint inverse problem, solved with
  diffusion posterior sampling over both sources at once.
- **Prompts settle the ambiguity.** Cross-negative prompting conditions each source on its own
  prompt and uses the other source's prompt as the negative.
- **Coarse to fine.** Stable Diffusion separates the semantics in latent space, then a
  pixel-space ImageNet diffusion model restores fine detail.
- **Spatially varying alpha.** The alpha map is a smooth 6-coefficient polynomial, optimized
  together with the images.

This repository is a paper-only extraction. It intentionally excludes exploratory DeepFloyd,
FLUX, SD3.5, and speed-search code, raw experiment galleries, private datasets, and baseline
source trees.

## Method

<img src="assets/method.webp" alt="Pipeline: Stage 1 text-guided latent diffusion produces N candidate decompositions, PickScore selects one, and Stage 2 refines it in pixel space." width="100%">

1. Stage 1 uses Stable Diffusion v1.4 at 512 x 512 for text-guided semantic separation.
2. Three stochastic candidates are generated with fixed per-pair seeds.
3. A frozen, permutation-invariant PickScore policy selects one candidate without ground truth.
4. Stage 2 uses OpenAI's unconditional 256 x 256 ImageNet diffusion model for pixel-space refinement.
5. PSNR, grayscale SSIM, and LPIPS are evaluated after resolving the two-source ordering by minimum total MSE.

Both stages alternate data-consistency updates of the two samples with an update of the alpha
coefficients every 5 steps. The canonical pipeline takes approximately 50.6--71.0 minutes per
input on one NVIDIA RTX 3090, excluding selector overhead.

## Results

Known scalar alpha on the held-out Vehicles and Flowers domains (Table 1 of the paper; TF =
training-free). The full table, covering seen domains and the unknown-scalar and
spatially-varying scenarios, is in the paper and on the [project page](https://danbotchan.github.io/image-deblending-diffusion/).

| Method | TF | Vehicles PSNR ↑ | SSIM ↑ | LPIPS ↓ | Flowers PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|:-:|--:|--:|--:|--:|--:|--:|
| DAD | | 11.088 | 0.432 | 0.504 | 11.446 | 0.387 | 0.522 |
| GIP | | 14.075 | 0.509 | 0.508 | 14.090 | 0.482 | 0.504 |
| BIDeN | | 15.437 | 0.552 | 0.385 | 15.690 | 0.542 | 0.379 |
| LCD | | 17.208 | 0.689 | 0.293 | 17.186 | 0.655 | 0.298 |
| DoubleDIP | ✓ | 13.696 | 0.480 | 0.510 | 13.709 | 0.437 | 0.525 |
| **Ours** | ✓ | **20.229** | **0.817** | **0.121** | **20.417** | **0.806** | **0.117** |

## Installation

The recorded environment uses Python 3.9, PyTorch 2.5.1+cu121, and CUDA 12.1.

```bash
conda create -n deblend-paper python=3.9 -y
conda activate deblend-paper
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

See `requirements-lock-cu121.txt` for the recorded package versions. The Stable Diffusion
checkpoint is license-gated and must be obtained separately; see `docs/MODELS.md`.

## Data layout

Set the dataset root before running an experiment:

```bash
export DEBLEND_DATA_ROOT=/absolute/path/to/BID_eval_v2
export DEBLEND_OUTPUT_ROOT=$PWD/outputs        # optional, default ./outputs
```

The exact annotation manifests (pairs, prompts, alpha coefficients, seeds) are versioned under
`manifests/comparison` and `manifests/generalization`. Their relative image paths resolve below
`DEBLEND_DATA_ROOT`, which must contain the source images:

```text
$DEBLEND_DATA_ROOT/
└── pairs/
    ├── pair_000/img1.png, img2.png, blended_polynomial.png, ...
    └── ...
```

See `docs/DATA.md` for the CSV schema and dataset preparation policy.

## Reproduce the paper

One driver runs every experiment end to end (Stage 1, ground-truth-free selection, Stage 2,
evaluation) and writes the corresponding table to `summary.md`:

```bash
# Check the installation in a few minutes: 1 pair, ~10x fewer steps (numbers are meaningless)
python scripts/reproduce.py main --scenario spatial --smoke

# Main result and two-stage ablation for one scenario, sharded over four GPUs
python scripts/reproduce.py main --scenario known_scalar --gpus 0,1,2,3

# Held-out Vehicles/Flowers domains
python scripts/reproduce.py main --scenario spatial --split generalization \
  --data-root /absolute/path/to/BID_generalization --gpus 0,1,2,3
```

Scenarios are `known_scalar`, `unknown_scalar` and `spatial`. The supplementary experiments
are `text-guidance`, `timestep`, `recurrence`, `selector`, `candidate-budget` and
`auto-prompts`; `docs/EXPERIMENTS.md` maps each paper table to its command and protocol.
Add `--dry-run` to print every command without running it. Rerunning a command resumes it.

### Running the stages by hand

The driver chains the same four scripts (shown here for `known_scalar` with the default paths):

```bash
python scripts/first_stage.py +paper=known_scalar_stage1

python scripts/select_candidates.py \
  outputs/known_scalar/stage1/results \
  --output-csv outputs/known_scalar/selected.csv

python scripts/second_stage.py +paper=known_scalar_stage2 \
  datasets.annotations_path=$PWD/outputs/known_scalar/selected.csv

python scripts/evaluate_artifact_tree.py \
  outputs/known_scalar/stage2/results \
  --output-csv outputs/known_scalar/metrics.csv \
  --output-json outputs/known_scalar/metrics.json \
  --experiment known_scalar \
  --project-root "$PWD" \
  --expected-pairs 300
```

The optimized-alpha scenarios use `100` Stage-1 steps starting at `80`; known scalar uses
`80` steps starting at `64`. All Stage-2 scenarios use `600` steps starting at `350` and
eight recurrence iterations. One Stage-1 candidate takes roughly 10-18 minutes and one Stage-2
refinement roughly 18 minutes on an RTX 3090.

## Documentation

| File | Contents |
|---|---|
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Every paper table and its command and protocol |
| [`docs/DATA.md`](docs/DATA.md) | Manifest schema, alpha coefficients, seeds |
| [`docs/MODELS.md`](docs/MODELS.md) | Pretrained checkpoints and where to put them |
| [`docs/BASELINES.md`](docs/BASELINES.md) | Baseline methods, adaptations, the DoubleDIP adapter |
| [`docs/GENERALIZATION.md`](docs/GENERALIZATION.md) | The held-out Vehicles/Flowers split |
| [`docs/VERIFICATION.md`](docs/VERIFICATION.md) | How the release was checked against the paper |

## Tests

```bash
pytest
```

The tests need no GPU. They cover artifact discovery, source-order reassignment,
same-candidate Stage-1/Stage-2 lineage, metric aggregation, the selection rules, and that
every command emitted by `scripts/reproduce.py` composes to the intended Hydra config.

## Baselines

Baseline implementations are not vendored. `docs/BASELINES.md` records the upstream methods,
the paper adaptations, and the DoubleDIP adapter in `baselines/doubledip/`.

## Citation

```bibtex
@inproceedings{botchan2027deblending,
  title     = {Image Deblending with Diffusion Models},
  author    = {Botchan, Dan and Fried, Ohad and Hel-Or, Yacov},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on
               Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Acknowledgements

This work builds on [Stable Diffusion](https://github.com/CompVis/stable-diffusion),
OpenAI's [guided-diffusion](https://github.com/openai/guided-diffusion) and
[PickScore](https://github.com/yuvalkirstain/PickScore). See `THIRD_PARTY_NOTICES.md` for the
upstream code and licenses.

## License status

The project-owned code is currently all-rights-reserved until the authors choose an open-source license. Upstream-derived files and pretrained weights retain their own terms. See `LICENSE` and `THIRD_PARTY_NOTICES.md` before redistribution.
