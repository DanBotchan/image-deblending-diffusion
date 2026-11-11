# Baselines

Upstream baseline code is not vendored. Each baseline is pinned to an upstream commit; our
changes ship either as a patch against that commit (BIDeN, DAD) or, when the code is our own,
as source under `baselines/` (GIP PyTorch port, LCD reimplementation, DoubleDIP adapter).

Every evaluation command below was checked against the outputs behind Table 1: on the first
three pairs of each table cell, rerunning it from the stated checkpoint reproduces the stored
images to within one intensity level (0 for GIP and LCD).

| Method | Upstream | Our code | Checkpoints used for Table 1 |
|---|---|---|---|
| BIDeN | [JunlinHan/BID](https://github.com/JunlinHan/BID) @ `59b7037` (MIT) | `baselines/biden/biden.patch` | official pretrained `biden2` (known); retrained `experiment_scalar` / `alpha_map` |
| DAD | [jiupinjia/Deep-adversarial-decomposition](https://github.com/jiupinjia/Deep-adversarial-decomposition) @ `c983bc6` | `baselines/dad/dad.patch` | three models trained from scratch |
| GIP | method of [Zou et al.](https://github.com/jiupinjia/GANs-for-Inverse-Problems) (TensorFlow, not used) | `baselines/gip/` (our PyTorch implementation) | one model, alpha = 0.5 |
| LCD | method of Montenegro & Cardoso (2025) | `baselines/lcd/` (our implementation) | one model |
| DoubleDIP | [yossigandelsman/DoubleDIP](https://github.com/yossigandelsman/DoubleDIP) @ `4f12629` | `baselines/doubledip/run_doubledip.py` | training-free |

The trained baseline checkpoints are not yet hosted and will be uploaded soon.

The supervised baselines are trained on the BIDeN training set (`BID_train/trainA`,
`BID_train/trainB`) at 256 x 256. Their evaluation scripts read image paths relative to the
manifest's directory, so copy the release manifest next to the images first:

```bash
cp manifests/comparison/*.csv "$DEBLEND_DATA_ROOT"/          # or manifests/generalization/
```

Outputs use the main pipeline layout and are scored with
`scripts/evaluate_artifact_tree.py <results> --expected-pairs 300 ...` (100 for generalization).

## Inconsistencies in the Table 1 baseline runs

The following settings differ between the comparison split and the held-out split in the runs
behind the submitted Table 1. The commands below reproduce the table as run; the
consistent alternative is given next to each.

| Cell | As run | Consistent with the comparison split | Effect (held-out domains, Vehicles / Flowers PSNR) |
|---|---|---|---|
| BIDeN, unknown scalar, held-out | spatial model `BID_random_alpha/.../alpha_map` (best) | scalar model `experiment_scalar` (latest) | 14.52 / 15.65 -> 15.14 / 16.16 |
| DAD, known scalar, held-out | outputs clipped to [0, 1] without the x2 rescale | `--clip_max 0.5` | 11.09 / 11.45 -> 14.43 / 14.42 |
| LCD, known scalar, held-out | sampler `ours_epsilon` (stated in the paper) | sampler `ours` | 17.21 / 17.19 -> 16.62 / 16.62 |

## BIDeN

```bash
git clone https://github.com/JunlinHan/BID.git && cd BID && git checkout 59b7037
git apply /path/to/release/baselines/biden/biden.patch     # adds alpha_train.py / alpha_test.py
```

Training (unknown scalar; spatial adds `--alpha_mode spatial --alpha_min 0 --alpha_max 1
--n_epochs 100 --n_epochs_decay 100 --name alpha_map`):

```bash
python alpha_train.py --dataroot BID_train --model alphabiden --dataset_mode alpha_blend \
  --alpha_mode scalar --alpha_min 0.2 --alpha_max 0.8 --batch_size 8 \
  --n_epochs 500 --n_epochs_decay 500 --name experiment_scalar --output_dir ckpt/BID_random_alpha
```

The known-scalar row uses the authors' released `biden2` weights. Evaluation (all cells):

```bash
python alpha_test.py --dataroot $DEBLEND_DATA_ROOT/<manifest>.csv --model alphabiden \
  --checkpoints_dir <dir> --name <name> --epoch <epoch> --alpha_mode <scalar|spatial> \
  --results_dir <out> --eval --gpu_ids 0 --test_input AB --alpha_min 0 --alpha_max 1
```

| Scenario | `--checkpoints_dir` / `--name` / `--epoch` |
|---|---|
| known scalar | directory containing `biden2/` / `biden2` / `latest` |
| unknown scalar | `ckpt/BID_random_alpha/checkpoints` / `experiment_scalar` / `latest` |
| spatial | `ckpt/BID_random_alpha_map_v2/checkpoints` / `alpha_map` / `best` |

## DAD

```bash
git clone https://github.com/jiupinjia/Deep-adversarial-decomposition.git DAD
cd DAD && git checkout c983bc6 && git apply /path/to/release/baselines/dad/dad.patch
```

Known scalar was trained with `my_train.py`, which uses upstream's mixing (alpha ~ U[0.4, 0.6])
and predicts alpha-scaled sources; unknown scalar and spatial with `alpha_train.py`
(alpha in [0.2, 0.8], respectively polynomial maps in [0, 1]); all with `--net_G unet_256`
for 200 epochs, keeping the best-validation checkpoint. Other training arguments were not
recorded.

```bash
python my_train.py --data_path BID_train --net_G unet_256 --checkpoint_dir ckpt/known
python alpha_train.py --data_path BID_train --net_G unet_256 --alpha_mode scalar \
  --alpha_min 0.2 --alpha_max 0.8 --output_dir ckpt/unknown
python alpha_train.py --data_path BID_train --net_G unet_256 --alpha_mode spatial \
  --alpha_min 0.0 --alpha_max 1.0 --output_dir ckpt/spatial

# Evaluation; --clip_max 0.5 for the known-scalar model, 1.0 otherwise
python alpha_eval_unmix.py --dataset $DEBLEND_DATA_ROOT/<manifest>.csv --net_G unet_256 \
  --ckptdir <dir with best_ckpt.pt> --save_dir <out> --clip_max <0.5|1.0>
```

## GIP

The original GIP code is TensorFlow; the paper uses our PyTorch implementation of the method
for alpha blending (`baselines/gip/igan_deblend_fixed.py`): G predicts x2 from y and
x1 = (y - (1 - alpha) x2) / alpha with alpha = 0.5, trained with the bidirectional
adversarial constraint. The Table 1 checkpoint is the lowest-generator-loss checkpoint, reached
at iteration 44,690 (batch size 16; the iteration budget was not recorded).

```bash
python baselines/gip/igan_deblend_fixed.py --data_path BID_train --checkpoint_dir ckpt/gip/checkpoints \
  --alpha 0.5 --max_iters 50000
python baselines/gip/eval_deblend_fixed.py --checkpoint ckpt/gip/checkpoints/best_ckpt.pth \
  --csv $DEBLEND_DATA_ROOT/dataset.csv --output <out>
```

## LCD

`baselines/lcd/` is our implementation of Montenegro & Cardoso's cold-diffusion
decomposition (U-Net predicting the first source; mixing schedule alpha_t from 0 to 0.8 over
1,000 steps). Table 1 uses the EMA weights after 47,000 steps at learning rate 1e-4.

```bash
cd baselines/lcd
python train_predict_original_BID.py --data_path BID_train --train_lr 1e-4 \
  --results_folder ckpt/lcd                       # Table 1 used model-00047000.pt
python inference_BID_eval.py --model_path ckpt/lcd/model-00047000.pt \
  --dataset $DEBLEND_DATA_ROOT/dataset.csv --save_dir <out> --image_size 256 \
  --sampling_method ours --prediction_type x_orig  # held-out split as run: ours_epsilon
```

## DoubleDIP best-of-three

```bash
git clone https://github.com/yossigandelsman/DoubleDIP.git /path/to/DoubleDIP
git -C /path/to/DoubleDIP checkout 4f12629c740402a57776f070b99e9aee880daf1f

S=spatial   # known_scalar | unknown_scalar | spatial
python baselines/doubledip/run_doubledip.py --doubledip-root /path/to/DoubleDIP \
  --data-root "$DEBLEND_DATA_ROOT" --manifest manifests/comparison/dataset_random_alpha_map.csv \
  --scenario $S --output-dir outputs/doubledip/$S/samples       # --pair-start/--pair-end to shard
python scripts/select_candidates.py outputs/doubledip/$S/samples --selector pickscore \
  --output-csv outputs/doubledip/$S/pickscore.csv               # or --selector clip
python scripts/evaluate_artifact_tree.py outputs/doubledip/$S/samples \
  --candidates-csv outputs/doubledip/$S/pickscore.csv --experiment doubledip \
  --project-root "$PWD" --expected-pairs 300 \
  --output-csv outputs/doubledip/$S/metrics.csv --output-json outputs/doubledip/$S/metrics.json
```

The adapter keeps the optimization used for the paper (two image DIPs plus an alpha DIP in
the optimized-alpha scenarios, reconstruction MSE + 0.1 x exclusion loss + an early alpha
prior, best re-blend iterate, 8,000 iterations). By default it never reads ground truth before
evaluation and selection uses the same permutation-invariant rule as our method. The paper's
best-of-three table instead reordered each output pair to the ground-truth sources and scored
only that order; reproduce it with `run_doubledip.py --order-by-ground-truth` and
`select_candidates.py --direct-order`.
