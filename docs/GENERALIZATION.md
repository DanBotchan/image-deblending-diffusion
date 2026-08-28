# Held-out domain generalization

The exact 100-pair Vehicle--Flower manifests are in `manifests/generalization/`. Point the same paper configs at those images and manifests while writing to a separate output root:

```bash
export DEBLEND_DATA_ROOT=/absolute/path/to/BID_generalization
export DEBLEND_MANIFEST_ROOT=$PWD/manifests/generalization
export DEBLEND_OUTPUT_ROOT=$PWD/outputs/generalization

python scripts/first_stage.py +paper=known_scalar_stage1 general.N_pairs=100
python scripts/select_candidates.py \
  outputs/generalization/known_scalar/stage1/results \
  --output-csv outputs/generalization/known_scalar/selected.csv
export DEBLEND_STAGE2_MANIFEST=$PWD/outputs/generalization/known_scalar/selected.csv
python scripts/second_stage.py +paper=known_scalar_stage2 general.N_pairs=100
```

Repeat with `unknown_scalar` and `spatial`. The configs keep the same method parameters; only the source domains, manifest root, pair count, and output root change.
