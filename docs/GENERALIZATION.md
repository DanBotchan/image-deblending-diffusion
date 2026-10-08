# Held-out domain generalization

The exact 100-pair Vehicle--Flower manifests are in `manifests/generalization/`. Run the main
experiment on them with the same method configuration, pointing `--data-root` at the
generalization images:

```bash
for S in known_scalar unknown_scalar spatial; do
  python scripts/reproduce.py main --scenario $S --split generalization \
    --data-root /absolute/path/to/BID_generalization --gpus 0,1,2,3
done
```

Results are written to `$DEBLEND_OUTPUT_ROOT/generalization/main/<scenario>/summary.md`. Only the
source domains, manifests and pair count differ from the comparison split.
