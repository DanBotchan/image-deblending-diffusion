# Baseline policy

Baseline repositories are not vendored. The public release should pin each upstream revision and ship only small patches, dataset adapters, and invocation scripts needed to reproduce the paper protocol.

| Method | Paper protocol | Release status |
|---|---|---|
| BIDeN | Official pretrained model for known alpha; retrained for unknown scalar and polynomial spatial alpha | Adapter and exact upstream revision still required |
| DAD | Retrained from scratch for each of the three alpha scenarios | Adapter and exact upstream revision still required |
| GIP | Trained/evaluated only for known alpha 0.5 because inversion requires known scalar alpha | Adapter and exact upstream revision still required |
| LCD | Cold-diffusion anymorph training for known alpha 0.5 only | Adapter and exact upstream revision still required |
| DoubleDIP | 256 x 256, 8,000 iterations, learning rate 1e-3, three random seeds, permutation-invariant PickScore best-of-three | Adapter in `baselines/doubledip/`, runs against upstream `yossigandelsman/DoubleDIP` commit `4f12629c740402a57776f070b99e9aee880daf1f` |

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
prior, best re-blend iterate). By default it never reads ground truth before evaluation and
selection uses the same permutation-invariant rule as our method. The paper's best-of-three
table instead reordered each output pair to the ground-truth sources and scored only that
order; reproduce it with `run_doubledip.py --order-by-ground-truth` and
`select_candidates.py --direct-order`.

## Resolved: DoubleDIP iteration count

Earlier manuscript versions gave two DoubleDIP iteration counts (4,000 and 8,000). The run logs record `num_iter=8000` for the best-of-three runs used in the main comparison; the paper and this release use 8,000.
