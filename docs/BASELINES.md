# Baseline policy

Baseline repositories are not vendored. The public release should pin each upstream revision and ship only small patches, dataset adapters, and invocation scripts needed to reproduce the paper protocol.

| Method | Paper protocol | Release status |
|---|---|---|
| BIDeN | Official pretrained model for known alpha; retrained for unknown scalar and polynomial spatial alpha | Adapter and exact upstream revision still required |
| DAD | Retrained from scratch for each of the three alpha scenarios | Adapter and exact upstream revision still required |
| GIP | Trained/evaluated only for known alpha 0.5 because inversion requires known scalar alpha | Adapter and exact upstream revision still required |
| LCD | Cold-diffusion anymorph training for known alpha 0.5 only | Adapter and exact upstream revision still required |
| DoubleDIP | 256 x 256, learning rate 1e-3, three random seeds, permutation-invariant PickScore best-of-three | Upstream `yossigandelsman/DoubleDIP` commit `4f12629c740402a57776f070b99e9aee880daf1f`; local patch still required |

## Blocking protocol discrepancy

The paper currently contains two incompatible DoubleDIP iteration counts:

- `appendix/3_1_baseline_model_training.tex` says 4,000 iterations.
- `appendix/3_4_2_baseline_oversampling.tex` says 8,000 iterations for the best-of-three runs used by the main comparison.

Do not publish a baseline command until the raw run configuration is checked and the manuscript is made consistent. The release should encode one verified value rather than silently choosing one.
