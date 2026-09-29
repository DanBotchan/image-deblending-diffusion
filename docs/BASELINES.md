# Baseline policy

Baseline repositories are not vendored. The public release should pin each upstream revision and ship only small patches, dataset adapters, and invocation scripts needed to reproduce the paper protocol.

| Method | Paper protocol | Release status |
|---|---|---|
| BIDeN | Official pretrained model for known alpha; retrained for unknown scalar and polynomial spatial alpha | Adapter and exact upstream revision still required |
| DAD | Retrained from scratch for each of the three alpha scenarios | Adapter and exact upstream revision still required |
| GIP | Trained/evaluated only for known alpha 0.5 because inversion requires known scalar alpha | Adapter and exact upstream revision still required |
| LCD | Cold-diffusion anymorph training for known alpha 0.5 only | Adapter and exact upstream revision still required |
| DoubleDIP | 256 x 256, 8,000 iterations, learning rate 1e-3, three random seeds, permutation-invariant PickScore best-of-three | Upstream `yossigandelsman/DoubleDIP` commit `4f12629c740402a57776f070b99e9aee880daf1f`; local patch still required |

## Resolved: DoubleDIP iteration count

Earlier manuscript versions gave two DoubleDIP iteration counts (4,000 in the baseline-training section, 8,000 in the best-of-three section). The raw run logs (`Experiments/DoubleDIP_3samples_queue.log`, `Experiments/DoubleDIP_3samples_HANDOFF.md`) record `num_iter=8000` for the best-of-three runs used in the main comparison. The release and the thesis use 8,000.
