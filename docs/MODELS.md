# Models

The paper uses exactly two pretrained diffusion priors:

- Stage 1: Stable Diffusion v1.4, operating on 64 x 64 latents for 512 x 512 images.
- Stage 2: OpenAI guided-diffusion's unconditional ImageNet 256 x 256 model.

Neither checkpoint is committed. Follow `checkpoints/README.md`, retain the upstream licenses, and set `DEBLEND_SD14_CKPT` or `DEBLEND_DDPM_CKPT` when storing them elsewhere.

PickScore is loaded through Transformers with frozen model and processor revisions. Use `scripts/select_candidates.py --offline` after both repositories have been cached locally, or omit `--offline` to allow the first run to download them.
