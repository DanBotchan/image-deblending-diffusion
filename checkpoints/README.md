# Model checkpoints

Checkpoints are intentionally excluded from Git.

1. Accept the Stable Diffusion v1.4 license at <https://huggingface.co/CompVis/stable-diffusion-v-1-4-original>, download `sd-v1-4.ckpt`, and place it at `checkpoints/sd-v1-4.ckpt`.
2. Download OpenAI's unconditional 256x256 ImageNet model from <https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt> and place it at `checkpoints/256x256_diffusion_uncond.pt`.

Alternative locations can be supplied without editing YAML:

```bash
export DEBLEND_SD14_CKPT=/path/to/sd-v1-4.ckpt
export DEBLEND_DDPM_CKPT=/path/to/256x256_diffusion_uncond.pt
```

SHA-256 hashes of the verified checkpoint files will be added soon.
