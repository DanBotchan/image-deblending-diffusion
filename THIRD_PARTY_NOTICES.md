# Third-party notices

This repository contains adapted upstream source and relies on separately downloaded pretrained weights. Upstream copyright and license terms remain in force.

| Release path or dependency | Upstream project | Role |
|---|---|---|
| `models/latent_diffusion/` | [CompVis Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [CompVis latent-diffusion](https://github.com/CompVis/latent-diffusion) | Stage-1 architecture and checkpoint loading |
| portions of `models/latent_diffusion/` | [CompVis taming-transformers](https://github.com/CompVis/taming-transformers) | VAE support |
| `models/pixel_diffusion/` | [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) | Stage-2 UNet and diffusion process |
| `clip` dependency | [OpenAI CLIP](https://github.com/openai/CLIP), commit `a1d071733d7111c9c014f024669f959182114e33` | Stage-1 text encoder |
| PickScore model | [`yuvalkirstain/PickScore_v1`](https://huggingface.co/yuvalkirstain/PickScore_v1), revision `a4e4367c6dfa7288a00c550414478f865b875800` | Candidate selection |
| PickScore processor | [`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K), revision `1c2b8495b28150b8a4922ee1c8edee224c284c0c` | Candidate selection preprocessing |
| Stable Diffusion v1.4 weights | [`CompVis/stable-diffusion-v-1-4-original`](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) | Stage-1 prior; license acceptance required |
| `baselines/biden/biden.patch` | [JunlinHan/BID](https://github.com/JunlinHan/BID) @ `59b7037`, MIT | Patch against upstream; apply to a checkout |
| `baselines/dad/dad.patch` | [jiupinjia/Deep-adversarial-decomposition](https://github.com/jiupinjia/Deep-adversarial-decomposition) @ `c983bc6`, no license file | Patch against upstream; apply to a checkout |
| `baselines/doubledip/` | [yossigandelsman/DoubleDIP](https://github.com/yossigandelsman/DoubleDIP) @ `4f12629`, academic use | Adapter importing an upstream checkout |
| `baselines/gip/`, `baselines/lcd/` | Methods of Zou et al. (GIP) and Montenegro & Cardoso (LCD) | Our implementations; no upstream code |
| 256x256 unconditional ImageNet checkpoint | [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) | Stage-2 prior |

Applicable license texts recovered so far are stored in `licenses/`. Before a public release, add any remaining upstream notices and verify every adapted file's provenance, and have the copyright holders choose a license for project-owned code. The current root `LICENSE` is deliberately conservative and does not relicense upstream work.
