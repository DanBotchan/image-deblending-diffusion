"""
Stage-specific helper functions for the deblend pipeline.

Provides model loading, alpha map creation, text embeddings,
and sampler construction for both Stage 1 (latent) and Stage 2 (pixel).
"""
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import torch
from PIL import Image

from alphas.alpha_maps.registry import get_alpha_map
from . import Device, TRANSFORM
from .config import load_details_json, load_alphas_json
from .config import ExpConfig  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_model_latent(exp_config):
    """Load Stage 1 Stable Diffusion model."""
    import torch.nn as nn
    from models.latent_diffusion.util import instantiate_from_config

    model_config = exp_config['model']
    ckpt = model_config['model_path']
    print(f"Loading latent diffusion model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd.get("state_dict", pl_sd)
    model = instantiate_from_config(model_config['model'])
    m, u = model.load_state_dict(sd, strict=False)
    if m:
        print(f"Missing keys: {len(m)}")
    if u:
        print(f"Unexpected keys: {len(u)}")
    model = nn.DataParallel(model)
    model = model.to(exp_config.get_device())
    model.eval()
    # Disable gradient checkpointing for inference — trades memory for speed
    for module in model.module.model.diffusion_model.modules():
        if hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = False

    if exp_config['general'].get('compile_model', False):
        print("Compiling latent UNet with torch.compile...")
        model.module.model.diffusion_model = torch.compile(
            model.module.model.diffusion_model, mode='reduce-overhead'
        )
    return model


def get_model_pixel(exp_config):
    """Load Stage 2 DDPM pixel diffusion model."""
    from models.pixel_diffusion.unet import create_model

    model_config = exp_config['model']
    model = create_model(
        image_size=model_config['image_size'],
        num_channels=model_config['num_channels'],
        num_res_blocks=model_config['num_res_blocks'],
        channel_mult=model_config['channel_mult'],
        learn_sigma=model_config['learn_sigma'],
        class_cond=model_config['class_cond'],
        use_checkpoint=model_config['use_checkpoint'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        num_head_channels=model_config['num_head_channels'],
        num_heads_upsample=model_config['num_heads_upsample'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
        dropout=model_config['dropout'],
        resblock_updown=model_config['resblock_updown'],
        use_fp16=model_config['use_fp16'],
        use_new_attention_order=model_config['use_new_attention_order'],
        model_path=model_config['model_path'],
    )
    model = model.to(exp_config.get_device())
    model.eval()
    if exp_config['general'].get('compile_model', False):
        print("Compiling pixel UNet with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    return model


# ---------------------------------------------------------------------------
# Pixel-space sampler construction (Stage 2)
# ---------------------------------------------------------------------------

def get_pixel_sampler(exp_config):
    """Create Stage 2 DDPM sampler from config."""
    # Load the project's pixel sampler package before querying the shared
    # registry. Its import registers the configured LADiBI sampler.
    import samplers.pixel  # noqa: F401
    from models.pixel_diffusion.samplers.sampler_utils import create_sampler

    sampler_config = exp_config['sampler']
    return create_sampler(
        sampler=sampler_config['type'],
        steps=sampler_config['steps'],
        noise_schedule=sampler_config['noise_schedule'],
        model_mean_type=sampler_config['model_mean_type'],
        model_var_type=sampler_config['model_var_type'],
        dynamic_threshold=sampler_config['dynamic_threshold'],
        clip_denoised=sampler_config['clip_denoised'],
        rescale_timesteps=sampler_config['rescale_timesteps'],
        timestep_respacing=sampler_config['timestep_respacing'],
    )


# ---------------------------------------------------------------------------
# Alpha map creation
# ---------------------------------------------------------------------------

def _get_img_size(exp_config) -> int:
    """Get image size from dataset config."""
    return exp_config['datasets']['img_size']


def create_alpha_map_gt_stage1(exp_config,
                               data_item: Dict[str, Any]):
    """
    Create ground-truth alpha map for Stage 1.
    Type and init params come from config (not data_item).
    """
    alphas = exp_config['alphas']
    return get_alpha_map(
        name=alphas['gt_map_type'],
        init_alpha=data_item.get('alpha_gt', alphas['gt_init_alpha_params']),
        device=exp_config.get_device(),
        size=_get_img_size(exp_config),
        alpha_min=alphas['gt_min'],
        alpha_max=alphas['gt_max'],
    )


def create_alpha_map_gt_stage2(exp_config,
                               data_item: Dict[str, Any]):
    """
    Create ground-truth alpha map for Stage 2.
    Type and init params come from the data_item (loaded from Stage 1 output).
    """
    alphas = exp_config['alphas']
    return get_alpha_map(
        name=data_item['alpha_gt_type'][0],
        init_alpha=data_item['alpha_gt'],
        device=exp_config.get_device(),
        size=_get_img_size(exp_config),
        alpha_min=alphas['gt_min'],
        alpha_max=alphas['gt_max'],
    )


def create_alpha_map_pred_stage1(exp_config, alpha_map_gt):
    """
    Create predicted alpha map for Stage 1.
    Optionally initialized from GT map if pred_map_type == 'same'.
    """
    alphas = exp_config['alphas']
    img_size = _get_img_size(exp_config)
    if alphas['pred_map_type'] == 'same':
        return get_alpha_map(
            name=alpha_map_gt.map_type,
            alpha_min=alphas['pred_min'],
            alpha_max=alphas['pred_max'],
            init_alpha=alpha_map_gt.get_scaled_parameters(),
            device=exp_config.get_device(),
            size=img_size,
            requires_grad=True,
        )
    return get_alpha_map(
        name=alphas['pred_map_type'],
        init_alpha=alphas['pred_init_alpha_params'],
        device=exp_config.get_device(),
        size=img_size,
        alpha_min=alphas['pred_min'],
        alpha_max=alphas['pred_max'],
        requires_grad=True,
    )


def create_alpha_map_pred_stage2(exp_config,
                                 data_item: Dict[str, Any],
                                 requires_grad: bool):
    """
    Create predicted alpha map for Stage 2.
    Type and init params come from the data_item.
    """
    alphas = exp_config['alphas']
    return get_alpha_map(
        name=data_item['alpha_pred_type'][0],
        init_alpha=data_item['alpha_pred'],
        device=exp_config.get_device(),
        size=_get_img_size(exp_config),
        alpha_min=alphas['pred_min'],
        alpha_max=alphas['pred_max'],
        requires_grad=requires_grad,
    )


# ---------------------------------------------------------------------------
# Text embeddings (Stage 1 only)
# ---------------------------------------------------------------------------

def get_text_emb(exp_config, model, data_item: Dict[str, Any]):
    """
    Build positive and negative CLIP text embeddings for Stage 1.

    Returns (pos_text_emb, neg_text_emb).
    """
    device = exp_config.get_device()

    if exp_config['general']['cfg'] == 0.0:
        pos_text_emb = model.module.get_learned_conditioning(['', ''])
        neg_text_emb = model.module.get_learned_conditioning(['', ''])
        return pos_text_emb.to(device), neg_text_emb.to(device)

    for key in ['image1_prompt', 'image2_prompt', 'image1_class', 'image2_class']:
        if not isinstance(data_item[key], list):
            data_item[key] = [data_item[key]]

    prompts_config = exp_config['prompts']
    prompts = data_item['image1_prompt'] + data_item['image2_prompt']
    classes = data_item['image1_class'] + data_item['image2_class']

    pos_text_emb = model.module.get_learned_conditioning(
        [prompts_config['positive_template'].format(prompt=p, cls=c)
         for p, c in zip(prompts, classes)]
    ).to(device)

    neg_text_emb = model.module.get_learned_conditioning(
        [prompts_config['negative_template'].format(neg_prompt=p, neg_cls=c)
         for p, c in zip(prompts[::-1], classes[::-1])]
    ).to(device)

    return pos_text_emb, neg_text_emb


# ---------------------------------------------------------------------------
# Stage 2: load prior-stage outputs
# ---------------------------------------------------------------------------

def _read_image_to_tensor(img_path: Union[str, Path],
                          img_size: int,
                          device: Device) -> torch.Tensor:
    img = np.array(Image.open(str(img_path)).convert('RGB').resize((img_size, img_size)))
    return TRANSFORM(img).unsqueeze(0).to(device)


def _load_resample_gt(img1_path: str,
                      img2_path: str,
                      device: Device,
                      img_size: int = 256) -> Dict:
    return {
        'gt1': _read_image_to_tensor(img1_path, img_size, device),
        'img1_path': img1_path,
        'gt2': _read_image_to_tensor(img2_path, img_size, device),
        'img2_path': img2_path,
    }


def _get_samples(path: Path,
                 device: Device,
                 img_size: int = 256) -> Dict:
    return {
        'sample1': _read_image_to_tensor(path / 'sample1.png', img_size, device),
        'sample1_path': path / 'sample1.png',
        'sample2': _read_image_to_tensor(path / 'sample2.png', img_size, device),
        'sample2_path': path / 'sample2.png',
    }


def load_resample_info(path: Path, device: Device) -> Dict:
    """Load Stage 1 output to initialize Stage 2."""
    info = {}
    info['details'] = load_details_json(path)
    info['alphas'] = load_alphas_json(path)
    info.update(_load_resample_gt(
        img1_path=info['details']['image1_path'],
        img2_path=info['details']['image2_path'],
        device=device,
    ))
    info.update(_get_samples(path=path, device=device))
    info['details']['sample1_path'] = str(info['sample1_path'])
    info['details']['sample2_path'] = str(info['sample2_path'])
    return info


def create_resample_alpha_maps(resample_info: Dict, device: Device) -> Dict:
    """Reconstruct alpha maps from saved Stage 1 parameters."""
    alphas = resample_info['alphas']
    gt_map = get_alpha_map(
        name=alphas['gt']['type'],
        init_alpha=alphas['gt']['init_parameters'],
        device=device,
    )
    init_pred_map = get_alpha_map(
        name=alphas['pred']['type'],
        init_alpha=alphas['pred']['init_parameters'],
        device=device,
        requires_grad=False,
    )
    pred_map = get_alpha_map(
        name=alphas['pred']['type'],
        init_alpha=alphas['pred']['init_parameters'],
        device=device,
        requires_grad=True,
    )
    return {'gt': gt_map, 'pred': pred_map, 'init_pred_map': init_pred_map}


