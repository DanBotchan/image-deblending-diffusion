"""Stage 2: pixel-space refinement of selected Stage-1 candidates (guided-diffusion 256x256).

Run with an exact paper configuration and a selection manifest, e.g.
``python scripts/second_stage.py +paper=spatial_stage2 datasets.annotations_path=selected.csv``.
"""
import os
import sys
from pathlib import Path

# transformers must not import TensorFlow (version clashes in mixed environments).
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time  # noqa: E402
from typing import Dict, Any

import hydra
import torch
from omegaconf import OmegaConf, DictConfig, MissingMandatoryValue
from tqdm import tqdm

from alphas.alpha_utils import build_controller
from conditions import get_cond_method
from data.utils import get_data_as_dataloader
from models.latent_diffusion.scale_factor_methods import get_scale_scheduler
from operators import get_operator
from utils import set_seed, get_logger, TRANSFORM, flush
from utils.config import (get_config_class,
                          reload_from_disk,
                          save_details_json,
                          save_alphas_params,
                          )
from utils.exp_ops import (get_model_pixel,
                           get_pixel_sampler,
                           create_alpha_map_gt_stage2,
                           create_alpha_map_pred_stage2,
                           )
from utils import save_results
from utils.metrics import compute_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    try:
        cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run_pipeline(cfg=cfg)
    except MissingMandatoryValue as e:
        print(f"✗ Missing required parameter: {e}")


def run_pipeline(cfg: Dict[str, Any]) -> None:
    logger = get_logger()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    exp_config = get_config_class(cfg=cfg, device=device)

    init_seed = set_seed(config=exp_config)
    logger.info(f"Seed is set to: {init_seed}")
    logger.info(f"Device set to {device_str}.")

    model = get_model_pixel(exp_config)
    data_loader = get_data_as_dataloader(exp_config=exp_config, transform=TRANSFORM)

    exp_config.set_paths(mkdirs=True)
    exp_config.save_config()

    sampler = get_pixel_sampler(exp_config)

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True)
    for pair_idx, data_item in pbar:
        pair_num = int(data_item.get('pair_idx', pair_idx))
        exp_config.set_pair(pair_num)
        alpha_controller = build_controller(exp_config=exp_config)
        guidance_sched = get_scale_scheduler(exp_config=exp_config)

        smp_seed = set_seed(config=exp_config, seed=data_item['smp_seed'])
        smp_num = data_item['smp_idx'].item()
        exp_config.set_smp(smp_num=smp_num)

        if exp_config['general']['skip_finished']:
            if (exp_config.get_path('results', include_smp=True) /
                    'sample1.png').exists():
                continue

        alpha_map_gt = create_alpha_map_gt_stage2(exp_config=exp_config,
                                                  data_item=data_item)
        alpha_map_pred = create_alpha_map_pred_stage2(exp_config=exp_config,
                                                      data_item=data_item,
                                                      requires_grad=True)
        init_alpha_map_pred = create_alpha_map_pred_stage2(exp_config=exp_config,
                                                           data_item=data_item,
                                                           requires_grad=False)

        sampler.init_experiment(exp_config=exp_config,
                                init_alpha_map=init_alpha_map_pred)

        operator = get_operator(name='blend_am', alpha=alpha_map_gt)
        cond_method = get_cond_method(exp_config=exp_config,
                                      operator=operator,
                                      guidance_scheduler=guidance_sched,
                                      alpha_controller=alpha_controller)

        pbar.set_description(f"Pair: {pair_num}, SMP: {smp_num}, seed: {smp_seed}")

        ref_img = torch.cat([data_item['image1'], data_item['image2']]).to(device)
        init_smp = torch.cat([data_item['sample1'], data_item['sample2']]).to(device)
        measurement = reload_from_disk(img=operator(ref_img)[0], device=device)

        start_time = time.time()
        outputs = sampler.p_sample_loop(
            model=model,
            init_smp=init_smp,
            ref_img=ref_img,
            measurement=measurement,
            alpha=alpha_map_pred,
            start_idx=exp_config['sampler']['start_idx'],
            num_steps=exp_config['sampler']['timestep_respacing'],
            num_recurrence=exp_config['general']['n_recurrence'],
            lpips_scale=exp_config['general']['lpips'],
            eta=exp_config['general']['eta'],
            cond_method=cond_method,
            device=device,
        )
        sample_time = time.time() - start_time

        results_path = exp_config.get_path('results', include_smp=True)
        results_path.mkdir(exist_ok=True, parents=True)

        metrics = compute_metrics(sample=outputs['sample'], ref_img=ref_img, device=device)
        save_details_json(data_item=data_item,
                          save_to=results_path,
                          sample_time=sample_time,
                          smp_seed=smp_seed,
                          metrics=metrics)
        save_results(sample=outputs['sample'],
                     operator=operator,
                     alpha=alpha_map_pred,
                     measurement=measurement,
                     results_path=results_path)
        save_alphas_params(gt=alpha_map_gt,
                           pred=alpha_map_pred,
                           results_path=results_path,
                           alpha_list=alpha_controller.get_optimizer_states(),
                           init_pred_map=init_alpha_map_pred)
        alpha_controller.reset_optimizer()
        del alpha_map_pred, alpha_map_gt, init_alpha_map_pred, outputs
        flush()


if __name__ == "__main__":
    main()
