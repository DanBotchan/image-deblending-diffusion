"""Stage 1: joint latent-diffusion sampling of both sources (Stable Diffusion v1.4).

Run with an exact paper configuration, e.g. ``python scripts/first_stage.py +paper=spatial_stage1``.
Writes ``<general.output_dir>/results/pair_XXX/smp_YYYY/`` per candidate.
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
from samplers.latent.registry import get_sampler
from utils import set_seed, get_logger, TRANSFORM, flush
from utils.config import (get_config_class,
                          reload_from_disk,
                          save_details_json,
                          save_alphas_params,
                          )
from utils.exp_ops import (get_model_latent,
                           get_text_emb,
                           create_alpha_map_gt_stage1,
                           create_alpha_map_pred_stage1,
                           )
from utils import save_results
from utils.metrics import compute_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    try:
        cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        run_pipeline(cfg=cfg)
        import gc
        torch.cuda.empty_cache()
        gc.collect()
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

    model = get_model_latent(exp_config)
    data_loader = get_data_as_dataloader(exp_config=exp_config, transform=TRANSFORM)

    exp_config.set_paths(mkdirs=True)
    exp_config.save_config()

    sampler = get_sampler(name=exp_config['sampler']['type'], model=model)
    torch.set_grad_enabled(False)

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True)
    for pair_idx, data_item in pbar:
        pair_num = int(data_item.get('pair_idx', pair_idx))
        exp_config.set_pair(pair_num)
        alpha_controller = build_controller(exp_config=exp_config)
        alpha_map_gt = create_alpha_map_gt_stage1(exp_config=exp_config,
                                                  data_item=data_item)
        operator = get_operator(name='blend_am', alpha=alpha_map_gt)
        guidance_sched = get_scale_scheduler(exp_config=exp_config)
        cond_method = get_cond_method(exp_config=exp_config,
                                      operator=operator,
                                      guidance_scheduler=guidance_sched,
                                      alpha_controller=alpha_controller)
        pos_text_emb, neg_text_emb = get_text_emb(exp_config=exp_config,
                                                   model=model,
                                                   data_item=data_item)
        ref_img = torch.cat([data_item['image1'], data_item['image2']]).to(device)
        measurement = reload_from_disk(img=operator(ref_img)[0], device=device)

        for smp_num in range(exp_config['general']['N_samples_for_pair']):
            smp_seed = set_seed(config=exp_config, seed=data_item['pair_seeds'][smp_num])
            pbar.set_description(
                f"Pair: {pair_num}, SMP: {smp_num}, seed: {smp_seed}")
            exp_config.set_smp(smp_num=smp_num)
            sampler.init_experiment(exp_config=exp_config)

            if exp_config['general']['skip_finished']:
                if (exp_config.get_path('results', include_smp=True) /
                        'sample1.png').exists():
                    continue

            alpha_map_pred = create_alpha_map_pred_stage1(exp_config=exp_config,
                                                          alpha_map_gt=alpha_map_gt)

            if exp_config['general']['alpha_initialize']:
                alpha_map_pred = sampler.initialize_operator(
                    measurement=measurement,
                    pos_text_emb=pos_text_emb,
                    neg_text_emb=neg_text_emb,
                    alpha=alpha_map_pred,
                    start_idx=exp_config['sampler']['start_idx'],
                    num_steps=exp_config['operator_init']['num_steps'],
                    num_M=exp_config['operator_init']['num_M'],
                    lpips_scale=exp_config['general']['lpips'],
                    cfg_scale=exp_config['general']['cfg'],
                    eta=exp_config['general']['eta'],
                    cond_method=cond_method,
                )

            start_time = time.time()
            outputs = sampler.p_sample_loop(
                ref_img=ref_img,
                measurement=measurement,
                pos_text_emb=pos_text_emb,
                neg_text_emb=neg_text_emb,
                alpha=alpha_map_pred,
                start_idx=exp_config['sampler']['start_idx'],
                num_steps=exp_config['sampler']['timestep_respacing'],
                num_recurrence=exp_config['general']['n_recurrence'],
                lpips_scale=exp_config['general']['lpips'],
                cfg_scale=exp_config['general']['cfg'],
                eta=exp_config['general']['eta'],
                cond_method=cond_method,
            )
            sample = model.module.decode_first_stage(outputs['sample'])
            sample_time = time.time() - start_time

            results_path = exp_config.get_path('results', include_smp=True)
            results_path.mkdir(exist_ok=True, parents=True)

            metrics = compute_metrics(sample=sample, ref_img=ref_img, device=device)
            save_details_json(data_item=data_item,
                              save_to=results_path,
                              sample_time=sample_time,
                              smp_seed=smp_seed,
                              metrics=metrics)
            save_results(sample=sample,
                         operator=operator,
                         alpha=alpha_map_pred,
                         measurement=measurement,
                         results_path=results_path)
            save_alphas_params(gt=alpha_map_gt,
                               pred=alpha_map_pred,
                               results_path=results_path,
                               alpha_list=alpha_controller.get_optimizer_states())
            alpha_controller.reset_optimizer()
            del alpha_map_pred
            flush()


if __name__ == "__main__":
    main()
