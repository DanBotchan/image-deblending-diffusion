#!/usr/bin/env python3
"""DoubleDIP baseline adapted to alpha deblending (paper Sec. 5 / supp. "Best-of-N").

DoubleDIP's code is not redistributed here. Clone the official repository and
point ``--doubledip-root`` at it (tested at commit 4f12629):

    git clone https://github.com/yossigandelsman/DoubleDIP.git
    git -C DoubleDIP checkout 4f12629c740402a57776f070b99e9aee880daf1f

Two DIP generators produce the sources; in the unknown-scalar and spatial
scenarios a third DIP produces the alpha map (scalar = its centre pixel).
Loss per iteration, as run for the paper:

    MSE(alpha * x1 + (1 - alpha) * x2, y) + 0.1 * Exclusion(x1, x2)
        + 0.5 * MSE(alpha, 0.5)   [alpha DIP only, first 1000 iterations]

Adam, lr 1e-3, 8000 iterations at 256x256. The returned iterate is the one
whose re-blend best matches the observation (no ground truth is used).
Each (pair, sample) is seeded with ``seed + pair * num_samples + sample``.

The paper's runs additionally reordered each output pair to best match the
ground-truth sources (by PSNR, mapping alpha to 1 - alpha) before selection,
and selected with ``scripts/select_candidates.py --direct-order``; pass
``--order-by-ground-truth`` to reproduce that. Without it, ground truth is
never read before evaluation.

Outputs follow the main pipeline layout (``pair_XXX/smp_YYYY/sample{1,2}.png``
plus ``details.json``/``alphas.json``), so ``scripts/select_candidates.py``
and ``scripts/evaluate_artifact_tree.py`` apply unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

ALPHA_MODES = {"known_scalar": "constant", "unknown_scalar": "scalar", "spatial": "spatial"}


def import_upstream(root: Path):
    if not (root / "net" / "skip_model.py").is_file():
        raise SystemExit(f"{root} is not a DoubleDIP checkout (missing net/skip_model.py)")
    sys.path.insert(0, str(root))
    from net import skip
    from net.losses import ExclusionLoss
    from net.noise import get_noise
    from utils.image_io import np_to_pil, np_to_torch, pil_to_np, torch_to_np
    return skip, ExclusionLoss, get_noise, np_to_pil, np_to_torch, pil_to_np, torch_to_np


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def polynomial_alpha(row: pd.Series, size: int) -> np.ndarray:
    """Ground-truth alpha map used to synthesize the observation."""
    a, b, c, d, e, f = (float(row[k]) for k in "abcdef")
    y, x = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij")
    return np.clip(a + b * x + c * y + d * x * y + e * x ** 2 + f * y ** 2, 0, 1)[None]


def separate(observation: np.ndarray, mode: str, args, upstream) -> tuple[np.ndarray, ...]:
    skip, ExclusionLoss, get_noise, _, np_to_torch, _, torch_to_np = upstream
    dtype = torch.cuda.FloatTensor
    target = np_to_torch(observation).type(dtype)
    size = observation.shape[1:]

    def dip(channels: int):
        return skip(2, channels, num_channels_down=[8, 16, 32, 64, 128],
                    num_channels_up=[8, 16, 32, 64, 128], num_channels_skip=[0, 0, 0, 4, 4],
                    upsample_mode="bilinear", filter_size_down=5, filter_size_up=5,
                    need_sigmoid=True, need_bias=True, pad="reflection",
                    act_fun="LeakyReLU").type(dtype)

    nets = [dip(3), dip(3)] + ([dip(1)] if mode != "constant" else [])
    inputs = [get_noise(2, "noise", size).type(dtype).detach() for _ in nets]
    mse, exclusion = nn.MSELoss().type(dtype), ExclusionLoss().type(dtype)
    optimizer = torch.optim.Adam([p for net in nets for p in net.parameters()], lr=args.lr)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    best, best_psnr = None, -np.inf
    h, w = size
    for step in range(args.num_iter):
        optimizer.zero_grad()
        x1, x2 = nets[0](inputs[0]), nets[1](inputs[1])
        if mode == "constant":
            alpha = args.alpha
        else:
            alpha = nets[2](inputs[2])
            if mode == "scalar":
                alpha = alpha[:, :, h // 2:h // 2 + 1, w // 2:w // 2 + 1]
            alpha = alpha * 0.9 + 0.05
        loss = mse(alpha * x1 + (1 - alpha) * x2, target) + 0.1 * exclusion(x1, x2)
        if mode != "constant" and step < 1000:
            loss = loss + 0.5 * mse(alpha, torch.full_like(alpha, 0.5))
        loss.backward()

        # Keep the iterate whose re-blend best explains the observation.
        out1, out2 = (np.clip(torch_to_np(x), 0, 1) for x in (x1, x2))
        alpha_np = (np.full((1, h, w), alpha) if mode == "constant"
                    else np.clip(torch_to_np(alpha), 0, 1))
        error = np.mean((alpha_np * out1 + (1 - alpha_np) * out2 - observation) ** 2)
        psnr = 10 * np.log10(1.0 / error) if error > 0 else np.inf
        if psnr > best_psnr:
            best, best_psnr = (out1, out2, alpha_np), psnr
        optimizer.step()
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doubledip-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="manifests/<split>/dataset*.csv for the scenario")
    parser.add_argument("--scenario", choices=sorted(ALPHA_MODES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--num-iter", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5, help="Known alpha (known_scalar)")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-end", type=int)
    parser.add_argument("--order-by-ground-truth", action="store_true",
                        help="Reorder outputs to the ground-truth sources (paper protocol)")
    args = parser.parse_args()

    upstream = import_upstream(args.doubledip_root.resolve())
    np_to_pil, pil_to_np = upstream[3], upstream[5]
    mode = ALPHA_MODES[args.scenario]
    frame = pd.read_csv(args.manifest)
    end = len(frame) if args.pair_end is None else min(args.pair_end, len(frame))

    for pair in tqdm(range(args.pair_start, end), desc="pairs"):
        row = frame.iloc[pair]
        paths = [args.data_root / row["img1_path"], args.data_root / row["img2_path"]]
        sources = [pil_to_np(Image.open(p).convert("RGB").resize(
            (args.img_size, args.img_size), Image.Resampling.BICUBIC)) for p in paths]
        alpha_gt = polynomial_alpha(row, args.img_size)
        observation = alpha_gt * sources[0] + (1 - alpha_gt) * sources[1]
        for sample in range(args.num_samples):
            out = args.output_dir / f"pair_{pair:03d}" / f"smp_{sample:04d}"
            if (out / "sample1.png").is_file():
                continue
            seed = args.seed + pair * args.num_samples + sample
            seed_everything(seed)
            image1, image2, alpha = separate(observation, mode, args, upstream)
            if args.order_by_ground_truth:
                def psnr(a, b):
                    return 10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12))
                if (psnr(sources[0], image2) + psnr(sources[1], image1)
                        > psnr(sources[0], image1) + psnr(sources[1], image2)):
                    image1, image2, alpha = image2, image1, 1 - alpha
            out.mkdir(parents=True, exist_ok=True)
            np_to_pil(image1).save(out / "sample1.png")
            np_to_pil(image2).save(out / "sample2.png")
            np_to_pil(alpha).save(out / "pred_map.png")
            (out / "alphas.json").write_text(json.dumps({
                "gt_alpha_mean": float(alpha_gt.mean()), "pred_alpha_mean": float(alpha.mean()),
            }, indent=2) + "\n")
            (out / "details.json").write_text(json.dumps({
                "image1_path": str(paths[0].resolve()), "image2_path": str(paths[1].resolve()),
                "image1_class": row["img1_class"], "image2_class": row["img2_class"],
                "image1_prompt": row["img1_prompt"], "image2_prompt": row["img2_prompt"],
                "smp_seed": seed, "alpha_mode": mode, "num_iter": args.num_iter,
            }, indent=2) + "\n")
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
