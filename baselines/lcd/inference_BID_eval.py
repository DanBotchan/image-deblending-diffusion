"""
Inference script for IID decomposition on BID evaluation dataset

This script evaluates a trained IID model on the BID evaluation dataset,
following the same format and structure as the DAD evaluation script.
"""
import argparse
import ast
import json
import os
import time
import random
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any

import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from iid_diffusion import Unet, IIDGaussianDiffusion
from alphas.alpha_maps import get_alpha_map


def convert_to_json_serializable(obj):
    """Convert tensors and other non-JSON-serializable objects to JSON-serializable format."""
    if torch.is_tensor(obj):
        return obj.tolist() if obj.numel() > 1 else obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def save_json(file_path: Union[str, Path], data: Dict[str, Any]):
    """Save dictionary as JSON file."""
    # Convert all tensors to JSON-serializable format
    serializable_data = convert_to_json_serializable(data)
    with open(file_path, 'w') as fp:
        json.dump(serializable_data, fp, indent=4)


def tensor2im(tensor):
    """Convert tensor to image numpy array (0-255 range)."""
    # Assumes tensor is in [0, 1] range
    img = tensor.detach().cpu().squeeze().numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    img = (img * 255).astype('uint8')
    return img


def save_pil_image(tensor, path):
    """Save tensor as PIL image."""
    img = tensor2im(tensor)
    Image.fromarray(img).save(path)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: This may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BIDEvalDataset(Dataset):
    """Dataset loader for BID evaluation following DAD format."""

    def __init__(self, csv_root, transform) -> None:
        self.csv_root = Path(csv_root)
        self.root = self.csv_root.parent
        self.data = pd.read_csv(self.csv_root)
        self.transform = transform

    def __getitem__(self, index: int) -> Dict[str, Union['torch.Tensor', str]]:
        pair = self.data.iloc[index]
        image1_path = self.root / pair['img1_path']
        image2_path = self.root / pair['img2_path']

        image1 = self.transform(Image.open(image1_path).convert('RGB'))
        image2 = self.transform(Image.open(image2_path).convert('RGB'))

        # Get alpha parameters (a, b, c, d, e, f for parametric alpha)
        if 'a' in pair.index:
            alpha_gt = pair[['a', 'b', 'c', 'd', 'e', 'f']].values.tolist()
        else:  # scalar manifests may store only 'alpha'
            alpha_gt = [float(pair['alpha']), 0.0, 0.0, 0.0, 0.0, 0.0]

        return {
            'image1': image1,
            'image1_path': str(image1_path),
            'image1_class': pair['img1_class'],
            'image1_prompt': pair['img1_prompt'],
            'image2': image2,
            'image2_path': str(image2_path),
            'image2_class': pair['img2_class'],
            'image2_prompt': pair['img2_prompt'],
            'index': index,
            'alpha_gt': alpha_gt,
            'pair_seeds': ast.literal_eval(pair['z_seeds']),
            'pair_idx': index,
        }

    def __len__(self) -> int:
        return len(self.data)


def get_alpha_value_from_params(alpha_params):
    """
    Extract a single alpha value from parametric alpha.
    For IID decomposition, we use a constant alpha (average of the parametric map).

    Args:
        alpha_params: List of [a, b, c, d, e, f] parameters

    Returns:
        float: Alpha value between 0 and 1
    """
    # For simplicity, use the constant term 'a' or compute average
    # The parametric form is: alpha(x,y) = a + b*x + c*y + d*x^2 + e*xy + f*y^2
    # For a constant alpha, we use the center value or just 'a'
    # Assuming normalized coordinates [-1, 1], at center (0, 0): alpha = a
    return float(alpha_params[0])


def load_model(args, device):
    """Load trained IID model from checkpoint."""
    print(f'Loading model from {args.model_path}...')

    # Create model architecture
    model = Unet(
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        channels=args.channels,
        with_time_emb=not args.no_time_embed,
        residual=args.residual
    ).to(device)

    diffusion = IIDGaussianDiffusion(
        model,
        image_size=args.image_size,
        channels=args.channels,
        timesteps=args.timesteps,
        alpha_max=args.alpha_max,
        loss_type='l1',
        prediction_type=args.prediction_type
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)

    if 'ema' in checkpoint:
        # Use EMA weights if available (recommended)
        diffusion.load_state_dict(checkpoint['ema'])
        print('Loaded EMA weights')
    elif 'model' in checkpoint:
        diffusion.load_state_dict(checkpoint['model'])
        print('Loaded model weights')
    else:
        diffusion.load_state_dict(checkpoint)
        print('Loaded weights')

    diffusion.eval()
    return diffusion


def run_eval(args, model, device):
    """Run evaluation on BID dataset."""
    print('Running evaluation...')

    # Create dataset and dataloader
    dataset = BIDEvalDataset(
        csv_root=args.dataset,
        transform=ToTensor()  # Converts to [0, 1]
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Process each pair
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        pair_name = f"pair_{data['pair_idx'].item():03d}/smp_0000"
        save_path = save_dir / pair_name
        save_path.mkdir(parents=True, exist_ok=True)

        # Get images (in [0, 1] range from ToTensor)
        img1 = data['image1'].to(device)
        img2 = data['image2'].to(device)

        # Convert to [-1, 1] range for IID model
        img1_normalized = (img1 * 2) - 1
        img2_normalized = (img2 * 2) - 1

        # Get alpha parameters and create alpha map
        alpha_params = data['alpha_gt']  # [a, b, c, d, e, f]

        # Convert to list if it's a tensor
        if torch.is_tensor(alpha_params):
            alpha_params_list = alpha_params.tolist()
        else:
            alpha_params_list = list(alpha_params)

        # Create parametric 2D alpha map using the same class as DAD
        alpha_map = get_alpha_map(
            name='param_2d_order',
            init_alpha=alpha_params_list,
            device=device,
            alpha_min=0.0,
            alpha_max=1.0,
            size=args.image_size,
        )

        # Get the full spatial alpha map [H, W]
        alpha_spatial = alpha_map.forward()  # Shape: [H, W]

        # Compute mean alpha value for IID decomposition (which expects scalar alpha)
        alpha_value = alpha_spatial.mean().item()

        # Create ground truth blended image using the spatial alpha map
        # Note: alpha_map follows DAD convention: blend = alpha * img1 + (1 - alpha) * img2
        # But IID uses: x_t = (1 - alpha) * x_orig + alpha * x_add
        # So we need to swap the images or invert alpha

        # Expand alpha_spatial to match image dimensions [1, 1, H, W]
        alpha_spatial_expanded = alpha_spatial.unsqueeze(0).unsqueeze(0)

        # Create blended image (following DAD convention: alpha * img1 + (1-alpha) * img2)
        gt_blended = alpha_spatial_expanded * img1 + (1 - alpha_spatial_expanded) * img2
        gt_blended_normalized = alpha_spatial_expanded * img1_normalized + (1 - alpha_spatial_expanded) * img2_normalized

        # Get seed and set it before sampling
        seed = data['pair_seeds'][0][0]
        if torch.is_tensor(seed):
            seed = seed.item()
        seed = int(seed)

        # Set random seed for reproducibility
        set_seed(seed)

        # Perform decomposition
        start_time = time.time()
        with torch.no_grad():
            pred1, pred2 = model.sample(
                gt_blended_normalized,
                alpha_init=alpha_value,
                sampling_method=args.sampling_method
            )
        inference_time = time.time() - start_time

        # Convert predictions back to [0, 1] range
        pred1 = (pred1 + 1) / 2
        pred2 = (pred2 + 1) / 2

        # Clip to valid range
        pred1 = torch.clamp(pred1, 0, 1)
        pred2 = torch.clamp(pred2, 0, 1)

        # Create reconstructed blended image from predictions
        sample_blended = (1 - alpha_value) * pred1 + alpha_value * pred2

        # Save images
        save_pil_image(pred1, save_path / 'sample1.png')
        save_pil_image(pred2, save_path / 'sample2.png')
        save_pil_image(gt_blended, save_path / 'blended_gt.png')
        save_pil_image(sample_blended, save_path / 'blended_sample.png')

        # Save metadata
        save_json(
            save_path / 'details.json',
            {
                "image1_path": data['image1_path'][0],
                "image1_class": data['image1_class'][0],
                "image1_prompt": data['image1_prompt'][0],
                "image2_path": data['image2_path'][0],
                "image2_class": data['image2_class'][0],
                "image2_prompt": data['image2_prompt'][0],
                "sample1_path": str(save_path / 'sample1.png'),
                "sample2_path": str(save_path / 'sample2.png'),
                "sample_time": inference_time,
                "smp_seed": seed,  # Use the seed that was set
                "sampling_method": args.sampling_method,
                "alpha_value": alpha_value,
                "model_path": args.model_path,
            }
        )

        # Save alpha information using the parametric alpha map format (matching DAD)
        alpha_info = {
            "type": "param_2d_order",
            "parameters": [t.item() for t in alpha_map.get_scaled_parameters()],
            "raw_parameters": [t.item() for t in alpha_map.get_parameters()],
            "init_parameters": [t.item() for t in alpha_map.init_params],
            "mean_alpha": alpha_value,  # Add mean alpha for reference
        }

        save_json(
            save_path / 'alphas.json',
            {
                "pred": alpha_info,
                "gt": alpha_info,
                "init_pred_map": alpha_info
            }
        )

        # Save alpha map visualization
        fig, ax = plt.subplots(1)
        alpha_img = alpha_spatial.detach().cpu().numpy()
        img_plot = ax.imshow(alpha_img, vmin=0, vmax=1, cmap='gray')
        ax.set_title(f'Alpha Map (mean: {alpha_value:.3f})')
        plt.colorbar(img_plot, ax=ax)
        fig.savefig(save_path / 'pred_map.png')
        fig.savefig(save_path / 'gt_map.png')
        plt.close(fig)

    print(f'\nEvaluation complete! Results saved to {save_dir}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate IID model on BID dataset')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to BID evaluation CSV file')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save evaluation results')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--prediction_type', type=str, default='x_orig',
                        choices=['x_add', 'x_orig', 'x_diff'],
                        help='What the model was trained to predict (default: x_orig)')

    # Model architecture (must match training)
    parser.add_argument('--dim', type=int, default=64,
                        help='Base dimension for U-Net (default: 64)')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Dimension multipliers for U-Net')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of image channels (default: 3)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (default: 256)')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('--alpha_max', type=float, default=0.8,
                        help='Maximum alpha value (default: 0.8)')

    # Inference parameters
    parser.add_argument('--sampling_method', type=str, default='ours',
                        choices=['one_step', 'cold_diffusion', 'ours', 'ours_epsilon'],
                        help='Sampling method to use (default: ours)')

    # Other
    parser.add_argument('--no_time_embed', action='store_true',
                        help='Disable time embedding in U-Net')
    parser.add_argument('--residual', action='store_true',
                        help='Use residual connection in U-Net')

    args = parser.parse_args()

    print("=" * 80)
    print("IID Model Evaluation on BID Dataset")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_path}")
    print(f"Prediction type: {args.prediction_type}")
    print(f"Sampling method: {args.sampling_method}")
    print(f"Image size: {args.image_size}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args, device)

    # Run evaluation
    run_eval(args, model, device)

    print("\nDone!")


if __name__ == '__main__':
    main()
