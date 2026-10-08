"""
Evaluation Script for Image Deblending GAN - Fixed Version
Evaluates trained model on test dataset and saves results

Fixed issues:
1. Model architecture matches training script (UNetGenerator)
2. Checkpoint key fixed: 'G_state' instead of 'generator'
3. Inference logic corrected: G predicts X2 (background), then recovers X1
4. Supports both fixed alpha and GT alpha from CSV
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import math
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from tqdm import tqdm
import time
import argparse


# ============================================================================
# UTILITIES
# ============================================================================

def tensor2im(image_tensor, imtype=np.uint8):
    """Convert torch tensor to numpy image"""
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype))
        return image_numpy

    # Convert from [C, H, W] to [H, W, C]
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]  # Take first in batch

    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:  # Grayscale
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # [H, W, C]

    # Clip to [0, 1] range
    image_numpy = np.clip(image_numpy, 0, 1)
    image_numpy = (image_numpy * 255.0).astype(imtype)

    return image_numpy


def save_image(image_numpy, save_path, aspect_ratio=1.0):
    """Save numpy image to file"""
    image_pil = Image.fromarray(image_numpy)

    h, w, _ = image_numpy.shape
    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)

    image_pil.save(save_path)


def save_json(filepath, data):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def plot_map(alpha_map):
    """Plot alpha map as heatmap"""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(alpha_map, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Alpha Map')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    return fig


# ============================================================================
# ALPHA MAP
# ============================================================================

class ParamSecondOrderMap:
    """Second-order parametric alpha map for spatially-varying blending"""

    def __init__(self, size, init_alpha=None, alpha_min=0.0, alpha_max=1.0):
        """
        Initialize parametric alpha map.

        Args:
            size: (height, width) of the alpha map
            init_alpha: Optional tuple of 6 parameters (a, b, c, d, e, f)
                       If None, defaults to uniform 0.5
            alpha_min: Minimum alpha value
            alpha_max: Maximum alpha value
        """
        self.size = size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.min_slope = alpha_max - alpha_min
        self.init_alpha = init_alpha

        if init_alpha:
            a_t, b_t, c_t, d_t, e_t, f_t = init_alpha
        else:
            # Default to uniform 0.5
            a_t = 0.5
            b_t = 0.0
            c_t = 0.0
            d_t = 0.0
            e_t = 0.0
            f_t = 0.0

        # Transform to unconstrained space using atanh
        self.a = math.atanh(2 * ((a_t - alpha_min) / self.min_slope - 0.5))
        self.b = math.atanh(2 * ((b_t - (alpha_min - a_t)) / self.min_slope - 0.5))
        self.e = math.atanh(2 * (((e_t + (a_t + b_t)) - alpha_min) / self.min_slope - 0.5))
        self.c = math.atanh(2 * ((c_t - (alpha_min - a_t)) / self.min_slope - 0.5))
        self.f = math.atanh(2 * (((f_t + (a_t + c_t)) - alpha_min) / self.min_slope - 0.5))
        self.d = math.atanh(2 * (((d_t + (a_t + b_t + c_t + e_t + f_t)) - alpha_min) /
                                 self.min_slope - 0.5))

        # Create meshgrid for pixel coordinates
        img_range_x = np.arange(size[1], dtype=np.float32)
        img_range_y = np.arange(size[0], dtype=np.float32)
        self.X, self.Y = np.meshgrid(img_range_x, img_range_y)

    def tanh(self, x):
        """Scaled tanh activation"""
        return 0.5 * np.tanh(x) + 0.5

    def get_parameters(self):
        """Get raw unconstrained parameters"""
        return [self.a, self.b, self.c, self.d, self.e, self.f]

    def get_scaled_parameters(self):
        """Get scaled parameters in constrained space"""
        tanh_a = self.min_slope * self.tanh(self.a) + self.alpha_min
        tanh_b = (self.alpha_min - tanh_a) + self.min_slope * self.tanh(self.b)
        tanh_e = (self.alpha_min - (tanh_a + tanh_b)) + self.min_slope * self.tanh(self.e)
        tanh_c = (self.alpha_min - tanh_a) + self.min_slope * self.tanh(self.c)
        tanh_f = (self.alpha_min - (tanh_a + tanh_c)) + self.min_slope * self.tanh(self.f)
        total_base = tanh_a + tanh_b + tanh_c + tanh_e + tanh_f
        tanh_d = (self.alpha_min - total_base) + self.min_slope * self.tanh(self.d)
        return [tanh_a, tanh_b, tanh_c, tanh_d, tanh_e, tanh_f]

    def forward(self):
        """
        Generate the alpha map.

        Returns:
            alpha_map: [H, W] array with values in [alpha_min, alpha_max]
        """
        scaled = self.get_scaled_parameters()
        tanh_a, tanh_b, tanh_c, tanh_d, tanh_e, tanh_f = scaled

        # Normalize coordinates to [0, 1]
        X_norm = self.X / (self.size[1] - 1)
        Y_norm = self.Y / (self.size[0] - 1)

        # Compute second-order polynomial
        pixels = (tanh_a +
                 tanh_b * X_norm +
                 tanh_c * Y_norm +
                 tanh_d * X_norm * Y_norm +
                 tanh_e * (X_norm**2) +
                 tanh_f * (Y_norm**2))

        return pixels


# ============================================================================
# DATASET
# ============================================================================

class MyBIDDataset(Dataset):
    """Dataset for loading image pairs from CSV"""

    def __init__(self, csv_root, transform=None):
        """
        Args:
            csv_root: Path to CSV file
            transform: Optional transform to apply to images
        """
        self.csv_root = Path(csv_root)
        self.root = self.csv_root.parent
        self.data = pd.read_csv(self.csv_root)
        self.transform = transform

    def __getitem__(self, index: int):
        pair = self.data.iloc[index]
        image1_path = self.root / pair['img1_path']
        image2_path = self.root / pair['img2_path']

        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

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
            'pair_seeds': ast.literal_eval(pair['z_seeds']) if 'z_seeds' in pair else None,
            'pair_idx': index,
        }

    def __len__(self):
        return len(self.data)


# ============================================================================
# MODEL ARCHITECTURE (MUST MATCH TRAINING)
# ============================================================================

class UNetGenerator(nn.Module):
    """
    U-Net style generator that predicts X2 (background/degradation parameter) from Y.

    FIXED: This now matches the training script architecture exactly!
    - Uses kernel size 4 (not 3)
    - Uses padding 2 (not 1)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_filters: int = 64):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder blocks
        self.enc1 = self._make_conv_block(in_channels, base_filters, num_layers=2)
        self.pool1 = nn.Conv2d(base_filters, base_filters, 4, stride=2, padding=1)

        self.enc2 = self._make_conv_block(base_filters, base_filters, num_layers=2)
        self.pool2 = nn.Conv2d(base_filters, base_filters, 4, stride=2, padding=1)

        self.enc3 = self._make_conv_block(base_filters, base_filters, num_layers=2)
        self.pool3 = nn.Conv2d(base_filters, base_filters, 4, stride=2, padding=1)

        self.enc4 = self._make_conv_block(base_filters, base_filters, num_layers=2)
        self.pool4 = nn.Conv2d(base_filters, base_filters, 4, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = self._make_conv_block(base_filters, base_filters, num_layers=2)

        # Decoder blocks with skip connections
        self.dec1 = self._make_conv_block(base_filters, base_filters, num_layers=2)
        self.dec2 = self._make_conv_block(base_filters, base_filters, num_layers=2)
        self.dec3 = self._make_conv_block(base_filters, base_filters, num_layers=2)
        self.dec4 = self._make_conv_block(base_filters, base_filters, num_layers=2)

        # Output layer with sigmoid activation
        self.output = nn.Conv2d(base_filters, out_channels, 4, padding=2)

    def _make_conv_block(self, in_ch: int, out_ch: int, num_layers: int):
        """Create a block of Conv-BN-ReLU layers"""
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 4, padding=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Predict X2 (degradation parameter) from blended image Y.

        Args:
            y: Blended image [B, C, H, W]

        Returns:
            z: Predicted X2 [B, C, H, W]
        """
        original_size = y.shape[2:]

        # Encoder with skip connections
        e1 = self.enc1(y)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with skip connections
        d1 = self.dec1(b)
        d1 = F.interpolate(d1, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d1 = d1 + e4

        d2 = self.dec2(d1)
        d2 = F.interpolate(d2, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d2 = d2 + e3

        d3 = self.dec3(d2)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = d3 + e2

        d4 = self.dec4(d3)
        d4 = F.interpolate(d4, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d4 = d4 + e1

        # Output
        out = self.output(d4)
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        out = torch.sigmoid(out)  # Bound to [0, 1]

        return out


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(
    checkpoint_path,
    csv_path,
    output_dir,
    img_size=256,
    device='cuda',
    use_fixed_alpha=True,
    fixed_alpha_value=0.5
):
    """
    Evaluate trained deblending model.

    Args:
        checkpoint_path: Path to trained model checkpoint
        csv_path: Path to evaluation CSV
        output_dir: Directory to save results
        img_size: Image size for evaluation
        device: Device to run on ('cuda' or 'cpu')
        use_fixed_alpha: Whether to use fixed alpha (0.5) or ground truth alpha
        fixed_alpha_value: Value for fixed alpha if use_fixed_alpha=True
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n{'='*80}")
    print(f"Loading model from: {checkpoint_path}")
    print(f"{'='*80}")

    G = UNetGenerator(in_channels=3, out_channels=3, base_filters=64).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # FIXED: Use correct key 'G_state' instead of 'generator'
    if 'G_state' in checkpoint:
        state_dict = checkpoint['G_state']
    elif 'generator' in checkpoint:
        state_dict = checkpoint['generator']
    else:
        raise KeyError(f"Checkpoint does not contain 'G_state' or 'generator' key. Keys: {checkpoint.keys()}")

    # Handle DataParallel wrapped models
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    G.load_state_dict(state_dict)
    G.eval()

    print("✓ Model loaded successfully")
    print(f"  - Parameters: {sum(p.numel() for p in G.parameters()):,}")

    # Create dataset
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    print(f"\n{'='*80}")
    print(f"Loading dataset from: {csv_path}")
    print(f"{'='*80}")

    dataset = MyBIDDataset(csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"✓ Dataset loaded")
    print(f"  - Samples: {len(dataset)}")
    print(f"  - Alpha mode: {'Fixed α=' + str(fixed_alpha_value) if use_fixed_alpha else 'Ground Truth'}")
    print(f"  - Output dir: {output_dir}")

    # Evaluate
    print(f"\n{'='*80}")
    print("Starting evaluation...")
    print(f"{'='*80}\n")

    for idx, data in enumerate(tqdm(dataloader, desc="Evaluating")):
        # Create output directory for this sample
        save_path = output_dir / f"pair_{idx:03d}" / "smp_0000"
        save_path.mkdir(parents=True, exist_ok=True)

        # Get data
        image1 = data['image1'].to(device)  # [1, 3, H, W] - X1 (foreground)
        image2 = data['image2'].to(device)  # [1, 3, H, W] - X2 (background)

        # Extract alpha GT parameters - handle different formats
        alpha_gt_raw = data['alpha_gt']
        if isinstance(alpha_gt_raw, torch.Tensor):
            if alpha_gt_raw.dim() == 2:  # [1, 6]
                alpha_gt_params = alpha_gt_raw[0].tolist()
            else:  # [6]
                alpha_gt_params = alpha_gt_raw.tolist()
        elif isinstance(alpha_gt_raw, list):
            alpha_gt_params = alpha_gt_raw[0] if isinstance(alpha_gt_raw[0], list) else alpha_gt_raw
        else:
            alpha_gt_params = list(alpha_gt_raw)

        # Create alpha maps
        if use_fixed_alpha:
            # Use fixed alpha (e.g., 0.5)
            alpha_pred = ParamSecondOrderMap(
                size=(img_size, img_size),
                init_alpha=[fixed_alpha_value, 0, 0, 0, 0, 0],
                alpha_min=0.0,
                alpha_max=1.0
            )
        else:
            # Use ground truth alpha
            alpha_pred = ParamSecondOrderMap(
                size=(img_size, img_size),
                init_alpha=alpha_gt_params,
                alpha_min=0.0,
                alpha_max=1.0
            )

        # Ground truth alpha (always create for comparison)
        alpha_gt = ParamSecondOrderMap(
            size=(img_size, img_size),
            init_alpha=alpha_gt_params,
            alpha_min=0.0,
            alpha_max=1.0
        )

        # Get alpha maps as tensors [1, 1, H, W]
        alpha_pred_map = torch.from_numpy(alpha_pred.forward()).unsqueeze(0).unsqueeze(0).float().to(device)
        alpha_gt_map = torch.from_numpy(alpha_gt.forward()).unsqueeze(0).unsqueeze(0).float().to(device)

        # Create ground truth blended image using GT alpha
        # Y = α*X1 + (1-α)*X2
        gt_blended = alpha_gt_map * image1 + (1 - alpha_gt_map) * image2

        # ====================================================================
        # INFERENCE (FIXED LOGIC)
        # ====================================================================
        start_time = time.time()
        with torch.no_grad():
            # FIXED: Generator predicts X2 (background), not X1!
            pred_X2 = G(gt_blended)
            pred_X2 = torch.clamp(pred_X2, 0, 1)

            # Recover X1 using predicted X2 and alpha
            # Formula: X1 = (Y - (1-α)*X2) / α
            pred_X1 = (gt_blended - (1.0 - alpha_pred_map) * pred_X2) / (alpha_pred_map + 1e-8)
            pred_X1 = torch.clamp(pred_X1, 0, 1)

            # Reconstruct blended from predictions
            # Y' = α*X1' + (1-α)*X2'
            sample_blended = alpha_pred_map * pred_X1 + (1 - alpha_pred_map) * pred_X2

        inference_time = time.time() - start_time

        # ====================================================================
        # SAVE RESULTS (Same structure as original)
        # ====================================================================

        # Save predicted images
        save_image(tensor2im(pred_X1), save_path / 'sample1.png', aspect_ratio=1.0)
        save_image(tensor2im(pred_X2), save_path / 'sample2.png', aspect_ratio=1.0)
        save_image(tensor2im(gt_blended), save_path / 'blended_gt.png', aspect_ratio=1.0)
        save_image(tensor2im(sample_blended), save_path / 'blended_sample.png', aspect_ratio=1.0)

        # Save ground truth (for comparison)
        save_image(tensor2im(image1), save_path / 'gt1.png', aspect_ratio=1.0)
        save_image(tensor2im(image2), save_path / 'gt2.png', aspect_ratio=1.0)

        # Save details JSON
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
                "smp_seed": None,
                "alpha_mode": "fixed" if use_fixed_alpha else "ground_truth",
                "alpha_value": fixed_alpha_value if use_fixed_alpha else "gt",
            }
        )

        # Save alpha parameters
        alpha_pred_dict = {
            "type": "param_2d_order",
            "parameters": [float(p) for p in alpha_pred.get_scaled_parameters()],
            "raw_parameters": [float(p) for p in alpha_pred.get_parameters()],
            "init_parameters": alpha_pred.init_alpha if alpha_pred.init_alpha else [fixed_alpha_value, 0, 0, 0, 0, 0],
        }

        alpha_gt_dict = {
            "type": "param_2d_order",
            "parameters": [float(p) for p in alpha_gt.get_scaled_parameters()],
            "raw_parameters": [float(p) for p in alpha_gt.get_parameters()],
            "init_parameters": [a.item() for a in alpha_gt.init_alpha],
        }

        save_json(
            save_path / 'alphas.json',
            {
                "pred": alpha_pred_dict,
                "gt": alpha_gt_dict,
                "init_pred_map": alpha_pred_dict
            }
        )

        # Save alpha visualizations
        alpha_fig = plot_map(alpha_pred.forward())
        alpha_fig.savefig(save_path / 'pred_map.png', bbox_inches='tight', dpi=100)
        plt.close(alpha_fig)

        alpha_fig = plot_map(alpha_gt.forward())
        alpha_fig.savefig(save_path / 'gt_map.png', bbox_inches='tight', dpi=100)
        plt.close(alpha_fig)

    print(f"\n{'='*80}")
    print(f"✓ Evaluation complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Total samples: {len(dataset)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Image Deblending GAN - Fixed Version')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (best_ckpt.pth)')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to evaluation CSV file')
    parser.add_argument('--output', type=str, default='./eval_results',
                        help='Output directory for results')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for evaluation (default: 256)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--use_gt_alpha', action='store_true',
                        help='Use ground truth alpha instead of fixed alpha')
    parser.add_argument('--fixed_alpha', type=float, default=0.5,
                        help='Fixed alpha value to use (default: 0.5)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Image Deblending GAN - Evaluation (Fixed Version)")
    print("="*80)

    evaluate_model(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        output_dir=args.output,
        img_size=args.img_size,
        device=args.device,
        use_fixed_alpha=not args.use_gt_alpha,
        fixed_alpha_value=args.fixed_alpha
    )
