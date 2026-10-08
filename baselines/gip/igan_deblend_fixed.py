"""
Fixed Image Deblending GAN - PyTorch Implementation
Implements the paper "Adversarial Training for Solving Inverse Problems in Image Processing"
correctly for alpha blending with bidirectional mapping.

Key corrections:
1. Z = G(Y) predicts X2 (background/second image - the "degradation parameter")
2. Bidirectional mapping: Apply predicted X2 to DIFFERENT X1 samples
3. Proper independent sampling for bidirectional constraint
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import glob
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    def __init__(self):
        # Task settings
        self.task_name = 'alpha_deblending'
        self.gan_model = 'Vanilla_GAN'  # 'Vanilla_GAN', 'LS_GAN', 'W_GAN'

        # Training hyperparameters
        self.batch_size = 16
        self.max_iters = 20000
        self.with_bidirectional_mapping = True
        self.with_paired_loss = False  # Set to True for supervised training
        self.paired_loss_weight = 100.0

        # Data paths
        self.data_path = None       # set from --data_path
        self.checkpoint_dir = None  # set from --checkpoint_dir
        self.sample_dir = None
        self.log_dir = None

        # Image settings
        self.img_size = 256
        self.n_channels = 3

        # Alpha blending (fixed uniform alpha for now)
        self.alpha = 0.5

        # Network settings
        self.g_base_filters = 64
        self.d_base_filters = 32

        # Learning rates
        self.g_lr = 1e-4
        self.d_lr = 1e-5

        # Logging
        self.log_interval = 20
        self.save_interval = 500

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def create_dirs(self):
        """Create necessary directories"""
        for path in [self.checkpoint_dir, self.sample_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)

# ============================================================================
# DATA LOADER
# ============================================================================

class DeblendingDataset:
    """Dataset for loading and blending two image domains"""

    def __init__(self, root_path: str, img_size: int, n_channels: int = 3):
        self.root_path = Path(root_path)
        self.img_size = img_size
        self.n_channels = n_channels

        # Load image paths
        self.images_A = self._load_image_paths(self.root_path / 'trainA')
        self.images_B = self._load_image_paths(self.root_path / 'trainB')

        assert len(self.images_A) > 0, f"No images found in {self.root_path / 'trainA'}"
        assert len(self.images_B) > 0, f"No images found in {self.root_path / 'trainB'}"

        print(f"Loaded {len(self.images_A)} images from domain A")
        print(f"Loaded {len(self.images_B)} images from domain B")

    def _load_image_paths(self, folder: Path) -> list:
        """Load all image paths from folder"""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(str(folder / ext)))
        return sorted(paths)

    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess a single image"""
        img = Image.open(path)

        # Handle different image modes
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB' and img.mode != 'L':
            img = img.convert('RGB')

        # Convert to grayscale or RGB
        if self.n_channels == 1:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        # Resize
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Convert to numpy array [H, W, C] in range [0, 1]
        img = np.array(img, dtype=np.float32) / 255.0

        # Add channel dimension for grayscale
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

        return img

    def get_random_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get random pair of images from both domains"""
        img_a_path = random.choice(self.images_A)
        img_b_path = random.choice(self.images_B)

        img_a = self._load_image(img_a_path)
        img_b = self._load_image(img_b_path)

        return img_a, img_b


class DeblendingDataLoader:
    """Data loader that creates blended images for training"""

    def __init__(self, dataset: DeblendingDataset, batch_size: int, alpha: float = 0.5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.alpha = alpha

        print(f"DataLoader initialized with alpha={alpha}")

    def get_batch(self, paired: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a batch of blended images.

        Returns:
            X1: Images from domain A [B, C, H, W]
            X2: Images from domain B [B, C, H, W]
            Y: Blended images Y = alpha*X1 + (1-alpha)*X2 [B, C, H, W]
        """
        X1_batch = []
        X2_batch = []

        for _ in range(self.batch_size):
            x1, x2 = self.dataset.get_random_pair()
            X1_batch.append(x1)
            X2_batch.append(x2)

        # Convert to tensors [B, H, W, C] -> [B, C, H, W]
        X1 = torch.from_numpy(np.array(X1_batch)).permute(0, 3, 1, 2).float()
        X2 = torch.from_numpy(np.array(X2_batch)).permute(0, 3, 1, 2).float()

        # Create blended images
        Y = self.alpha * X1 + (1.0 - self.alpha) * X2

        return X1, X2, Y

    def get_independent_batches(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get two independent batches for bidirectional mapping.

        Returns:
            X1: First batch from domain A
            X2: First batch from domain B
            Y: Blended images from first batch
            X1_prime: Second independent batch from domain A (for bidirectional mapping)
        """
        # First batch (paired for creating Y)
        X1, X2, Y = self.get_batch()

        # Second independent batch (only need X1 for bidirectional mapping)
        X1_prime, _, _ = self.get_batch()

        return X1, X2, Y, X1_prime

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class UNetGenerator(nn.Module):
    """
    U-Net style generator that predicts X2 (background/degradation parameter) from Y.

    Architecture based on the paper:
    - 12-layer encoder-decoder with skip connections
    - Batch normalization
    - Bilinear upsampling (instead of transposed convolution)
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


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator.

    Architecture based on the paper:
    - 4 convolutional layers with strided convolutions
    - Leaky ReLU activations
    - Global average pooling before final classification
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, base_filters, 4, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2_block = self._make_conv_block(base_filters, base_filters * 2, num_layers=2)
        self.conv2_down = nn.Conv2d(base_filters * 2, base_filters * 2, 4, stride=2, padding=1)

        self.conv3_block = self._make_conv_block(base_filters * 2, base_filters * 4, num_layers=2)
        self.conv3_down = nn.Conv2d(base_filters * 4, base_filters * 4, 4, stride=2, padding=1)

        self.conv4_block = self._make_conv_block(base_filters * 4, base_filters * 8, num_layers=2)
        self.conv4_down = nn.Conv2d(base_filters * 8, base_filters * 8, 4, stride=2, padding=1)

        # Final classification
        self.fc = nn.Linear(base_filters * 8, 1)

    def _make_conv_block(self, in_ch: int, out_ch: int, num_layers: int):
        """Create a block of Conv-LeakyReLU layers"""
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 4, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discriminate whether input belongs to real or fake domain.

        Args:
            x: Input image [B, C, H, W]

        Returns:
            logit: Classification logit [B, 1]
        """
        x = self.lrelu(self.conv1(x))

        x = self.conv2_block(x)
        x = self.lrelu(self.conv2_down(x))

        x = self.conv3_block(x)
        x = self.lrelu(self.conv3_down(x))

        x = self.conv4_block(x)
        x = self.lrelu(self.conv4_down(x))

        # Global average pooling
        x = torch.mean(x, dim=[2, 3])

        # Classification
        logit = self.fc(x)

        return logit

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def compute_discriminator_loss(D_real: torch.Tensor, D_fake: torch.Tensor,
                               gan_type: str) -> torch.Tensor:
    """
    Compute discriminator loss.

    Args:
        D_real: Discriminator output on real samples
        D_fake: Discriminator output on fake samples
        gan_type: Type of GAN ('Vanilla_GAN', 'LS_GAN', 'W_GAN')
    """
    if gan_type == 'Vanilla_GAN':
        loss_real = F.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real))
        loss_fake = F.binary_cross_entropy_with_logits(D_fake, torch.zeros_like(D_fake))
        return loss_real + loss_fake

    elif gan_type == 'LS_GAN':
        loss_real = 0.5 * torch.mean((D_real - 1) ** 2)
        loss_fake = 0.5 * torch.mean(D_fake ** 2)
        return loss_real + loss_fake

    elif gan_type == 'W_GAN':
        return -(torch.mean(D_real) - torch.mean(D_fake))

    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")


def compute_generator_loss(D1_fake: torch.Tensor, D2_fake: Optional[torch.Tensor],
                          gan_type: str, with_bidirectional: bool) -> torch.Tensor:
    """
    Compute generator loss.

    Args:
        D1_fake: Discriminator output on recovered X1
        D2_fake: Discriminator output on generated blends (for bidirectional mapping)
        gan_type: Type of GAN
        with_bidirectional: Whether to include bidirectional mapping loss
    """
    if gan_type == 'Vanilla_GAN':
        loss1 = F.binary_cross_entropy_with_logits(D1_fake, torch.ones_like(D1_fake))
        if with_bidirectional and D2_fake is not None:
            loss2 = F.binary_cross_entropy_with_logits(D2_fake, torch.ones_like(D2_fake))
            return loss1 + loss2
        return loss1

    elif gan_type == 'LS_GAN':
        loss1 = 0.5 * torch.mean((D1_fake - 1) ** 2)
        if with_bidirectional and D2_fake is not None:
            loss2 = 0.5 * torch.mean((D2_fake - 1) ** 2)
            return loss1 + loss2
        return loss1

    elif gan_type == 'W_GAN':
        if with_bidirectional and D2_fake is not None:
            return -torch.mean(D1_fake + D2_fake)
        return -torch.mean(D1_fake)

    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")

# ============================================================================
# TRAINER
# ============================================================================

class DeblendingTrainer:
    """Trainer for alpha deblending using adversarial training"""

    def __init__(self, config: Config):
        self.config = config
        config.create_dirs()

        # Set random seeds
        self._set_seeds(42)

        # Create dataset and dataloader
        print("\n" + "="*80)
        print("Loading dataset...")
        print("="*80)
        self.dataset = DeblendingDataset(
            config.data_path,
            config.img_size,
            config.n_channels
        )
        self.dataloader = DeblendingDataLoader(
            self.dataset,
            config.batch_size,
            config.alpha
        )

        # Create models
        print("\n" + "="*80)
        print("Creating models...")
        print("="*80)
        self.G = UNetGenerator(
            config.n_channels,
            config.n_channels,
            config.g_base_filters
        ).to(config.device)

        self.D1 = PatchGANDiscriminator(
            config.n_channels,
            config.d_base_filters
        ).to(config.device)

        self.D2 = PatchGANDiscriminator(
            config.n_channels,
            config.d_base_filters
        ).to(config.device)

        # Initialize weights
        self._init_weights()

        # Print model info
        self._print_model_info()

        # Create optimizers
        self._create_optimizers()

        # Create tensorboard writer
        self.writer = SummaryWriter(config.log_dir)

        # Training state
        self.g_iter = 0
        self.d_iter = 0
        self.best_g_loss = float('inf')

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed}")

    def _init_weights(self):
        """Initialize network weights"""
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.G.apply(init_func)
        self.D1.apply(init_func)
        self.D2.apply(init_func)

    def _print_model_info(self):
        """Print model parameter counts"""
        g_params = sum(p.numel() for p in self.G.parameters())
        d1_params = sum(p.numel() for p in self.D1.parameters())
        d2_params = sum(p.numel() for p in self.D2.parameters())

        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator1 parameters: {d1_params:,}")
        print(f"Discriminator2 parameters: {d2_params:,}")
        print(f"Total parameters: {g_params + d1_params + d2_params:,}")

    def _create_optimizers(self):
        """Create optimizers based on GAN type"""
        if self.config.gan_model == 'W_GAN':
            self.opt_G = optim.RMSprop(self.G.parameters(), lr=self.config.g_lr)
            self.opt_D1 = optim.RMSprop(self.D1.parameters(), lr=self.config.d_lr)
            self.opt_D2 = optim.RMSprop(self.D2.parameters(), lr=self.config.d_lr)
        else:
            self.opt_G = optim.Adam(self.G.parameters(), lr=self.config.g_lr, betas=(0.5, 0.999))
            self.opt_D1 = optim.Adam(self.D1.parameters(), lr=self.config.d_lr, betas=(0.5, 0.999))
            self.opt_D2 = optim.Adam(self.D2.parameters(), lr=self.config.d_lr, betas=(0.5, 0.999))

    def _get_d_threshold(self) -> float:
        """Get discriminator threshold for adaptive training"""
        if self.config.gan_model == 'Vanilla_GAN':
            return 0.5
        elif self.config.gan_model == 'LS_GAN':
            return 0.1
        elif self.config.gan_model == 'W_GAN':
            return -5.0
        else:
            return -1e9

    def train_discriminators(self, X1: torch.Tensor, X2: torch.Tensor,
                           Y: torch.Tensor, X1_prime: torch.Tensor) -> Tuple[float, float]:
        """
        Train discriminators.

        Args:
            X1: Images from domain A [B, C, H, W]
            X2: Images from domain B [B, C, H, W]
            Y: Blended images [B, C, H, W]
            X1_prime: Independent batch from domain A [B, C, H, W]

        Returns:
            D1_loss, D2_loss
        """
        # Train D1 (discriminates recovered X1)
        self.D1.train()
        self.G.eval()

        self.opt_D1.zero_grad()

        # Real samples from domain A
        D1_real = self.D1(X1)

        # Fake samples: Recover X1 from Y
        with torch.no_grad():
            Z = self.G(Y)  # Predict X2
            X1_recovered = (Y - (1.0 - self.config.alpha) * Z) / (self.config.alpha + 1e-8)

        D1_fake = self.D1(X1_recovered.detach())

        # Compute loss
        D1_loss = compute_discriminator_loss(D1_real, D1_fake, self.config.gan_model)
        D1_loss.backward()
        self.opt_D1.step()

        # Clip weights for W-GAN
        if self.config.gan_model == 'W_GAN':
            for p in self.D1.parameters():
                p.data.clamp_(-0.01, 0.01)

        # Train D2 (discriminates generated blends) - BIDIRECTIONAL MAPPING
        D2_loss_val = 0.0
        if self.config.with_bidirectional_mapping:
            self.D2.train()
            self.opt_D2.zero_grad()

            # Real samples from blend domain Y
            D2_real = self.D2(Y)

            # Fake samples: Apply predicted X2 to DIFFERENT X1
            with torch.no_grad():
                Z = self.G(Y)  # Predict X2 (same as above)
                # KEY FIX: Use X1_prime (different foreground), not X2!
                Y_gen = self.config.alpha * X1_prime + (1.0 - self.config.alpha) * Z

            D2_fake = self.D2(Y_gen.detach())

            # Compute loss
            D2_loss = compute_discriminator_loss(D2_real, D2_fake, self.config.gan_model)
            D2_loss.backward()
            self.opt_D2.step()

            # Clip weights for W-GAN
            if self.config.gan_model == 'W_GAN':
                for p in self.D2.parameters():
                    p.data.clamp_(-0.01, 0.01)

            D2_loss_val = D2_loss.item()

        return D1_loss.item(), D2_loss_val

    def train_generator(self, X1: torch.Tensor, X2: torch.Tensor,
                       Y: torch.Tensor, X1_prime: torch.Tensor) -> float:
        """
        Train generator.

        Args:
            X1: Images from domain A (ground truth)
            X2: Images from domain B (ground truth)
            Y: Blended images
            X1_prime: Independent batch from domain A

        Returns:
            G_loss value
        """
        self.G.train()
        self.D1.eval()
        self.D2.eval()

        self.opt_G.zero_grad()

        # Predict X2 from blended Y
        Z = self.G(Y)

        # Recover X1 using the predicted X2
        # Formula: X1 = (Y - (1-alpha)*X2) / alpha
        X1_recovered = (Y - (1.0 - self.config.alpha) * Z) / (self.config.alpha + 1e-8)

        # Bidirectional: Generate new blends using predicted X2 and different X1
        # Formula: Y' = alpha*X1' + (1-alpha)*Z
        Y_gen = self.config.alpha * X1_prime + (1.0 - self.config.alpha) * Z

        # Discriminator outputs
        D1_fake = self.D1(X1_recovered)
        D2_fake = self.D2(Y_gen) if self.config.with_bidirectional_mapping else None

        # Adversarial loss
        G_loss = compute_generator_loss(
            D1_fake, D2_fake,
            self.config.gan_model,
            self.config.with_bidirectional_mapping
        )

        # Add paired loss if enabled (supervised training)
        if self.config.with_paired_loss:
            # Loss on predicted X2
            loss_x2 = F.l1_loss(Z, X2)
            # Loss on recovered X1
            loss_x1 = F.l1_loss(X1_recovered, X1)

            paired_loss = self.config.paired_loss_weight * (loss_x1 + loss_x2)
            G_loss = G_loss + paired_loss

            # Log individual losses
            self.writer.add_scalar('Loss/Paired_X1', loss_x1.item(), self.g_iter)
            self.writer.add_scalar('Loss/Paired_X2', loss_x2.item(), self.g_iter)

        G_loss.backward()
        self.opt_G.step()

        return G_loss.item()

    def save_samples(self, X1: torch.Tensor, X2: torch.Tensor, Y: torch.Tensor):
        """Save sample outputs for visualization"""
        self.G.eval()
        with torch.no_grad():
            Z = self.G(Y)
            X1_recovered = (Y - (1.0 - self.config.alpha) * Z) / (self.config.alpha + 1e-8)

        # Helper function to save grid
        def save_grid(tensor, filename):
            # Take first 4 samples
            tensor = tensor[:4].cpu()
            # Clamp for visualization
            tensor = torch.clamp(tensor, 0, 1)
            # Create grid
            grid = torch.cat([tensor[i] for i in range(min(4, len(tensor)))], dim=2)
            # Convert to numpy
            img = grid.permute(1, 2, 0).numpy()
            # Save
            plt.imsave(filename, img if img.shape[2] == 3 else img.squeeze(), cmap='gray' if img.shape[2] == 1 else None)

        # Save images
        iter_str = str(self.g_iter).zfill(6)
        save_grid(Y, f"{self.config.sample_dir}/{iter_str}_input_Y.png")
        save_grid(Z, f"{self.config.sample_dir}/{iter_str}_predicted_X2.png")
        save_grid(X1_recovered, f"{self.config.sample_dir}/{iter_str}_recovered_X1.png")
        save_grid(X1, f"{self.config.sample_dir}/{iter_str}_gt_X1.png")
        save_grid(X2, f"{self.config.sample_dir}/{iter_str}_gt_X2.png")

    def save_checkpoint(self, filename: str = 'checkpoint.pth'):
        """Save training checkpoint"""
        checkpoint = {
            'g_iter': self.g_iter,
            'd_iter': self.d_iter,
            'best_g_loss': self.best_g_loss,
            'G_state': self.G.state_dict(),
            'D1_state': self.D1.state_dict(),
            'D2_state': self.D2.state_dict(),
            'opt_G_state': self.opt_G.state_dict(),
            'opt_D1_state': self.opt_D1.state_dict(),
            'opt_D2_state': self.opt_D2.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, filename))

    def load_checkpoint(self, filename: str = 'checkpoint.pth'):
        """Load training checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
            self.g_iter = checkpoint['g_iter']
            self.d_iter = checkpoint['d_iter']
            self.best_g_loss = checkpoint['best_g_loss']
            self.G.load_state_dict(checkpoint['G_state'])
            self.D1.load_state_dict(checkpoint['D1_state'])
            self.D2.load_state_dict(checkpoint['D2_state'])
            self.opt_G.load_state_dict(checkpoint['opt_G_state'])
            self.opt_D1.load_state_dict(checkpoint['opt_D1_state'])
            self.opt_D2.load_state_dict(checkpoint['opt_D2_state'])
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"Resuming from iteration {self.g_iter}")
            return True
        return False

    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        print(f"Device: {self.config.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Max iterations: {self.config.max_iters}")
        print(f"Alpha: {self.config.alpha}")
        print(f"Bidirectional mapping: {self.config.with_bidirectional_mapping}")
        print(f"Paired loss: {self.config.with_paired_loss}")
        print("="*80 + "\n")

        d_thresh = self._get_d_threshold()
        mm = 1  # Generator training multiplier

        while self.g_iter < self.config.max_iters:
            # ================================================================
            # Get data
            # ================================================================
            X1, X2, Y, X1_prime = self.dataloader.get_independent_batches()
            X1 = X1.to(self.config.device)
            X2 = X2.to(self.config.device)
            Y = Y.to(self.config.device)
            X1_prime = X1_prime.to(self.config.device)

            # ================================================================
            # Train Discriminators
            # ================================================================
            D1_loss, D2_loss = self.train_discriminators(X1, X2, Y, X1_prime)
            self.d_iter += 1

            # Adjust generator training frequency
            if D1_loss < d_thresh or D2_loss < d_thresh:
                mm = min(mm + 5, 20)
            else:
                mm = 1

            # ================================================================
            # Train Generator (multiple times if needed)
            # ================================================================
            for _ in range(mm):
                # Get fresh batch
                X1, X2, Y, X1_prime = self.dataloader.get_independent_batches()
                X1 = X1.to(self.config.device)
                X2 = X2.to(self.config.device)
                Y = Y.to(self.config.device)
                X1_prime = X1_prime.to(self.config.device)

                G_loss = self.train_generator(X1, X2, Y, X1_prime)
                self.g_iter += 1

                # Log to tensorboard
                self.writer.add_scalar('Loss/G', G_loss, self.g_iter)
                self.writer.add_scalar('Loss/D1', D1_loss, self.g_iter)
                self.writer.add_scalar('Loss/D2', D2_loss, self.g_iter)
                self.writer.add_scalar('Training/G_multiplier', mm, self.g_iter)

                # ============================================================
                # Logging and Saving
                # ============================================================
                if self.g_iter % self.config.log_interval == 0:
                    print(f"[Iter {self.g_iter}/{self.config.max_iters}] "
                          f"G: {G_loss:.4f} | D1: {D1_loss:.4f} | D2: {D2_loss:.4f} | mm: {mm}")

                    # Save samples
                    self.save_samples(X1, X2, Y)

                if self.g_iter % self.config.save_interval == 0 and self.g_iter > 0:
                    # Save regular checkpoint
                    self.save_checkpoint(f'ckpt_{self.g_iter}.pth')

                    # Save best model
                    if G_loss < self.best_g_loss:
                        self.best_g_loss = G_loss
                        self.save_checkpoint('best_ckpt.pth')
                        print(f"*** New best model saved! G_loss: {G_loss:.6f} ***")

                if self.g_iter >= self.config.max_iters:
                    break

        # Save final checkpoint
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        self.save_checkpoint('final_ckpt.pth')
        self.writer.close()
        print(f"Best G_loss: {self.best_g_loss:.6f}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train alpha deblending GAN')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_iters', type=int, default=20000, help='Maximum iterations')
    parser.add_argument('--alpha', type=float, default=0.5, help='Blending alpha value')
    parser.add_argument('--paired_loss', action='store_true', help='Use paired supervision')
    parser.add_argument('--no_bidirectional', action='store_true', help='Disable bidirectional mapping')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Create config
    config = Config()
    config.data_path = args.data_path
    config.checkpoint_dir = args.checkpoint_dir
    config.batch_size = args.batch_size
    config.max_iters = args.max_iters
    config.alpha = args.alpha
    config.with_paired_loss = args.paired_loss
    config.with_bidirectional_mapping = not args.no_bidirectional

    # Update sample and log dirs based on checkpoint dir
    config.sample_dir = os.path.join(os.path.dirname(config.checkpoint_dir), 'samples')
    config.log_dir = os.path.join(os.path.dirname(config.checkpoint_dir), 'logs')

    # Create trainer
    trainer = DeblendingTrainer(config)

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
