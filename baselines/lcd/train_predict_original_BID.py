"""
Training script for IID decomposition on BID dataset - Predict Original Image

This script trains a diffusion model to predict the ORIGINAL image (x_orig)
from a blended image, using the BID dataset with separate trainA/trainB folders.

trainA = x_orig (original images)
trainB = x_add (added images)

Supports distributed training across multiple GPUs using torch.distributed.
"""
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from iid_diffusion import Unet, IIDGaussianDiffusion, Trainer
from iid_diffusion.utils import BIDPairedDataset


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif torch.cuda.device_count() > 1:
        # If not using torchrun, but multiple GPUs available, use all of them
        rank = 0
        world_size = 1
        local_rank = 0
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Train IID decomposition model on BID dataset (predict original image)')

    # Dataset parameters
    parser.add_argument('--data_path', type=str,
                        required=True,
                        help='Path to BID training data folder (should contain trainA and trainB)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (default: 256)')

    # Model parameters
    parser.add_argument('--dim', type=int, default=64,
                        help='Base dimension for U-Net (default: 64)')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Dimension multipliers for U-Net (default: [1, 2, 4, 8])')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of image channels (default: 3)')

    # Diffusion parameters
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('--alpha_max', type=float, default=0.8,
                        help='Maximum alpha value (default: 0.8)')
    parser.add_argument('--loss_type', type=str, default='l1',
                        choices=['l1', 'l2'],
                        help='Loss type (default: l1)')

    # Sampling parameters
    parser.add_argument('--sampling_method', type=str, default='ours',
                        choices=['ours', 'ours_epsilon', 'cold_diffusion', 'one_step'],
                        help='Sampling method for decomposition (default: ours)')

    # Training parameters
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='Batch size per GPU for training (default: 8)')
    parser.add_argument('--train_lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--train_num_steps', type=int, default=1000000,
                        help='Total training steps (default: 1000000)')
    parser.add_argument('--gradient_accumulate_every', type=int, default=2,
                        help='Gradient accumulation steps (default: 2)')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='EMA decay rate (default: 0.995)')
    parser.add_argument('--save_and_sample_every', type=int, default=1000,
                        help='Save checkpoint every N steps (default: 1000)')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to generate (default: 16)')

    # Paths
    parser.add_argument('--results_folder', type=str, default='./results_BID_predict_original',
                        help='Folder to save results (default: ./results_BID_predict_original)')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to checkpoint to resume from (default: None)')

    # Other
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation (default: True)')
    parser.add_argument('--no_augment', action='store_false', dest='augment',
                        help='Disable data augmentation')
    parser.add_argument('--no_time_embed', action='store_true',
                        help='Disable time embedding in U-Net')
    parser.add_argument('--residual', action='store_true',
                        help='Use residual connection in U-Net')

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    if is_main_process:
        print("=" * 80)
        print("Training IID Decomposition Model on BID Dataset")
        print("Prediction Type: x_orig (Original Image)")
        print("=" * 80)
        print(f"Dataset: BID")
        print(f"Data path: {args.data_path}")
        print(f"  - trainA (x_orig): {args.data_path}/trainA")
        print(f"  - trainB (x_add): {args.data_path}/trainB")
        print(f"Image size: {args.image_size}x{args.image_size}")
        print(f"Timesteps: {args.timesteps}")
        print(f"Alpha max: {args.alpha_max}")
        print(f"Batch size per GPU: {args.train_batch_size}")
        print(f"Total batch size: {args.train_batch_size * world_size * args.gradient_accumulate_every}")
        print(f"Learning rate: {args.train_lr}")
        print(f"Augmentation: {args.augment}")
        print(f"Sampling method: {args.sampling_method}")
        print(f"Results folder: {args.results_folder}")
        print("=" * 80)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process:
            print(f"Using device: {device}")
            print(f"Number of GPUs: {world_size}")
            print(f"GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
            print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        if is_main_process:
            print(f"Using device: {device}")

    # Create dataset
    if is_main_process:
        print(f"\nLoading BID dataset from {args.data_path}...")
    dataset = BIDPairedDataset(
        args.data_path,
        args.image_size,
        augment=args.augment
    )

    # Create model
    if is_main_process:
        print("\nCreating U-Net model...")
    model = Unet(
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        channels=args.channels,
        with_time_emb=not args.no_time_embed,
        residual=args.residual
    ).to(device)

    # Count parameters
    if is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable_params:,}")

    # Create diffusion model
    if is_main_process:
        print("\nCreating diffusion model...")
    diffusion = IIDGaussianDiffusion(
        model,
        image_size=args.image_size,
        channels=args.channels,
        timesteps=args.timesteps,
        alpha_max=args.alpha_max,
        loss_type=args.loss_type,
        prediction_type='x_orig'  # Predict original image (BEST)
    ).to(device)

    # Use DistributedDataParallel if multiple GPUs available
    if world_size > 1:
        if is_main_process:
            print(f"Using {world_size} GPUs with DistributedDataParallel")
        diffusion = DDP(diffusion, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        # Fallback to DataParallel if DDP not initialized
        if is_main_process:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        diffusion = torch.nn.DataParallel(diffusion)

    # Create trainer
    if is_main_process:
        print("\nCreating trainer...")
    trainer = Trainer(
        diffusion,
        dataset,
        train_batch_size=args.train_batch_size,
        gradient_accumulate_every=args.gradient_accumulate_every,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        ema_decay=args.ema_decay,
        save_and_sample_every=args.save_and_sample_every,
        num_samples=args.num_samples,
        results_folder=args.results_folder,
        load_path=args.load_path,
        sampling_method=args.sampling_method,
    )

    # Print training info
    if is_main_process:
        print("\nTraining Configuration:")
        print(f"  Total steps: {args.train_num_steps:,}")
        print(f"  Gradient accumulation: {args.gradient_accumulate_every}")
        print(f"  Effective batch size per GPU: {args.train_batch_size * args.gradient_accumulate_every}")
        print(f"  Total effective batch size: {args.train_batch_size * args.gradient_accumulate_every * world_size}")
        print(f"  Save/sample every: {args.save_and_sample_every} steps")
        print(f"  Number of samples: {args.num_samples}")
        print(f"  Sampling method: {args.sampling_method}")

    # Start training
    try:
        if is_main_process:
            print("\n" + "=" * 80)
            print("Starting training...")
            print("=" * 80 + "\n")
        trainer.train()
    finally:
        # Clean up distributed training
        cleanup_distributed()

    if is_main_process:
        print("\nTraining complete!")


if __name__ == '__main__':
    main()
