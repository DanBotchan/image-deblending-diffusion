"""
Trainer for IID image decomposition models
"""
import os
import copy
from pathlib import Path
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from iid_diffusion.utils import cycle, num_to_groups


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer:
    """
    Trainer for IID decomposition models.

    Args:
        diffusion_model: IIDGaussianDiffusion model
        dataset: Training dataset
        *,
        train_batch_size: Batch size for training
        gradient_accumulate_every: Gradient accumulation steps
        train_lr: Learning rate
        train_num_steps: Total training steps
        ema_decay: EMA decay rate
        save_and_sample_every: Save checkpoint every N steps
        results_folder: Where to save results
        load_path: Path to checkpoint to resume from
    """

    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        train_batch_size=32,
        gradient_accumulate_every=2,
        train_lr=3e-4,
        train_num_steps=1000000,
        ema_decay=0.995,
        save_and_sample_every=1000,
        num_samples=16,
        results_folder='./results',
        load_path=None,
        sampling_method='ours_epsilon',
    ):
        super().__init__()

        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = 10

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.sampling_method = sampling_method

        self.dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
        self.dl = cycle(self.dl)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        if load_path is not None:
            self.load(load_path)

    def save(self, step=None):
        """Save model checkpoint."""
        # Get underlying modules (unwrap DataParallel if needed)
        model_module = self.get_module(self.model)
        ema_module = self.get_module(self.ema_model)

        data = {
            'step': self.step,
            'model': model_module.state_dict(),
            'ema': ema_module.state_dict(),
            'opt': self.opt.state_dict(),
        }

        if step is not None:
            torch.save(data, str(self.results_folder / f'model-{step:08d}.pt'))
        else:
            torch.save(data, str(self.results_folder / 'model.pt'))

    def load(self, path):
        """Load model checkpoint."""
        print(f"Loading model from {path}")
        data = torch.load(path, map_location='cpu')

        self.step = data['step']

        # Get underlying modules (unwrap DataParallel if needed)
        model_module = self.get_module(self.model)
        ema_module = self.get_module(self.ema_model)

        model_module.load_state_dict(data['model'])
        ema_module.load_state_dict(data['ema'])
        self.opt.load_state_dict(data['opt'])

    def get_module(self, model):
        """Get the underlying module from DataParallel/DistributedDataParallel if wrapped."""
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            return model.module
        return model

    def train(self):
        """Main training loop."""
        device = next(self.model.parameters()).device

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    x_orig = data['x_orig'].to(device)
                    x_add = data['x_add'].to(device)

                    loss = self.model(x_orig, x_add)

                    # Handle DataParallel returning tensor instead of scalar
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()

                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()

                    loss.backward()

                pbar.set_description(f'loss: {total_loss:.4f}')

                self.opt.step()
                self.opt.zero_grad()

                # Update EMA
                if self.step % self.update_ema_every == 0:
                    self.ema.update_model_average(self.ema_model, self.model)

                # Save and sample
                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.save(self.step)
                    self.sample(self.step)

                self.step += 1
                pbar.update(1)

        print('Training complete!')
        self.save()

    @torch.no_grad()
    def sample(self, step=None):
        """
        Generate samples during training.

        Creates blended images and decomposes them to verify training progress.
        """
        self.ema_model.eval()

        # Get a batch of data
        data = next(self.dl)
        device = next(self.model.parameters()).device

        x_orig = data['x_orig'][:self.num_samples].to(device)
        x_add = data['x_add'][:self.num_samples].to(device)

        # Create blended images with alpha=0.5
        alpha = 0.5
        x_blend = (1 - alpha) * x_orig + alpha * x_add

        # Decompose using the model (handle DataParallel/DistributedDataParallel)
        ema_module = self.get_module(self.ema_model)
        x_orig_pred, x_add_pred = ema_module.sample(
            x_blend,
            alpha_init=alpha,
            sampling_method=self.sampling_method
        )

        # Save visualizations
        # Grid: [original GT | added GT | blended | original pred | added pred]
        all_images = torch.cat([
            x_orig,
            x_add,
            x_blend,
            x_orig_pred,
            x_add_pred
        ], dim=0)

        suffix = f'{step:08d}' if step is not None else ''
        save_path = str(self.results_folder / f'sample_{suffix}.png')

        # Convert from [-1, 1] to [0, 1]
        all_images = (all_images + 1) / 2

        save_image(
            all_images,
            save_path,
            nrow=self.num_samples
        )

        self.ema_model.train()
