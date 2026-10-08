"""
IID Gaussian Diffusion for Identically Distributed Superimposed Image Decomposition
Based on "Leveraging Cold Diffusion for the Decomposition of Identically Distributed Superimposed Images"
Montenegro & Cardoso, IEEE OJSP 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot  as plt

def show_tensor(tens):
    plt.imshow((tens.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2)
    plt.show()

class IIDGaussianDiffusion(nn.Module):
    """
    Diffusion model for decomposing superimposed images.

    Forward process: x_t = (1 - alpha_t) * x_orig + alpha_t * x_add

    Args:
        denoise_fn: U-Net model
        image_size: Size of images
        channels: Number of image channels (default: 3)
        timesteps: Number of diffusion steps (default: 1000)
        alpha_max: Maximum alpha value (default: 0.8)
        loss_type: 'l1' or 'l2' (default: 'l1')
        prediction_type: What the model predicts - 'x_add', 'x_orig', or 'x_diff'
    """

    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels=3,
        timesteps=1000,
        alpha_max=0.8,
        loss_type='l1',
        prediction_type='x_orig'  # 'x_add', 'x_orig', 'x_diff'
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)
        self.alpha_max = alpha_max
        self.loss_type = loss_type
        self.prediction_type = prediction_type

        # Linear schedule for alpha: alpha_t = (t / t_max) * alpha_max
        # At t=0: alpha=0 (pure original image)
        # At t=t_max: alpha=alpha_max (mostly added image)
        alphas = torch.linspace(0, alpha_max, timesteps)
        self.register_buffer('alphas', alphas)

    def q_sample(self, x_orig, x_add, t):
        """
        Forward diffusion process: blend two images according to timestep.

        x_t = (1 - alpha_t) * x_orig + alpha_t * x_add (Equation 2 from paper)

        Args:
            x_orig: Original image (higher pixel intensity)
            x_add: Added image (lower pixel intensity)
            t: Timestep tensor [batch_size]

        Returns:
            Blended image at timestep t
        """
        alpha_t = self.alphas[t]

        # Reshape alpha_t for broadcasting: [batch_size, 1, 1, 1]
        alpha_t = alpha_t.view(-1, 1, 1, 1)

        # Linear blending
        x_t = (1 - alpha_t) * x_orig + alpha_t * x_add

        return x_t

    def p_losses(self, x_orig, x_add, t):
        """
        Calculate training loss.

        Args:
            x_orig: Original image
            x_add: Added image
            t: Timestep

        Returns:
            Loss value
        """
        # Get blended image at timestep t
        x_blend = self.q_sample(x_orig, x_add, t)

        # Model prediction
        x_pred = self.denoise_fn(x_blend, t)

        # Determine ground truth based on prediction type
        if self.prediction_type == 'x_add':
            target = x_add
        elif self.prediction_type == 'x_orig':
            target = x_orig
        elif self.prediction_type == 'x_diff':
            # Difference between consecutive timesteps
            # For t > 0: x_{t-1} - x_t
            # For t = 0: 0 (no difference)
            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            alpha_t_minus_1 = torch.where(
                t > 0,
                self.alphas[torch.clamp(t - 1, min=0)],
                torch.zeros_like(self.alphas[t])
            ).view(-1, 1, 1, 1)

            x_t_minus_1 = (1 - alpha_t_minus_1) * x_orig + alpha_t_minus_1 * x_add
            target = x_t_minus_1 - x_blend
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Compute loss (Equation 3 from paper)
        if self.loss_type == 'l1':
            loss = (target - x_pred).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(target, x_pred)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

        return loss

    def forward(self, x_orig, x_add):
        """
        Training forward pass.

        Args:
            x_orig: Original image batch
            x_add: Added image batch

        Returns:
            Loss value
        """
        b, c, h, w = x_orig.shape
        device = x_orig.device

        assert h == self.image_size and w == self.image_size, \
            f'Image size must be {self.image_size}x{self.image_size}'

        # Random timestep for each sample in batch
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x_orig, x_add, t)

    @torch.no_grad()
    def sample(
        self,
        x_init,
        alpha_init=0.5,
        sampling_method='ours_epsilon',
        return_all_steps=False
    ):
        """
        Sample/decompose a superimposed image.

        Args:
            x_init: Initial superimposed image
            alpha_init: Weight of the blending (default 0.5 for equal average)
            sampling_method: 'ours', 'ours_epsilon', 'cold_diffusion', 'one_step'
            return_all_steps: If True, return intermediate steps

        Returns:
            x_orig_pred: Predicted original image
            x_add_pred: Predicted added image
            (optional) all_steps: List of intermediate predictions
        """
        self.denoise_fn.eval()

        batch_size = x_init.shape[0]
        device = x_init.device

        # Find initial timestep: t_init = (alpha_init / alpha_max) * t_max
        t_init = int((alpha_init / self.alpha_max) * self.num_timesteps)
        t_init = max(0, min(t_init, self.num_timesteps - 1))

        if sampling_method == 'one_step':
            return self._sample_one_step(x_init, t_init)
        elif sampling_method == 'cold_diffusion':
            return self._sample_cold_diffusion(x_init, t_init, return_all_steps)
        elif sampling_method == 'ours':
            return self._sample_ours(x_init, t_init, alpha_init, return_all_steps, remove_error=False)
        elif sampling_method == 'ours_epsilon':
            return self._sample_ours(x_init, t_init, alpha_init, return_all_steps, remove_error=True)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    def _sample_one_step(self, x_init, t_init):
        """Direct one-step prediction."""
        batch_size = x_init.shape[0]
        device = x_init.device

        t = torch.full((batch_size,), t_init, dtype=torch.long, device=device)
        x_pred = self.denoise_fn(x_init, t)

        if self.prediction_type == 'x_orig':
            x_orig_pred = x_pred
            alpha_init = self.alphas[t_init]
            x_add_pred = (x_init - (1 - alpha_init) * x_orig_pred) / alpha_init
        elif self.prediction_type == 'x_add':
            x_add_pred = x_pred
            alpha_init = self.alphas[t_init]
            x_orig_pred = (x_init - alpha_init * x_add_pred) / (1 - alpha_init)
        elif self.prediction_type == 'x_diff':
            # Multiply by number of steps
            x_diff_total = x_pred * t_init
            x_orig_pred = x_init + x_diff_total
            alpha_init = self.alphas[t_init]
            x_add_pred = (x_init - (1 - alpha_init) * x_orig_pred) / alpha_init

        return x_orig_pred, x_add_pred

    def _sample_cold_diffusion(self, x_init, t_init, return_all_steps):
        """Standard Cold Diffusion sampling (Algorithm from Bansal et al.)"""
        batch_size = x_init.shape[0]
        device = x_init.device

        x_t = x_init
        all_steps = [x_t] if return_all_steps else None

        for t_idx in range(t_init, 0, -1):
            t = torch.full((batch_size,), t_idx, dtype=torch.long, device=device)

            # Predict
            if self.prediction_type == 'x_orig':
                x_orig_pred = self.denoise_fn(x_t, t)
                x_add_pred = (x_t - (1 - self.alphas[t_idx]) * x_orig_pred) / self.alphas[t_idx]
            elif self.prediction_type == 'x_add':
                x_add_pred = self.denoise_fn(x_t, t)
                x_orig_pred = (x_t - self.alphas[t_idx] * x_add_pred) / (1 - self.alphas[t_idx])
            elif self.prediction_type == 'x_diff':
                x_diff = self.denoise_fn(x_t, t)
                x_t = x_t + x_diff
                if return_all_steps:
                    all_steps.append(x_t)
                continue

            # Apply forward process to get x_{t-1}
            alpha_t_minus_1 = self.alphas[t_idx - 1]
            x_t = (1 - alpha_t_minus_1) * x_orig_pred + alpha_t_minus_1 * x_add_pred

            if return_all_steps:
                all_steps.append(x_t)

        # Final prediction at t=0
        t = torch.full((batch_size,), 0, dtype=torch.long, device=device)
        if self.prediction_type != 'x_diff':
            if self.prediction_type == 'x_orig':
                x_orig_final = self.denoise_fn(x_t, t)
                x_add_final = (x_init - (1 - self.alphas[t_init]) * x_orig_final) / self.alphas[t_init]
            else:
                x_add_final = self.denoise_fn(x_t, t)
                x_orig_final = (x_init - self.alphas[t_init] * x_add_final) / (1 - self.alphas[t_init])
        else:
            x_orig_final = x_t
            x_add_final = (x_init - (1 - self.alphas[t_init]) * x_orig_final) / self.alphas[t_init]

        if return_all_steps:
            return x_orig_final, x_add_final, all_steps
        return x_orig_final, x_add_final

    def _sample_ours(self, x_init, t_init, alpha_init, return_all_steps, remove_error):
        """
        Proposed sampling method from the paper.

        Implements equations 5-7 for error removal when remove_error=True.
        """
        batch_size = x_init.shape[0]
        device = x_init.device

        x_t = x_init
        all_steps = [x_t] if return_all_steps else None

        # Convert prediction to x_add for processing (Section III.C.2)
        for t_idx in range(t_init, 0, -1):
            t = torch.full((batch_size,), t_idx, dtype=torch.long, device=device)
            alpha_t = self.alphas[t_idx]
            alpha_t_minus_1 = self.alphas[t_idx - 1]

            # Get model prediction
            x_pred = self.denoise_fn(x_t, t)

            # Convert to x_add_pred if needed (Equation 8)
            if self.prediction_type == 'x_orig':
                x_add_pred = (x_t - (1 - alpha_t) * x_pred) / alpha_t
            elif self.prediction_type == 'x_add':
                x_add_pred = x_pred
            elif self.prediction_type == 'x_diff':
                x_t = x_t + x_pred
                if return_all_steps:
                    all_steps.append(x_t)
                continue

            # Equation 5: Compute x_{t-1} without error removal
            coef_t = (1 - alpha_t_minus_1) / (1 - alpha_t)
            coef_add = (alpha_t - alpha_t_minus_1) / (1 - alpha_t)
            x_t_minus_1 = coef_t * x_t - coef_add * x_add_pred

            # Error removal (Equations 6-7)
            if remove_error:
                # Estimate x_orig (Equation 6)
                x_orig_est = (x_t - alpha_t * x_add_pred) / (1 - alpha_t)

                # Estimate x_init
                x_init_est = (1 - alpha_init) * x_orig_est + alpha_init * x_add_pred

                # Compute error (Equation 7)
                if abs(alpha_init - alpha_t) > 1e-2:
                    epsilon = ((1 - alpha_t) / (alpha_init - alpha_t)) * (x_init_est - x_init)

                    # Remove error from x_{t-1}
                    x_t_minus_1 = x_t_minus_1 - ((alpha_t_minus_1 - alpha_t) / (1 - alpha_t)) * epsilon

            x_t = x_t_minus_1

            if return_all_steps:
                all_steps.append(x_t)

        # Final prediction at t=0
        x_orig_pred = x_t

        # Get x_add from equation 4
        x_add_pred = (x_init - (1 - alpha_init) * x_orig_pred) / alpha_init

        if return_all_steps:
            return x_orig_pred, x_add_pred, all_steps
        return x_orig_pred, x_add_pred
