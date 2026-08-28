import torch

from utils.image import show_tensor  # noqa: F401
from .base import AlphaOptimizer
from .registry import register_alpha_optimizer



@register_alpha_optimizer('param_base')
class ParamOptimizer(AlphaOptimizer):
    def optimize_step(self,
                      alpha_map,
                      measurement,
                      x_0_hat,
                      operator,
                      alpha_scale,
                      ):
        blend = operator.forward(x_0_hat, alpha=alpha_map).unsqueeze(0)
        diff = measurement - blend
        reconstruction_loss = torch.norm(diff)
        total_loss = reconstruction_loss
        grads = torch.autograd.grad(total_loss, alpha_map.get_parameters())
        with torch.no_grad():
            for param, grad in zip(alpha_map.get_parameters(), grads):
                param.data -= alpha_scale * grad

        self.record_state(alpha_map())
        return total_loss.item()


@register_alpha_optimizer('param_adam')
class ParamAdamOptimizer(AlphaOptimizer):
    def __init__(self,
                 lr: float = 0.01,
                 betas: float = (0.9, 0.999),
                 eps: float = 1e-8,
                 **kwargs):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = {}  # First moment estimate
        self.v = {}  # Second moment estimate
        self.t = {}  # Timestep counter for each parameter

    def _init_param_state(self,
                          param: torch.Tensor):
        """Initialize state for a parameter if not already done."""
        param_id = id(param)
        if param_id not in self.m:
            self.m[param_id] = torch.zeros_like(param.data)
            self.v[param_id] = torch.zeros_like(param.data)
            self.t[param_id] = 0

    def optimize_step(self,
                      alpha_map,
                      measurement,
                      x_0_hat,
                      operator,
                      alpha_scale,
                      iterations=1,
                      ) -> float:
        """
        Optimize alpha map parameters using Adam algorithm.

        Args:
            alpha_map: The alpha map to optimize
            measurement: Target image
            x_0_hat: Current estimate of source images
            operator: Blending operator
            alpha_scale: Scale factor for updates
            iterations: Number of optimization iterations to perform

        Returns:
            float: Final loss value
        """
        # Initialize parameter states if needed
        for param in alpha_map.get_parameters():
            self._init_param_state(param)

        loss = None
        for _ in range(iterations):
            blend = operator.forward(x_0_hat,
                                     alpha=alpha_map).unsqueeze(0)
            diff = torch.sub(measurement, blend)
            reconstruction_loss = torch.norm(diff)
            loss = reconstruction_loss.item()
            del blend
            grads = torch.autograd.grad(reconstruction_loss,
                                        alpha_map.get_parameters())
            with torch.no_grad():
                for param, grad in zip(alpha_map.get_parameters(), grads):
                    param_id = id(param)
                    self.m[param_id] = (self.betas[0] * self.m[param_id] +
                                        (1 - self.betas[0]) * grad)
                    self.v[param_id] = (self.betas[1] * self.v[param_id] +
                                        (1 - self.betas[1]) * grad.pow(2))
                    self.t[param_id] += 1

                    m_hat = self.m[param_id] / (
                            1 - self.betas[0]**self.t[param_id])
                    v_hat = self.v[param_id] / (
                            1 - self.betas[1]**self.t[param_id])

                    # Update parameter
                    param.data -= alpha_scale * self.lr * m_hat / (
                            v_hat.sqrt() + self.eps)

        return loss


@register_alpha_optimizer('param_adam_ms')
class ParamAdamOptimizerMS(ParamAdamOptimizer):
    """
    Adam optimizer for alpha map parameters with smoothness regularization.
    Implements the Adam algorithm for parameter optimization.
    """

    def __init__(self, *args, smoothness_weight=1e-3, **kwargs):
        """
        Initialize the Adam optimizer.

        Args:
            lr (float): Learning rate
            betas (tuple): Coefficients for computing running averages of gradient and its square
            eps (float): Term added to denominator for numerical stability
            smoothness_weight (float): Weight for smoothness regularization (default: 1e-3)
        """
        super().__init__(*args, **kwargs)

        self.scales = [1.0, 0.5, 0.25]
        self.scale_weights = [0.25, 0.5, 1.0]
        self.downsample = torch.nn.AvgPool2d(2)
        self.smoothness_weight = smoothness_weight

    def optimize_step(self,
                      alpha_map,
                      measurement,
                      x_0_hat,
                      operator,
                      alpha_scale,
                      iterations=1,
                      ) -> float:
        """
        Optimize alpha map parameters using Adam algorithm with smoothness regularization.

        Args:
            alpha_map: The alpha map to optimize
            measurement: Target image
            x_0_hat: Current estimate of source images
            operator: Blending operator
            alpha_scale: Scale factor for updates
            iterations: Number of optimization iterations to perform

        Returns:
            float: Final loss value
        """
        # Initialize parameter states if needed
        for param in alpha_map.get_parameters():
            self._init_param_state(param)

        loss = None
        # Perform multiple optimization iterations
        for _ in range(iterations):
            # Forward pass - use inplace operations where possible
            blend = operator.forward(x_0_hat, alpha=alpha_map)

            # Multi-scale reconstruction loss
            losses = []
            current_pred = blend
            current_target = measurement
            for scale, weight in zip(self.scales, self.scale_weights):
                if scale != 1.0:
                    current_pred = self.downsample(current_pred)
                    current_target = self.downsample(current_target)

                diff = current_pred - current_target
                closs = weight * torch.linalg.norm(diff)
                losses.append(closs)

            reconstruction_loss = torch.stack(losses).sum()

            # Add smoothness regularization
            smoothness_loss = alpha_map.get_smoothness_loss(weight=self.smoothness_weight)

            # Total loss
            total_loss = reconstruction_loss + smoothness_loss

            # Store loss value before grad computation
            loss = total_loss.item()

            # Free memory for unused tensors
            del blend

            # Compute gradients with respect to total loss
            grads = torch.autograd.grad(total_loss, alpha_map.get_parameters())

            # Update parameters using Adam
            with torch.no_grad():
                for param, grad in zip(alpha_map.get_parameters(), grads):
                    param_id = id(param)

                    # Update biased first moment estimate
                    self.m[param_id] = self.betas[0] * self.m[param_id] + (1 - self.betas[0]) * grad

                    # Update biased second moment estimate
                    self.v[param_id] = self.betas[1] * self.v[param_id] + (1 - self.betas[1]) * grad.pow(2)

                    # Increment timestep
                    self.t[param_id] += 1

                    # Compute bias-corrected first and second moment estimates
                    m_hat = self.m[param_id] / (1 - self.betas[0]**self.t[param_id])
                    v_hat = self.v[param_id] / (1 - self.betas[1]**self.t[param_id])

                    # Update parameter
                    param.data -= alpha_scale * self.lr * m_hat / (v_hat.sqrt() + self.eps)

            # Record state after each iteration
            #self.record_state(alpha_map())

        return loss

    def get_loss_components(self, alpha_map, measurement, x_0_hat, operator):
        """
        Get individual loss components for monitoring/debugging.

        Args:
            alpha_map: The alpha map
            measurement: Target image
            x_0_hat: Current estimate of source images
            operator: Blending operator

        Returns:
            dict: Dictionary with loss components
        """
        with torch.no_grad():
            # Reconstruction loss
            blend = operator.forward(x_0_hat, alpha=alpha_map)

            losses = []
            current_pred = blend
            current_target = measurement
            for scale, weight in zip(self.scales, self.scale_weights):
                if scale != 1.0:
                    current_pred = self.downsample(current_pred)
                    current_target = self.downsample(current_target)

                diff = current_pred - current_target
                closs = weight * torch.linalg.norm(diff)
                losses.append(closs)

            reconstruction_loss = torch.stack(losses).sum()

            # Smoothness loss
            smoothness_loss = alpha_map.get_smoothness_loss(weight=self.smoothness_weight)

            return {
                'reconstruction_loss': reconstruction_loss.item(),
                'smoothness_loss': smoothness_loss.item(),
                'total_loss': (reconstruction_loss + smoothness_loss).item(),
                'smoothness_weight': self.smoothness_weight
            }