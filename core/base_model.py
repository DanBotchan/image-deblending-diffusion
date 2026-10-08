"""
Base model wrapper interface for the deblend package.

Provides abstract base class for model wrappers that unifies the interface
between latent-space (Stable Diffusion) and pixel-space (DDPM) models.
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseModelWrapper(ABC):
    """
    Abstract base class for diffusion model wrappers.

    Provides a unified interface for both latent-space models (Stable Diffusion)
    and pixel-space models (DDPM). This allows samplers to work with either
    model type through the same API.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize the model wrapper.

        Args:
            model: The underlying diffusion model
            device: Device to run the model on
        """
        self.model = model
        self.device = device

    @abstractmethod
    def predict_noise(self,
                      x_t: torch.Tensor,
                      t: torch.Tensor,
                      cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict noise from noisy input at timestep t.

        Args:
            x_t: Noisy input (latent or pixel space)
            t: Timestep tensor
            cond: Optional conditioning (e.g., text embeddings)

        Returns:
            Predicted noise tensor
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to diffusion space.

        For latent models (SD): Encodes pixels to latent space
        For pixel models (DDPM): Identity operation

        Args:
            x: Input tensor in pixel space

        Returns:
            Tensor in diffusion space
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from diffusion space to pixel space.

        For latent models (SD): Decodes latents to pixels
        For pixel models (DDPM): Identity operation

        Args:
            z: Tensor in diffusion space

        Returns:
            Tensor in pixel space
        """
        pass

    def decode_with_grad(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from diffusion space with gradient computation enabled.

        Default implementation calls decode(). Override if gradients
        need special handling (e.g., for latent diffusion).

        Args:
            z: Tensor in diffusion space

        Returns:
            Tensor in pixel space (with gradients)
        """
        return self.decode(z)

    @property
    def is_latent_model(self) -> bool:
        """Whether this is a latent-space model."""
        return False

    @property
    def latent_scale_factor(self) -> float:
        """Scale factor for latent space (1.0 for pixel models)."""
        return 1.0

    def to(self, device: torch.device) -> "BaseModelWrapper":
        """Move model to specified device."""
        self.model = self.model.to(device)
        self.device = device
        return self

    def eval(self) -> "BaseModelWrapper":
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def train(self) -> "BaseModelWrapper":
        """Set model to training mode."""
        self.model.train()
        return self

    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()
