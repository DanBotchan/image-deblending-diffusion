"""
Base class for measurement operators.
"""
from abc import ABC, abstractmethod

import torch


class LinearOperator(ABC):
    """
    Abstract base class for linear measurement operators.

    Linear operators represent the forward measurement model A in
    the inverse problem y = A(x), where y is the measurement and
    x is the signal to recover.
    """

    @abstractmethod
    def forward(self,
                data: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Apply the forward operator A to the data.

        Args:
            data: Input tensor
            **kwargs: Additional parameters

        Returns:
            Measured output tensor
        """
        pass

    def __call__(self,
                 data: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Apply the forward operator."""
        return self.forward(data, **kwargs)
