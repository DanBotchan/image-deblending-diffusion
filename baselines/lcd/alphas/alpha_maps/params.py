import math
import random

import torch

from .base import AlphaMap
from .registry import register_alpha_map


@register_alpha_map('param_2d_order')
class ParamSecondOrderMap(AlphaMap):
    def _initialize_parameters(self):
        if self.init_alpha:
            a_t, b_t, c_t, d_t, e_t, f_t = self.init_alpha
        else:
            a_t = random.uniform(self.alpha_min, self.alpha_max)
            b_t = random.uniform(self.alpha_min - a_t, self.alpha_max - a_t)
            e_t = random.uniform(self.alpha_min - (a_t + b_t),
                                 self.alpha_max - (a_t + b_t))
            c_t = random.uniform(self.alpha_min - a_t, self.alpha_max - a_t)
            f_t = random.uniform(self.alpha_min - (a_t + c_t),
                                 self.alpha_max - (a_t + c_t))
            d_t = random.uniform(
                self.alpha_min - (a_t + b_t + c_t + e_t + f_t),
                self.alpha_max - (a_t + b_t + c_t + e_t + f_t))

        a = math.atanh(2 * ((a_t - self.alpha_min) / self.min_slope - 0.5))
        b = math.atanh(
            2 * ((b_t - (self.alpha_min - a_t)) / self.min_slope - 0.5))
        e = math.atanh(2 * (
                ((e_t + (a_t + b_t)) - self.alpha_min) / self.min_slope - 0.5))
        c = math.atanh(
            2 * ((c_t - (self.alpha_min - a_t)) / self.min_slope - 0.5))
        f = math.atanh(2 * (
                ((f_t + (a_t + c_t)) - self.alpha_min) / self.min_slope - 0.5))
        d = math.atanh(2 * (((d_t + (
                a_t + b_t + c_t + e_t + f_t)) - self.alpha_min) /
                            self.min_slope - 0.5))

        self.a = torch.tensor(a, device=self.device).requires_grad_(self.requires_grad)
        self.b = torch.tensor(b, device=self.device).requires_grad_(self.requires_grad)
        self.c = torch.tensor(c, device=self.device).requires_grad_(self.requires_grad)
        self.d = torch.tensor(d, device=self.device).requires_grad_(self.requires_grad)
        self.e = torch.tensor(e, device=self.device).requires_grad_(self.requires_grad)
        self.f = torch.tensor(f, device=self.device).requires_grad_(self.requires_grad)

        # Create coordinate grid for evaluation
        img_range_x = torch.arange(self.size[1], device=self.device)
        img_range_y = torch.arange(self.size[0], device=self.device)
        self.Y, self.X = torch.meshgrid(img_range_y, img_range_x,
                                        indexing='ij')
        self.pixels = self.forward()
        self.init_params = [p.clone().detach() for p in
                            self.get_scaled_parameters()]

    def get_parameters(self):
        return [self.a, self.b, self.c, self.d, self.e, self.f]

    def get_scaled_parameters(self):
        self.calc_tanh()
        return [self.tanh_a, self.tanh_b, self.tanh_c, self.tanh_d,
                self.tanh_e, self.tanh_f]

    def calc_tanh(self):

        self.tanh_a = self.min_slope * self.tanh(self.a) + self.alpha_min
        self.tanh_b = (
                              self.alpha_min - self.tanh_a) + self.min_slope\
                      * self.tanh(
            self.b)
        self.tanh_e = (self.alpha_min - (
                self.tanh_a + self.tanh_b)) + self.min_slope * self.tanh(
            self.e)
        self.tanh_c = (
                              self.alpha_min - self.tanh_a) + self.min_slope * self.tanh(
            self.c)
        self.tanh_f = (self.alpha_min - (
                self.tanh_a + self.tanh_c)) + self.min_slope * self.tanh(
            self.f)
        total_base = self.tanh_a + self.tanh_b + self.tanh_c + self.tanh_e + self.tanh_f
        self.tanh_d = (
                              self.alpha_min - total_base) + self.min_slope * self.tanh(
            self.d)

    def forward(self):
        self.calc_tanh()
        # Normalize coordinates to [0, 1]
        X_norm = self.X / (self.size[1] - 1)
        Y_norm = self.Y / (self.size[0] - 1)
        self.pixels = (
                self.tanh_a + self.tanh_b * X_norm + self.tanh_c * Y_norm + self.tanh_d * X_norm * Y_norm + self.tanh_e * (
                X_norm**2) + self.tanh_f * (Y_norm**2))
        return self.pixels

    def __str__(self):
        a, b, c, d, e, f = self.get_scaled_parameters()
        return f'α(X,Y) = {a:.02} + {b:.02}X + {c:.02}Y + {d:.02}XY + {e:.02}X² + {f:.02}Y²'


@register_alpha_map('param_spatial_full')
class ParamSpatialFullMap(AlphaMap):
    """
    Full spatial resolution alpha map with HxW parameters.
    Initialized using the same logic as ParamSecondOrderMap but optimizes every pixel independently.
    """

    def _initialize_parameters(self):
        if self.init_alpha is None:
            # Random initialization using polynomial
            self._initialize_from_polynomial()
        elif isinstance(self.init_alpha, list):
            if len(self.init_alpha) == 6:
                # List of 6 polynomial coefficients - initialize from polynomial
                self._initialize_from_polynomial()
            else:
                # Assume it's a spatial map (nested list HxW)
                self._initialize_from_spatial_map()
        else:
            raise ValueError(f"Unsupported init_alpha format: {type(self.init_alpha)}")

        # Store initial state and compute initial pixels
        self.pixels = self.forward()
        self.init_params = [self.alpha_params.clone().detach()]

    def _initialize_from_polynomial(self):
        """Initialize from 6 polynomial coefficients using ParamSecondOrderMap."""
        # Store original init_alpha for saving
        self._original_init_alpha = self.init_alpha

        # Create temporary 2nd-order map with same initialization parameters
        temp_poly_map = ParamSecondOrderMap(
            map_type='param_2d_order',
            size=self.size[0],  # assuming square
            device=self.device,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            init_alpha=self.init_alpha,
            requires_grad=False  # we don't need gradients for initialization
        )

        # Get the evaluated polynomial as our starting alpha map
        initial_alpha_map = temp_poly_map.forward()  # Shape: (H, W)

        # Convert to unconstrained parameters
        self._convert_alpha_map_to_params(initial_alpha_map)

    def _initialize_from_spatial_map(self):
        """Initialize from loaded spatial alpha map (constrained values)."""
        # Convert nested list to tensor
        if isinstance(self.init_alpha[0], list):
            # Nested list [[row1], [row2], ...]
            alpha_map = torch.tensor(self.init_alpha, device=self.device, dtype=torch.float32)
        else:
            # Flattened list - reshape to spatial dimensions
            alpha_map = torch.tensor(self.init_alpha, device=self.device, dtype=torch.float32)
            alpha_map = alpha_map.reshape(self.size)

        # Convert to unconstrained parameters
        self._convert_alpha_map_to_params(alpha_map)

    def _convert_alpha_map_to_params(self, alpha_map):
        """Convert constrained alpha map to unconstrained parameters."""
        # Reverse the tanh transformation: alpha -> unconstrained
        # alpha = alpha_min + min_slope * (0.5 * tanh(x) + 0.5)
        # Solving for x: x = atanh(2 * ((alpha - alpha_min) / min_slope) - 1)
        normalized_alpha = (alpha_map - self.alpha_min) / self.min_slope  # [0, 1]
        tanh_input = 2 * normalized_alpha - 1  # [-1, 1]
        # Clamp to avoid numerical issues with atanh
        tanh_input = torch.clamp(tanh_input, -0.999, 0.999)
        unconstrained_init = torch.atanh(tanh_input)

        # Create the learnable HxW parameter tensor
        self.alpha_params = torch.tensor(unconstrained_init, device=self.device).requires_grad_(self.requires_grad)

    def get_parameters(self):
        """Return the full spatial parameter tensor."""
        return [self.alpha_params]

    def get_scaled_parameters(self):
        """Return the constrained alpha values."""
        self.calc_tanh()
        return [self.tanh_alpha]

    def calc_tanh(self):
        """Convert unconstrained parameters to valid alpha values [alpha_min, alpha_max]."""
        # Follow the same pattern as other alpha maps
        self.tanh_alpha = self.min_slope * self.tanh(self.alpha_params) + self.alpha_min

    def forward(self):
        """Generate the alpha map from current parameters."""
        self.calc_tanh()
        self.pixels = self.tanh_alpha
        return self.pixels

    def get_scale_functions(self):
        """Return functions to convert from constrained to unconstrained space."""

        # For compatibility - though less meaningful for spatial maps
        def inverse_transform(alpha_tensor):
            normalized = (alpha_tensor - self.alpha_min) / self.min_slope
            tanh_input = torch.clamp(2 * normalized - 1, -0.999, 0.999)
            return torch.atanh(tanh_input)

        return [inverse_transform]

    def get_save_parameters(self):
        """Get parameters in format compatible with JSON saving."""
        # Return the constrained alpha map as nested list for JSON
        alpha_map = self.forward()  # Get constrained values
        return alpha_map.detach().cpu().tolist()  # Convert to nested list

    def get_save_raw_parameters(self):
        """Get raw (unconstrained) parameters for JSON saving."""
        return self.alpha_params.detach().cpu().tolist()

    def get_save_init_parameters(self):
        """Get initial parameters for JSON saving."""
        if hasattr(self, '_original_init_alpha') and self._original_init_alpha is not None:
            return self._original_init_alpha
        else:
            # If initialized from spatial map, return the initial map
            return self.init_params[0].detach().cpu().tolist()

    def get_smoothness_loss(self, weight=100000.0):
        """
        Compute Laplacian smoothness regularization loss on the alpha map.
        This penalizes second derivatives and encourages polynomial-like behavior.

        Args:
            weight: Regularization weight (default: 1e-1)

        Returns:
            Smoothness loss tensor
        """
        alpha_map = self.forward()  # Get constrained alpha map

        # Compute discrete Laplacian: ∇²α = α[i+1,j] + α[i-1,j] + α[i,j+1] + α[i,j-1] - 4α[i,j]
        padded = torch.nn.functional.pad(alpha_map, (1, 1, 1, 1), mode='constant', value=0)

        laplacian = (padded[2:, 1:-1] +  # α[i+1,j]
                     padded[:-2, 1:-1] +  # α[i-1,j]
                     padded[1:-1, 2:] +  # α[i,j+1]
                     padded[1:-1, :-2] -  # α[i,j-1]
                     4 * alpha_map)  # -4α[i,j]

        return weight * laplacian.pow(2).mean()

    def __str__(self):
        """String representation showing the spatial dimensions."""
        num_params = self.alpha_params.numel()
        memory_mb = num_params * 4 / (1024 * 1024)  # 4 bytes per float32
        return f'α(X,Y) = Spatial[{self.size[0]}×{self.size[1]}]'