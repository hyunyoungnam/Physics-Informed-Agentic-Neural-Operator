"""
Transolver Neural Operator implementation.

Transolver uses Physics-Attention to handle unstructured meshes efficiently.
The key idea is to soft-assign N mesh points to S slices (S << N),
perform attention among slices, then redistribute back to points.

Reference: "Transolver: A Fast Transformer Solver for PDEs on General Geometries"
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

from .base import SurrogateModel, TransolverConfig, PredictionResult
from ..data.zero_copy import numpy_to_tensor


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation by global physics parameters.

    Applied once per transformer layer after attention, using the raw (B, n_params)
    parameter vector so each layer can independently modulate hidden features
    based on the global physics state (K_I, traction, etc.).
    This allows the slice-assignment softmax to differentiate physical regions
    (near-tip vs far-field) based on the global loading condition.
    """

    def __init__(self, n_params: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)   params: (B, n_params)
        out = self.net(params)               # (B, 2D)
        gamma, beta = out.chunk(2, dim=-1)   # each (B, D)
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


class PhysicsAttention(nn.Module):
    """
    Physics-Attention (Slice-Attention) mechanism from Transolver.

    Reduces computational complexity from O(N^2) to O(S^2 + NS) where S << N.
    This enables efficient attention on large unstructured meshes.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        slice_num: int,
        dropout: float = 0.0
    ):
        """
        Initialize PhysicsAttention.

        Args:
            d_model: Hidden dimension
            n_heads: Number of attention heads
            slice_num: Number of physics slices (S)
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.slice_num = slice_num
        self.head_dim = d_model // n_heads

        # Projection for computing slice assignments
        self.slice_proj = nn.Linear(d_model, slice_num)

        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Physics-Attention.

        Args:
            x: Input features (B, N, d_model)
            coords: Coordinates (B, N, coord_dim) - used for positional context

        Returns:
            Output features (B, N, d_model)
        """
        B, N, D = x.shape

        # Step 1: Compute slice assignments (soft assignment of N points to S slices)
        # slice_weights: (B, N, S)
        slice_logits = self.slice_proj(x) / (self.temperature + 1e-6)
        slice_weights = F.softmax(slice_logits, dim=-1)

        # Step 2: Aggregate points into slices
        # slices: (B, S, D)
        slices = einsum(slice_weights, x, 'b n s, b n d -> b s d')

        # Step 3: Multi-head attention among slices (S << N, so O(S^2) is fast)
        Q = self.q_proj(slices)  # (B, S, D)
        K = self.k_proj(slices)
        V = self.v_proj(slices)

        # Reshape for multi-head attention
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.n_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.n_heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.n_heads)

        # Attention scores and output
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        slice_out = torch.matmul(attn, V)  # (B, H, S, head_dim)
        slice_out = rearrange(slice_out, 'b h s d -> b s (h d)')  # (B, S, D)
        slice_out = self.out_proj(slice_out)

        # Step 4: Deslicing - redistribute slice features back to points
        # out: (B, N, D)
        out = einsum(slice_weights, slice_out, 'b n s, b s d -> b n d')

        return out


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name.lower()]


class TransolverBlock(nn.Module):
    """Single Transolver block: PhysicsAttention + FFN with residual connections."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        slice_num: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu"
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PhysicsAttention(d_model, n_heads, slice_num, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.mlp(self.norm2(x))
        return x


class TransolverModel(SurrogateModel, nn.Module):
    """
    Transolver model for surrogate modeling on unstructured meshes.

    Takes input parameters and mesh coordinates, outputs field predictions
    (e.g., displacement, stress, temperature).
    """

    def __init__(self, config: TransolverConfig):
        """
        Initialize Transolver model.

        Args:
            config: Model configuration
        """
        nn.Module.__init__(self)
        SurrogateModel.__init__(self, config)

        self._input_dim: Optional[int] = None
        self._coord_dim: Optional[int] = None
        self._num_points: Optional[int] = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build(self, input_dim: int, coord_dim: int, num_points: int) -> None:
        """
        Build the model architecture.

        Args:
            input_dim: Dimension of input parameters
            coord_dim: Dimension of coordinates (typically 2 or 3)
            num_points: Number of mesh points
        """
        self._input_dim = input_dim
        self._coord_dim = coord_dim
        self._num_points = num_points

        cfg = self.config

        # Input projection: concatenate coordinates and expanded parameters
        self.input_proj = nn.Linear(coord_dim + input_dim, cfg.d_model)

        # Transolver blocks
        self.layers = nn.ModuleList([
            TransolverBlock(
                cfg.d_model,
                cfg.n_heads,
                cfg.slice_num,
                cfg.mlp_ratio,
                cfg.dropout,
                cfg.activation
            )
            for _ in range(cfg.n_layers)
        ])

        # FiLM layers — one per transformer block; conditions hidden features on global params
        # n_params = input_dim (params only, not coords), applied using original (B, n_params) vector
        self.film_layers = nn.ModuleList([
            FiLMLayer(input_dim, cfg.d_model) for _ in range(cfg.n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(cfg.d_model, cfg.output_dim)

        self.to(self._device)

    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            params: Input parameters (B, n_params)
            coords: Mesh coordinates (B, N, coord_dim)

        Returns:
            Predicted field values (B, N, output_dim)
        """
        B, N, _ = coords.shape

        # Expand parameters to each point: (B, n_params) -> (B, N, n_params)
        params_expanded = params.unsqueeze(1).expand(-1, N, -1)

        # Concatenate coordinates and parameters
        x = torch.cat([coords, params_expanded], dim=-1)  # (B, N, coord_dim + n_params)

        # Project to hidden dimension
        x = self.input_proj(x)  # (B, N, d_model)

        # Apply Transolver blocks with per-layer FiLM conditioning on global params
        for i, layer in enumerate(self.layers):
            x = layer(x, coords)
            x = self.film_layers[i](x, params)  # params: (B, n_params) — before expansion

        # Project to output
        out = self.output_proj(x)  # (B, N, output_dim)

        return out

    def predict(
        self,
        params: np.ndarray,
        coords: np.ndarray
    ) -> PredictionResult:
        """
        Make predictions with the trained model.

        Args:
            params: Input parameters (N_samples, n_params) or (n_params,)
            coords: Query coordinates (num_points, coord_dim)

        Returns:
            PredictionResult with predictions
        """
        self.eval()

        # Handle single sample
        if params.ndim == 1:
            params = params[np.newaxis, :]

        # Convert to tensors
        params_t = numpy_to_tensor(params, self._device)
        coords_t = numpy_to_tensor(coords, self._device)

        # Expand coords for batch: (N, coord_dim) -> (B, N, coord_dim)
        if coords_t.ndim == 2:
            coords_t = coords_t.unsqueeze(0).expand(params_t.shape[0], -1, -1)

        with torch.no_grad():
            predictions = self.forward(params_t, coords_t)

        values = predictions.cpu().numpy()

        # Squeeze if single sample
        if values.shape[0] == 1:
            values = values[0]

        return PredictionResult(
            values=values,
            uncertainty=None,
            coordinates=coords,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'config': self.config.to_dict(),
            'state_dict': self.state_dict(),
            'input_dim': self._input_dim,
            'coord_dim': self._coord_dim,
            'num_points': self._num_points,
            'is_trained': self._is_trained,
        }
        torch.save(checkpoint, path)

    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self._device)

        # Rebuild model with saved dimensions
        self.build(
            checkpoint['input_dim'],
            checkpoint['coord_dim'],
            checkpoint['num_points']
        )

        self.load_state_dict(checkpoint['state_dict'])
        self._is_trained = checkpoint['is_trained']
