"""
Base classes for surrogate models.

Defines the interface for surrogate models that predict FEM outputs
from input parameters without running full simulations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class CrackConfig:
    """
    Crack geometry and parameter-index configuration for CrackFractureLoss.

    Tells the trainer which parameter indices hold K_I, E, nu so it can
    pass physical values to the physics loss after output denormalization.

    Attributes:
        tip_x:           Crack tip x-coordinate
        tip_y:           Crack tip y-coordinate
        e_param_idx:     Index of Young's modulus in the raw parameter vector
        nu_param_idx:    Index of Poisson's ratio in the raw parameter vector
        ki_param_idx:    Index of K_I in the raw parameter vector
        r_ki_min:        Inner radius of K_I extraction annulus
        r_ki_max:        Outer radius of K_I extraction annulus
        r_j:             Outer radius of J-integral domain
        crack_face_tol:  Half-width tolerance for crack-face element detection
        horizon_factor:  PD horizon δ = horizon_factor × h_avg (default 3.0)
    """
    tip_x: float
    tip_y: float
    e_param_idx: int = 0        # param order: E, nu, traction, K_I, G_c, crack_len, l_0
    nu_param_idx: int = 1
    traction_param_idx: int = 2  # index of applied traction in raw parameter vector
    ki_param_idx: int = 3
    r_ki_min: float = 0.02
    r_ki_max: float = 0.10
    r_j: float = 0.15
    crack_face_tol: float = 0.02
    horizon_factor: float = 3.0

import numpy as np


class SurrogateType(Enum):
    """Types of surrogate models."""
    TRANSOLVER = auto()
    ENSEMBLE = auto()


@dataclass
class TransolverConfig:
    """
    Configuration for Transolver model.

    Attributes:
        slice_num: Number of physics slices (key hyperparameter)
        n_heads: Number of attention heads
        d_model: Hidden dimension
        n_layers: Number of transformer layers
        mlp_ratio: FFN expansion ratio
        dropout: Dropout rate
        learning_rate: Initial learning rate
        batch_size: Training batch size
        epochs: Maximum training epochs
        patience: Early stopping patience
        output_dim: Dimension of output field (e.g., 3 for displacement)
        checkpoint_dir: Directory for saving checkpoints
        optimizer_type: Optimizer type ('adamw', 'adam', 'sgd')
        scheduler_type: LR scheduler type ('plateau', 'cosine', 'none')
        activation: Activation function ('gelu', 'relu', 'silu')
    """
    slice_num: int = 64
    n_heads: int = 8
    d_model: int = 128
    n_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 1000
    patience: int = 100
    output_dim: int = 3
    checkpoint_dir: Optional[Path] = None
    energy: float = 0.0          # elastic energy norm loss weight
    equilibrium: float = 0.0     # equilibrium PDE residual weight (∇·σ = 0)
    tip_weight: float = 0.0      # >0 upweights nodes near singularity tip by (1 + tip_weight/r)
    stress_intensity: float = 0.0  # K_I consistency loss weight
    traction_free: float = 0.0     # crack face traction-free BC loss weight
    near_tip: float = 0.0          # peridynamic equilibrium residual loss weight
    j_integral: float = 0.0        # J-integral conservation loss weight
    variational_weight: float = 0.0  # AT-2 variational elastic energy loss weight
    optimizer_type: str = "adamw"
    scheduler_type: str = "plateau"
    activation: str = "gelu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "slice_num": self.slice_num,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "output_dim": self.output_dim,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "energy": self.energy,
            "equilibrium": self.equilibrium,
            "tip_weight": self.tip_weight,
            "stress_intensity": self.stress_intensity,
            "traction_free": self.traction_free,
            "near_tip": self.near_tip,
            "j_integral": self.j_integral,
            "variational_weight": self.variational_weight,
            "optimizer_type": self.optimizer_type,
            "scheduler_type": self.scheduler_type,
            "activation": self.activation,
        }


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble model.

    Attributes:
        n_members: Number of ensemble members
        member_config: Configuration for each ensemble member (TransolverConfig or DeepONetConfig)
    """
    n_members: int = 5
    member_config: Any = field(default_factory=TransolverConfig)


# Alias for backward compatibility
SurrogateConfig = TransolverConfig


@dataclass
class PredictionResult:
    """
    Result of surrogate model prediction.

    Attributes:
        values: Predicted field values at query points
        uncertainty: Uncertainty estimate (if available)
        coordinates: Query point coordinates
        metadata: Additional prediction metadata
    """
    values: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    coordinates: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean(self) -> np.ndarray:
        """Get mean prediction."""
        return self.values

    @property
    def std(self) -> Optional[np.ndarray]:
        """Get standard deviation (uncertainty)."""
        return self.uncertainty

    def max_uncertainty_indices(self, top_k: int = 10) -> np.ndarray:
        """Get indices of points with highest uncertainty."""
        if self.uncertainty is None:
            return np.array([])
        return np.argsort(self.uncertainty.flatten())[-top_k:][::-1]


class SurrogateModel(ABC):
    """
    Abstract base class for surrogate models.

    A surrogate model learns to predict FEM simulation outputs
    (displacement, stress, temperature, etc.) from input parameters
    (geometry, material properties, boundary conditions).
    """

    def __init__(self, config: TransolverConfig):
        """
        Initialize surrogate model.

        Args:
            config: Model configuration
        """
        self.config = config
        self._is_trained = False

    @abstractmethod
    def build(
        self,
        input_dim: int,
        coord_dim: int,
        num_points: int
    ) -> None:
        """
        Build the model architecture.

        Args:
            input_dim: Dimension of input parameters
            coord_dim: Dimension of coordinates (typically 2 or 3)
            num_points: Number of mesh points
        """
        pass

    @abstractmethod
    def forward(
        self,
        params: "torch.Tensor",
        coords: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Forward pass through the model.

        Args:
            params: Input parameters (B, n_params)
            coords: Mesh coordinates (B, N, coord_dim)

        Returns:
            Predicted field values (B, N, output_dim)
        """
        pass

    @abstractmethod
    def predict(
        self,
        params: np.ndarray,
        coords: np.ndarray
    ) -> PredictionResult:
        """
        Make predictions with the trained model.

        Args:
            params: Input parameters (N, n_params) or (n_params,)
            coords: Query coordinates (num_points, coord_dim)

        Returns:
            PredictionResult with predictions and uncertainty
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load model from disk.

        Args:
            path: Path to load model from
        """
        pass

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained

    def compute_error(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute prediction errors.

        Args:
            predictions: Predicted values
            targets: Ground truth values

        Returns:
            Dictionary of error metrics
        """
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))

        # Relative errors (avoid division by zero)
        target_norm = np.linalg.norm(targets)
        if target_norm > 1e-10:
            relative_l2 = np.linalg.norm(predictions - targets) / target_norm
        else:
            relative_l2 = float('inf')

        # Max error
        max_error = np.max(np.abs(predictions - targets))

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "relative_l2": float(relative_l2),
            "max_error": float(max_error),
        }
