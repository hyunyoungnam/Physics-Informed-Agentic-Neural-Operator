"""
Surrogate model module for adaptive learning.

Provides surrogate model interfaces and training utilities for predicting
FEM simulation outputs from input parameters using Transolver / DeepONet.

Physics-informed loss functions live in piano.physics (not here).
"""

# ── Base types & configs ──────────────────────────────────────────────────────
from .base import (
    SurrogateModel,
    SurrogateConfig,
    TransolverConfig,
    EnsembleConfig,
    PredictionResult,
    SurrogateType,
)

# ── Model architectures ───────────────────────────────────────────────────────
from .transolver import TransolverModel, PhysicsAttention
from .ensemble import EnsembleModel
from .deeponet import DeepONetConfig, DeepONetModel

# ── Training ──────────────────────────────────────────────────────────────────
from .trainer import SurrogateTrainer, TrainingConfig, TrainingResult
from .agentic_trainer import (
    AgenticSurrogateTrainer,
    AgenticTrainingConfig,
    AgenticTrainingResult,
)

# ── Evaluation & active learning ──────────────────────────────────────────────
from .evaluator import SurrogateEvaluator, WeakRegion, UncertaintyAnalysis

__all__ = [
    # Base
    "SurrogateModel",
    "SurrogateConfig",
    "TransolverConfig",
    "EnsembleConfig",
    "PredictionResult",
    "SurrogateType",
    # Models
    "TransolverModel",
    "PhysicsAttention",
    "EnsembleModel",
    "DeepONetConfig",
    "DeepONetModel",
    # Training
    "SurrogateTrainer",
    "TrainingConfig",
    "TrainingResult",
    "AgenticSurrogateTrainer",
    "AgenticTrainingConfig",
    "AgenticTrainingResult",
    # Evaluation
    "SurrogateEvaluator",
    "WeakRegion",
    "UncertaintyAnalysis",
]
