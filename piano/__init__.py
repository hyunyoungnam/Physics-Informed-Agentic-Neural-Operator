"""
PIANO: Physics-Informed Agentic Neural Operator

A self-improving surrogate framework for computational mechanics that combines:
- Transolver / DeepONet neural operators for learning FEM field predictions
- Physics-informed losses (PINO, CrackFractureLoss, PeridynamicEquilibriumLoss, Variational)
- 6-agent HPO debate system for autonomous hyperparameter optimization
- Active learning loop with FEniCS phase-field fracture simulations

Example (Agentic Training):
    >>> from piano.surrogate.agentic_trainer import (
    ...     AgenticSurrogateTrainer, AgenticTrainingConfig
    ... )
    >>> config = AgenticTrainingConfig(
    ...     max_hpo_rounds=3,
    ...     use_physicist=True,
    ...     problem_type="crack",
    ... )
    >>> trainer = AgenticSurrogateTrainer(config, llm_provider)
    >>> result = trainer.train(params, coords, outputs)
"""

__version__ = "0.3.0"
__author__ = "H.-Y. Nam, Q. Jiang"

# ── Mesh management ───────────────────────────────────────────────────────────
from piano.mesh.base import MeshManager
from piano.mesh.mfem_manager import MFEMManager

# ── Solver API ────────────────────────────────────────────────────────────────
from piano.solvers.base import (
    SolverInterface,
    PhysicsType,
    PhysicsConfig,
    MaterialProperties,
    SolverResult,
)
from piano.solvers.mfem_solver import MFEMSolver

# ── Evaluation ────────────────────────────────────────────────────────────────
from piano.evaluation.pipeline import EvaluationPipeline, EvaluationResult

# ── Active learning orchestration ─────────────────────────────────────────────
from piano.orchestration.adaptive import (
    AdaptiveOrchestrator,
    AdaptiveConfig,
    AdaptiveResult,
)

# ── Surrogate models & training ───────────────────────────────────────────────
from piano.surrogate.base import TransolverConfig
from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
from piano.surrogate.agentic_trainer import (
    AgenticSurrogateTrainer,
    AgenticTrainingConfig,
    AgenticTrainingResult,
)

# ── Physics-informed losses ───────────────────────────────────────────────────
from piano.physics import (
    PINOElasticityLoss,
    CrackFractureLoss,
    PeridynamicEquilibriumLoss,
    VariationalElasticLoss,
)

# ── Dataset ───────────────────────────────────────────────────────────────────
from piano.data.dataset import FEMDataset, FEMSample, DatasetConfig

# ── Geometry ──────────────────────────────────────────────────────────────────
from piano.geometry import (
    CrackGeometry,
    EdgeCrack,
    CenterCrack,
    CrackMeshGenerator,
    generate_crack_mesh,
)

__all__ = [
    "__version__",
    "__author__",
    # Mesh
    "MeshManager",
    "MFEMManager",
    # Solvers
    "SolverInterface",
    "MFEMSolver",
    "PhysicsType",
    "PhysicsConfig",
    "MaterialProperties",
    "SolverResult",
    # Evaluation
    "EvaluationPipeline",
    "EvaluationResult",
    # Orchestration
    "AdaptiveOrchestrator",
    "AdaptiveConfig",
    "AdaptiveResult",
    # Surrogate
    "TransolverConfig",
    "SurrogateTrainer",
    "TrainingConfig",
    "AgenticSurrogateTrainer",
    "AgenticTrainingConfig",
    "AgenticTrainingResult",
    # Physics losses
    "PINOElasticityLoss",
    "CrackFractureLoss",
    "PeridynamicEquilibriumLoss",
    "VariationalElasticLoss",
    # Dataset
    "FEMDataset",
    "FEMSample",
    "DatasetConfig",
    # Geometry
    "CrackGeometry",
    "EdgeCrack",
    "CenterCrack",
    "CrackMeshGenerator",
    "generate_crack_mesh",
]
