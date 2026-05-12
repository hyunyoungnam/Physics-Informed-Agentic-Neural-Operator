"""
Physics-informed loss functions for surrogate training.

Four complementary terms covering elasticity and fracture mechanics:
- PINOElasticityLoss:       equilibrium residual + energy-norm error (label-free/labelled)
- CrackFractureLoss:        K_I consistency, traction-free BC, peridynamic equilibrium, J-integral
- PeridynamicEquilibriumLoss: bond-based PD residual (valid across full mesh and damage zones)
- VariationalElasticLoss:   AT-2 degraded strain energy (label-free, V-DeepONet style)
"""

from .pino_loss import PINOElasticityLoss
from .crack_pino_loss import CrackFractureLoss
from .peridynamic_loss import PeridynamicEquilibriumLoss
from .variational_loss import VariationalElasticLoss

__all__ = [
    "PINOElasticityLoss",
    "CrackFractureLoss",
    "PeridynamicEquilibriumLoss",
    "VariationalElasticLoss",
]
