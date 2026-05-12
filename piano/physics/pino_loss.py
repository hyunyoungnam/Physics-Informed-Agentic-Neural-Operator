"""
Physics-Informed Neural Operator (PINO) loss for 2D plane-stress linear elasticity.

Two complementary physics terms:

1. Equilibrium residual (label-free):
   For each mesh node, sum the element-wise nodal force contributions:
     R_i = Σ_e (B_e^T C B_e u_e A_e) at node i
   The true displacement satisfies R=0 at interior nodes (no body forces).
   Loss: ||R||² / N_nodes

2. Energy-norm error (with labels):
   Strain energy of the prediction error field u_err = u_pred - u_true:
     L_energy = Σ_e (eps_err_e^T C eps_err_e A_e) / Σ_e A_e
   This is the physics-weighted H1 seminorm of the prediction error —
   a displacement error field that satisfies equilibrium has zero strain energy.

Both terms use fully vectorized torch operations; only Delaunay triangulation
(called once per sample) uses numpy.

Reference:
    Li et al. (2021) "Physics-Informed Neural Operator for Learning Partial
    Differential Equations", ICLR 2024.
"""

from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import Delaunay


def _compute_B_matrices(
    xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized strain-displacement B matrices for linear triangle elements.

    For each triangle e with nodes at positions x1,y1 / x2,y2 / x3,y3, the
    constant strain-displacement matrix is the 3×6 matrix:

        B = (1 / 2A) * [[y23,  0,  y31,  0,  y12,  0 ],
                         [ 0,  x32,  0,  x13,  0,  x21],
                         [x32, y23, x13,  y31, x21,  y12]]

    where y_ij = y_i - y_j, x_ij = x_i - x_j, and A is the signed area.

    Args:
        xy: (M, 3, 2) node coordinates for M triangles

    Returns:
        B:     (M, 3, 6) B matrices
        areas: (M,)     positive element areas
    """
    x1, y1 = xy[:, 0, 0], xy[:, 0, 1]
    x2, y2 = xy[:, 1, 0], xy[:, 1, 1]
    x3, y3 = xy[:, 2, 0], xy[:, 2, 1]

    # Signed area (positive for CCW orientation)
    two_A = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)  # (M,)
    areas = torch.abs(two_A) * 0.5  # (M,)

    # Avoid degenerate elements
    inv_two_A = torch.where(areas > 1e-30, 1.0 / (two_A + 1e-30), torch.zeros_like(two_A))

    y23 = (y2 - y3) * inv_two_A
    y31 = (y3 - y1) * inv_two_A
    y12 = (y1 - y2) * inv_two_A
    x32 = (x3 - x2) * inv_two_A
    x13 = (x1 - x3) * inv_two_A
    x21 = (x2 - x1) * inv_two_A

    zeros = torch.zeros_like(y23)

    # Stack rows of B: (M, 3, 6)
    row0 = torch.stack([y23, zeros, y31, zeros, y12, zeros], dim=-1)
    row1 = torch.stack([zeros, x32, zeros, x13, zeros, x21], dim=-1)
    row2 = torch.stack([x32, y23, x13, y31, x21, y12], dim=-1)
    B = torch.stack([row0, row1, row2], dim=1)  # (M, 3, 6)

    return B, areas


def _build_C(nu: float) -> torch.Tensor:
    """Plane-stress constitutive matrix (3×3, Voigt) with E=1.0 (dimensionless)."""
    factor = 1.0 / (1.0 - nu ** 2)
    return torch.tensor(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ],
        dtype=torch.float32,
    ) * factor


class PINOElasticityLoss(nn.Module):
    """
    PINO loss for 2D plane-stress linear elasticity.

    E is fixed at 1.0 (dimensionless) because the trainer normalizes displacements —
    the physical Young's modulus cancels in the equilibrium residual.
    nu (Poisson's ratio) does affect the constitutive tensor's anisotropy and can be
    passed per-sample via forward() to reflect the actual material properties.

    Attributes:
        nominal_nu:    Default Poisson's ratio (used when forward() receives no nu)
        eq_weight:     Weight for equilibrium residual term
        energy_weight: Weight for energy-norm error term
    """

    def __init__(
        self,
        nominal_nu: float = 0.3,
        eq_weight: float = 0.1,
        energy_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.nominal_nu = nominal_nu
        self.eq_weight = eq_weight
        self.energy_weight = energy_weight
        self.register_buffer("C", _build_C(nominal_nu))
        self._coords_hash: int = -1
        self._elems: Optional[torch.Tensor] = None
        self._B: Optional[torch.Tensor] = None
        self._areas: Optional[torch.Tensor] = None

    def _ensure_mesh_cache(self, coords: torch.Tensor, device: torch.device) -> None:
        """Triangulate and cache B-matrices; re-runs only when coordinates change."""
        coords_np = coords.detach().cpu().numpy()
        h = hash(coords_np.tobytes())
        if h == self._coords_hash:
            self._elems = self._elems.to(device)
            self._B = self._B.to(device)
            self._areas = self._areas.to(device)
            return
        tri = Delaunay(coords_np)
        elems_np = tri.simplices
        xy = torch.tensor(coords_np[elems_np], dtype=torch.float32, device=device)
        B, areas = _compute_B_matrices(xy)
        self._elems = torch.tensor(elems_np, dtype=torch.long, device=device)
        self._B = B
        self._areas = areas
        self._coords_hash = h

    def forward(
        self,
        u_pred: torch.Tensor,
        u_true: torch.Tensor,
        coords: torch.Tensor,
        nu: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute physics loss for one training sample.

        Args:
            u_pred:  (N, D) predicted displacement field (D >= 2)
            u_true:  (N, D) ground-truth displacement field
            coords:  (N, 2) mesh node coordinates
            nu:      Per-sample Poisson's ratio; uses nominal_nu when None

        Returns:
            Scalar physics loss (differentiable w.r.t. u_pred)
        """
        device = u_pred.device
        C = _build_C(nu).to(device) if nu is not None and nu != self.nominal_nu else self.C
        N = coords.shape[0]

        self._ensure_mesh_cache(coords, device)
        elems = self._elems
        B, areas = self._B, self._areas
        M = elems.shape[0]

        # Only use displacement components (first 2 columns)
        u_pred_2 = u_pred[:, :2]  # (N, 2)
        u_true_2 = u_true[:, :2]  # (N, 2)

        # --- Term 1: Equilibrium residual  -----------------------------------
        # u_pred at element nodes, flattened to DOF vector: (M, 6)
        u_elem = u_pred_2[elems].reshape(M, 6)

        # Voigt strain per element: (M, 3)
        eps = torch.einsum("mij,mj->mi", B, u_elem)

        # Voigt stress per element: (M, 3)
        sig = torch.einsum("ij,mj->mi", C, eps)

        # Nodal force contributions from each element: B^T sigma * area
        # f_elem: (M, 6)  — interlaced [fx1, fy1, fx2, fy2, fx3, fy3]
        f_elem = torch.einsum("mji,mj->mi", B, sig) * areas.unsqueeze(-1)

        # Scatter to global nodal residual R: (N, 2)
        R = torch.zeros(N, 2, dtype=u_pred.dtype, device=device)
        # Reshape f_elem from (M, 6) to (M, 3, 2) then scatter per local node
        f_per_node = f_elem.reshape(M, 3, 2)        # (M, 3, 2)
        node_indices = elems.reshape(-1)             # (M*3,)
        f_flat = f_per_node.reshape(-1, 2)           # (M*3, 2)
        R.scatter_add_(0, node_indices.unsqueeze(1).expand_as(f_flat), f_flat)

        L_eq = (R ** 2).mean()

        # --- Term 2: Energy-norm error  --------------------------------------
        u_err = (u_pred_2 - u_true_2)[elems].reshape(M, 6)
        eps_err = torch.einsum("mij,mj->mi", B, u_err)
        C_eps_err = torch.einsum("ij,mj->mi", C, eps_err)
        energy_per_elem = areas * torch.sum(eps_err * C_eps_err, dim=-1)

        total_area = areas.sum().clamp(min=1e-30)
        L_energy = energy_per_elem.sum() / total_area

        return self.eq_weight * L_eq + self.energy_weight * L_energy
