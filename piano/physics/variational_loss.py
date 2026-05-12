"""
Variational AT-2 elastic energy loss for physics-informed training.

Based on the V-DeepONet formulation (Goswami et al. 2022):
  - Minimizes degraded elastic strain energy ∫ g(d) Ψ_e(u) dΩ subject to W_ext
  - g(d) = (1-d)^2 is the AT-2 degradation function
  - No displacement labels required; uses only predicted u + known damage d

Reference: Goswami et al. "A physics-informed variational DeepONet for weakly-
           supervised operator learning", CMAME 2022.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import Delaunay


class VariationalElasticLoss(nn.Module):
    """
    Degraded elastic strain energy loss: L = |E_elastic - W_ext| / E_ref

    E_elastic = Σ_e area_e · g(d̄_e) · Ψ_e(u_pred)
    W_ext     = traction · mean(u_y at top nodes) · domain_width
    g(d)      = (1 - d)^2   (AT-2 degradation)
    Ψ_e       = ½ ε_e : C : ε_e  (plane-stress strain energy density)

    Does not require displacement labels. If damage=None, assumes d=0 (undamaged).
    Mesh topology is cached on first call and rebuilt only when coords hash changes.
    """

    def __init__(self, E: float, nu: float) -> None:
        """
        Args:
            E:  Young's modulus [Pa]
            nu: Poisson's ratio
        """
        super().__init__()
        self.E = float(E)
        self.nu = float(nu)

        self._coords_hash: int = -1
        self._B: Optional[torch.Tensor] = None        # (M, 3, 6) B-matrices
        self._areas: Optional[torch.Tensor] = None    # (M,) element areas
        self._elems: Optional[torch.Tensor] = None    # (M, 3) connectivity
        self._top_mask: Optional[torch.Tensor] = None  # (N,) bool, top-boundary nodes
        self._domain_width: float = 1.0

    # ------------------------------------------------------------------
    def _build_C(self, device: torch.device) -> torch.Tensor:
        """Plane-stress constitutive matrix (3×3 Voigt notation)."""
        E, nu = self.E, self.nu
        factor = E / (1.0 - nu ** 2)
        C = factor * torch.tensor([
            [1.0,  nu,         0.0],
            [nu,   1.0,        0.0],
            [0.0,  0.0, (1.0 - nu) / 2.0],
        ], dtype=torch.float32, device=device)
        return C

    def _build_fem_cache(
        self,
        coords_np: np.ndarray,
        elements_np: Optional[np.ndarray],
        device: torch.device,
    ) -> None:
        """Build B-matrices, areas, top-node mask from mesh topology."""
        if elements_np is None:
            tri = Delaunay(coords_np[:, :2])
            elements_np = tri.simplices

        elems = elements_np.astype(np.int64)
        M = len(elems)
        x = coords_np[:, 0]
        y = coords_np[:, 1]

        # Build B-matrices (3, 6) per triangle — standard CST strain-displacement
        B_np = np.zeros((M, 3, 6), dtype=np.float32)
        areas_np = np.zeros(M, dtype=np.float32)

        for e, (i, j, k) in enumerate(elems):
            xi, yi = x[i], y[i]
            xj, yj = x[j], y[j]
            xk, yk = x[k], y[k]

            # Shape function derivatives (constant per CST element)
            bi = yj - yk
            bj = yk - yi
            bk = yi - yj
            ci = xk - xj
            cj = xi - xk
            ck = xj - xi

            area = 0.5 * abs(bi * (xj - xk) + bj * (xk - xi) + bk * (xi - xj))
            # fallback signed area formula
            area2 = abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            area = 0.5 * area2
            areas_np[e] = area

            if area < 1e-20:
                continue

            inv2A = 1.0 / (2.0 * area)
            # B = (1/2A) [[bi 0 bj 0 bk 0],
            #             [0 ci 0 cj 0 ck],
            #             [ci bi cj bj ck bk]]
            B_np[e, 0, 0] = bi * inv2A
            B_np[e, 0, 2] = bj * inv2A
            B_np[e, 0, 4] = bk * inv2A
            B_np[e, 1, 1] = ci * inv2A
            B_np[e, 1, 3] = cj * inv2A
            B_np[e, 1, 5] = ck * inv2A
            B_np[e, 2, 0] = ci * inv2A
            B_np[e, 2, 1] = bi * inv2A
            B_np[e, 2, 2] = cj * inv2A
            B_np[e, 2, 3] = bj * inv2A
            B_np[e, 2, 4] = ck * inv2A
            B_np[e, 2, 5] = bk * inv2A

        # Top-boundary nodes: y ≥ 99th percentile of y-coords
        y_thresh = np.percentile(y, 99)
        top_mask_np = (y >= y_thresh)
        domain_width = float(x.max() - x.min())

        self._B = torch.tensor(B_np, dtype=torch.float32, device=device)
        self._areas = torch.tensor(areas_np, dtype=torch.float32, device=device)
        self._elems = torch.tensor(elems, dtype=torch.long, device=device)
        self._top_mask = torch.tensor(top_mask_np, dtype=torch.bool, device=device)
        self._domain_width = max(domain_width, 1e-8)

    # ------------------------------------------------------------------
    def forward(
        self,
        u_pred: torch.Tensor,
        coords: torch.Tensor,
        elements: Optional[torch.Tensor] = None,
        traction: float = 0.0,
        damage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            u_pred:   (N, 2) denormalized displacement [m]
            coords:   (N, 2) mesh coordinates [m]
            elements: (M, 3) triangle connectivity (long), or None → Delaunay
            traction: applied traction [Pa] for external work term
            damage:   (N,) nodal damage field ∈ [0, 1]; None assumes d=0

        Returns:
            Scalar loss (dimensionless)
        """
        device = u_pred.device
        coords_np = coords.detach().cpu().numpy()
        elements_np = elements.cpu().numpy() if elements is not None else None

        # Rebuild FEM cache when mesh changes
        coords_hash = hash(coords_np.tobytes())
        if coords_hash != self._coords_hash:
            self._build_fem_cache(coords_np, elements_np, device)
            self._coords_hash = coords_hash
            # Move cached tensors to current device
            self._B = self._B.to(device)
            self._areas = self._areas.to(device)
            self._elems = self._elems.to(device)
            self._top_mask = self._top_mask.to(device)

        M = self._elems.shape[0]

        # Degradation g(d̄_e) = (1 - mean_nodal_damage)^2 per element
        if damage is not None:
            d_nodes = damage.clamp(0.0, 1.0)  # (N,)
            d_elem = d_nodes[self._elems].mean(dim=1)  # (M,)
            g_e = (1.0 - d_elem) ** 2          # (M,)
        else:
            g_e = torch.ones(M, dtype=torch.float32, device=device)

        # Assemble element displacement vectors u_e: (M, 6) → [ux_i, uy_i, ux_j, uy_j, ux_k, uy_k]
        i_idx = self._elems[:, 0]
        j_idx = self._elems[:, 1]
        k_idx = self._elems[:, 2]
        u_e = torch.stack([
            u_pred[i_idx, 0], u_pred[i_idx, 1],
            u_pred[j_idx, 0], u_pred[j_idx, 1],
            u_pred[k_idx, 0], u_pred[k_idx, 1],
        ], dim=1)  # (M, 6)

        # Strain: ε_e = B_e @ u_e  →  (M, 3)
        eps = torch.einsum('mij,mj->mi', self._B, u_e)  # (M, 3)

        # Stress: σ_e = C @ ε_e  →  (M, 3)
        C = self._build_C(device)
        sig = torch.einsum('ij,mj->mi', C, eps)         # (M, 3)

        # Strain energy density: Ψ_e = ½ ε:σ  →  (M,)
        psi = 0.5 * (eps * sig).sum(dim=1)              # (M,)

        # Degraded elastic energy
        elastic = (self._areas * g_e * psi).sum()

        # External work: traction × mean top-node uy × domain_width
        if traction != 0.0 and self._top_mask.any():
            u_y_top = u_pred[self._top_mask, 1].mean()
            w_ext = traction * u_y_top * self._domain_width
        else:
            w_ext = torch.tensor(0.0, device=device)

        # Reference energy for dimensionless normalization
        # E_ref ~ ½ σ²/E × domain_area (characteristic elastic energy)
        domain_area = float(self._areas.sum().item())
        e_ref = max(0.5 * (traction ** 2) / (self.E + 1e-12) * domain_area, 1e-12)

        return (elastic - w_ext).abs() / e_ref
