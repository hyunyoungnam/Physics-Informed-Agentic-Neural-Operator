"""
Bond-based peridynamic equilibrium residual loss (2D, static).

For each node i, static equilibrium requires:

    L_i = Σ_{j ∈ H(i)} (1 − d_ij)² · s_ij · ê_ij = 0

where
    H(i)    = horizon ball of radius δ = horizon_factor × h_avg
    ξ_ij    = x_j − x_i                          reference bond vector
    η_ij    = u_j − u_i                          relative displacement
    s_ij    = (|ξ_ij + η_ij| − |ξ_ij|) / |ξ_ij| bond stretch (scalar)
    ê_ij    = (ξ_ij + η_ij) / |ξ_ij + η_ij|     deformed unit bond direction
    d_ij    = max(d_i, d_j)                       bond damage ∈ [0, 1]

Advantages over the Williams asymptotic residual:
- Valid everywhere on the mesh, not just in the K-dominant near-tip zone
- Naturally zero inside fully damaged bonds ((1−1)²=0) — no spurious gradient there
- No singularity assumption; handles large process zones from phase field damage
- Micromodulus c(δ) cancels in the dimensionless normalization

Loss (dimensionless, ≈ 1 for random field, 0 for equilibrated field):

    L_peri = mean_i(‖L_i‖²) / (s_var · n_avg)

where s_var = mean_bonds(deg · s²), n_avg = |bonds| / N.

Reference:
    Silling (2000) "Reformulation of elasticity theory for discontinuities
        and long-range forces." J. Mech. Phys. Solids 48(1):175-209.
    Bobaru & Hu (2012) "The meaning, selection, and use of the peridynamic
        horizon." Int. J. Fract. 162(1-2):229-234.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
from typing import Optional


class PeridynamicEquilibriumLoss(nn.Module):
    """
    Bond-based peridynamic equilibrium residual for 2D static fracture.

    Args:
        horizon_factor: δ = horizon_factor × h_avg  (default 3.0, standard PD choice)
    """

    def __init__(self, horizon_factor: float = 3.0):
        super().__init__()
        self.horizon_factor = horizon_factor

        self._coords_hash: int = -1
        self._src: Optional[torch.Tensor] = None
        self._dst: Optional[torch.Tensor] = None
        self._xi: Optional[torch.Tensor] = None       # (E, 2) reference bond vectors
        self._xi_norm: Optional[torch.Tensor] = None  # (E,)   |ξ|
        self._N: int = 0

    # ------------------------------------------------------------------
    # Bond list construction (cached per mesh topology)
    # ------------------------------------------------------------------

    def _build_bonds(self, coords_np: np.ndarray, device: torch.device) -> None:
        N = len(coords_np)
        tree = cKDTree(coords_np)

        # Estimate average nodal spacing from nearest-neighbour distance
        dists, _ = tree.query(coords_np, k=2)   # k=2: first hit is self (d=0)
        h_avg = float(dists[:, 1].mean())
        delta = self.horizon_factor * h_avg

        pairs = tree.query_pairs(delta, output_type="ndarray")   # (P, 2) int array
        if pairs.shape[0] == 0:
            delta *= 2.0
            pairs = tree.query_pairs(delta, output_type="ndarray")

        if pairs.shape[0] == 0:
            # Degenerate mesh — no bonds; loss will return 0
            self._src = torch.zeros(0, dtype=torch.long, device=device)
            self._dst = torch.zeros(0, dtype=torch.long, device=device)
            self._xi = torch.zeros(0, 2, dtype=torch.float32, device=device)
            self._xi_norm = torch.zeros(0, dtype=torch.float32, device=device)
            self._N = N
            self._coords_hash = hash(coords_np.tobytes())
            return

        # Convert undirected pairs → directed edges (both directions per pair)
        src = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
        dst = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int64)

        xi = coords_np[dst] - coords_np[src]         # (E, 2)
        xi_norm = np.linalg.norm(xi, axis=1)         # (E,)

        self._src = torch.tensor(src, dtype=torch.long, device=device)
        self._dst = torch.tensor(dst, dtype=torch.long, device=device)
        self._xi = torch.tensor(xi, dtype=torch.float32, device=device)
        self._xi_norm = torch.tensor(xi_norm, dtype=torch.float32, device=device)
        self._N = N
        self._coords_hash = hash(coords_np.tobytes())

    def _ensure_cache(self, coords: torch.Tensor, device: torch.device) -> None:
        coords_np = coords.detach().cpu().numpy()
        h = hash(coords_np.tobytes())
        if h != self._coords_hash:
            self._build_bonds(coords_np, device)
        else:
            self._src = self._src.to(device)
            self._dst = self._dst.to(device)
            self._xi = self._xi.to(device)
            self._xi_norm = self._xi_norm.to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        u_pred: torch.Tensor,                   # (N, 2) predicted displacement
        coords: torch.Tensor,                   # (N, 2) mesh node coordinates
        damage: Optional[torch.Tensor] = None,  # (N,) ∈ [0,1], optional
    ) -> torch.Tensor:
        """
        Compute the peridynamic equilibrium residual loss.

        Returns a dimensionless scalar ≈ 0 for an equilibrated field and ≈ 1
        for a field with bond-stretch imbalance at the scale of the rms stretch.
        """
        device = u_pred.device
        self._ensure_cache(coords, device)

        if len(self._src) == 0:
            return torch.tensor(0.0, device=device)

        # Relative displacement per bond: η = u_j − u_i
        eta = u_pred[self._dst] - u_pred[self._src]          # (E, 2)

        # Deformed bond vector: ξ + η
        xi_eta = self._xi + eta                               # (E, 2)
        xi_eta_norm = xi_eta.norm(dim=1).clamp(min=1e-12)    # (E,)

        # Bond stretch: s = (|ξ+η| − |ξ|) / |ξ|
        s = (xi_eta_norm - self._xi_norm) / self._xi_norm.clamp(min=1e-12)   # (E,)

        # Deformed unit bond direction: ê = (ξ+η) / |ξ+η|
        e_hat = xi_eta / xi_eta_norm.unsqueeze(1)             # (E, 2)

        # Bond damage degradation: (1 − d_max)²
        if damage is not None:
            d_max = torch.maximum(damage[self._src], damage[self._dst])
            deg = (1.0 - d_max.clamp(0.0, 1.0)) ** 2         # (E,)
        else:
            deg = torch.ones(len(self._src), dtype=torch.float32, device=device)

        # Bond force contribution: (1 − d)² · s · ê
        f_bond = (deg * s).unsqueeze(1) * e_hat               # (E, 2)

        # Nodal equilibrium residual: L_i = Σ_{j ∈ H(i)} f_ij
        L_i = torch.zeros(self._N, 2, dtype=torch.float32, device=device)
        L_i = L_i.scatter_add(0, self._src.unsqueeze(1).expand(-1, 2), f_bond)  # (N, 2)

        # Dimensionless normalization:
        # For uncorrelated bonds: E[‖L_i‖²] = n_avg · s_var  →  loss ≈ 1
        # For equilibrated field: L_i = 0                    →  loss = 0
        s_var = (deg * s ** 2).mean().detach().clamp(min=1e-12)
        n_avg = len(self._src) / float(self._N)
        loss = (L_i ** 2).sum(dim=1).mean() / (s_var * n_avg)

        return loss
