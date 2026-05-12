"""
Fracture mechanics PINO loss for crack / V-notch problems.

Four complementary physics-informed loss terms:

1. K_I Consistency
   Extracts the stress intensity factor from the predicted displacement field
   via least-squares Williams correlation and enforces it equals the input K_I.

2. Crack Face Traction-Free BC
   Enforces σ_yy = σ_xy = 0 on elements adjacent to the crack face
   (y ≈ tip_y, x < tip_x).

3. Peridynamic Equilibrium Residual  (replaces former Williams Asymptotic Residual)
   Bond-based peridynamic equilibrium: Σ_j (1−d_ij)² s_ij ê_ij = 0 at every node.
   Works on the full mesh (not limited to the K-dominant zone); naturally handles
   large phase-field damage zones where the Williams LEFM expansion breaks down.

4. J-Integral Conservation
   Uses the domain form of the J-integral and enforces J = K_I²/E (plane stress).

All terms operate in **physical (SI) units**. The caller (trainer) is responsible
for denormalizing predicted displacements before passing them here.

Reference:
  Silling (2000), "Reformulation of elasticity theory for discontinuities..."
  Williams (1957), "On the stress distribution at the base of a stationary crack"
  Li et al. (2021), "Physics-Informed Neural Operator for Learning PDEs", ICLR 2024
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import Delaunay

from .pino_loss import _compute_B_matrices
from .peridynamic_loss import PeridynamicEquilibriumLoss


class CrackFractureLoss(nn.Module):
    """
    PINO loss for Mode I fracture mechanics problems.

    Assumes:
    - Dominant Mode I loading (K_II ≈ 0)
    - Plane stress conditions
    - Crack/notch bisector along y = tip_y, extending from x = 0 to x = tip_x
    - Crack propagation direction: +x

    Args:
        tip_x:          Crack tip x-coordinate
        tip_y:          Crack tip y-coordinate
        r_ki_min:       Inner radius of K_I extraction annulus
        r_ki_max:       Outer radius of K_I extraction annulus
        r_j:            Outer radius of J-integral domain
        crack_face_tol: Half-width tolerance for detecting crack-face elements
        stress_intensity: Weight for K_I consistency loss
        traction_free:    Weight for crack face BC loss
        near_tip:         Weight for peridynamic equilibrium residual loss
        j_integral:       Weight for J-integral conservation loss
        horizon_factor:   PD horizon δ = horizon_factor × h_avg  (default 3.0)
    """

    def __init__(
        self,
        tip_x: float,
        tip_y: float,
        r_ki_min: float = 0.02,
        r_ki_max: float = 0.10,
        r_j: float = 0.15,
        crack_face_tol: float = 0.02,
        stress_intensity: float = 1.0,
        traction_free: float = 1.0,
        near_tip: float = 1.0,        # weight for peridynamic equilibrium residual
        j_integral: float = 1.0,
        horizon_factor: float = 3.0,  # PD horizon δ = horizon_factor × h_avg
    ):
        super().__init__()
        self.tip_x = tip_x
        self.tip_y = tip_y
        self.r_ki_min = r_ki_min
        self.r_ki_max = r_ki_max
        self.r_j = r_j
        self.crack_face_tol = crack_face_tol
        self.stress_intensity = stress_intensity
        self.traction_free = traction_free
        self.near_tip = near_tip
        self.j_integral = j_integral

        # Peridynamic equilibrium loss (replaces Williams asymptotic residual)
        self._pd_loss = PeridynamicEquilibriumLoss(horizon_factor=horizon_factor)

        # FEM mesh topology cache — computed once per unique coordinate set
        self._coords_hash: int = -1
        self._elems: torch.Tensor = None
        self._B: torch.Tensor = None
        self._areas: torch.Tensor = None
        self._centroids: torch.Tensor = None

    # ------------------------------------------------------------------
    # Mesh helpers
    # ------------------------------------------------------------------

    def _ensure_mesh_cache(self, coords: torch.Tensor, device: torch.device) -> None:
        """Triangulate coords and cache B-matrices (runs once per unique mesh)."""
        coords_np = coords.detach().cpu().numpy()
        h = hash(coords_np.tobytes())

        if h == self._coords_hash:
            # Move cached tensors to current device if needed
            self._elems = self._elems.to(device)
            self._B = self._B.to(device)
            self._areas = self._areas.to(device)
            self._centroids = self._centroids.to(device)
            return

        tri = Delaunay(coords_np)
        elems_np = tri.simplices  # (M, 3)

        xy = torch.tensor(coords_np[elems_np], dtype=torch.float32, device=device)
        B, areas = _compute_B_matrices(xy)

        self._elems = torch.tensor(elems_np, dtype=torch.long, device=device)
        self._B = B
        self._areas = areas
        self._centroids = xy.mean(dim=1)  # (M, 2)
        self._coords_hash = h

    def _build_C(self, E: float, nu: float, device: torch.device) -> torch.Tensor:
        """Plane-stress constitutive matrix C (3×3, Voigt notation)."""
        factor = E / (1.0 - nu ** 2)
        C = torch.tensor(
            [
                [1.0,  nu,             0.0],
                [nu,   1.0,            0.0],
                [0.0,  0.0, (1.0 - nu) / 2.0],
            ],
            dtype=torch.float32,
            device=device,
        ) * factor
        return C

    @staticmethod
    def _kappa(nu: float) -> float:
        """Kolosov constant for plane stress."""
        return (3.0 - nu) / (1.0 + nu)

    # ------------------------------------------------------------------
    # Term 1: K_I Consistency
    # ------------------------------------------------------------------

    def _ki_consistency(
        self,
        u_pred: torch.Tensor,  # (N, 2)
        coords: torch.Tensor,  # (N, 2)
        K_I: float,
        E: float,
        nu: float,
    ) -> torch.Tensor:
        """
        Least-squares K_I extraction from u_y in annulus [r_ki_min, r_ki_max].

        K_I_fit = 2μ * Σ(u_y_i * f_y_i) / Σ(f_y_i²)
        where f_y(r,θ) = sqrt(r/2π) * sin(θ/2) * (κ+1 - 2cos²(θ/2))

        Loss: ((K_I_fit - K_I_input) / |K_I_input|)²
        """
        mu = E / (2.0 * (1.0 + nu))
        kappa = self._kappa(nu)

        dx = coords[:, 0] - self.tip_x
        dy = coords[:, 1] - self.tip_y
        r = (dx ** 2 + dy ** 2).sqrt()

        mask = (r >= self.r_ki_min) & (r <= self.r_ki_max)
        if mask.sum() < 3:
            return torch.tensor(0.0, device=u_pred.device)

        r_m = r[mask].clamp(min=1e-12)
        theta = torch.atan2(dy[mask], dx[mask])

        sqrt_r_2pi = (r_m / (2.0 * torch.pi)).sqrt()
        sin_h = torch.sin(theta / 2.0)
        cos_h = torch.cos(theta / 2.0)

        # Williams basis for u_y (Mode I)
        f_y = sqrt_r_2pi * sin_h * (kappa + 1.0 - 2.0 * cos_h ** 2)

        u_y = u_pred[mask, 1]
        denom = (f_y ** 2).sum().clamp(min=1e-30)
        K_I_fit = 2.0 * mu * (u_y * f_y).sum() / denom

        return ((K_I_fit - K_I) / (abs(K_I) + 1e-10)) ** 2

    # ------------------------------------------------------------------
    # Term 2: Crack Face Traction-Free BC
    # ------------------------------------------------------------------

    def _crack_face_bc(
        self,
        u_pred: torch.Tensor,  # (N, 2)
        K_I: float,
        E: float,
        nu: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        σ_yy = 0 and σ_xy = 0 on crack-face elements.

        Crack face: centroid satisfies |y_c - tip_y| < tol, x_c < tip_x.
        Loss: Σ A_e(σ_yy² + σ_xy²) / (Σ A_e · σ_ref²)   [dimensionless]

        σ_ref = K_I / √(2π r_ki_min) — characteristic stress at the inner
        extraction radius. This gives O(1) loss when crack face stresses are
        at the K_I-field scale, regardless of E.
        """
        cx = self._centroids[:, 0]
        cy = self._centroids[:, 1]
        mask = (torch.abs(cy - self.tip_y) < self.crack_face_tol) & (
            cx < self.tip_x - self.crack_face_tol
        )

        if mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        C = self._build_C(E, nu, device)

        u_elem = u_pred[self._elems[mask]].reshape(-1, 6)
        eps = torch.einsum("mij,mj->mi", self._B[mask], u_elem)  # (M_cf, 3)
        sig = torch.einsum("ij,mj->mi", C, eps)                  # (M_cf, 3)

        a_cf = self._areas[mask]
        total_a = a_cf.sum().clamp(min=1e-30)
        sig_ref_sq = K_I ** 2 / (2.0 * torch.pi * self.r_ki_min)
        return (a_cf * (sig[:, 1] ** 2 + sig[:, 2] ** 2)).sum() / (
            total_a * sig_ref_sq
        )

    # ------------------------------------------------------------------
    # Term 4: J-Integral Conservation
    # ------------------------------------------------------------------

    def _j_integral(
        self,
        u_pred: torch.Tensor,  # (N, 2)
        K_I: float,
        E: float,
        nu: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Domain form of J-integral: J = ∫_Ω [σ_ij ∂u_i/∂x₁ - W δ₁j] ∂q/∂x_j dΩ

        Hat function: q(r) = 1 - r/r_j  (q=1 at tip, q=0 at r=r_j)
        Reference:    J_ref = K_I²/E  (plane stress)
        Loss:         ((J - J_ref) / |J_ref|)²
        """
        cx = self._centroids[:, 0]
        cy = self._centroids[:, 1]
        dx_c = cx - self.tip_x
        dy_c = cy - self.tip_y
        r_c = (dx_c ** 2 + dy_c ** 2).sqrt().clamp(min=1e-12)

        in_domain = r_c < self.r_j
        if in_domain.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Gradient of hat function q(r) = 1 - r/r_j at element centroids
        # ∂q/∂x = -1/r_j * (x_c - tip_x) / r_c   (zero outside domain)
        dq_dx = torch.where(in_domain, -dx_c / (self.r_j * r_c), torch.zeros_like(dx_c))
        dq_dy = torch.where(in_domain, -dy_c / (self.r_j * r_c), torch.zeros_like(dy_c))

        C = self._build_C(E, nu, device)

        # All-element strain, stress, and strain energy
        u_elem = u_pred[self._elems].reshape(-1, 6)          # (M, 6)
        eps = torch.einsum("mij,mj->mi", self._B, u_elem)    # (M, 3): [ε_xx, ε_yy, γ_xy]
        sig = torch.einsum("ij,mj->mi", C, eps)              # (M, 3): [σ_xx, σ_yy, σ_xy]
        W = 0.5 * (eps * sig).sum(dim=-1)                    # (M,)  strain energy density

        # Displacement gradients ∂u_x/∂x and ∂u_y/∂x
        # B[:,0,0::2] = [∂N₁/∂x, ∂N₂/∂x, ∂N₃/∂x]  (from row 0 of B, even cols)
        dN_dx = self._B[:, 0, 0::2]                  # (M, 3)
        u_x_nodes = u_elem[:, 0::2]                  # (M, 3)  x-displacements at nodes
        u_y_nodes = u_elem[:, 1::2]                  # (M, 3)  y-displacements at nodes

        du_x_dx = (dN_dx * u_x_nodes).sum(-1)        # (M,)  ∂u_x/∂x
        du_y_dx = (dN_dx * u_y_nodes).sum(-1)        # (M,)  ∂u_y/∂x

        sig_xx, sig_yy, sig_xy = sig[:, 0], sig[:, 1], sig[:, 2]

        # Integrand per element (domain form, x₁ = x):
        # f_x = (σ_xx ∂u_x/∂x + σ_xy ∂u_y/∂x - W) * ∂q/∂x
        # f_y = (σ_xy ∂u_x/∂x + σ_yy ∂u_y/∂x)     * ∂q/∂y
        integrand = (
            (sig_xx * du_x_dx + sig_xy * du_y_dx - W) * dq_dx
            + (sig_xy * du_x_dx + sig_yy * du_y_dx) * dq_dy
        )

        J = (self._areas * integrand).sum()

        J_ref = K_I ** 2 / E
        return ((J - J_ref) / (abs(J_ref) + 1e-10)) ** 2

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        u_pred: torch.Tensor,  # (N, 2)  physical displacement
        coords: torch.Tensor,  # (N, 2)  mesh coordinates
        K_I: float,
        E: float,
        nu: float,
    ) -> torch.Tensor:
        """
        Compute combined fracture mechanics physics loss.

        Args:
            u_pred:  Predicted displacement field in physical units (N, 2).
                     **Incompatible with von Mises scalar output (N, 1).**
                     Only enable this loss when the surrogate predicts
                     displacement (output_dim=2), not stress scalars.
            coords:  Mesh node coordinates (N, 2)
            K_I:     Mode I stress intensity factor [Pa√m]
            E:       Young's modulus [Pa]
            nu:      Poisson's ratio

        Returns:
            Scalar loss (dimensionless, weighted sum of all active terms)
        """
        device = u_pred.device
        self._ensure_mesh_cache(coords, device)

        total = torch.tensor(0.0, device=device)

        if self.stress_intensity > 0.0:
            total = total + self.stress_intensity * self._ki_consistency(
                u_pred, coords, K_I, E, nu
            )

        if self.traction_free > 0.0:
            total = total + self.traction_free * self._crack_face_bc(
                u_pred, K_I, E, nu, device
            )

        if self.near_tip > 0.0:
            total = total + self.near_tip * self._pd_loss(u_pred, coords)

        if self.j_integral > 0.0:
            total = total + self.j_integral * self._j_integral(
                u_pred, K_I, E, nu, device
            )

        return total
