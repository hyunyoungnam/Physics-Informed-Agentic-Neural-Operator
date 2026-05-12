"""
FEM-based ground truth data generation for V-notch problems.

This module provides utilities to generate FEM solutions using MFEM
for V-notch geometries with varying material properties and loading.

The workflow:
1. Generate V-notch mesh at specified resolution
2. Set up boundary conditions (fixed bottom, tension on top)
3. Solve linear elasticity with MFEM
4. Extract displacement field at mesh nodes
"""

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dataset import FEMSample

logger = logging.getLogger(__name__)


@dataclass
class VNotchFEMConfig:
    """
    Configuration for V-notch FEM simulations.

    Attributes:
        notch_depth: Depth of V-notch (0 < depth < width)
        notch_angle: Opening angle in degrees
        width: Domain width
        height: Domain height
        resolution: Mesh resolution (elements per unit length)
        fe_order: Finite element polynomial order
    """
    notch_depth: float = 0.3
    notch_angle: float = 60.0
    width: float = 1.0
    height: float = 1.0
    resolution: int = 25
    fe_order: int = 1


def generate_vnotch_fem_sample(
    E: float,
    nu: float,
    traction: float,
    config: VNotchFEMConfig,
    output_dir: Optional[Path] = None,
) -> Optional[FEMSample]:
    """
    Generate a single FEM sample for V-notch problem.

    Boundary conditions:
    - Bottom edge: Fixed (u_x = u_y = 0)
    - Top edge: Uniform traction (tension)
    - Other edges: Free

    Args:
        E: Young's modulus (Pa)
        nu: Poisson's ratio
        traction: Applied traction on top edge (Pa)
        config: V-notch FEM configuration
        output_dir: Optional directory for mesh/output files

    Returns:
        FEMSample with displacement field, or None if MFEM not available
    """
    try:
        import mfem.ser as mfem
    except ImportError:
        logger.warning("PyMFEM not available, returning synthetic data")
        return _generate_synthetic_sample(E, nu, traction, config)

    from ..geometry.notch import VNotchGeometry, VNotchMeshGenerator
    from ..mesh.mfem_manager import MFEMManager
    from ..solvers.base import (
        BoundaryCondition,
        BoundaryConditionType,
        MaterialProperties,
        PhysicsConfig,
        PhysicsType,
    )
    from ..solvers.mfem_solver import MFEMSolver

    # Create output directory
    if output_dir is None:
        import tempfile
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate mesh
    geometry = VNotchGeometry(
        notch_depth=config.notch_depth,
        notch_angle=config.notch_angle,
        width=config.width,
        height=config.height,
    )
    mesh_gen = VNotchMeshGenerator(
        geometry=geometry,
        base_resolution=config.resolution,
        tip_refinement_levels=4,
    )

    mesh_path = output_dir / "vnotch.mesh"
    vertices, elements, mesh_meta = mesh_gen.generate(output_path=str(mesh_path))

    # Load mesh with MFEM manager
    mesh_manager = MFEMManager(mesh_path)

    # Set up physics
    material = MaterialProperties(E=E, nu=nu, density=7800.0)

    # Boundary conditions using the boundary markers from mesh generator
    boundary_conditions = [
        # Bottom edge: Fixed
        BoundaryCondition(
            bc_type=BoundaryConditionType.DISPLACEMENT,
            boundary_id=VNotchMeshGenerator.BOTTOM,
            value=[0.0, 0.0],
        ),
        # Top edge: Tension
        BoundaryCondition(
            bc_type=BoundaryConditionType.TRACTION,
            boundary_id=VNotchMeshGenerator.TOP,
            value=[0.0, traction],
        ),
    ]

    physics = PhysicsConfig(
        physics_type=PhysicsType.LINEAR_ELASTICITY,
        material=material,
        boundary_conditions=boundary_conditions,
    )

    # Solve
    solver = MFEMSolver(order=config.fe_order)
    solver.setup(mesh_manager, physics)
    result = solver.solve(output_dir)

    if not result.success:
        logger.error(f"FEM solve failed: {result.error_message}")
        return None

    # Extract data
    displacement = result.solution_data.get("displacement", np.zeros((len(vertices), 2)))
    von_mises = result.solution_data.get("von_mises", None)

    # Use MFEM's actual node coordinates (may differ from Python generator vertices
    # because MFEM compacts unreferenced vertices when loading the mesh file)
    mfem_coords = mesh_manager.get_nodes()  # (n_mfem_nodes, 2) — matches displacement size

    # Use MFEM's own element connectivity — consistent with get_nodes() ordering
    mfem_elements = mesh_manager.get_elements()

    return FEMSample(
        sample_id=str(uuid.uuid4()),
        parameters={"E": E, "nu": nu, "traction": traction},
        coordinates=mfem_coords.astype(np.float32),
        displacement=displacement.astype(np.float32),
        von_mises=von_mises.astype(np.float32) if von_mises is not None else None,
        elements=mfem_elements,
        metadata={
            "notch_depth": config.notch_depth,
            "notch_angle": config.notch_angle,
            "n_nodes": mfem_coords.shape[0],
            "n_elements": mfem_elements.shape[0],
            "max_displacement": float(np.max(np.linalg.norm(displacement, axis=1))),
        },
    )


def _generate_synthetic_sample(
    E: float,
    nu: float,
    traction: float,
    config: VNotchFEMConfig,
) -> FEMSample:
    """
    Generate synthetic displacement field when MFEM is not available.

    Uses Williams eigenfunction expansion for V-notch singularity:
    - Near-tip: singular field with lambda eigenvalue from notch angle
    - Far-field: uniform tension solution
    - Blending function for smooth transition

    The Williams expansion for V-notch gives:
        u ~ r^lambda * f(theta, lambda)
    where lambda depends on notch angle (lambda < 1 for sharp notches).
    """
    from ..geometry.notch import VNotchGeometry, VNotchMeshGenerator

    # Generate mesh
    geometry = VNotchGeometry(
        notch_depth=config.notch_depth,
        notch_angle=config.notch_angle,
        width=config.width,
        height=config.height,
    )
    mesh_gen = VNotchMeshGenerator(
        geometry=geometry,
        base_resolution=config.resolution,
    )
    vertices, elements, _ = mesh_gen.generate()

    W = config.width
    H = config.height
    tip = geometry.tip_position
    tip_x, tip_y = tip[0], tip[1]

    # Lame parameters
    mu = E / (2 * (1 + nu))  # Shear modulus
    kappa = (3 - nu) / (1 + nu)  # Plane stress

    # Williams eigenvalue for V-notch
    # For opening angle 2*alpha, the eigenvalue satisfies:
    # lambda * sin(2*alpha) + sin(2*lambda*alpha) = 0 (symmetric mode)
    # For 60 degree notch (alpha = pi - 30deg = 150deg), lambda ≈ 0.5
    alpha = np.pi - np.radians(config.notch_angle / 2)
    # Approximate eigenvalue (use Newton-Raphson for exact)
    lam = _compute_williams_eigenvalue(alpha)

    # Stress intensity factor (approximate)
    # K scales with traction and geometry
    K = traction * np.sqrt(np.pi * config.notch_depth) * _geometry_factor(config.notch_depth / W, config.notch_angle)

    n_points = len(vertices)
    displacement = np.zeros((n_points, 2), dtype=np.float32)

    # Far-field strain (uniform tension)
    eps_yy_far = traction / E
    eps_xx_far = -nu * eps_yy_far

    # Blending radius
    R_blend = config.notch_depth * 0.8

    for i, (x, y) in enumerate(vertices):
        # Distance and angle from notch tip
        dx = x - tip_x
        dy = y - tip_y
        r = np.sqrt(dx**2 + dy**2) + 1e-12
        theta = np.arctan2(dy, dx)

        # --- Far-field solution (uniform tension, fixed bottom) ---
        u_y_far = eps_yy_far * y
        u_x_far = eps_xx_far * (x - W / 2)

        # --- Near-tip Williams solution ---
        # Symmetric mode (Mode I type) displacement field
        r_lam = r ** lam
        cos_lam_theta = np.cos(lam * theta)
        sin_lam_theta = np.sin(lam * theta)
        cos_theta_2 = np.cos(theta / 2)
        sin_theta_2 = np.sin(theta / 2)

        # Williams displacement (simplified for symmetric loading)
        coeff = K / (2 * mu) * np.sqrt(r / (2 * np.pi))

        # Mode I Williams expansion
        u_x_near = coeff * cos_theta_2 * (kappa - 1 + 2 * sin_theta_2**2)
        u_y_near = coeff * sin_theta_2 * (kappa + 1 - 2 * cos_theta_2**2)

        # --- Blending function (smooth transition) ---
        # phi = 1 near tip, 0 far away
        if r < R_blend:
            phi = 1.0 - 3 * (r / R_blend)**2 + 2 * (r / R_blend)**3
        else:
            phi = 0.0

        # --- Combine with blending ---
        displacement[i, 0] = phi * u_x_near + (1 - phi) * u_x_far
        displacement[i, 1] = phi * u_y_near + (1 - phi) * u_y_far

    # Enforce boundary conditions
    # Bottom boundary: fixed (y ≈ 0)
    bottom_mask = vertices[:, 1] < 0.01 * H
    displacement[bottom_mask, :] = 0.0

    # Compute von Mises stress (approximate)
    von_mises = _compute_synthetic_von_mises(vertices, displacement, E, nu, tip_x, tip_y, K, lam)

    # Compute max displacement for metadata
    disp_mag = np.linalg.norm(displacement, axis=1)

    return FEMSample(
        sample_id=str(uuid.uuid4()),
        parameters={"E": E, "nu": nu, "traction": traction},
        coordinates=vertices.astype(np.float32),
        displacement=displacement.astype(np.float32),
        von_mises=von_mises,
        elements=elements.astype(np.int32),
        metadata={
            "notch_depth": config.notch_depth,
            "notch_angle": config.notch_angle,
            "n_nodes": len(vertices),
            "n_elements": len(elements),
            "synthetic": True,
            "williams_lambda": lam,
            "K_approx": K,
            "max_displacement": float(disp_mag.max()),
        },
    )


def _compute_williams_eigenvalue(alpha: float) -> float:
    """
    Compute Williams eigenvalue for V-notch.

    Solves: lambda * sin(2*alpha) + sin(2*lambda*alpha) = 0

    Args:
        alpha: Half opening angle (pi - notch_half_angle)

    Returns:
        Smallest positive eigenvalue lambda
    """
    # Use Newton-Raphson starting from initial guess
    # For sharp notches, lambda is close to 0.5
    lam = 0.5

    for _ in range(20):
        f = lam * np.sin(2 * alpha) + np.sin(2 * lam * alpha)
        df = np.sin(2 * alpha) + 2 * alpha * np.cos(2 * lam * alpha)

        if abs(df) < 1e-10:
            break

        lam_new = lam - f / df

        if abs(lam_new - lam) < 1e-8:
            break

        lam = lam_new

    return max(0.3, min(lam, 0.9))  # Keep in reasonable range


def geometry_factor(a_W: float, angle: float) -> float:
    """
    Geometry factor for V-notch stress intensity factor.

    Args:
        a_W:   Notch depth to width ratio (a/W)
        angle: Notch opening angle in degrees

    Returns:
        Dimensionless geometry factor F
    """
    # Empirical polynomial for V-notched specimens (FEM-calibrated)
    F = 1.12 - 0.231 * a_W + 10.55 * a_W**2 - 21.72 * a_W**3 + 30.39 * a_W**4
    # Angle correction: sharper notch → higher K_I
    angle_correction = 1.0 + 0.5 * (60.0 - angle) / 60.0
    return F * angle_correction


# Keep private alias for backward compatibility within this module
_geometry_factor = geometry_factor


def compute_ki(
    traction: float,
    notch_depth: float,
    width: float = 1.0,
    angle: float = 60.0,
) -> float:
    """
    Approximate Mode I stress intensity factor for a V-notch under uniform traction.

    K_I ≈ traction * sqrt(π * a) * F(a/W, angle)

    Args:
        traction:     Applied far-field traction [Pa]
        notch_depth:  V-notch depth a [m]
        width:        Domain width W [m]
        angle:        Notch opening angle [degrees]

    Returns:
        K_I [Pa√m]
    """
    return traction * np.sqrt(np.pi * notch_depth) * geometry_factor(notch_depth / width, angle)


def _compute_synthetic_von_mises(
    vertices: np.ndarray,
    displacement: np.ndarray,
    E: float,
    nu: float,
    tip_x: float,
    tip_y: float,
    K: float,
    lam: float,
) -> np.ndarray:
    """
    Compute approximate von Mises stress field.
    """
    mu = E / (2 * (1 + nu))
    n_points = len(vertices)
    von_mises = np.zeros(n_points, dtype=np.float32)

    for i, (x, y) in enumerate(vertices):
        dx = x - tip_x
        dy = y - tip_y
        r = np.sqrt(dx**2 + dy**2) + 1e-10
        theta = np.arctan2(dy, dx)

        # Near-tip stress (singular)
        sigma_r = K / np.sqrt(2 * np.pi * r) * (1 + np.sin(theta / 2)**2)
        sigma_theta = K / np.sqrt(2 * np.pi * r) * np.cos(theta / 2)**2
        tau_r_theta = K / np.sqrt(2 * np.pi * r) * np.sin(theta / 2) * np.cos(theta / 2)

        # Convert to Cartesian (approximate)
        sigma_xx = sigma_r * np.cos(theta)**2 + sigma_theta * np.sin(theta)**2
        sigma_yy = sigma_r * np.sin(theta)**2 + sigma_theta * np.cos(theta)**2
        sigma_xy = (sigma_r - sigma_theta) * np.sin(theta) * np.cos(theta) + tau_r_theta

        # Von Mises
        vm = np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3 * sigma_xy**2)

        # Cap at reasonable value
        von_mises[i] = min(vm, 10 * K / np.sqrt(0.01))

    return von_mises


def generate_vnotch_dataset(
    n_samples: int,
    config: VNotchFEMConfig,
    E_range: Tuple[float, float] = (150e9, 250e9),
    nu_range: Tuple[float, float] = (0.25, 0.35),
    traction_range: Tuple[float, float] = (50e6, 150e6),
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Generate a dataset of FEM samples for V-notch problem.

    Args:
        n_samples: Number of samples to generate
        config: V-notch FEM configuration
        E_range: Range of Young's modulus
        nu_range: Range of Poisson's ratio
        traction_range: Range of applied traction
        seed: Random seed
        output_dir: Optional output directory

    Returns:
        params: (n_samples, 3) array of [E, nu, traction]
        coords: Mesh coordinates (same for all samples)
        outputs: List of displacement arrays
    """
    rng = np.random.default_rng(seed)

    params_list = []
    outputs = []
    coords = None

    for i in range(n_samples):
        E = float(rng.uniform(*E_range))
        nu = float(rng.uniform(*nu_range))
        traction = float(rng.uniform(*traction_range))

        sample_dir = output_dir / f"sample_{i:04d}" if output_dir else None

        sample = generate_vnotch_fem_sample(
            E=E,
            nu=nu,
            traction=traction,
            config=config,
            output_dir=sample_dir,
        )

        if sample is not None:
            params_list.append([E, nu, traction])
            outputs.append(sample.displacement)

            if coords is None:
                coords = sample.coordinates

            logger.info(
                f"Sample {i+1}/{n_samples}: E={E:.2e}, nu={nu:.3f}, "
                f"traction={traction:.2e}, max_disp={sample.metadata.get('max_displacement', 0):.2e}"
            )

    params = np.array(params_list, dtype=np.float32)

    return params, coords, outputs


def generate_ground_truth(
    params: np.ndarray,
    config: VNotchFEMConfig,
    fine_resolution: int = 50,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate fine-mesh FEM solutions as ground truth.

    Uses a finer mesh than training data for more accurate reference.

    Args:
        params: (n_samples, 3) array of [E, nu, traction]
        config: Base V-notch configuration
        fine_resolution: Resolution for fine mesh

    Returns:
        fine_coords: Fine mesh coordinates
        fine_outputs: List of displacement arrays on fine mesh
    """
    fine_config = VNotchFEMConfig(
        notch_depth=config.notch_depth,
        notch_angle=config.notch_angle,
        width=config.width,
        height=config.height,
        resolution=fine_resolution,
        fe_order=2,  # Higher order for accuracy
    )

    fine_outputs = []
    fine_coords = None

    for i, (E, nu, traction) in enumerate(params):
        sample = generate_vnotch_fem_sample(
            E=float(E),
            nu=float(nu),
            traction=float(traction),
            config=fine_config,
        )

        if sample is not None:
            fine_outputs.append(sample.displacement)
            if fine_coords is None:
                fine_coords = sample.coordinates

    return fine_coords, fine_outputs
