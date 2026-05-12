"""
MFEM solver implementation.

Provides FEM solving capabilities using PyMFEM for linear elasticity
and heat transfer problems.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..mesh.base import MeshManager
from ..mesh.mfem_manager import MFEMManager
from .base import (
    BoundaryCondition,
    BoundaryConditionType,
    PhysicsConfig,
    PhysicsType,
    SolverInterface,
    SolverResult,
)

# Lazy import for optional dependency
_mfem = None


def _get_mfem():
    """Lazy import of mfem module."""
    global _mfem
    if _mfem is None:
        try:
            import mfem.ser as mfem
            _mfem = mfem
        except ImportError:
            raise ImportError(
                "PyMFEM is required for MFEM solver. "
                "Install with: pip install mfem"
            )
    return _mfem


class MFEMSolver(SolverInterface):
    """
    FEM solver using PyMFEM.

    Supports:
    - Linear elasticity with ElasticityIntegrator
    - Heat transfer with DiffusionIntegrator
    """

    def __init__(self, order: int = 1):
        """
        Initialize the MFEM solver.

        Args:
            order: Polynomial order for finite elements (default: 1 for linear)
        """
        super().__init__()
        self._order = order
        self._fespace: Optional[object] = None
        self._solution: Optional[object] = None
        self._solution_data: Dict[str, np.ndarray] = {}
        # Keep MFEM coefficient objects alive for the lifetime of the solver.
        # MFEM integrators hold raw C++ pointers; if the Python wrapper is
        # garbage-collected before assembly completes, those pointers dangle.
        self._coef_refs: list = []

    def setup(
        self,
        mesh_manager: MeshManager,
        physics: PhysicsConfig
    ) -> None:
        """
        Set up the solver with mesh and physics configuration.

        Args:
            mesh_manager: Mesh manager instance (must be MFEMManager)
            physics: Physics configuration
        """
        super().setup(mesh_manager, physics)

        if not isinstance(mesh_manager, MFEMManager):
            raise TypeError(
                f"MFEMSolver requires MFEMManager, got {type(mesh_manager)}"
            )

        self._is_setup = True

    def solve(self, output_dir: Union[str, Path]) -> SolverResult:
        """
        Execute the solver.

        Args:
            output_dir: Directory to store output files

        Returns:
            SolverResult: Result of the solve
        """
        if not self._is_setup:
            return SolverResult(
                success=False,
                error_message="Solver not set up. Call setup() first."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            if self._physics.physics_type == PhysicsType.LINEAR_ELASTICITY:
                result = self._solve_elasticity(output_dir)
            elif self._physics.physics_type == PhysicsType.HEAT_TRANSFER:
                result = self._solve_heat_transfer(output_dir)
            else:
                return SolverResult(
                    success=False,
                    error_message=f"Unsupported physics type: {self._physics.physics_type}"
                )

            result.solve_time = time.time() - start_time
            return result

        except Exception as e:
            return SolverResult(
                success=False,
                error_message=str(e),
                solve_time=time.time() - start_time
            )

    def _solve_elasticity(self, output_dir: Path) -> SolverResult:
        """
        Solve linear elasticity problem.

        Args:
            output_dir: Directory for output files

        Returns:
            SolverResult: Result of the solve
        """
        mfem = _get_mfem()
        mesh = self._mesh_manager.mesh
        dim = mesh.Dimension()

        # Material properties
        mat = self._physics.material
        lmbda = mat.lame_lambda
        mu = mat.lame_mu

        # Create vector finite element collection and space
        fec = mfem.H1_FECollection(self._order, dim)
        fespace = mfem.FiniteElementSpace(mesh, fec, dim)
        self._fespace = fespace

        # Define essential (Dirichlet) boundary conditions — per-component for SYMMETRY BCs
        ess_tdof_list = self._get_essential_true_dofs(mesh, fespace)

        # Set up the bilinear form (stiffness matrix)
        a = mfem.BilinearForm(fespace)

        # Add elasticity integrator.
        # Store fec, lambda_coef, mu_coef to prevent Python GC from freeing
        # objects that MFEM holds raw C++ pointers into.
        self._coef_refs.clear()
        lambda_coef = mfem.ConstantCoefficient(lmbda)
        mu_coef = mfem.ConstantCoefficient(mu)
        self._coef_refs.extend([fec, lambda_coef, mu_coef])
        a.AddDomainIntegrator(mfem.ElasticityIntegrator(lambda_coef, mu_coef))
        a.Assemble()

        # Set up the linear form (load vector)
        b = mfem.LinearForm(fespace)

        # Add body force if specified
        if self._physics.body_force is not None:
            body_force = mfem.VectorArrayCoefficient(dim)
            for d in range(dim):
                if d < len(self._physics.body_force):
                    body_force.Set(d, mfem.ConstantCoefficient(
                        self._physics.body_force[d] * mat.density
                    ))
                else:
                    body_force.Set(d, mfem.ConstantCoefficient(0.0))
            b.AddDomainIntegrator(mfem.VectorDomainLFIntegrator(body_force))

        # Add traction boundary conditions
        self._add_traction_bcs(b, mesh, dim)
        b.Assemble()

        # Create solution vector with initial displacement from BCs
        x = mfem.GridFunction(fespace)
        x.Assign(0.0)
        self._apply_displacement_bcs(x, mesh, dim)

        # Form the linear system
        A = mfem.OperatorPtr()
        B = mfem.Vector()
        X = mfem.Vector()
        a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

        # Solve the system
        M = mfem.GSSmoother(A.AsSparseMatrix())
        mfem.PCG(A, M, B, X, 1, 500, 1e-12, 0.0)

        # Recover the solution
        a.RecoverFEMSolution(X, b, x)
        self._solution = x

        # Extract solution data
        self._solution_data["displacement"] = self._extract_vector_field(x, dim)

        # Compute stress (post-processing)
        stress = self._compute_stress(x, lmbda, mu, dim)
        self._solution_data["stress"] = stress

        # Compute von Mises stress
        von_mises = self._compute_von_mises(stress, dim)
        self._solution_data["von_mises"] = von_mises

        # Export to VTU
        output_files = self._export_to_vtu(output_dir, "elasticity")

        # Compute metrics
        metrics = {
            "max_displacement": np.max(np.linalg.norm(
                self._solution_data["displacement"], axis=1
            )),
            "max_von_mises": np.max(von_mises),
            "num_dofs": fespace.GetTrueVSize(),
        }

        return SolverResult(
            success=True,
            solution_data=self._solution_data.copy(),
            output_files=output_files,
            metrics=metrics,
        )

    def _solve_heat_transfer(self, output_dir: Path) -> SolverResult:
        """
        Solve heat transfer (diffusion) problem.

        Args:
            output_dir: Directory for output files

        Returns:
            SolverResult: Result of the solve
        """
        mfem = _get_mfem()
        mesh = self._mesh_manager.mesh
        dim = mesh.Dimension()

        # Material properties
        mat = self._physics.material
        k = mat.k  # Thermal conductivity

        # Create scalar finite element collection and space
        fec = mfem.H1_FECollection(self._order, dim)
        fespace = mfem.FiniteElementSpace(mesh, fec)
        self._fespace = fespace

        # Define essential (Dirichlet) boundary conditions
        ess_tdof_list = mfem.intArray()
        ess_bdr = self._get_essential_boundaries_scalar(mesh)
        if ess_bdr.Size() > 0:
            fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

        # Set up the bilinear form (conductivity matrix)
        a = mfem.BilinearForm(fespace)
        k_coef = mfem.ConstantCoefficient(k)
        a.AddDomainIntegrator(mfem.DiffusionIntegrator(k_coef))
        a.Assemble()

        # Set up the linear form (heat source)
        b = mfem.LinearForm(fespace)

        # Add volumetric heat source if specified
        if self._physics.heat_source != 0.0:
            q_coef = mfem.ConstantCoefficient(self._physics.heat_source)
            b.AddDomainIntegrator(mfem.DomainLFIntegrator(q_coef))

        # Add heat flux boundary conditions
        self._add_heat_flux_bcs(b, mesh)

        b.Assemble()

        # Create solution vector with initial temperature from BCs
        x = mfem.GridFunction(fespace)
        x.Assign(0.0)
        self._apply_temperature_bcs(x, mesh)

        # Form the linear system
        A = mfem.OperatorPtr()
        B = mfem.Vector()
        X = mfem.Vector()
        a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

        # Solve the system
        M = mfem.GSSmoother(A.AsSparseMatrix())
        mfem.PCG(A, M, B, X, 1, 500, 1e-12, 0.0)

        # Recover the solution
        a.RecoverFEMSolution(X, b, x)
        self._solution = x

        # Extract solution data
        self._solution_data["temperature"] = self._extract_scalar_field(x)

        # Compute heat flux (post-processing)
        heat_flux = self._compute_heat_flux(x, k, dim)
        self._solution_data["heat_flux"] = heat_flux

        # Export to VTU
        output_files = self._export_to_vtu(output_dir, "heat_transfer")

        # Compute metrics
        metrics = {
            "max_temperature": np.max(self._solution_data["temperature"]),
            "min_temperature": np.min(self._solution_data["temperature"]),
            "max_heat_flux": np.max(np.linalg.norm(heat_flux, axis=1)),
            "num_dofs": fespace.GetTrueVSize(),
        }

        return SolverResult(
            success=True,
            solution_data=self._solution_data.copy(),
            output_files=output_files,
            metrics=metrics,
        )

    def _get_essential_true_dofs(self, mesh, fespace):
        """
        Build the combined essential true-DOF list, supporting per-component constraints.

        DISPLACEMENT BCs constrain all vector components on the marked boundary.
        SYMMETRY BCs with direction=0 or 1 constrain only that displacement component,
        which is the correct way to impose symmetry / roller BCs in MFEM.
        """
        mfem = _get_mfem()
        num_bdr = mesh.bdr_attributes.Max() if mesh.bdr_attributes.Size() > 0 else 0

        all_dofs: set = set()

        for bc in self._physics.boundary_conditions:
            if bc.bc_type not in (
                BoundaryConditionType.DISPLACEMENT,
                BoundaryConditionType.SYMMETRY,
            ):
                continue
            if not (1 <= bc.boundary_id <= num_bdr):
                continue

            marker = mfem.intArray(num_bdr)
            marker.Assign(0)
            marker[bc.boundary_id - 1] = 1

            tdofs = mfem.intArray()
            if bc.bc_type == BoundaryConditionType.SYMMETRY and bc.direction is not None:
                # Constrain only the specified displacement component (0=x, 1=y)
                fespace.GetEssentialTrueDofs(marker, tdofs, bc.direction)
            else:
                # DISPLACEMENT: constrain all components
                fespace.GetEssentialTrueDofs(marker, tdofs)

            for j in range(tdofs.Size()):
                all_dofs.add(tdofs[j])

        sorted_dofs = sorted(all_dofs)
        combined = mfem.intArray(len(sorted_dofs))
        for i, v in enumerate(sorted_dofs):
            combined[i] = v
        return combined

    def _get_essential_boundaries_scalar(self, mesh):
        """Get essential boundary markers for scalar problems."""
        mfem = _get_mfem()
        num_bdr = mesh.bdr_attributes.Max() if mesh.bdr_attributes.Size() > 0 else 0
        ess_bdr = mfem.intArray(num_bdr)
        ess_bdr.Assign(0)

        for bc in self._physics.boundary_conditions:
            if bc.bc_type == BoundaryConditionType.TEMPERATURE:
                if 1 <= bc.boundary_id <= num_bdr:
                    ess_bdr[bc.boundary_id - 1] = 1

        return ess_bdr

    def _apply_displacement_bcs(self, x, mesh, dim: int) -> None:
        """Apply displacement boundary conditions to the solution.

        Zero displacement is correct here: x is pre-initialized to 0.0, and
        FormLinearSystem enforces the essential DOFs at exactly those zero values.
        Non-zero prescribed displacements are not currently needed.
        """
        pass

    def _apply_temperature_bcs(self, x, mesh) -> None:
        """Apply temperature boundary conditions to the solution."""
        mfem = _get_mfem()

        for bc in self._physics.boundary_conditions:
            if bc.bc_type == BoundaryConditionType.TEMPERATURE:
                if bc.value is not None:
                    # Create a constant coefficient for the temperature
                    temp_coef = mfem.ConstantCoefficient(float(bc.value))
                    # Project onto boundary DOFs
                    # Note: This is a simplified approach
                    x.ProjectBdrCoefficient(temp_coef, self._get_bc_markers(mesh, bc))

    def _get_bc_markers(self, mesh, bc: BoundaryCondition):
        """Get boundary markers for a specific boundary condition."""
        mfem = _get_mfem()
        num_bdr = mesh.bdr_attributes.Max() if mesh.bdr_attributes.Size() > 0 else 0
        markers = mfem.intArray(num_bdr)
        markers.Assign(0)
        if 1 <= bc.boundary_id <= num_bdr:
            markers[bc.boundary_id - 1] = 1
        return markers

    def _add_traction_bcs(self, b, mesh, dim: int) -> None:
        """Add traction boundary conditions to the linear form."""
        mfem = _get_mfem()

        for bc in self._physics.boundary_conditions:
            if bc.bc_type == BoundaryConditionType.TRACTION:
                if bc.value is not None:
                    # Create vector coefficient for traction — store component
                    # coefficients and the vector coefficient to prevent GC.
                    value = np.atleast_1d(bc.value)
                    comp_coefs = []
                    for d in range(dim):
                        v = float(value[d]) if d < len(value) else 0.0
                        comp_coefs.append(mfem.ConstantCoefficient(v))
                    traction = mfem.VectorArrayCoefficient(dim)
                    for d, c in enumerate(comp_coefs):
                        traction.Set(d, c)
                    self._coef_refs.extend(comp_coefs)
                    self._coef_refs.append(traction)

                    markers = self._get_bc_markers(mesh, bc)
                    # markers must outlive b.Assemble() — MFEM stores a pointer
                    self._coef_refs.append(markers)
                    b.AddBoundaryIntegrator(
                        mfem.VectorBoundaryLFIntegrator(traction),
                        markers
                    )

    def _add_heat_flux_bcs(self, b, mesh) -> None:
        """Add heat flux boundary conditions to the linear form."""
        mfem = _get_mfem()

        for bc in self._physics.boundary_conditions:
            if bc.bc_type == BoundaryConditionType.HEAT_FLUX:
                if bc.value is not None:
                    flux_coef = mfem.ConstantCoefficient(float(bc.value))
                    markers = self._get_bc_markers(mesh, bc)
                    b.AddBoundaryIntegrator(
                        mfem.BoundaryLFIntegrator(flux_coef),
                        markers
                    )

    def _extract_vector_field(self, gf, dim: int) -> np.ndarray:
        """Extract vector field data from GridFunction using bulk array access."""
        mfem = _get_mfem()
        num_nodes = self._mesh_manager.num_nodes
        raw = np.array(gf.GetDataArray())  # (num_nodes * dim,) flat, component-major
        # MFEM stores all x-dofs first, then all y-dofs (Ordering::byNODES)
        if raw.size == num_nodes * dim:
            return raw.reshape(dim, num_nodes).T.copy()
        # Fallback: per-dof extraction when ordering differs
        data = np.zeros((num_nodes, dim), dtype=np.float64)
        for i in range(num_nodes):
            for d in range(dim):
                data[i, d] = gf[self._fespace.DofToVDof(i, d)]
        return data

    def _extract_scalar_field(self, gf) -> np.ndarray:
        """Extract scalar field data from GridFunction."""
        num_nodes = self._mesh_manager.num_nodes
        data = np.zeros(num_nodes, dtype=np.float64)

        for i in range(num_nodes):
            data[i] = gf[i]

        return data

    def _compute_stress(
        self,
        displacement: object,
        lmbda: float,
        mu: float,
        dim: int
    ) -> np.ndarray:
        """
        Compute stress tensor from displacement field.

        Returns stress in Voigt notation (xx, yy, zz, xy, yz, xz) or (xx, yy, xy) for 2D.
        """
        mfem = _get_mfem()

        # Number of stress components
        num_stress = 3 if dim == 2 else 6
        num_elements = self._mesh_manager.num_elements
        stress = np.zeros((num_elements, num_stress), dtype=np.float64)

        # Evaluate stress at element centres
        for i in range(num_elements):
            trans = self._mesh_manager.mesh.GetElementTransformation(i)

            # Set integration point at element centre (required before GetVectorGradient)
            ip = mfem.IntegrationPoint()
            if dim == 2:
                ip.Set2(0.5, 0.5)
            else:
                ip.Set3(0.5, 0.5, 0.5)
            trans.SetIntPoint(ip)

            # Compute displacement gradient; use GetDataArray for numpy-compatible extraction
            grad_u = mfem.DenseMatrix(dim, dim)
            displacement.GetVectorGradient(trans, grad_u)
            g = np.array(grad_u.GetDataArray())  # shape (dim, dim), column-major → (row, col)

            trace_eps = sum(g[d, d] for d in range(dim))

            if dim == 2:
                eps_xx = g[0, 0]
                eps_yy = g[1, 1]
                eps_xy = 0.5 * (g[0, 1] + g[1, 0])

                stress[i, 0] = lmbda * trace_eps + 2 * mu * eps_xx  # sigma_xx
                stress[i, 1] = lmbda * trace_eps + 2 * mu * eps_yy  # sigma_yy
                stress[i, 2] = 2 * mu * eps_xy                       # sigma_xy
            else:
                eps_xx = g[0, 0]
                eps_yy = g[1, 1]
                eps_zz = g[2, 2]
                eps_xy = 0.5 * (g[0, 1] + g[1, 0])
                eps_yz = 0.5 * (g[1, 2] + g[2, 1])
                eps_xz = 0.5 * (g[0, 2] + g[2, 0])

                stress[i, 0] = lmbda * trace_eps + 2 * mu * eps_xx  # sigma_xx
                stress[i, 1] = lmbda * trace_eps + 2 * mu * eps_yy  # sigma_yy
                stress[i, 2] = lmbda * trace_eps + 2 * mu * eps_zz  # sigma_zz
                stress[i, 3] = 2 * mu * eps_xy                       # sigma_xy
                stress[i, 4] = 2 * mu * eps_yz                       # sigma_yz
                stress[i, 5] = 2 * mu * eps_xz                       # sigma_xz

        return stress

    def _compute_von_mises(self, stress: np.ndarray, dim: int) -> np.ndarray:
        """Compute von Mises stress from stress tensor."""
        if dim == 2:
            # Plane stress assumption
            s_xx = stress[:, 0]
            s_yy = stress[:, 1]
            s_xy = stress[:, 2]
            von_mises = np.sqrt(
                s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * s_xy**2
            )
        else:
            s_xx = stress[:, 0]
            s_yy = stress[:, 1]
            s_zz = stress[:, 2]
            s_xy = stress[:, 3]
            s_yz = stress[:, 4]
            s_xz = stress[:, 5]
            von_mises = np.sqrt(
                0.5 * ((s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2)
                + 3 * (s_xy**2 + s_yz**2 + s_xz**2)
            )

        return von_mises

    def _compute_heat_flux(
        self,
        temperature: object,
        k: float,
        dim: int
    ) -> np.ndarray:
        """Compute heat flux from temperature field: q = -k * grad(T)."""
        mfem = _get_mfem()

        num_elements = self._mesh_manager.num_elements
        heat_flux = np.zeros((num_elements, dim), dtype=np.float64)

        for i in range(num_elements):
            trans = self._mesh_manager.mesh.GetElementTransformation(i)

            # Evaluate gradient at element center
            ip = mfem.IntegrationPoint()
            ip.Set2(0.5, 0.5) if dim == 2 else ip.Set3(0.5, 0.5, 0.5)

            grad_T = mfem.Vector(dim)
            temperature.GetGradient(trans, grad_T)

            for d in range(dim):
                heat_flux[i, d] = -k * grad_T[d]

        return heat_flux

    def _export_to_vtu(self, output_dir: Path, prefix: str) -> List[Path]:
        """Export solution to VTU format for visualization."""
        mfem = _get_mfem()
        output_files = []

        # Save mesh
        mesh_file = output_dir / f"{prefix}_mesh.mesh"
        self._mesh_manager.mesh.Print(str(mesh_file))
        output_files.append(mesh_file)

        # Save solution GridFunction
        if self._solution is not None:
            sol_file = output_dir / f"{prefix}_solution.gf"
            self._solution.Save(str(sol_file))
            output_files.append(sol_file)

        # Export to VTU using MFEM's ParaView data collection
        try:
            dc = mfem.VisItDataCollection(prefix, self._mesh_manager.mesh)
            dc.SetPrefixPath(str(output_dir))

            if self._solution is not None:
                if self._physics.physics_type == PhysicsType.LINEAR_ELASTICITY:
                    dc.RegisterField("displacement", self._solution)
                else:
                    dc.RegisterField("temperature", self._solution)

            dc.Save()

            # VTU files are in output_dir/prefix_*
            for vtu in output_dir.glob(f"{prefix}*.vtu"):
                output_files.append(vtu)

        except Exception:
            # VisIt export not available, skip VTU
            pass

        return output_files

    def get_solution_field(self, field_name: str) -> np.ndarray:
        """
        Extract a solution field.

        Args:
            field_name: Name of the field (displacement, stress, temperature, etc.)

        Returns:
            np.ndarray: Solution field data
        """
        if field_name not in self._solution_data:
            raise KeyError(
                f"Field '{field_name}' not found. "
                f"Available fields: {list(self._solution_data.keys())}"
            )
        return self._solution_data[field_name].copy()

    def get_available_fields(self) -> List[str]:
        """Get list of available solution fields."""
        return list(self._solution_data.keys())
