"""
Adaptive learning orchestrator.

Implements an autonomous active learning loop for efficient FEM dataset generation:
1. Generate initial FEM simulations (Latin Hypercube Sampling)
2. Train surrogate model (FNO/Transolver - implementation pending)
3. Evaluate surrogate and identify high-error/high-uncertainty regions
4. Select new samples using acquisition functions (informative sampling)
5. Repeat until convergence criteria are met

The key innovation is "informative sampling" - using acquisition functions
to prioritize samples that maximize information gain about the underlying
physics, rather than sampling uniformly.

Reference:
    Settles (2009): "Active Learning Literature Survey"
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..data.dataset import DatasetConfig, FEMDataset, FEMSample
from ..surrogate.base import SurrogateConfig, SurrogateModel, TransolverConfig
from ..surrogate.evaluator import SurrogateEvaluator, UncertaintyAnalysis, WeakRegion
from ..surrogate.trainer import SurrogateTrainer, TrainingConfig, TrainingResult
from ..surrogate.acquisition import (
    AcquisitionFunction,
    AcquisitionType,
    get_acquisition_function,
)
from ..surrogate.agentic_trainer import (
    AgenticSurrogateTrainer,
    AgenticTrainingConfig,
    AgenticTrainingResult,
)
from ..agents.roles.adaptive_proposer import AdaptiveProposerAgent, AdaptiveProposal
from ..agents.base import AgentContext
from .metrics import ActiveLearningMetrics, ConvergenceMonitor

logger = logging.getLogger(__name__)


def _williams_enrich(coords: np.ndarray, crack_tip_x: float, crack_y: float = 0.5) -> np.ndarray:
    """Append Williams near-tip polar features to (N,2) mesh coordinates.

    Returns (N, 8): [x, y, r, log_r, sinθ, cosθ, sin(θ/2), cos(θ/2)]
    The half-angle basis (sin/cos of θ/2) encodes the crack-face displacement
    jump discontinuity required by mode-I Williams expansion.
    """
    dx = coords[:, 0] - crack_tip_x
    dy = coords[:, 1] - crack_y
    r  = np.hypot(dx, dy).clip(1e-8).astype(np.float32)
    th = np.arctan2(dy, dx).astype(np.float32)
    return np.concatenate([
        coords,
        r[:, None],
        np.log(r)[:, None],
        np.sin(th)[:, None],
        np.cos(th)[:, None],
        np.sin(th / 2)[:, None],
        np.cos(th / 2)[:, None],
    ], axis=1)


class StoppingCriterion(Enum):
    """Reasons for stopping the active learning loop."""
    CONVERGED = auto()  # Error below threshold
    PATIENCE_EXHAUSTED = auto()  # No improvement for N iterations
    BUDGET_EXHAUSTED = auto()  # Maximum samples reached
    MAX_ITERATIONS = auto()  # Maximum iterations reached
    LOW_UNCERTAINTY = auto()  # Uncertainty below threshold
    DIMINISHING_RETURNS = auto()  # Efficiency dropped below threshold
    USER_INTERRUPTED = auto()  # User requested stop


@dataclass
class AdaptiveConfig:
    """
    Configuration for adaptive active learning.

    Essential parameters only - derived values computed at runtime.

    Attributes:
        base_mesh_path: Optional path kept for MFEM-based scripts (not used by the orchestrator)
        output_dir: Directory for outputs
        parameter_bounds: Bounds for each parameter {name: (min, max)}
        initial_samples: Number of initial LHS samples
        max_samples: Hard budget limit on total samples
        convergence_threshold: Error threshold for convergence
        patience: Iterations without improvement before stopping
        n_ensemble: Number of ensemble models for uncertainty
        use_agentic_hpo: Whether to use LLM-based hyperparameter optimization
        use_agentic_proposer: Whether to use LLM-based sample proposal
        max_hpo_rounds: Maximum HPO rounds per training session
        llm_model: LLM model for agents
    """
    base_mesh_path: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    parameter_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"delta_R": (-0.5, 0.5)}
    )
    initial_samples: int = 20
    max_samples: int = 200
    convergence_threshold: float = 0.05
    patience: int = 3
    n_ensemble: int = 5
    random_seed: int = 42
    use_agentic_hpo: bool = False
    use_agentic_proposer: bool = False
    max_hpo_rounds: int = 3
    llm_model: str = "claude-sonnet-4-6"
    use_budget_agent: bool = False
    use_mesh_strategy_agent: bool = False
    tip_coords: Optional[np.ndarray] = None
    phase_field_resolution: int = 30
    phase_field_n_load_steps: int = 20
    # Surrogate config override — if None, a default is chosen per backend
    surrogate_config_override: Optional[Any] = None

    # Derived at runtime
    @property
    def parameter_names(self) -> List[str]:
        return list(self.parameter_bounds.keys())

    @property
    def samples_per_iteration(self) -> int:
        return 5

    @property
    def max_iterations(self) -> int:
        return (self.max_samples - self.initial_samples) // self.samples_per_iteration + 1

    @property
    def surrogate_config(self):
        if self.surrogate_config_override is not None:
            return self.surrogate_config_override
        # Compact Transolver sized for small phase-field FEA datasets
        return TransolverConfig(
            d_model=64, n_layers=3, n_heads=4, slice_num=16,
            mlp_ratio=2.0, dropout=0.1, learning_rate=5e-4,
            batch_size=4, epochs=300, patience=100,
            scheduler_type="cosine", optimizer_type="adamw",
        )

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.base_mesh_path is not None:
            self.base_mesh_path = Path(self.base_mesh_path)


@dataclass
class AdaptiveResult:
    """
    Result of adaptive active learning run.

    Attributes:
        success: Whether learning converged successfully
        n_iterations: Number of iterations completed
        final_error: Final surrogate error
        total_samples: Total samples generated
        dataset_path: Path to final dataset
        surrogate_path: Path to trained surrogate
        history: Training history per iteration
        stopping_criterion: Why the loop stopped
        sample_efficiency: Error reduction per sample
        metrics_path: Path to saved metrics
    """
    success: bool
    n_iterations: int = 0
    final_error: float = float('inf')
    initial_error: float = float('inf')
    total_samples: int = 0
    dataset_path: Optional[Path] = None
    surrogate_path: Optional[Path] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    stopping_criterion: Optional[StoppingCriterion] = None
    sample_efficiency: float = 0.0
    error_reduction_percent: float = 0.0
    metrics_path: Optional[Path] = None


class AdaptiveOrchestrator:
    """
    Orchestrates autonomous active learning for surrogate model training.

    Implements an informative sampling strategy using acquisition functions
    to prioritize FEM simulations that maximize model improvement.

    Workflow:
    1. Generate initial samples with Latin Hypercube Sampling
    2. Run FEM simulations to get ground truth
    3. Train surrogate model (FNO/Transolver - implementation pending)
    4. Evaluate surrogate and compute acquisition scores
    5. Select new samples using acquisition function (informative sampling)
    6. Repeat until convergence criteria are met

    The key difference from uniform sampling is that new samples are
    selected to maximize information gain, focusing computational
    resources on regions where the model is weak.
    """

    def __init__(self, config: AdaptiveConfig, llm_provider: Optional[Any] = None):
        """
        Initialize orchestrator.

        Args:
            config: Adaptive learning configuration
            llm_provider: LLM provider for agentic HPO (optional)
        """
        self.config = config
        self.llm_provider = llm_provider
        self.dataset: Optional[FEMDataset] = None
        self.surrogate: Optional[SurrogateModel] = None
        self.trainer: Optional[SurrogateTrainer] = None
        self.evaluator: Optional[SurrogateEvaluator] = None

        # Active learning components
        self.metrics = ActiveLearningMetrics()
        self.convergence_monitor = ConvergenceMonitor(
            target_error=config.convergence_threshold,
            patience=config.patience,
            min_improvement=0.01,  # Hardcoded sensible default
            max_samples=config.max_samples,
        )
        self._acquisition_fn: Optional[AcquisitionFunction] = None
        self._coordinates: Optional[np.ndarray] = None
        self._best_error: float = float('inf')
        self._no_improvement_count: int = 0

        # Setup output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.meshes_dir = self.config.output_dir / "meshes"
        self.meshes_dir.mkdir(exist_ok=True)
        self.dataset_dir = self.config.output_dir / "dataset"
        self.dataset_dir.mkdir(exist_ok=True)
        self.surrogate_dir = self.config.output_dir / "surrogate"
        self.surrogate_dir.mkdir(exist_ok=True)
        self.metrics_dir = self.config.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        # Initialize dataset
        dataset_config = DatasetConfig(
            name="adaptive_fem_dataset",
            parameter_names=config.parameter_names,
            parameter_bounds=config.parameter_bounds,
            output_fields=["displacement", "von_mises"],
            coordinate_dim=2,  # Will be updated from mesh
            storage_dir=self.dataset_dir,
        )
        self.dataset = FEMDataset(dataset_config)

        # Initialize acquisition function (for non-agentic mode)
        self._acquisition_fn = get_acquisition_function("uncertainty")

        # Initialize Budget Agent if enabled
        self.budget_agent: Optional[Any] = None
        if config.use_budget_agent:
            from piano.agents.roles.budget import BudgetAgent
            self.budget_agent = BudgetAgent(
                model=config.llm_model,
                convergence_threshold=config.convergence_threshold,
                max_samples=config.max_samples,
                patience=config.patience,
                base_samples_per_iter=config.samples_per_iteration,
            )
            if llm_provider:
                self.budget_agent.set_llm_provider(llm_provider)

        # Initialize Mesh Strategy Agent if enabled
        self.mesh_strategy_agent: Optional[Any] = None
        if config.use_mesh_strategy_agent:
            from piano.agents.roles.mesh_strategy import MeshStrategyAgent
            self.mesh_strategy_agent = MeshStrategyAgent(model=config.llm_model)
            if llm_provider:
                self.mesh_strategy_agent.set_llm_provider(llm_provider)

        # Initialize agentic proposer if enabled
        self.proposer: Optional[AdaptiveProposerAgent] = None
        if config.use_agentic_proposer:
            self.proposer = AdaptiveProposerAgent(model=config.llm_model)
            if llm_provider:
                self.proposer.set_llm_provider(llm_provider)

        # Setup RNG
        np.random.seed(config.random_seed)

        logger.info(f"Initialized AdaptiveOrchestrator")
        logger.info(f"  Agentic Proposer: {config.use_agentic_proposer}")
        logger.info(f"  Convergence: {config.convergence_threshold}")
        logger.info(f"  Patience: {config.patience}")
        logger.info(f"  Max samples: {config.max_samples}")

    def run(
        self,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
    ) -> AdaptiveResult:
        """
        Run the autonomous active learning loop.

        This method implements the core active learning algorithm:
        1. Initialize with diverse samples (LHS)
        2. Train surrogate and evaluate
        3. Use acquisition function to select informative samples
        4. Repeat until convergence criteria met

        Args:
            callback: Optional callback(iteration, metrics) for progress reporting

        Returns:
            AdaptiveResult with final metrics, paths, and efficiency statistics
        """
        history = []
        stopping_criterion = None
        initial_error = float('inf')

        try:
            # Phase 1: Generate initial samples using Latin Hypercube Sampling
            logger.info("="*60)
            logger.info("PHASE 1: Initial Sampling (Latin Hypercube)")
            logger.info("="*60)
            initial_params = self._generate_initial_parameters()
            self._run_simulations(initial_params)

            # Main active learning loop
            for iteration in range(self.config.max_iterations):
                logger.info("")
                logger.info("="*60)
                logger.info(f"ITERATION {iteration + 1}/{self.config.max_iterations}")
                logger.info("="*60)

                # Phase 2: Train surrogate
                logger.info("Phase 2: Training surrogate model...")
                train_result = self._train_surrogate()

                if not train_result.success:
                    logger.error(f"Surrogate training failed: {train_result.error_message}")
                    return AdaptiveResult(
                        success=False,
                        error_message=train_result.error_message,
                        n_iterations=iteration + 1,
                        total_samples=len(self.dataset),
                        stopping_criterion=StoppingCriterion.USER_INTERRUPTED,
                    )

                # Phase 3: Evaluate surrogate and compute uncertainty
                logger.info("Phase 3: Evaluating surrogate model...")
                analysis = self._evaluate_surrogate()
                uncertainty_stats = self._compute_uncertainty_stats()

                # Get current error metrics
                test_error = train_result.metrics.get("relative_l2", float('inf'))
                if iteration == 0:
                    initial_error = test_error

                # Log metrics
                iteration_metrics = {
                    "iteration": iteration + 1,
                    "n_samples": len(self.dataset),
                    "train_loss": train_result.train_loss,
                    "test_loss": train_result.test_loss,
                    "test_error": test_error,
                    "n_weak_regions": len(analysis.weak_regions),
                    "mean_uncertainty": uncertainty_stats.get("mean_uncertainty", 0),
                    "max_uncertainty": uncertainty_stats.get("max_uncertainty", 0),
                }
                history.append(iteration_metrics)

                # Update metrics tracker
                self.metrics.log_iteration(
                    iteration=iteration + 1,
                    n_samples_total=len(self.dataset),
                    n_samples_new=self.config.samples_per_iteration if iteration > 0 else self.config.initial_samples,
                    train_error=train_result.train_loss,
                    test_error=test_error,
                    mean_uncertainty=uncertainty_stats.get("mean_uncertainty", 0),
                    max_uncertainty=uncertainty_stats.get("max_uncertainty", 0),
                    n_weak_regions=len(analysis.weak_regions),
                )

                if callback:
                    callback(iteration + 1, iteration_metrics)

                logger.info(f"  Test error: {test_error:.6f}")
                logger.info(f"  Mean uncertainty: {uncertainty_stats.get('mean_uncertainty', 0):.6f}")
                logger.info(f"  Weak regions: {len(analysis.weak_regions)}")

                # Check convergence criteria
                mean_uncertainty = uncertainty_stats.get("mean_uncertainty", 0)

                if self.budget_agent is not None:
                    budget_decision = self.budget_agent.decide_sync(
                        iteration=iteration + 1,
                        test_error=test_error,
                        mean_uncertainty=mean_uncertainty,
                        n_samples=len(self.dataset),
                        n_hpo_rounds=iteration_metrics.get("n_hpo_rounds", 0),
                    )
                    logger.info(f"BudgetAgent: {budget_decision.decision} — {budget_decision.reasoning[:100]}")
                    if budget_decision.should_stop() or budget_decision.should_switch_hpo():
                        stopping_criterion = (
                            StoppingCriterion.CONVERGED
                            if budget_decision.should_stop()
                            else StoppingCriterion.DIMINISHING_RETURNS
                        )
                        logger.info(f"Stopping per BudgetAgent: {stopping_criterion.name}")
                        break
                    n_new_samples = budget_decision.samples_next
                else:
                    stopping_criterion = self._check_convergence(
                        test_error=test_error,
                        n_samples=len(self.dataset),
                        n_weak_regions=len(analysis.weak_regions),
                        mean_uncertainty=mean_uncertainty,
                        iteration=iteration,
                    )
                    if stopping_criterion is not None:
                        logger.info(f"Stopping: {stopping_criterion.name}")
                        break
                    n_new_samples = self._compute_adaptive_budget(test_error)

                # Phase 3b: Mesh strategy (optional)
                if (
                    self.mesh_strategy_agent is not None
                    and self._coordinates is not None
                    and self.config.tip_coords is not None
                ):
                    node_uncertainties = self._get_node_uncertainties()
                    from piano.agents.base import AgentContext
                    mesh_ctx = AgentContext(iteration=iteration + 1)
                    mesh_decision = self.mesh_strategy_agent.decide_sync(
                        context=mesh_ctx,
                        coordinates=self._coordinates,
                        errors=node_uncertainties,
                        tip_coords=self.config.tip_coords,
                        iteration=iteration + 1,
                        n_samples=len(self.dataset),
                    )
                    logger.info(f"MeshStrategy: {mesh_decision.to_summary()}")
                    if mesh_decision.needs_h_refinement():
                        self._apply_h_refinement()

                # Phase 4: Select new samples
                logger.info("Phase 4: Selecting informative samples...")
                if self.config.use_agentic_proposer and self.proposer is not None:
                    new_params = self._select_informative_samples(n_new_samples)
                else:
                    new_params = self._suggest_new_parameters(analysis, budget=n_new_samples)

                _acq_name = "agentic_proposer" if self.config.use_agentic_proposer else self._acquisition_fn.name
                logger.info(f"  Selected {len(new_params)} new samples using {_acq_name}")

                # Phase 5: Run simulations for new samples
                logger.info("Phase 5: Running FEM simulations...")
                self._run_simulations(new_params)

            # Final stopping criterion if loop completed
            if stopping_criterion is None:
                stopping_criterion = StoppingCriterion.MAX_ITERATIONS

            # Save final artifacts
            logger.info("")
            logger.info("="*60)
            logger.info("SAVING RESULTS")
            logger.info("="*60)

            dataset_path = self.dataset.save()
            surrogate_path = self.surrogate_dir / "final_model"
            if self.surrogate:
                self.surrogate.save(surrogate_path)

            # Save metrics
            metrics_path = self.metrics_dir / "active_learning_metrics.json"
            self.metrics.save(metrics_path)

            # Compute final statistics
            final_error = history[-1]["test_error"] if history else float('inf')
            error_reduction = initial_error - final_error
            sample_efficiency = self.metrics.compute_efficiency()
            error_reduction_percent = (error_reduction / initial_error * 100) if initial_error > 0 else 0

            logger.info(f"Final error: {final_error:.6f}")
            logger.info(f"Error reduction: {error_reduction:.6f} ({error_reduction_percent:.1f}%)")
            logger.info(f"Sample efficiency: {sample_efficiency:.6f}")
            logger.info(f"Total samples: {len(self.dataset)}")
            logger.info(self.metrics.summary())

            return AdaptiveResult(
                success=True,
                n_iterations=len(history),
                final_error=final_error,
                initial_error=initial_error,
                total_samples=len(self.dataset),
                dataset_path=dataset_path,
                surrogate_path=surrogate_path,
                history=history,
                stopping_criterion=stopping_criterion,
                sample_efficiency=sample_efficiency,
                error_reduction_percent=error_reduction_percent,
                metrics_path=metrics_path,
            )

        except Exception as e:
            logger.exception("Active learning failed")
            return AdaptiveResult(
                success=False,
                error_message=str(e),
                n_iterations=len(history),
                total_samples=len(self.dataset) if self.dataset else 0,
                history=history,
                stopping_criterion=StoppingCriterion.USER_INTERRUPTED,
            )

    def _check_convergence(
        self,
        test_error: float,
        n_samples: int,
        n_weak_regions: int,
        mean_uncertainty: float,
        iteration: int
    ) -> Optional[StoppingCriterion]:
        """
        Check all convergence criteria.

        Returns StoppingCriterion if should stop, None otherwise.
        """
        min_improvement = 0.01  # Hardcoded sensible default

        # Check target error
        if test_error <= self.config.convergence_threshold:
            return StoppingCriterion.CONVERGED

        # Check sample budget
        if n_samples >= self.config.max_samples:
            return StoppingCriterion.BUDGET_EXHAUSTED

        # Check improvement (patience)
        improvement = self._best_error - test_error
        if improvement > min_improvement:
            self._best_error = test_error
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.config.patience:
            return StoppingCriterion.PATIENCE_EXHAUSTED

        # Check uncertainty — only after iteration 2 and only when we have a real signal
        # (0.0 means no ensemble yet, not that the model is already confident)
        if mean_uncertainty > 0 and iteration >= 2 and mean_uncertainty < self.config.convergence_threshold:
            return StoppingCriterion.LOW_UNCERTAINTY

        # Check diminishing returns
        if self.metrics.detect_diminishing_returns(
            window=self.config.patience,
            threshold=min_improvement
        ):
            return StoppingCriterion.DIMINISHING_RETURNS

        return None

    def _compute_adaptive_budget(self, current_error: float) -> int:
        """
        Dynamically compute number of samples for next iteration.

        Uses more samples when error is high, fewer as we converge.
        """
        base_samples = self.config.samples_per_iteration

        # Scale budget based on how far from convergence
        error_ratio = current_error / self.config.convergence_threshold

        if error_ratio > 5:
            # Far from convergence - use more samples
            scale = 1.5
        elif error_ratio > 2:
            # Moderate distance - standard budget
            scale = 1.0
        elif error_ratio > 1:
            # Close to convergence - use fewer, more targeted samples
            scale = 0.7
        else:
            # Very close - minimal samples
            scale = 0.5

        budget = int(base_samples * scale)
        budget = max(3, min(budget, base_samples * 2))

        return budget

    def _select_informative_samples(
        self,
        n_samples: int
    ) -> List[Dict[str, float]]:
        """
        Select new samples using LLM-based adaptive proposer.

        The AdaptiveProposerAgent analyzes uncertainty and weak regions,
        then proposes new simulation parameters with reasoning.
        """
        import asyncio

        if self.evaluator is None or self._coordinates is None:
            raise RuntimeError("Evaluator not initialized")

        if self.proposer is None:
            raise RuntimeError("AdaptiveProposer not initialized. Set use_agentic_proposer=True")

        # Get uncertainty analysis from evaluator
        analysis = self.evaluator.analyze_uncertainty(
            self.surrogate,
            self._coordinates
        )

        # Get current sample count
        n_current = len(self.dataset) if self.dataset else 0

        # Create agent context
        context = AgentContext()

        # Run async proposer synchronously
        loop = asyncio.new_event_loop()
        try:
            proposals = loop.run_until_complete(
                self.proposer.propose_targeted(
                    context=context,
                    uncertainty_analysis=analysis,
                    parameter_bounds=self.config.parameter_bounds,
                    n_samples=n_current,
                    n_valid=n_current,
                    n_proposals=n_samples,
                )
            )
        finally:
            loop.close()

        # Convert proposals to parameter dicts
        new_params = []
        for proposal in proposals:
            if proposal.parameters:
                # Ensure all required parameters are present
                param_dict = {}
                for name in self.config.parameter_names:
                    if name in proposal.parameters:
                        param_dict[name] = proposal.parameters[name]
                    else:
                        # Use midpoint of bounds if not specified
                        bounds = self.config.parameter_bounds[name]
                        param_dict[name] = (bounds[0] + bounds[1]) / 2
                new_params.append(param_dict)
                logger.info(f"  Proposal: {param_dict}")
                logger.info(f"    Reasoning: {proposal.reasoning[:100]}...")

        return new_params

    def _compute_uncertainty_stats(self) -> Dict[str, float]:
        """Compute uncertainty statistics across parameter space."""
        if self.evaluator is None or self._coordinates is None:
            return {"mean_uncertainty": 0.0, "max_uncertainty": 0.0}

        return self.evaluator.estimate_remaining_uncertainty(
            coordinates=self._coordinates,
            n_probe_samples=100
        )

    def _generate_initial_parameters(self) -> List[Dict[str, float]]:
        """Generate initial parameter samples using Latin Hypercube Sampling."""
        n_samples = self.config.initial_samples
        n_params = len(self.config.parameter_names)

        # Latin Hypercube Sampling for better coverage
        samples = []
        for i, name in enumerate(self.config.parameter_names):
            min_val, max_val = self.config.parameter_bounds[name]
            # Create stratified samples
            edges = np.linspace(min_val, max_val, n_samples + 1)
            points = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points)
            samples.append(points)

        samples = np.array(samples).T  # Shape: (n_samples, n_params)

        # Convert to list of dicts
        param_list = []
        for i in range(n_samples):
            params = {
                name: float(samples[i, j])
                for j, name in enumerate(self.config.parameter_names)
            }
            param_list.append(params)

        return param_list

    def _generate_random_parameters(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate random parameter samples."""
        param_list = []
        for _ in range(n_samples):
            params = {}
            for name in self.config.parameter_names:
                min_val, max_val = self.config.parameter_bounds[name]
                params[name] = np.random.uniform(min_val, max_val)
            param_list.append(params)
        return param_list

    def _suggest_new_parameters(
        self,
        analysis: UncertaintyAnalysis,
        budget: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Suggest new parameters based on uncertainty analysis and acquisition function."""
        n = budget or self.config.samples_per_iteration

        if self.evaluator is None:
            return self._generate_random_parameters(n)

        if self._coordinates is not None:
            return self.evaluator.suggest_samples_active(
                budget=n,
                coordinates=self._coordinates,
                acquisition_type=self._acquisition_fn.name,
            )

        return self.evaluator.suggest_samples(analysis, budget=n)

    def _run_simulations(self, param_list: List[Dict[str, float]]) -> None:
        """Run FEM simulations for given parameters."""
        for i, params in enumerate(param_list):
            logger.info(f"Running simulation {i+1}/{len(param_list)}: {params}")

            try:
                sample = self._run_single_simulation(params)
                self.dataset.add_sample(sample)
                logger.info(f"Sample {sample.sample_id} added (valid={sample.is_valid})")

            except Exception as e:
                logger.warning(f"Simulation failed for params {params}: {e}")
                # Add failed sample
                sample = FEMSample(
                    sample_id=f"failed_{i}_{np.random.randint(10000)}",
                    parameters=params,
                    coordinates=np.array([]),
                    is_valid=False,
                    metadata={"error": str(e)},
                )
                self.dataset.add_sample(sample)

    def _run_single_simulation(self, params: Dict[str, float]) -> FEMSample:
        return self._run_phase_field_simulation(params)

    def _run_phase_field_simulation(self, params: Dict[str, float]) -> FEMSample:
        """Run a single phase field FEA sample via DOLFINx."""
        from ..data.phase_field_generator import PhaseFieldFEMConfig, generate_phase_field_sample

        E            = params["E"]
        nu           = params["nu"]
        G_c          = params["G_c"]
        crack_length = params["crack_length"]
        load_ratio   = params["load_ratio"]

        # Griffith critical traction, scaled by load_ratio so every sample is near-critical
        Y      = 1.12
        K_Ic   = np.sqrt(E * G_c / (1.0 - nu ** 2))
        sigma_c = K_Ic / (Y * np.sqrt(np.pi * crack_length))
        traction = sigma_c * load_ratio

        cfg = PhaseFieldFEMConfig(
            geometry_type="edge_crack",
            crack_length=crack_length,
            resolution=self.config.phase_field_resolution,
            n_load_steps=self.config.phase_field_n_load_steps,
            output_dir=self.meshes_dir,
        )

        sample = generate_phase_field_sample(
            E=E, nu=nu, traction=traction, G_c=G_c, config=cfg
        )

        if sample is None or not sample.is_valid:
            raise RuntimeError(
                f"Phase field simulation failed — params: E={E:.2e}, nu={nu:.3f}, "
                f"traction={traction:.2e}, G_c={G_c:.0f}, crack={crack_length:.3f}"
            )

        # Expose traction and load_ratio so agents can reason about loading
        sample.parameters["traction"]   = float(traction)
        sample.parameters["load_ratio"] = float(load_ratio)
        return sample

    def _train_surrogate(self) -> TrainingResult:
        """Train surrogate on the current dataset."""
        return self._train_surrogate_phase_field()

    def _train_surrogate_phase_field(self) -> TrainingResult:
        """Phase field training path: Williams-enriched coords, stacked [u_x, u_y, log1p(σ_vm)]."""
        try:
            parameters, coords_list, disp_list = self.dataset.prepare_training_data(
                output_field="displacement", valid_only=True,
            )
            _, _, vm_list = self.dataset.prepare_training_data(
                output_field="von_mises", valid_only=True,
            )
        except ValueError as e:
            return TrainingResult(success=False, error_message=str(e))

        # Stack outputs: [u_x, u_y, log1p(σ_vm)] — (N, 3) per sample
        # log1p collapses the 4-decade stress spike at the tip to a smooth 1-decade field
        outputs_list = [
            np.concatenate([disp, np.log1p(vm)], axis=1).astype(np.float32)
            for disp, vm in zip(disp_list, vm_list)
        ]

        # Williams enrichment: append [r, log_r, sinθ, cosθ, sin(θ/2), cos(θ/2)] to (x,y)
        param_names = self.dataset.config.parameter_names
        cl_idx = param_names.index("crack_length")
        enriched_coords = [
            _williams_enrich(coords.astype(np.float32), float(param_row[cl_idx]))
            for coords, param_row in zip(coords_list, parameters)
        ]

        # Store enriched first-sample coords so the acquisition function can probe
        # the input space at inference time (variable-mesh: just use first sample)
        self._coordinates = enriched_coords[0]

        if self.config.use_agentic_hpo:
            return self._train_with_agentic_hpo(parameters, enriched_coords, outputs_list)
        return self._train_standard(parameters, enriched_coords, outputs_list)

    def _train_standard(
        self,
        parameters: np.ndarray,
        coordinates_list: List[np.ndarray],
        outputs_list: List[np.ndarray],
    ) -> TrainingResult:
        """Standard training without agentic HPO."""
        training_config = TrainingConfig(
            surrogate_config=self.config.surrogate_config,
            use_ensemble=self.config.n_ensemble > 1,
            n_ensemble=self.config.n_ensemble,
            normalize_inputs=True,
            normalize_outputs=True,
            train_test_split=0.2,
            random_seed=self.config.random_seed,
            save_dir=self.surrogate_dir,
        )

        self.trainer = SurrogateTrainer(training_config)
        result = self.trainer.train(parameters, coordinates_list, outputs_list)

        if result.success:
            self.surrogate = self.trainer.model

        return result

    def _train_with_agentic_hpo(
        self,
        parameters: np.ndarray,
        coordinates_list: List[np.ndarray],
        outputs_list: List[np.ndarray],
    ) -> TrainingResult:
        """Training with LLM-based hyperparameter optimization."""
        logger.info("Using agentic HPO for surrogate training")

        agentic_config = AgenticTrainingConfig(
            base_config=self.config.surrogate_config,
            max_hpo_rounds=self.config.max_hpo_rounds,
            trigger_threshold=self.config.convergence_threshold,
            use_ensemble=self.config.n_ensemble > 1,
            n_ensemble=self.config.n_ensemble,
            llm_model=self.config.llm_model,
            random_seed=self.config.random_seed,
        )

        agentic_trainer = AgenticSurrogateTrainer(
            config=agentic_config,
            llm_provider=self.llm_provider,
        )

        agentic_result = agentic_trainer.train(
            parameters, coordinates_list, outputs_list
        )

        if agentic_result.success and agentic_result.final_result:
            self.surrogate = agentic_trainer.model
            logger.info(
                f"Agentic HPO complete: {agentic_result.n_hpo_rounds} rounds, "
                f"{agentic_result.improvement_percent:.1f}% improvement"
            )
            return agentic_result.final_result
        else:
            return TrainingResult(
                success=False,
                error_message=agentic_result.error_message or "Agentic training failed",
            )

    def _evaluate_surrogate(self) -> UncertaintyAnalysis:
        """Evaluate surrogate model and identify weak regions using ensemble uncertainty."""
        if self.surrogate is None:
            return UncertaintyAnalysis()

        if self._coordinates is None:
            return UncertaintyAnalysis()

        # Initialize evaluator
        self.evaluator = SurrogateEvaluator(
            model=self.surrogate,
            parameter_names=self.config.parameter_names,
            parameter_bounds=self.config.parameter_bounds,
        )

        # Probe ensemble uncertainty across parameter space — no ground truth needed
        return self.evaluator.analyze_uncertainty(
            coordinates=self._coordinates,
            n_probe_samples=100,
            uncertainty_threshold=self.config.convergence_threshold,
        )

    def _get_node_uncertainties(self) -> np.ndarray:
        """Return per-node ensemble uncertainty at the median parameter point."""
        if self.surrogate is None or self._coordinates is None:
            return np.zeros(len(self._coordinates) if self._coordinates is not None else 0)

        # Probe at the parameter-space centre
        centre = np.array([[
            (lo + hi) / 2.0
            for lo, hi in (
                self.config.parameter_bounds[n]
                for n in self.config.parameter_names
            )
        ]])
        result = self.surrogate.predict(centre, self._coordinates)
        if result.uncertainty is not None:
            unc = result.uncertainty
            return unc.mean(axis=-1) if unc.ndim > 1 else unc
        return np.zeros(len(self._coordinates))

    def _apply_h_refinement(self) -> None:
        """Request finer resolution for subsequent phase-field samples."""
        new_res = min(self.config.phase_field_resolution + 10, 80)
        if new_res > self.config.phase_field_resolution:
            self.config.phase_field_resolution = new_res
            logger.info(f"h-refinement: phase_field_resolution → {new_res}")

    def get_dataset(self) -> FEMDataset:
        """Get the current dataset."""
        return self.dataset

    def get_surrogate(self):
        """Get the trained surrogate model."""
        return self.surrogate
