"""
Agentic Surrogate Trainer.

Wraps SurrogateTrainer with LLM-based hyperparameter optimization.
Uses a 4-agent system:
- HyperparameterCriticAgent: Diagnoses training issues
- ArchitectAgent: Proposes architecture/optimizer changes + flags code changes
- PhysicistAgent: Proposes physics loss configuration changes
- EngineerAgent: Implements code-level changes via Claude Code CLI
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from piano.surrogate.base import TransolverConfig, CrackConfig
from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig, TrainingResult

if TYPE_CHECKING:
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent,
        TrainingHistory,
        TrainingIssue,
    )
    from piano.agents.roles.architect import ArchitectAgent, ArchitectureProposal
    from piano.agents.roles.physicist import PhysicistAgent, PhysicsProposal


logger = logging.getLogger(__name__)


@dataclass
class AgenticTrainingConfig:
    """
    Configuration for agentic training.

    Attributes:
        base_config: Initial surrogate configuration
        max_hpo_rounds: Maximum HPO rounds per training session
        trigger_threshold: Error threshold to trigger HPO
        use_ensemble: Whether to use ensemble for uncertainty
        n_ensemble: Number of ensemble members
        llm_model: LLM model for agents
        random_seed: Random seed
        problem_type: Type of physics problem (crack, plate, thermal, etc.)
        has_singularity: Whether the problem has stress singularities
        use_engineer: Whether to invoke EngineerAgent for code-level changes
        working_dir: Working directory for EngineerAgent (defaults to cwd)
        n_candidates: Number of debate candidates per HPO round (1 = no ensemble)
        eval_epochs: Brief-training epochs for candidate selection
    """
    base_config: TransolverConfig = field(default_factory=TransolverConfig)
    max_hpo_rounds: int = 3
    trigger_threshold: float = 0.1
    use_ensemble: bool = True
    n_ensemble: int = 5
    llm_model: str = "claude-sonnet-4-6"
    random_seed: int = 42
    problem_type: str = "crack"
    has_singularity: bool = True
    use_engineer: bool = False
    working_dir: Optional[str] = None
    n_candidates: int = 1
    eval_epochs: int = 20
    use_data_analyst: bool = True
    data_analysis_report_path: Optional[str] = None
    tip_coords: Optional[Any] = None
    parameter_names: Optional[Any] = None
    crack_config: Optional[CrackConfig] = None


@dataclass
class AgenticTrainingResult:
    """
    Result of agentic training.

    Attributes:
        success: Whether training succeeded
        final_result: Final TrainingResult
        n_hpo_rounds: Number of HPO rounds executed
        config_history: History of configurations tried
        best_config: Best configuration found
        improvement_percent: Improvement from HPO
    """
    success: bool
    final_result: Optional[TrainingResult] = None
    n_hpo_rounds: int = 0
    config_history: List[Dict[str, Any]] = field(default_factory=list)
    best_config: Optional[TransolverConfig] = None
    improvement_percent: float = 0.0
    error_message: Optional[str] = None


class AgenticSurrogateTrainer:
    """
    Agentic wrapper for SurrogateTrainer.

    Implements adaptive HPO via 4-round structured agent debate:
      Round 1 (OBSERVATION)  — Analyst + Critic describe training state
      Round 2 (ANALYSIS)     — Architect + Physicist reason about root causes
      Round 3 (SYNTHESIS)    — Architect + Physicist propose concrete changes
      Round 4 (FINALIZATION) — Critic validates proposals before applying

    Features:
    - Multi-round debate prevents premature proposals
    - Peer validation catches capacity/physics ordering violations
    - History tracking: each round sees previous HPO attempts
    """

    def __init__(
        self,
        config: AgenticTrainingConfig,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize agentic trainer.

        Args:
            config: Agentic training configuration
            llm_provider: LLM provider for agents (optional)
        """
        self.config = config
        self.llm_provider = llm_provider

        # Lazy import agents to avoid circular imports
        from piano.agents.roles.hyperparameter_critic import HyperparameterCriticAgent
        from piano.agents.roles.architect import ArchitectAgent
        from piano.agents.roles.physicist import PhysicistAgent
        from piano.agents.roles.result_analyst import ResultAnalystAgent
        from piano.agents.roles.engineer import EngineerAgent

        # Initialize agents
        self.critic = HyperparameterCriticAgent(model=config.llm_model)
        self.architect = ArchitectAgent(model=config.llm_model)
        self.physicist = PhysicistAgent(model=config.llm_model)
        self.analyst = ResultAnalystAgent(model=config.llm_model)

        if llm_provider:
            self.critic.set_llm_provider(llm_provider)
            self.architect.set_llm_provider(llm_provider)
            self.physicist.set_llm_provider(llm_provider)
            self.analyst.set_llm_provider(llm_provider)

        # Data Analyst (Phase 0 — runs once before first training)
        self.data_analyst = None
        if config.use_data_analyst and llm_provider:
            from piano.agents.roles.data_analyst import DataAnalystAgent
            self.data_analyst = DataAnalystAgent(model=config.llm_model)
            self.data_analyst.set_llm_provider(llm_provider)
        self._data_analysis = None  # Cached result, shared via context

        # Knowledge Retriever (wired into debate)
        from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent
        self.knowledge_retriever = KnowledgeRetrieverAgent()

        # Debate orchestrator (4-round structured debate — always active)
        from piano.orchestration.debate import DebateOrchestrator
        self.debate = DebateOrchestrator(
            analyst=self.analyst,
            critic=self.critic,
            architect=self.architect,
            physicist=self.physicist,
            knowledge_retriever=self.knowledge_retriever,
        )

        # Selector Ensemble (3-LLM voting for multi-candidate selection)
        self.selector: Optional[Any] = None
        if llm_provider and config.n_candidates > 1:
            from piano.agents.roles.selector_ensemble import SelectorEnsembleAgent
            self.selector = SelectorEnsembleAgent(llm_provider=llm_provider)

        # Engineer agent (optional — uses Claude Code CLI independently of llm_provider)
        self.engineer: Optional[EngineerAgent] = None
        if config.use_engineer:
            import os
            working_dir = config.working_dir or os.getcwd()
            self.engineer = EngineerAgent(working_dir=working_dir)

        # State
        self._current_config = config.base_config
        self._config_history: List[Dict[str, Any]] = []
        self._best_result: Optional[TrainingResult] = None
        self._best_config: Optional[TransolverConfig] = None
        self._trainer: Optional[SurrogateTrainer] = None
        self._all_train_losses: List[float] = []
        self._all_test_losses: List[float] = []
        # Stored training data for brief-training during ensemble selection
        self._params: Optional[np.ndarray] = None
        self._coordinates: Optional[List[np.ndarray]] = None
        self._outputs: Optional[List[np.ndarray]] = None

    def train(
        self,
        parameters: np.ndarray,
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> AgenticTrainingResult:
        """
        Train surrogate with adaptive HPO.

        Args:
            parameters: Input parameters (N_samples, n_params)
            coordinates: Per-sample coordinates
            outputs: Per-sample outputs
            callback: Optional progress callback

        Returns:
            AgenticTrainingResult with final model and HPO history
        """
        try:
            return self._train_with_hpo(parameters, coordinates, outputs, callback)
        except Exception as e:
            logger.exception("Agentic training failed")
            return AgenticTrainingResult(
                success=False,
                error_message=str(e),
            )

    def _train_with_hpo(
        self,
        parameters: np.ndarray,
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> AgenticTrainingResult:
        """Internal training loop with HPO."""
        self._params = parameters
        self._coordinates = coordinates
        self._outputs = outputs
        dataset_size = len(parameters)
        initial_error = float('inf')
        n_rounds = 0

        # Phase 0: Dataset analysis (runs once, before first training)
        if self.data_analyst is not None and self._data_analysis is None:
            logger.info("Phase 0: Dataset analysis...")
            from pathlib import Path
            report_path = (
                Path(self.config.data_analysis_report_path)
                if self.config.data_analysis_report_path
                else None
            )
            tip = (
                np.array(self.config.tip_coords)
                if self.config.tip_coords is not None
                else None
            )
            try:
                self._data_analysis = self.data_analyst.analyze_sync(
                    coordinates=coordinates,
                    outputs=outputs,
                    parameters=parameters,
                    tip_coords=tip,
                    parameter_names=self.config.parameter_names,
                    report_path=report_path,
                )
                logger.info(
                    f"  Dataset quality: {self._data_analysis.dataset_quality}, "
                    f"near-tip fraction: {self._data_analysis.near_tip_fraction:.3f}"
                )
            except Exception as e:
                logger.warning(f"DataAnalystAgent failed (non-fatal): {e}")

        # Initial training
        logger.info("Starting initial training...")
        result = self._train_once(parameters, coordinates, outputs, callback)

        if not result.success:
            return AgenticTrainingResult(
                success=False,
                final_result=result,
                error_message=result.error_message,
            )

        initial_error = result.test_loss
        self._best_result = result
        self._best_config = self._current_config
        self._record_attempt(result, "initial")

        # Check if HPO is needed
        history = self._extract_history(result)

        if not self.critic.should_trigger_hpo(history, self.config.trigger_threshold):
            logger.info(f"Training converged well (loss={result.test_loss:.6f}), no HPO needed")
            return AgenticTrainingResult(
                success=True,
                final_result=result,
                n_hpo_rounds=0,
                config_history=self._config_history,
                best_config=self._current_config,
                improvement_percent=0.0,
            )

        # HPO loop
        logger.info(f"HPO triggered (threshold={self.config.trigger_threshold})")

        for round_idx in range(self.config.max_hpo_rounds):
            n_rounds = round_idx + 1
            logger.info(f"HPO Round {n_rounds}/{self.config.max_hpo_rounds}")

            # Get new config
            new_config = self._get_new_config(history, dataset_size)

            if new_config is None:
                logger.warning("Could not generate new config, stopping HPO")
                break

            self._current_config = new_config

            # Train with new config
            result = self._train_once(parameters, coordinates, outputs, callback)

            if not result.success:
                logger.warning(f"Training failed with new config: {result.error_message}")
                self._record_attempt(result, "failed")
                continue

            self._record_attempt(result, "success")
            history = self._extract_history(result)

            # Track best
            if result.test_loss < self._best_result.test_loss:
                logger.info(f"New best! {self._best_result.test_loss:.6f} -> {result.test_loss:.6f}")
                self._best_result = result
                self._best_config = self._current_config

            # Check if good enough
            if not self.critic.should_trigger_hpo(history, self.config.trigger_threshold):
                logger.info("HPO converged, stopping early")
                break

        # Compute improvement
        final_error = self._best_result.test_loss
        improvement = (initial_error - final_error) / initial_error * 100 if initial_error > 0 else 0

        logger.info(f"HPO complete: {n_rounds} rounds, {improvement:.1f}% improvement")

        return AgenticTrainingResult(
            success=True,
            final_result=self._best_result,
            n_hpo_rounds=n_rounds,
            config_history=self._config_history,
            best_config=self._best_config,
            improvement_percent=improvement,
        )

    def _train_once(
        self,
        parameters: np.ndarray,
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> TrainingResult:
        """Single training run with current config."""
        training_config = TrainingConfig(
            surrogate_config=self._current_config,
            use_ensemble=self.config.use_ensemble,
            n_ensemble=self.config.n_ensemble,
            normalize_inputs=True,
            normalize_outputs=True,
            train_test_split=0.2,
            random_seed=self.config.random_seed,
            crack_config=self.config.crack_config,
        )

        self._trainer = SurrogateTrainer(training_config)
        return self._trainer.train(parameters, coordinates, outputs, callback)

    def _extract_history(self, result: TrainingResult) -> "TrainingHistory":
        """Extract TrainingHistory from TrainingResult, accumulating across rounds."""
        from piano.agents.roles.hyperparameter_critic import TrainingHistory

        history = result.history or {}

        round_train = history.get('train_loss', [])
        round_test = history.get('test_loss', [])
        pino_losses = history.get('pino_loss', [])

        self._all_train_losses.extend(round_train)
        self._all_test_losses.extend(round_test)

        has_nan = any(np.isnan(l) for l in round_train + round_test)

        # Per-term PINO losses (only from the current round — most recent context)
        pino_term_losses = {}
        for key in ('elasticity_loss', 'crack_loss'):
            vals = history.get(key, [])
            if vals:
                pino_term_losses[key.replace('_loss', '')] = vals

        ensemble_std = history.get('ensemble_std', 0.0)

        return TrainingHistory(
            train_losses=self._all_train_losses,
            test_losses=self._all_test_losses,
            pino_losses=pino_losses,
            pino_term_losses=pino_term_losses,
            ensemble_std=ensemble_std,
            epochs_completed=len(self._all_train_losses),
            best_test_loss=min(self._all_test_losses) if self._all_test_losses else float('inf'),
            final_train_loss=round_train[-1] if round_train else float('inf'),
            final_test_loss=round_test[-1] if round_test else float('inf'),
            has_nan=has_nan,
            metrics=result.metrics,
        )

    def _get_new_config(
        self,
        history: "TrainingHistory",
        dataset_size: int,
    ) -> Optional[TransolverConfig]:
        """Get new config from LLM agents."""
        if self.llm_provider is None:
            raise RuntimeError("LLM provider is required for agentic HPO")

        try:
            return self._get_config_from_agents(history, dataset_size)
        except Exception as e:
            logger.error(f"Agent-based HPO failed: {e}")
            raise

    def _get_config_from_agents(
        self,
        history: "TrainingHistory",
        dataset_size: int,
    ) -> TransolverConfig:
        """Get new config via 4-round agent debate, with optional ensemble selection."""
        n_cand = self.config.n_candidates

        if n_cand > 1:
            # Ensemble: generate N candidates, select via SelectorEnsemble (or brief-train fallback)
            debate_results = self.debate.run_ensemble_debates_sync(
                history=history,
                current_config=self._current_config,
                dataset_size=dataset_size,
                config_history=self._config_history,
                problem_type=self.config.problem_type,
                has_singularity=self.config.has_singularity,
                n_candidates=n_cand,
            )
            if self.selector is not None:
                # SelectorEnsemble: 3-LLM majority vote
                history_summary = (
                    f"Test loss: {history.final_test_loss:.6f}, "
                    f"Train loss: {history.final_train_loss:.6f}, "
                    f"Issue: {history.best_test_loss:.6f} best"
                )
                sel_result = self.selector.select_sync(debate_results, history_summary)
                winner_idx = sel_result.selected_index
                winner_result = debate_results[winner_idx]
                best_config = self._merge_proposals(
                    winner_result.arch_proposal.config, winner_result.physics_changes
                )
                logger.info(
                    f"  SelectorEnsemble winner: candidate[{winner_idx}] "
                    f"(confidence={sel_result.confidence:.2f})"
                )
            else:
                # Fallback: brief-train each candidate
                best_config = None
                best_loss = float('inf')
                for i, result in enumerate(debate_results):
                    candidate = self._merge_proposals(result.arch_proposal.config, result.physics_changes)
                    brief_loss = self._train_brief(candidate)
                    logger.info(f"  Ensemble candidate {i+1}/{n_cand}: brief_loss={brief_loss:.6f}")
                    if brief_loss < best_loss:
                        best_loss = brief_loss
                        best_config = candidate
                        winner_result = result
                logger.info(f"  Ensemble winner (brief-train): loss={best_loss:.6f}")
            selected = winner_result
        else:
            selected = self.debate.run_debate_sync(
                history=history,
                current_config=self._current_config,
                dataset_size=dataset_size,
                config_history=self._config_history,
                problem_type=self.config.problem_type,
                has_singularity=self.config.has_singularity,
            )
            best_config = None  # set below

        arch_proposal = selected.arch_proposal

        if (
            self.engineer is not None
            and arch_proposal.code_change_description
            and arch_proposal.code_change_description.lower() != "none"
        ):
            logger.info(f"EngineerAgent: {arch_proposal.code_change_description[:120]}...")
            eng_result = self.engineer.implement_change_sync(
                change_description=arch_proposal.code_change_description,
                context_files=[
                    "piano/surrogate/deeponet.py",
                    "piano/surrogate/pino_loss.py",
                    "piano/surrogate/crack_pino_loss.py",
                    "piano/surrogate/trainer.py",
                ],
                validation_command="python -m pytest tests/ -x -q 2>&1 | tail -20",
            )
            if eng_result.success:
                logger.info(f"EngineerAgent: {eng_result.changes_made[:200]}")
            else:
                logger.warning(f"EngineerAgent failed: {eng_result.error}")

        if best_config is None:
            best_config = self._merge_proposals(arch_proposal.config, selected.physics_changes)
        return best_config

    def _train_brief(self, config: Any) -> float:
        """Brief-train a candidate config for eval_epochs; return final test loss."""
        if self._params is None:
            return float('inf')

        import copy
        brief_config = copy.deepcopy(config)
        brief_config.epochs = self.config.eval_epochs
        brief_config.patience = self.config.eval_epochs

        training_config = TrainingConfig(
            surrogate_config=brief_config,
            use_ensemble=False,
            normalize_inputs=True,
            normalize_outputs=True,
            train_test_split=0.2,
            random_seed=self.config.random_seed,
            crack_config=self.config.crack_config,
        )
        trainer = SurrogateTrainer(training_config)
        result = trainer.train(self._params, self._coordinates, self._outputs)
        return result.test_loss if result.success else float('inf')

    def _merge_proposals(
        self,
        arch_config: Any,
        physics_changes: Dict[str, Any],
    ) -> Any:
        """
        Merge architecture and physics proposals into a single config.

        The Physicist's changes override the arch config for all physics weights.
        The Architect owns only NN/optimizer params; it preserves but does not modify physics weights.
        """
        from piano.surrogate.deeponet import DeepONetConfig

        _physics_keys = {"energy", "equilibrium", "traction_free", "stress_intensity",
                         "near_tip", "j_integral"}

        config_dict = arch_config.to_dict()
        for key, value in physics_changes.items():
            if key in _physics_keys:
                config_dict[key] = value

        if config_dict.get("arch_type") == "deeponet":
            return DeepONetConfig(
                hidden_dim=config_dict.get('hidden_dim', 64),
                n_basis=config_dict.get('n_basis', 32),
                n_layers=config_dict.get('n_layers', 3),
                dropout=config_dict.get('dropout', 0.0),
                trunk_dropout=config_dict.get('trunk_dropout', 0.1),
                learning_rate=config_dict.get('learning_rate', 1e-3),
                batch_size=config_dict.get('batch_size', 4),
                epochs=config_dict.get('epochs', 200),
                patience=config_dict.get('patience', 50),
                optimizer_type=config_dict.get('optimizer_type', 'adamw'),
                scheduler_type=config_dict.get('scheduler_type', 'cosine'),
                activation=config_dict.get('activation', 'gelu'),
                output_dim=config_dict.get('output_dim', 1),
                energy=config_dict.get('energy', 0.0),
                equilibrium=config_dict.get('equilibrium', 0.0),
                tip_weight=config_dict.get('tip_weight', 0.0),
                stress_intensity=config_dict.get('stress_intensity', 0.0),
                traction_free=config_dict.get('traction_free', 0.0),
                near_tip=config_dict.get('near_tip', 0.0),
                j_integral=config_dict.get('j_integral', 0.0),
            )

        return TransolverConfig(
            slice_num=config_dict.get('slice_num', 32),
            n_heads=config_dict.get('n_heads', 8),
            d_model=config_dict.get('d_model', 256),
            n_layers=config_dict.get('n_layers', 6),
            mlp_ratio=config_dict.get('mlp_ratio', 4.0),
            dropout=config_dict.get('dropout', 0.0),
            learning_rate=config_dict.get('learning_rate', 1e-3),
            batch_size=config_dict.get('batch_size', 32),
            epochs=config_dict.get('epochs', 1000),
            patience=config_dict.get('patience', 100),
            output_dim=config_dict.get('output_dim', 3),
            energy=config_dict.get('energy', 0.0),
            equilibrium=config_dict.get('equilibrium', 0.0),
            stress_intensity=config_dict.get('stress_intensity', 0.0),
            traction_free=config_dict.get('traction_free', 0.0),
            near_tip=config_dict.get('near_tip', 0.0),
            j_integral=config_dict.get('j_integral', 0.0),
            optimizer_type=config_dict.get('optimizer_type', 'adamw'),
            scheduler_type=config_dict.get('scheduler_type', 'plateau'),
            activation=config_dict.get('activation', 'gelu'),
        )

    def _record_attempt(self, result: TrainingResult, status: str) -> None:
        """Record an HPO attempt."""
        self._config_history.append({
            "config": self._current_config.to_dict(),
            "changes": {},  # Could track specific changes here
            "result": f"{status}: loss={result.test_loss:.6f}" if result.success else status,
            "metrics": result.metrics if result.success else {},
        })

    @property
    def model(self):
        """Get the trained model from the best trainer."""
        return self._trainer.model if self._trainer else None

    @property
    def best_config(self) -> Optional[TransolverConfig]:
        """Get the best configuration found."""
        return self._best_config


async def train_with_agents_async(
    parameters: np.ndarray,
    coordinates: List[np.ndarray],
    outputs: List[np.ndarray],
    config: AgenticTrainingConfig,
    llm_provider: Any,
    callback: Optional[Callable[[int, float], None]] = None,
) -> AgenticTrainingResult:
    """
    Async version of agentic training.

    For use in async contexts like the orchestrator.
    """
    trainer = AgenticSurrogateTrainer(config, llm_provider)
    return trainer.train(parameters, coordinates, outputs, callback)
