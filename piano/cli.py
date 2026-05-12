#!/usr/bin/env python
"""
piano - CLI Entry Point

Subcommands:
  active-learn   Agentic active-learning loop (phase field FEA + Transolver surrogate)
  evolve         Legacy evolutionary MFEM tree search

Usage:
    piano active-learn --output outputs/run1 [options]
    piano evolve inputs/model.mesh [options]

Examples:
    piano active-learn --initial-samples 10 --max-samples 50 --agentic-hpo
    piano active-learn --output outputs/debug --initial-samples 5 --no-ensemble
    piano evolve inputs/model.mesh --generations 10 --population 5
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from piano.mesh.base import MeshManager
from piano.solvers.base import (
    SolverInterface,
    PhysicsType,
    PhysicsConfig,
    MaterialProperties,
)


def create_mesh_manager(file_path: Union[str, Path]) -> MeshManager:
    """
    Create a MeshManager from a file path.

    Args:
        file_path: Path to the mesh file (.mesh format)

    Returns:
        MeshManager instance
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == ".mesh":
        from piano.mesh.mfem_manager import MFEMManager
        return MFEMManager(file_path)
    else:
        raise ValueError(f"Unsupported mesh format: {extension}. Use .mesh files.")


def create_solver(
    physics_type: str = "elasticity",
    material: Optional[MaterialProperties] = None,
) -> tuple[SolverInterface, PhysicsConfig]:
    """
    Create an MFEM solver with physics configuration.

    Args:
        physics_type: Physics type ("elasticity" or "heat")
        material: Optional material properties

    Returns:
        Tuple of (SolverInterface, PhysicsConfig)
    """
    # Create physics configuration
    if physics_type.lower() in ("elasticity", "linear_elasticity"):
        physics_enum = PhysicsType.LINEAR_ELASTICITY
    elif physics_type.lower() in ("heat", "heat_transfer", "thermal"):
        physics_enum = PhysicsType.HEAT_TRANSFER
    else:
        raise ValueError(f"Unknown physics type: {physics_type}")

    if material is None:
        material = MaterialProperties()  # Use defaults

    physics = PhysicsConfig(
        physics_type=physics_enum,
        material=material,
    )

    # Create MFEM solver
    from piano.solvers.mfem_solver import MFEMSolver
    solver = MFEMSolver()

    return solver, physics


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(config_dir: Path = None) -> Dict[str, Any]:
    """
    Load all YAML configuration files from the configs directory.

    Returns a merged config dict with sections:
    - evolution: from evolution_config.yaml
    - agents: from agent_config.yaml
    """
    if config_dir is None:
        config_dir = Path(__file__).parent / "configs"

    config = {}

    # Load evolution config
    evolution_path = config_dir / "evolution_config.yaml"
    if evolution_path.exists():
        evolution_config = load_yaml_config(evolution_path)
        config.update(evolution_config)

    # Load agent config
    agent_path = config_dir / "agent_config.yaml"
    if agent_path.exists():
        agent_config = load_yaml_config(agent_path)
        config.update(agent_config)

    return config


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="piano — Agentic SciML for phase field fracture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand")

    # ── active-learn subcommand ───────────────────────────────────────────────
    al = sub.add_parser(
        "active-learn",
        help="Agentic active-learning loop: phase field FEA + Transolver + LLM agents",
    )
    al.add_argument("--output", "-o", default="outputs/active_learn",
                    help="Output directory (default: outputs/active_learn)")
    al.add_argument("--initial-samples", type=int, default=10,
                    help="Initial LHS samples (default: 10)")
    al.add_argument("--max-samples", type=int, default=50,
                    help="Total sample budget (default: 50)")
    al.add_argument("--convergence-threshold", type=float, default=0.05,
                    help="Relative L2 error target (default: 0.05)")
    al.add_argument("--n-ensemble", type=int, default=3,
                    help="Ensemble size for uncertainty (default: 3)")
    al.add_argument("--no-ensemble", action="store_true",
                    help="Disable ensemble (single model)")
    al.add_argument("--agentic-hpo", action="store_true",
                    help="Enable LLM-based hyperparameter optimisation")
    al.add_argument("--agentic-proposer", action="store_true",
                    help="Enable LLM-based active sample proposal")
    al.add_argument("--max-hpo-rounds", type=int, default=3)
    al.add_argument("--resolution", type=int, default=30,
                    help="Phase field mesh resolution (default: 30)")
    al.add_argument("--load-steps", type=int, default=20,
                    help="Phase field load steps (default: 20)")
    al.add_argument("--llm-model", default="claude-sonnet-4-6",
                    help="LLM model for agents")
    al.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    al.add_argument("--seed", type=int, default=42)

    # ── evolve subcommand (legacy) ────────────────────────────────────────────
    ev = sub.add_parser("evolve", help="Legacy evolutionary MFEM tree search")
    ev.add_argument("mesh_file", type=str, help="Path to .mesh file")
    ev.add_argument("--config", "-c", default=None)
    ev.add_argument("--generations", "-g", type=int, default=5)
    ev.add_argument("--population", "-p", type=int, default=5)
    ev.add_argument("--output", "-o", default="outputs")
    ev.add_argument("--run-solver", action="store_true")
    ev.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ev.add_argument("--dry-run", action="store_true")
    ev.add_argument("--test", action="store_true")
    ev.add_argument("--physics", choices=["elasticity", "heat"], default="elasticity")

    return parser.parse_args()


def run_active_learn(args) -> int:
    """Run the agentic active-learning loop with phase field FEA backend."""
    setup_logging(args.log_level)
    logger = logging.getLogger("piano.active_learn")

    from piano.orchestration.adaptive import AdaptiveOrchestrator, AdaptiveConfig
    from piano.agents.llm.anthropic_provider import AnthropicProvider

    # Phase field fracture parameter space:
    #   E, nu, G_c, crack_length sampled by LHS; load_ratio controls traction
    #   as a fraction of Griffith critical stress so every sample is near-critical.
    param_bounds = {
        "E":            (150e9, 250e9),
        "nu":           (0.25,  0.35),
        "G_c":          (1000.0, 5000.0),
        "crack_length": (0.20,  0.50),
        "load_ratio":   (0.60,  1.40),
    }

    config = AdaptiveConfig(
        output_dir=Path(args.output),
        parameter_bounds=param_bounds,
        initial_samples=args.initial_samples,
        max_samples=args.max_samples,
        convergence_threshold=args.convergence_threshold,
        patience=5,
        n_ensemble=1 if args.no_ensemble else args.n_ensemble,
        random_seed=args.seed,
        phase_field_resolution=args.resolution,
        phase_field_n_load_steps=args.load_steps,
        use_agentic_hpo=args.agentic_hpo,
        use_agentic_proposer=args.agentic_proposer,
        max_hpo_rounds=args.max_hpo_rounds,
        llm_model=args.llm_model,
    )

    llm_provider = None
    if args.agentic_hpo or args.agentic_proposer:
        try:
            llm_provider = AnthropicProvider(model=args.llm_model)
            logger.info(f"LLM provider: {args.llm_model}")
        except Exception as e:
            logger.warning(f"Could not initialise LLM provider: {e}. Agents disabled.")

    orchestrator = AdaptiveOrchestrator(config, llm_provider=llm_provider)

    logger.info("="*60)
    logger.info("piano  —  Agentic SciML active-learning loop")
    logger.info("="*60)
    logger.info(f"Backend:            Phase field FEA (DOLFINx AT-2)")
    logger.info(f"Surrogate:          Transolver (d_model=64, n_layers=3)")
    logger.info(f"Parameter space:    E, nu, G_c, crack_length, load_ratio")
    logger.info(f"Initial samples:    {args.initial_samples}")
    logger.info(f"Total budget:       {args.max_samples}")
    logger.info(f"Ensemble size:      {config.n_ensemble}")
    logger.info(f"Agentic HPO:        {args.agentic_hpo}")
    logger.info(f"Agentic proposer:   {args.agentic_proposer}")
    logger.info(f"Output:             {args.output}")
    logger.info("")

    def progress(iteration, metrics):
        logger.info(
            f"[iter {iteration}] samples={metrics['n_samples']}  "
            f"test_err={metrics['test_error']:.4f}  "
            f"uncertainty={metrics.get('mean_uncertainty', 0):.4f}"
        )

    result = orchestrator.run(callback=progress)

    logger.info("")
    if result.success:
        logger.info(f"Converged after {result.n_iterations} iterations")
        logger.info(f"Final error:   {result.final_error:.4f}")
        logger.info(f"Total samples: {result.total_samples}")
        logger.info(f"Dataset:       {result.dataset_path}")
        logger.info(f"Surrogate:     {result.surrogate_path}")
        return 0
    else:
        logger.error(f"Loop stopped: {result.stopping_criterion}")
        if result.error_message:
            logger.error(result.error_message)
        return 1


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.subcommand == "active-learn":
        return run_active_learn(args)

    if args.subcommand is None:
        print("Usage: piano <subcommand> [options]\n"
              "Subcommands: active-learn, evolve\n"
              "Run `piano <subcommand> --help` for details.")
        return 1

    # Load YAML configs
    yaml_config = load_configs()
    evolution_cfg = yaml_config.get("evolution", {})
    evaluation_cfg = yaml_config.get("evaluation", {})
    providers_cfg = yaml_config.get("providers", {})

    # CLI args override YAML config values
    generations = args.generations if args.generations != 5 else evolution_cfg.get("max_generations", 5)
    population = args.population if args.population != 5 else evolution_cfg.get("population_size", 5)
    output_dir = args.output if args.output != "outputs" else yaml_config.get("output", {}).get("dir", "outputs")

    # Setup logging
    log_file = Path(output_dir) / "logs" / "agentic.log"
    setup_logging(args.log_level, str(log_file))

    logger = logging.getLogger("AgenticSciML")
    logger.info("Starting AgenticSciML (MFEM)")
    logger.info(f"Loaded config from: configs/evolution_config.yaml, configs/agent_config.yaml")
    logger.info(f"Mesh file: {args.mesh_file}")
    logger.info(f"Generations: {generations}")
    logger.info(f"Population: {population}")
    logger.info(f"Physics: {args.physics}")

    # Validate inputs
    mesh_path = Path(args.mesh_file)
    if not mesh_path.exists():
        logger.error(f"Mesh file not found: {mesh_path}")
        return 1

    if mesh_path.suffix.lower() != ".mesh":
        logger.error(f"Invalid mesh format. Expected .mesh, got: {mesh_path.suffix}")
        return 1

    morphing_config_path = None
    if args.config:
        morphing_config_path = Path(args.config)
        if not morphing_config_path.exists():
            logger.error(f"Morphing config file not found: {morphing_config_path}")
            return 1

    # Import orchestrator
    try:
        from piano.orchestration.orchestrator import AgenticOrchestrator, OrchestrationConfig
        from piano.agents.llm.provider import create_provider, MockLLMProvider
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1

    # Create configuration from YAML + CLI overrides
    solver_cfg = evaluation_cfg.get("solver", {})
    debate_cfg = yaml_config.get("debate", {})
    claude_code_cfg = yaml_config.get("claude_code", {})

    config = OrchestrationConfig(
        max_generations=generations,
        population_size=population,
        num_parents=evolution_cfg.get("num_parents", 3),
        num_debate_rounds=debate_cfg.get("num_rounds", 4),
        consensus_threshold=debate_cfg.get("consensus_threshold", 0.7),
        max_attempts=3,
        run_solver=args.run_solver or solver_cfg.get("run_solver", False),
        solver_timeout=solver_cfg.get("timeout", 3600),
        openai_model=providers_cfg.get("openai", {}).get("default_model", "gpt-4-turbo"),
        anthropic_model=providers_cfg.get("anthropic", {}).get("default_model", "claude-3-opus-20240229"),
        # Claude Code settings (for Engineer and Debugger agents)
        use_claude_code=claude_code_cfg.get("enabled", True),
        claude_code_model=claude_code_cfg.get("model", "sonnet"),
        claude_code_max_turns=claude_code_cfg.get("max_turns", 25),
        claude_code_timeout=claude_code_cfg.get("timeout", 300),
        output_dir=output_dir,
    )

    logger.info(f"Config: debate_rounds={config.num_debate_rounds}, consensus={config.consensus_threshold}")

    # Create providers
    use_mock = args.test

    if use_mock:
        logger.info("Using mock LLM providers (--test mode)")
        # Comprehensive mock responses for all agent tasks
        openai_provider = MockLLMProvider({
            "propose": '''**Mutation Type**: morphing
**delta_R**: 0.75
**Reasoning**: Increasing hole radius by 30% to study stress concentration effects
**Expected Outcome**: Larger hole will increase stress concentration factor
**Risk Assessment**: Moderate - delta_R within safe bounds
**Material Changes**: None
**BC Changes**: None''',
            "critique": '''**Strengths**:
- Reasonable delta_R value within safe range
- No material changes reduces risk

**Weaknesses**:
- Could be more aggressive with morphing

**FINAL VOTE**: APPROVE''',
            "implement": '''**Mutation Type**: morphing
**delta_R**: 0.75
**Reasoning**: Synthesized from debate - increasing hole radius by 30%
**Expected Outcome**: Larger hole with good mesh quality
Implementation approach: Use morphing.py with delta_R=0.75''',
            "analysis": '''**Result Analysis**:
- Mesh quality: Good
- Jacobian minimum: 0.15
- Aspect ratio: 2.3
- Convergence: Expected''',
        })
        anthropic_provider = MockLLMProvider({
            "This is Round 4": '''**Summary of Debate**:
The proposal for morphing with delta_R=0.75 was discussed.

**Unresolved Issues**:
None significant.

**Final Assessment**:
The proposal meets all guidelines and poses acceptable risk.

**FINAL VOTE**: APPROVE

**Confidence**: high

**If Approved - Implementation Notes**:
- Use morphing.py with delta_R=0.75
- Run preflight check after implementation
- Monitor Jacobian near hole boundary''',
            "Respond to the Proposer": '''**Addressed Concerns**:
- The refinement adequately addresses previous feedback

**Remaining Issues**:
- None significant

**New Concerns**:
- None

**Assessment**:
The refined proposal is acceptable.

**Updated Vote**: LEAN_APPROVE''',
            "Critique the following mutation proposal": '''**Strengths**:
- Reasonable delta_R value (0.75)
- Within safe range [-1.0, 2.0]
- Clear reasoning for the change

**Concerns**:
- May require careful mesh quality monitoring

**Risk Analysis**:
- Mesh Quality Risk: low
- Convergence Risk: low
- Physical Plausibility: acceptable

**Preliminary Vote**: LEAN_APPROVE''',
            "Node Statistics": '''**Model Classification**:
Problem type: 2D plane stress with hole
Element type: Quad elements

**Geometry Analysis**:
- Plate with circular hole
- Symmetric geometry detected

**Mesh Quality Bounds**:
Minimum Jacobian: 0.1
Maximum aspect ratio: 10.0

**Allowable Modifications**:
delta_R: [-1.0, 2.0]''',
            "Generate a Guideline.md": '''# Guideline for 2D Plate with Hole Model

## Mesh Quality Requirements
- Minimum Jacobian determinant: 0.1
- Maximum aspect ratio: 10.0

## Morphing Parameters
- delta_R range: [-1.0, 2.0]
- Step size: 0.1

## Material Properties
- Young's modulus: 200 GPa
- Poisson's ratio: 0.3''',
            "diagnose": '''**Error Diagnosis**:
- Root cause: delta_R value too large causing element inversion

**Suggested Fix**:
- Reduce delta_R by 50%
- suggested_delta_R: 0.25''',
        })
    else:
        try:
            openai_provider = create_provider("openai")
            anthropic_provider = create_provider("anthropic")
            logger.info("LLM providers initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM providers: {e}")
            logger.warning("Using mock providers for testing (use --test to skip this)")
            use_mock = True
            openai_provider = MockLLMProvider({
                "propose": '**Mutation Type**: morphing\n**delta_R**: 0.5',
                "critique": '**FINAL VOTE**: APPROVE',
            })
            anthropic_provider = MockLLMProvider({
                "analyze": 'Problem type: 2D plane stress\nJacobian: 0.1\ndelta_R: [-1.0, 2.0]',
                "Guideline": '# Guideline\n## Mesh Quality\nJacobian > 0.1',
            })

    # Create orchestrator
    orchestrator = AgenticOrchestrator(
        config=config,
        openai_provider=openai_provider,
        anthropic_provider=anthropic_provider,
    )

    logger.info("Orchestrator created")

    if args.dry_run:
        logger.info("Dry run - initializing only")
        result = await orchestrator.initialize(
            base_inp_path=str(mesh_path),
            morphing_config_path=str(morphing_config_path) if morphing_config_path else None,
        )
        logger.info(f"Initialization result: {result.status.value}")
        return 0 if result.status.value == "completed" else 1

    # Run full evolution
    logger.info("Starting evolutionary run")

    try:
        result = await orchestrator.run(
            base_inp_path=str(mesh_path),
            morphing_config_path=str(morphing_config_path) if morphing_config_path else None,
            num_generations=generations,
        )

        # Log results
        logger.info(f"Run completed: {result['status']}")
        logger.info(f"Total generations: {result.get('total_generations', 0)}")
        logger.info(f"Total solutions: {result.get('total_solutions', 0)}")

        # Save summary
        summary_path = Path(args.output) / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Summary saved to: {summary_path}")

        # Print best solutions
        best_solutions = result.get("best_solutions", [])
        if best_solutions:
            logger.info("Best solutions:")
            for i, sol in enumerate(best_solutions[:3], 1):
                logger.info(f"  {i}. ID={sol['id'][:8]}... delta_R={sol['genome'].get('delta_R', 'N/A')}")

        return 0 if result["status"] == "completed" else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        orchestrator.save_state(str(Path(args.output) / "interrupted_state.json"))
        return 130

    except Exception as e:
        logger.error(f"Error during run: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
