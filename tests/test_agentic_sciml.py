"""
tests/test_agentic_sciml.py — Test the Agentic SciML Loop for Crack Problems.

Tests the complete agentic hyperparameter optimization pipeline:
  1. HyperparameterCriticAgent - diagnoses training issues
  2. ArchitectAgent - proposes architecture/optimizer changes
  3. PhysicistAgent - proposes physics loss configuration changes
  4. AgenticSurrogateTrainer - full training loop with 3-agent HPO

Focus: Static crack problems with stress singularity at crack tip.
The singularity (1/sqrt(r)) is challenging for neural operators and
benefits from agentic HPO to tune architecture and physics constraints.

Run:
    pytest tests/test_agentic_sciml.py -v
    pytest tests/test_agentic_sciml.py -v -m "not mfem"
    python tests/test_agentic_sciml.py [--n-samples N] [--epochs E]
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
CRACK_DATA = PROJECT_ROOT / "crack_data"

# Crack problem parameters
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "E": (150e9, 250e9),           # Young's modulus [Pa]
    "nu": (0.25, 0.35),            # Poisson's ratio
    "K_I": (1e6, 10e6),            # Mode I stress intensity factor [Pa*sqrt(m)]
    "crack_length": (0.2, 0.5),    # Crack length ratio a/W
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())


# =============================================================================
# Mock LLM Provider for Testing
# =============================================================================

class MockLLMResponse:
    """Mock LLM response object."""
    def __init__(self, content: str):
        self.content = content


class MockLLMProvider:
    """
    Mock LLM provider for testing agents without actual API calls.
    Simulates realistic responses for critic and architect agents.
    """

    def __init__(self, scenario: str = "underfitting"):
        """
        Initialize with a predefined scenario.
        For crack problems, underfitting is common due to singularity.
        """
        self.scenario = scenario
        self.call_count = 0
        self.call_history: List[Dict[str, str]] = []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4-turbo",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> MockLLMResponse:
        """Generate mock response based on scenario."""
        self.call_count += 1
        self.call_history.append({
            "system": system_prompt[:200],
            "user": user_prompt[:500],
        })

        if "reviewing a proposed configuration" in system_prompt.lower():
            return MockLLMResponse(self._critic_review_response(user_prompt))
        elif "curve analyst" in system_prompt.lower():
            return MockLLMResponse(self._analyst_response())
        elif "training analyst" in system_prompt.lower():
            return MockLLMResponse(self._critic_response(user_prompt))
        elif "equilibrium" in system_prompt.lower():
            return MockLLMResponse(self._physicist_response(user_prompt))
        elif "architect" in system_prompt.lower():
            return MockLLMResponse(self._architect_response(user_prompt))
        elif "scientific machine learning" in system_prompt.lower():
            return MockLLMResponse(self._proposer_response())
        else:
            return MockLLMResponse("Unknown agent type")

    def _critic_response(self, user_prompt: str = "") -> str:
        """Generate mock critic response based on actual training metrics in the prompt."""
        import re

        _responses = {
            "underfitting": """
DIAGNOSIS: The model is underfitting. Both training and test losses remain high. The neural operator lacks capacity to capture the stress singularity behavior.

PRIMARY_ISSUE: UNDERFITTING
SEVERITY: high

RECOMMENDATIONS:
- Increase d_model from 64 to 256 for more representational capacity
- Increase n_layers from 2 to 6 for deeper feature extraction
- Use SiLU activation instead of GELU (better for sharp gradients)
- Reduce PINO weight initially to allow data-driven learning of singularity

SHOULD_RETRAIN: true

METRICS_ANALYSIS:
- train_test_gap: Small but both losses are high
- convergence_rate: Plateaued at suboptimal level
""",
            "overfitting": """
DIAGNOSIS: Model is overfitting. Training loss is very low but test loss is much higher. The model memorizes training samples instead of learning general behavior.

PRIMARY_ISSUE: OVERFITTING
SEVERITY: high

RECOMMENDATIONS:
- Switch to DeepONet architecture which generalises better with small datasets
- Increase dropout to 0.2 for regularisation
- Reduce model capacity (d_model, n_layers)
- Decrease learning rate to slow down memorisation

SHOULD_RETRAIN: true

METRICS_ANALYSIS:
- train_test_gap: Very large — train/test ratio > 5
- convergence_rate: Train converged fast but test diverged
""",
            "slow_convergence": """
DIAGNOSIS: Training converges too slowly. Learning rate is too conservative.

PRIMARY_ISSUE: SLOW_CONVERGENCE
SEVERITY: medium

RECOMMENDATIONS:
- Increase learning rate from 1e-4 to 1e-3
- Use cosine scheduler with warm restarts
- Consider AdamW optimizer

SHOULD_RETRAIN: true
""",
            "stable": """
DIAGNOSIS: Training appears healthy. Model is learning the stress field reasonably well.

PRIMARY_ISSUE: NONE
SEVERITY: low

SHOULD_RETRAIN: false
""",
        }

        # Parse actual train/test loss from the prompt text
        train_match = re.search(r'Final train loss:\s*([\d.eE+\-]+)', user_prompt)
        test_match  = re.search(r'Final test loss:\s*([\d.eE+\-]+)', user_prompt)

        if train_match and test_match:
            try:
                train_loss = float(train_match.group(1))
                test_loss  = float(test_match.group(1))
                ratio = test_loss / (train_loss + 1e-12)

                if ratio > 5.0:
                    return _responses["overfitting"]
                elif train_loss < 0.01 and test_loss < 0.05:
                    return _responses["stable"]
                elif train_loss > 0.1 and test_loss > 0.1:
                    return _responses["underfitting"]
                elif test_loss < 0.5:
                    return _responses["stable"]
            except ValueError:
                pass

        return _responses.get(self.scenario, _responses["underfitting"])

    def _critic_review_response(self, user_prompt: str = "") -> str:
        """Review an Architect proposal — approve if it addresses the diagnosed issue."""
        import re
        issue = self._parse_issue(user_prompt)
        # Check whether the proposal contains a key relevant change
        has_arch_switch = "deeponet" in user_prompt.lower() or "transolver" in user_prompt.lower()
        has_dropout = re.search(r'dropout:\s*0\.[1-9]', user_prompt) is not None
        has_capacity_cut = re.search(r'hidden_dim:\s*[0-9]+|d_model:\s*[0-9]+', user_prompt) is not None

        if issue == "overfitting" and not (has_arch_switch or has_dropout):
            return (
                "FEASIBLE: no\n"
                "CONCERNS: Proposal does not add regularization — overfitting will persist.\n"
                "SUGGESTION: Add dropout ≥ 0.1 or reduce model capacity.\n"
            )
        return (
            "FEASIBLE: yes\n"
            "CONCERNS: none\n"
            "SUGGESTION: none\n"
        )

    def _analyst_response(self) -> str:
        """Generate mock analyst observation."""
        return """OVERALL_PATTERN: underfitting
SEVERITY: high
OBSERVATION: Both train and test losses remain elevated (train=0.45, test=0.48). The model has not made meaningful progress — test loss improved only 5% over 80 epochs. The train-test gap is small (0.03), ruling out overfitting.
PINO_STATUS: no PINO terms active — all weights are zero.
ENSEMBLE_STATUS: ensemble_std = 0.03, ratio ≈ 0.06 (low variance, members agree on poor predictions)."""

    def _parse_issue(self, user_prompt: str) -> str:
        """Extract the primary issue from the architect/physicist user prompt."""
        import re
        match = re.search(r'\*\*Primary Issue\*\*:\s*(\w+)', user_prompt)
        if match:
            return match.group(1).lower()
        for issue in ("overfitting", "underfitting", "slow_convergence",
                      "unstable_training", "loss_plateau", "stable"):
            if issue in user_prompt.lower():
                return issue
        return self.scenario

    def _architect_response(self, user_prompt: str = "") -> str:
        """Generate mock architect response based on the actual diagnosed issue."""
        issue = self._parse_issue(user_prompt)
        call_count = getattr(self, "_arch_call_count", 0)
        self._arch_call_count = call_count + 1

        overfitting_configs = [
            # Round 1: switch to DeepONet — correct architecture for fixed-geometry small data
            """
REASONING: Overfitting with fixed geometry and small dataset (<100 samples). Transolver has too many parameters for this regime. Switch to DeepONet whose branch-trunk structure generalises much better with 20-50 samples.

CHANGES:
- arch_type: deeponet
- hidden_dim: 64
- n_basis: 32
- n_layers: 3
- dropout: 0.1
- learning_rate: 5e-4
- optimizer_type: adamw
- scheduler_type: cosine

EXPECTED_IMPACT: DeepONet separates parameter dependence (branch) from spatial basis (trunk), reducing effective output dimensionality and closing the train/test gap.
TRADE_OFFS: No PINO physics loss support; trunk learns spatial patterns from data only.
CONFIDENCE: high
""",
            # Round 2: tune DeepONet — increase n_basis, add more dropout
            """
REASONING: DeepONet still overfitting — increase n_basis for richer spatial basis and add stronger dropout.

CHANGES:
- arch_type: deeponet
- hidden_dim: 64
- n_basis: 48
- n_layers: 3
- dropout: 0.2
- learning_rate: 2e-4
- optimizer_type: adamw
- scheduler_type: cosine

EXPECTED_IMPACT: More basis functions improve spatial expressiveness; stronger dropout regularises the branch network.
TRADE_OFFS: Slightly more parameters but still far fewer than Transolver.
CONFIDENCE: high
""",
            # Round 3: reduce hidden_dim to cut capacity further
            """
REASONING: Persistent overfitting — reduce hidden_dim to limit branch and trunk capacity.

CHANGES:
- arch_type: deeponet
- hidden_dim: 32
- n_basis: 32
- n_layers: 3
- dropout: 0.25
- learning_rate: 1e-4
- optimizer_type: adamw
- scheduler_type: cosine

EXPECTED_IMPACT: Smaller MLPs generalise better under severe data scarcity.
TRADE_OFFS: Risk of underfitting if reduced too far.
CONFIDENCE: medium
""",
        ]
        underfitting_configs = [
            """
REASONING: Add PINO physics constraint to regularize underfitting on small dataset.

CHANGES:
- d_model: 64
- n_layers: 2
- n_heads: 2
- slice_num: 8
- learning_rate: 8e-4
- dropout: 0.0
- energy: 0.05

EXPECTED_IMPACT: Physics regularization helps generalization.
CONFIDENCE: medium
""",
            """
REASONING: Still underfitting — increase model capacity moderately.

CHANGES:
- d_model: 48
- n_layers: 3
- n_heads: 4
- slice_num: 16
- activation: silu
- learning_rate: 5e-4
- dropout: 0.0
- epochs: 150
- energy: 0.05

EXPECTED_IMPACT: Larger model with physics captures stress concentration.
CONFIDENCE: medium
""",
            """
REASONING: Need longer training to escape plateau.

CHANGES:
- d_model: 48
- n_layers: 3
- n_heads: 4
- slice_num: 16
- learning_rate: 2e-4
- dropout: 0.0
- epochs: 200
- scheduler_type: cosine
- energy: 0.1

EXPECTED_IMPACT: Lower lr with more epochs finds better minimum.
CONFIDENCE: medium
""",
        ]
        responses = {
            "overfitting": overfitting_configs[min(call_count, len(overfitting_configs) - 1)],
            "underfitting": underfitting_configs[min(call_count, len(underfitting_configs) - 1)],
            "slow_convergence": """
REASONING: Need faster learning dynamics.

CHANGES:
- learning_rate: 3e-4
- scheduler_type: cosine
- optimizer_type: adamw

CONFIDENCE: medium
""",
            "unstable_training": """
REASONING: Reduce learning rate and increase batch size to stabilize training.

CHANGES:
- learning_rate: 1e-4
- batch_size: 8
- scheduler_type: plateau

CONFIDENCE: medium
""",
            "stable": """
REASONING: Training is stable — minor tuning only.

CHANGES:
- epochs: 50

CONFIDENCE: medium
""",
        }
        return responses.get(issue, underfitting_configs[min(call_count, len(underfitting_configs) - 1)])

    def _physicist_response(self, user_prompt: str = "") -> str:
        """
        Mock physicist: reads current active physics weights from prompt and
        advances through the sequential enabling schedule each call.
        Schedule: equilibrium → energy → traction_free → stress_intensity → near_tip
        """
        import re

        def _pw(name):
            m = re.search(rf'- {name}:\s*([\d.e+\-]+)', user_prompt)
            return float(m.group(1)) if m else 0.0

        eq = _pw('equilibrium')
        en = _pw('energy')
        tf = _pw('traction_free')
        si = _pw('stress_intensity')
        nt = _pw('near_tip')

        ratio_m = re.search(r'Latest ratio:\s*([\d.]+)%', user_prompt)
        ratio = float(ratio_m.group(1)) / 100.0 if ratio_m else 0.0

        # Priority 1: ratio too high — reduce all active weights by 4×
        if ratio > 0.10:
            lines = []
            for name, val in [('equilibrium', eq), ('energy', en),
                               ('traction_free', tf), ('stress_intensity', si), ('near_tip', nt)]:
                if val > 0:
                    lines.append(f'- {name}: {val / 4:.4e}')
            changes = '\n'.join(lines) if lines else ''
            return f"""
PHYSICS_DIAGNOSIS: Ratio {ratio:.1%} above 10% — reducing all active weights by 4×.
CHANGES:
{changes}
REASONING: Physics overriding FEM labels. Reducing to restore data dominance.
EXPECTED_IMPACT: Ratio drops to ~{ratio / 4:.1%}.
CONFIDENCE: high
"""

        # Priority 2: sequential enabling — advance to next inactive term
        if eq == 0.0:
            changes = "- equilibrium: 0.001"
            diag = "No physics active. Enabling equilibrium PDE residual at 1e-3."
        elif en == 0.0:
            changes = f"- equilibrium: {eq}\n- energy: 0.0005"
            diag = "Equilibrium stable. Enabling elastic energy norm at 5e-4."
        elif tf == 0.0:
            changes = f"- equilibrium: {eq}\n- energy: {en}\n- traction_free: 0.001"
            diag = "Equilibrium + energy stable. Enabling traction-free crack face BC at 1e-3."
        elif si == 0.0:
            changes = (f"- equilibrium: {eq}\n- energy: {en}\n"
                       f"- traction_free: {tf}\n- stress_intensity: 0.001")
            diag = "Traction-free BC stable. Enabling K_I consistency at 1e-3."
        elif nt == 0.0:
            changes = (f"- equilibrium: {eq}\n- energy: {en}\n- traction_free: {tf}\n"
                       f"- stress_intensity: {si}\n- near_tip: 0.001")
            diag = "K_I stable. Enabling near-tip Williams expansion at 1e-3."
        else:
            changes = ""
            diag = "All crack terms active and ratio in target range. Holding."

        return f"""
PHYSICS_DIAGNOSIS: {diag}
CHANGES:
{changes}
REASONING: Sequential enabling — each term stabilises before the next is added.
EXPECTED_IMPACT: Gradual physics regularization improves notch-tip generalization.
CONFIDENCE: high
"""

    def _proposer_response(self) -> str:
        """Generate mock adaptive proposer response."""
        # Generate proposals based on crack problem parameters
        return """
**Proposal 1**
Parameters: E=200e9, nu=0.30, K_I=6e6, crack_length=0.35
Target Region: High uncertainty near crack tip with long crack
Reasoning: The model shows highest uncertainty in the region with crack_length > 0.3 and high K_I values. This sample targets the singularity-dominated regime where the 1/sqrt(r) behavior is most pronounced.
Expected Improvement: Should reduce error in high-stress intensity region by ~15%
Priority: High

**Proposal 2**
Parameters: E=180e9, nu=0.28, K_I=4e6, crack_length=0.25
Target Region: Moderate crack length with lower stiffness
Reasoning: Current dataset lacks samples with lower Young's modulus combined with moderate crack lengths. This configuration will help the model generalize across material stiffness variations.
Expected Improvement: Better generalization for softer materials
Priority: Medium

**Proposal 3**
Parameters: E=220e9, nu=0.32, K_I=8e6, crack_length=0.45
Target Region: Long crack with high stress intensity
Reasoning: Extreme case near parameter bounds. The model needs exposure to near-critical crack configurations to accurately predict failure-prone scenarios.
Expected Improvement: Improved accuracy at parameter space boundaries
Priority: High

**Proposal 4**
Parameters: E=190e9, nu=0.30, K_I=3e6, crack_length=0.30
Target Region: Central parameter space with low K_I
Reasoning: Filling coverage gap in the low stress intensity region with moderate crack length.
Expected Improvement: More uniform error distribution across parameter space
Priority: Medium

**Proposal 5**
Parameters: E=210e9, nu=0.33, K_I=7e6, crack_length=0.40
Target Region: High Poisson's ratio with long crack
Reasoning: Higher Poisson's ratio affects stress distribution near crack tip. This sample explores material incompressibility effects on fracture behavior.
Expected Improvement: Better capture of Poisson's ratio sensitivity
Priority: Medium
"""


# =============================================================================
# Crack-Specific Synthetic Data Generation
# =============================================================================

def _generate_crack_mesh(n_points: int = 500, crack_length: float = 0.3,
                          crack_angle: float = 0.0, rng: np.random.Generator = None):
    """
    Generate synthetic mesh for plate with edge crack.
    Returns coordinates and triangulation.
    """
    from scipy.spatial import Delaunay

    if rng is None:
        rng = np.random.default_rng(42)

    # Crack tip position
    angle_rad = np.radians(crack_angle)
    tip_x = crack_length * np.cos(angle_rad)
    tip_y = 0.5 + crack_length * np.sin(angle_rad)

    points = []

    # Boundary points
    n_boundary = 40
    for i in range(n_boundary):
        t = i / n_boundary
        points.append([t, 0.0])
        points.append([1.0, t])
        points.append([1.0 - t, 1.0])
        points.append([0.0, 1.0 - t])

    # Interior points avoiding crack
    while len(points) < n_points:
        p = rng.uniform(0, 1, size=(2,))
        # Check if point is on crack line (from origin to tip)
        if p[0] < tip_x:
            # Distance from crack line
            if abs(p[1] - 0.5) > 0.02:  # Not too close to crack
                points.append(p.tolist())
        else:
            points.append(p.tolist())

    # Refined points near crack tip
    for level in range(4):
        r = 0.1 * (0.5 ** level)
        n_ring = 12 * (level + 1)
        for i in range(n_ring):
            theta = 2 * np.pi * i / n_ring
            x = tip_x + r * np.cos(theta)
            y = tip_y + r * np.sin(theta)
            if 0 < x < 1 and 0 < y < 1:
                points.append([x, y])

    # Points along crack faces
    for i in range(1, 20):
        t = i / 20 * crack_length
        x = t * np.cos(angle_rad)
        y_base = 0.5 + t * np.sin(angle_rad)
        points.append([x, y_base + 0.01])  # Upper face
        points.append([x, y_base - 0.01])  # Lower face

    coords = np.array(points, dtype=np.float32)

    # Triangulate
    tri = Delaunay(coords)

    # Filter triangles crossing crack
    valid = []
    for simplex in tri.simplices:
        centroid = coords[simplex].mean(axis=0)
        # Skip if centroid is on crack
        if centroid[0] < tip_x and abs(centroid[1] - 0.5) < 0.015:
            continue
        valid.append(simplex)

    return coords, np.array(valid)


def _williams_displacement(coords: np.ndarray, params: Dict,
                            tip_x: float, tip_y: float = 0.5) -> np.ndarray:
    """
    Generate Williams expansion displacement field near crack tip.
    This is the analytical solution for mode I crack tip displacement.

    u_x = K_I / (2*mu) * sqrt(r/(2*pi)) * cos(theta/2) * (kappa - 1 + 2*sin^2(theta/2))
    u_y = K_I / (2*mu) * sqrt(r/(2*pi)) * sin(theta/2) * (kappa + 1 - 2*cos^2(theta/2))

    where kappa = (3 - nu)/(1 + nu) for plane stress
    """
    E = params.get("E", 200e9)
    nu = params.get("nu", 0.3)
    K_I = params.get("K_I", 5e6)  # Stress intensity factor

    mu = E / (2 * (1 + nu))  # Shear modulus
    kappa = (3 - nu) / (1 + nu)  # Plane stress

    n_points = len(coords)
    disp = np.zeros((n_points, 2), dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        # Distance and angle from crack tip
        dx = x - tip_x
        dy = y - tip_y
        r = np.sqrt(dx**2 + dy**2) + 1e-10  # Avoid division by zero
        theta = np.arctan2(dy, dx)

        # Williams expansion (mode I)
        sqrt_r = np.sqrt(r / (2 * np.pi))
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)

        # Displacement components
        ux = K_I / (2 * mu) * sqrt_r * cos_half * (kappa - 1 + 2 * sin_half**2)
        uy = K_I / (2 * mu) * sqrt_r * sin_half * (kappa + 1 - 2 * cos_half**2)

        disp[i, 0] = ux
        disp[i, 1] = uy

    return disp


# =============================================================================
# 1. HyperparameterCriticAgent Tests
# =============================================================================

def test_critic_detect_issues_heuristic_overfitting():
    """Test heuristic overfitting detection."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory, TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

    # Smooth overfitting curve
    train_losses = [0.5 - 0.02 * i for i in range(25)]
    test_losses = [0.5 - 0.01 * i if i < 10 else 0.4 + 0.02 * (i - 10) for i in range(25)]

    history = TrainingHistory(
        train_losses=train_losses,
        test_losses=test_losses,
        epochs_completed=25,
        best_test_loss=min(test_losses),
        final_train_loss=train_losses[-1],
        final_test_loss=test_losses[-1],
    )

    issues = critic.detect_issues_heuristic(history)
    assert TrainingIssue.OVERFITTING in issues


def test_critic_detect_issues_heuristic_underfitting():
    """Test heuristic underfitting detection."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory, TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

    # Both losses high and plateaued
    history = TrainingHistory(
        train_losses=[0.5, 0.49, 0.48, 0.48, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47] * 2,
        test_losses=[0.52, 0.51, 0.50, 0.50, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49] * 2,
        epochs_completed=20,
        best_test_loss=0.49,
        final_train_loss=0.47,
        final_test_loss=0.49,
    )

    issues = critic.detect_issues_heuristic(history)
    assert TrainingIssue.UNDERFITTING in issues


def test_critic_detect_issues_heuristic_plateau():
    """Test heuristic loss plateau detection."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory, TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

    history = TrainingHistory(
        train_losses=[0.1] * 20,
        test_losses=[0.12] * 20,
        epochs_completed=20,
        best_test_loss=0.12,
        final_train_loss=0.1,
        final_test_loss=0.12,
    )

    issues = critic.detect_issues_heuristic(history)
    assert TrainingIssue.LOSS_PLATEAU in issues


def test_critic_detect_nan():
    """Test NaN detection."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory, TrainingIssue,
    )

    critic = HyperparameterCriticAgent()

    history = TrainingHistory(
        train_losses=[0.5, 0.4, float('nan')],
        test_losses=[0.5, 0.45, float('nan')],
        epochs_completed=3,
        has_nan=True,
    )

    issues = critic.detect_issues_heuristic(history)
    assert TrainingIssue.GRADIENT_EXPLOSION in issues


def test_critic_should_trigger_hpo():
    """Test HPO trigger logic."""
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory,
    )

    critic = HyperparameterCriticAgent()

    # Good training - should NOT trigger
    good_history = TrainingHistory(
        train_losses=[0.1, 0.05, 0.02, 0.01, 0.005],
        test_losses=[0.12, 0.06, 0.03, 0.015, 0.008],
        epochs_completed=5,
        final_test_loss=0.008,
    )
    assert not critic.should_trigger_hpo(good_history, threshold=0.01)

    # Bad training - SHOULD trigger
    bad_history = TrainingHistory(
        train_losses=[0.5, 0.4, 0.3, 0.2, 0.1],
        test_losses=[0.5, 0.5, 0.55, 0.6, 0.7],
        epochs_completed=5,
        final_test_loss=0.7,
        final_train_loss=0.1,
    )
    assert critic.should_trigger_hpo(bad_history, threshold=0.1)


@pytest.mark.asyncio
async def test_critic_analyze_training_with_mock_llm():
    """Test critic analysis with mock LLM."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory, TrainingIssue,
    )

    critic = HyperparameterCriticAgent()
    provider = MockLLMProvider(scenario="underfitting")
    critic.set_llm_provider(provider)

    context = AgentContext()
    history = TrainingHistory(
        train_losses=[0.5, 0.45, 0.42, 0.40],
        test_losses=[0.55, 0.50, 0.48, 0.47],
        epochs_completed=4,
        final_train_loss=0.40,
        final_test_loss=0.47,
    )

    result = await critic.analyze_training(
        context=context,
        training_history=history,
        config={"d_model": 64, "n_layers": 2},
    )

    assert result.primary_issue == TrainingIssue.UNDERFITTING
    assert result.severity == "high"
    assert result.should_retrain is True
    assert provider.call_count == 1


# =============================================================================
# 2. ArchitectAgent Tests
# =============================================================================

@pytest.mark.asyncio
async def test_architect_propose_config_underfitting():
    """Test architect proposes config for underfitting (crack tip singularity)."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.roles.hyperparameter_critic import CritiqueResult, TrainingIssue
    from piano.surrogate.base import TransolverConfig

    architect = ArchitectAgent()
    provider = MockLLMProvider(scenario="underfitting")
    architect.set_llm_provider(provider)

    context = AgentContext()
    current_config = TransolverConfig(
        d_model=64, n_layers=2, dropout=0.0, learning_rate=1e-4
    )
    critique = CritiqueResult(
        primary_issue=TrainingIssue.UNDERFITTING,
        severity="high",
        diagnosis="Model cannot capture crack tip singularity",
        recommendations=["Increase capacity", "Use SiLU activation"],
        should_retrain=True,
    )

    proposal = await architect.propose_config(
        context=context,
        current_config=current_config,
        critique=critique,
        dataset_size=10,
    )

    assert proposal.config is not None
    assert proposal.changes.get("d_model", 64) >= 64
    assert len(proposal.reasoning) > 0


def test_architect_apply_changes():
    """Test architect applies changes correctly."""
    from piano.agents.roles.architect import ArchitectAgent
    from piano.surrogate.base import TransolverConfig

    architect = ArchitectAgent()

    base_config = TransolverConfig(
        d_model=64, n_layers=2, dropout=0.0, learning_rate=1e-4,
    )

    changes = {
        "d_model": 256,
        "n_layers": 6,
        "dropout": 0.05,
        "learning_rate": 1e-3,
    }

    new_config = architect.apply_changes(base_config, changes)

    assert new_config.d_model == 256
    assert new_config.n_layers == 6
    assert new_config.dropout == 0.05
    assert new_config.learning_rate == 1e-3


# =============================================================================
# 2b. Physicist Agent Tests
# =============================================================================

@pytest.mark.asyncio
async def test_physicist_propose_physics_config():
    """Test physicist proposes physics loss configuration."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.agents.roles.hyperparameter_critic import CritiqueResult, TrainingIssue

    physicist = PhysicistAgent()
    provider = MockLLMProvider(scenario="underfitting")
    physicist.set_llm_provider(provider)

    context = AgentContext()
    current_config = {
        "equilibrium": 0.0,
        "energy": 0.0,
        "traction_free": 0.0,
        "stress_intensity": 0.0,
        "near_tip": 0.0,
        "j_integral": 0.0,
    }
    critique = CritiqueResult(
        primary_issue=TrainingIssue.UNDERFITTING,
        severity="high",
        diagnosis="Model cannot capture crack tip singularity",
        recommendations=["Wait for surrogate to stabilize before enabling physics terms"],
        should_retrain=True,
    )

    proposal = await physicist.propose_physics_config(
        context=context,
        current_config=current_config,
        critique=critique,
        dataset_size=10,
        problem_type="crack",
        has_singularity=True,
    )

    assert proposal.changes is not None
    assert len(proposal.physics_diagnosis) > 0
    # equilibrium is always-safe (no K_I dependency) — physicist enables it immediately
    assert proposal.changes.get("equilibrium", 0.0) > 0.0
    # K_I-dependent terms must stay off until equilibrium is stable
    assert proposal.changes.get("stress_intensity", 0.0) == 0.0
    assert proposal.changes.get("near_tip", 0.0) == 0.0
    assert proposal.changes.get("j_integral", 0.0) == 0.0


def test_physicist_detect_physics_issues():
    """Test physicist heuristic issue detection for weight calibration states."""
    from piano.agents.roles.physicist import PhysicistAgent, PhysicsIssue
    from piano.agents.roles.hyperparameter_critic import TrainingHistory

    physicist = PhysicistAgent()

    # Plateaued history: last 3 test losses are flat (< 2% improvement)
    history = TrainingHistory(
        train_losses=[0.5, 0.3, 0.2, 0.15, 0.10],
        test_losses=[0.6, 0.4, 0.3, 0.300, 0.300],
        epochs_completed=5,
    )
    all_zero = {"equilibrium": 0.0, "energy": 0.0,
                "traction_free": 0.0, "stress_intensity": 0.0, "near_tip": 0.0, "j_integral": 0.0}

    # Case 1: All weights 0, test loss plateaued → TEST_LOSS_PLATEAUED
    issues = physicist.detect_physics_issues(history, all_zero)
    assert PhysicsIssue.TEST_LOSS_PLATEAUED in issues

    # Case 2: equilibrium active, energy still 0, plateaued → TEST_LOSS_PLATEAUED
    issues2 = physicist.detect_physics_issues(
        history, {**all_zero, "equilibrium": 1e-3}
    )
    assert PhysicsIssue.TEST_LOSS_PLATEAUED in issues2

    # Case 3: equilibrium + energy active, traction_free still 0 → TEST_LOSS_PLATEAUED
    issues3 = physicist.detect_physics_issues(
        history, {**all_zero, "equilibrium": 1e-3, "energy": 5e-4}
    )
    assert PhysicsIssue.TEST_LOSS_PLATEAUED in issues3

    # Case 4: traction_free active, stress_intensity still 0 → NEXT_CRACK_TERM_READY
    issues4 = physicist.detect_physics_issues(
        history, {**all_zero, "equilibrium": 1e-3, "energy": 5e-4, "traction_free": 1e-3}
    )
    assert PhysicsIssue.NEXT_CRACK_TERM_READY in issues4

    # Case 5: Loss spike with a physics term active → PHYSICS_DESTABILIZING
    history_unstable = TrainingHistory(
        train_losses=[0.1, 0.09, 0.08, 0.5, 0.6],
        test_losses=[0.2, 0.19, 0.18, 0.9, 1.1],
        epochs_completed=5,
    )
    issues5 = physicist.detect_physics_issues(
        history_unstable, {**all_zero, "equilibrium": 0.1}
    )
    assert PhysicsIssue.PHYSICS_DESTABILIZING in issues5


def test_physicist_should_consult():
    """Test physicist consultation trigger logic."""
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.agents.roles.hyperparameter_critic import TrainingHistory

    physicist = PhysicistAgent()

    all_zero = {"equilibrium": 0.0, "energy": 0.0,
                "traction_free": 0.0, "stress_intensity": 0.0, "near_tip": 0.0, "j_integral": 0.0}

    # Should consult: test loss plateaued + all weights 0 → TEST_LOSS_PLATEAUED
    history_plateau = TrainingHistory(
        train_losses=[0.5, 0.3, 0.2, 0.15, 0.10],
        test_losses=[0.6, 0.4, 0.3, 0.300, 0.300],
        epochs_completed=5,
    )
    assert physicist.should_consult(history_plateau, all_zero)

    # Should not consult: test loss still improving, no ratio issues
    history_improving = TrainingHistory(
        train_losses=[0.5, 0.4, 0.3, 0.2, 0.15],
        test_losses=[0.6, 0.5, 0.4, 0.3, 0.2],
        epochs_completed=5,
    )
    assert not physicist.should_consult(history_improving, all_zero)


# =============================================================================
# 3. TransolverConfig Tests
# =============================================================================

def test_transolver_config_all_tunable_params():
    """Test that TransolverConfig includes all tunable parameters."""
    from piano.surrogate.base import TransolverConfig

    config = TransolverConfig(
        d_model=256, n_layers=6, n_heads=8, slice_num=32,
        mlp_ratio=4.0, dropout=0.1, activation="silu",
        optimizer_type="adamw", learning_rate=1e-3, scheduler_type="cosine",
        energy=0.1, equilibrium=0.1,
        batch_size=32, epochs=100, patience=50,
    )

    d = config.to_dict()
    assert d["d_model"] == 256
    assert d["activation"] == "silu"


# =============================================================================
# 4. Agentic Training Tests (Synthetic Crack Data)
# =============================================================================

@pytest.fixture(scope="module")
def synthetic_crack_dataset():
    """Create synthetic crack dataset for testing."""
    rng = np.random.default_rng(42)
    N_SAMPLES = 10

    # Generate crack mesh
    coords, triangles = _generate_crack_mesh(n_points=400, crack_length=0.3, rng=rng)

    # Generate parameter samples
    params = []
    outputs = []

    for i in range(N_SAMPLES):
        p = {
            "E": float(rng.uniform(150e9, 250e9)),
            "nu": float(rng.uniform(0.25, 0.35)),
            "K_I": float(rng.uniform(1e6, 10e6)),
            "crack_length": 0.3,
        }
        params.append([p["E"], p["nu"], p["K_I"], p["crack_length"]])

        # Williams expansion displacement
        disp = _williams_displacement(coords, p, tip_x=0.3, tip_y=0.5)
        outputs.append(disp)

    params_arr = np.array(params, dtype=np.float32)

    return params_arr, coords, triangles, outputs


def test_agentic_trainer_initialization():
    """Test AgenticSurrogateTrainer initialization."""
    from piano.surrogate.agentic_trainer import (
        AgenticSurrogateTrainer, AgenticTrainingConfig,
    )
    from piano.surrogate.base import TransolverConfig

    config = AgenticTrainingConfig(
        base_config=TransolverConfig(d_model=64, n_layers=2),
        max_hpo_rounds=3,
        trigger_threshold=0.1,
    )

    provider = MockLLMProvider()
    trainer = AgenticSurrogateTrainer(config, llm_provider=provider)

    assert trainer.config == config
    assert trainer.critic is not None
    assert trainer.architect is not None


def test_agentic_trainer_train_without_hpo(synthetic_crack_dataset):
    """Test agentic training when HPO is not needed."""
    from piano.surrogate.agentic_trainer import (
        AgenticSurrogateTrainer, AgenticTrainingConfig,
    )
    from piano.surrogate.base import TransolverConfig

    params, coords, _, outputs = synthetic_crack_dataset

    config = AgenticTrainingConfig(
        base_config=TransolverConfig(
            d_model=32, n_layers=1, n_heads=2, slice_num=4,
            epochs=5, patience=10, batch_size=4, output_dim=2,
        ),
        max_hpo_rounds=2,
        trigger_threshold=100.0,  # High = no HPO
        use_ensemble=False,
    )

    provider = MockLLMProvider(scenario="stable")
    trainer = AgenticSurrogateTrainer(config, llm_provider=provider)

    result = trainer.train(params, [coords] * len(params), outputs)

    assert result.success
    assert result.n_hpo_rounds == 0


# =============================================================================
# 5. Critic-Architect Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_critic_architect_loop_crack():
    """Test critic-architect loop for crack problem."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory,
    )
    from piano.agents.roles.architect import ArchitectAgent
    from piano.surrogate.base import TransolverConfig

    critic = HyperparameterCriticAgent()
    architect = ArchitectAgent()

    provider = MockLLMProvider(scenario="underfitting")
    critic.set_llm_provider(provider)
    architect.set_llm_provider(provider)

    context = AgentContext()

    # Small model struggling with singularity
    current_config = TransolverConfig(
        d_model=64, n_layers=2, dropout=0.0, learning_rate=1e-4,
    )

    # High loss plateau (underfitting)
    history = TrainingHistory(
        train_losses=[0.5, 0.45, 0.42, 0.40, 0.39],
        test_losses=[0.55, 0.50, 0.47, 0.45, 0.44],
        epochs_completed=5,
        final_train_loss=0.39,
        final_test_loss=0.44,
    )

    # Critic analyzes
    critique = await critic.analyze_training(
        context=context,
        training_history=history,
        config=current_config.to_dict(),
    )

    assert critique.should_retrain is True
    assert critique.primary_issue.name == "UNDERFITTING"

    # Architect proposes fix
    proposal = await architect.propose_config(
        context=context,
        current_config=current_config,
        critique=critique,
        dataset_size=10,
    )

    assert proposal.config is not None
    # Should increase capacity for crack singularity
    assert proposal.changes.get("d_model", 64) >= 64


# =============================================================================
# 6. Parametric Scenario Tests
# =============================================================================

@pytest.mark.parametrize("scenario,expected_issue", [
    ("underfitting", "UNDERFITTING"),
    ("overfitting", "OVERFITTING"),
    ("slow_convergence", "SLOW_CONVERGENCE"),
    ("stable", "NONE"),
])
@pytest.mark.asyncio
async def test_critic_scenarios(scenario, expected_issue):
    """Test critic identifies different scenarios."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory,
    )

    critic = HyperparameterCriticAgent()
    provider = MockLLMProvider(scenario=scenario)
    critic.set_llm_provider(provider)

    context = AgentContext()
    history = TrainingHistory(
        train_losses=[0.5, 0.4, 0.3, 0.2, 0.1],
        test_losses=[0.5, 0.45, 0.4, 0.35, 0.3],
        epochs_completed=5,
    )

    result = await critic.analyze_training(
        context=context,
        training_history=history,
        config={"d_model": 128},
    )

    assert result.primary_issue.name == expected_issue


# =============================================================================
# 7. AdaptiveProposer Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_adaptive_proposer_propose_targeted():
    """Test AdaptiveProposerAgent produces targeted proposals."""
    from piano.agents.roles.adaptive_proposer import AdaptiveProposerAgent
    from piano.agents.base import AgentContext
    from piano.surrogate.evaluator import UncertaintyAnalysis, WeakRegion

    proposer = AdaptiveProposerAgent()
    provider = MockLLMProvider(scenario="underfitting")
    proposer.set_llm_provider(provider)

    context = AgentContext()

    # Create mock uncertainty analysis with correct fields
    uncertainty = UncertaintyAnalysis(
        overall_uncertainty=0.15,
        max_uncertainty=0.35,
        weak_regions=[
            WeakRegion(
                parameter_ranges={"E": (180e9, 220e9), "nu": (0.28, 0.32), "K_I": (4e6, 6e6)},
                metric="uncertainty",
                metric_value=0.35,
                priority=1.0,
                sample_count=2,
                suggested_samples=3,
            ),
        ],
    )

    parameter_bounds = {
        "E": (150e9, 250e9),
        "nu": (0.25, 0.35),
        "K_I": (1e6, 10e6),
        "crack_length": (0.2, 0.5),
    }

    proposals = await proposer.propose_targeted(
        context=context,
        uncertainty_analysis=uncertainty,
        parameter_bounds=parameter_bounds,
        n_samples=10,
        n_valid=10,
        n_proposals=3,
    )

    # Should return proposals with parameters
    assert len(proposals) >= 1
    for proposal in proposals:
        assert proposal.parameters is not None
        assert "E" in proposal.parameters
        assert proposal.reasoning is not None


def test_adaptive_proposer_parse_response():
    """Test parsing of LLM response into proposals."""
    from piano.agents.roles.adaptive_proposer import AdaptiveProposerAgent

    proposer = AdaptiveProposerAgent()

    # Test parsing the mock response format
    response = """
    **Proposal 1**
    Parameters: E=200e9, nu=0.30, K_I=6e6, crack_length=0.35
    Target Region: High uncertainty near crack tip
    Reasoning: Testing the singularity region
    Priority: High

    **Proposal 2**
    Parameters: E=180e9, nu=0.28, K_I=4e6, crack_length=0.25
    Target Region: Low stiffness region
    Reasoning: Exploring softer materials
    Priority: Medium
    """

    proposals = proposer._parse_multiple_proposals(response, expected_count=2)

    assert len(proposals) >= 2
    assert proposals[0].parameters["E"] == 200e9
    assert proposals[0].parameters["crack_length"] == 0.35
    assert proposals[1].parameters["nu"] == 0.28


def test_orchestrator_select_informative_samples():
    """Test orchestrator sample selection with AdaptiveProposer."""
    from piano.orchestration.adaptive import AdaptiveOrchestrator, AdaptiveConfig
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AdaptiveConfig(
            base_mesh_path=Path(tmpdir) / "mesh.mesh",
            output_dir=Path(tmpdir) / "output",
            parameter_bounds={
                "E": (150e9, 250e9),
                "nu": (0.25, 0.35),
                "K_I": (1e6, 10e6),
                "crack_length": (0.2, 0.5),
            },
            use_agentic_proposer=True,
        )

        provider = MockLLMProvider(scenario="underfitting")
        orchestrator = AdaptiveOrchestrator(config, llm_provider=provider)

        # Verify proposer is initialized
        assert orchestrator.proposer is not None

        # Test that calling _select_informative_samples without evaluator raises
        # (this is expected since we haven't set up the full training pipeline)
        with pytest.raises(RuntimeError, match="Evaluator not initialized"):
            orchestrator._select_informative_samples(3)


def test_orchestrator_proposer_initialization():
    """Test that proposer is correctly initialized based on config."""
    from piano.orchestration.adaptive import AdaptiveOrchestrator, AdaptiveConfig
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # When use_agentic_proposer=False, proposer should be None
        config_disabled = AdaptiveConfig(
            base_mesh_path=Path(tmpdir) / "mesh.mesh",
            output_dir=Path(tmpdir) / "output",
            use_agentic_proposer=False,
        )
        orchestrator_disabled = AdaptiveOrchestrator(config_disabled)
        assert orchestrator_disabled.proposer is None

        # When use_agentic_proposer=True, proposer should be initialized
        config_enabled = AdaptiveConfig(
            base_mesh_path=Path(tmpdir) / "mesh2.mesh",
            output_dir=Path(tmpdir) / "output2",
            use_agentic_proposer=True,
        )
        provider = MockLLMProvider()
        orchestrator_enabled = AdaptiveOrchestrator(config_enabled, llm_provider=provider)
        assert orchestrator_enabled.proposer is not None


# =============================================================================
# 7b. Debate Orchestrator and Result Analyst Tests
# =============================================================================

def test_result_analyst_parse_response():
    """Result analyst correctly parses structured observation format."""
    from piano.agents.roles.result_analyst import ResultAnalystAgent

    analyst = ResultAnalystAgent()
    raw = """OVERALL_PATTERN: overfitting
SEVERITY: high
OBSERVATION: Train loss dropped to 0.01 but test loss stayed at 0.35. The gap widened over epochs 40-80.
PINO_STATUS: no PINO terms active.
ENSEMBLE_STATUS: ensemble_std = 0.04, ratio ≈ 0.11 (moderate variance)."""

    obs = analyst.parse_response(raw)
    assert obs.pattern == "overfitting"
    assert obs.severity == "high"
    assert "0.01" in obs.observation
    assert "PINO" in obs.pino_status
    assert "0.04" in obs.ensemble_status
    assert obs.to_debate_message().startswith("[ANALYST")


@pytest.mark.asyncio
async def test_result_analyst_observe():
    """Result analyst generates observation via mock LLM."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.result_analyst import ResultAnalystAgent
    from piano.agents.roles.hyperparameter_critic import TrainingHistory

    analyst = ResultAnalystAgent()
    provider = MockLLMProvider(scenario="underfitting")
    analyst.set_llm_provider(provider)

    history = TrainingHistory(
        train_losses=[0.5, 0.48, 0.46, 0.45, 0.45],
        test_losses=[0.52, 0.49, 0.48, 0.48, 0.48],
        pino_losses=[],
        epochs_completed=80,
        final_train_loss=0.45,
        final_test_loss=0.48,
        best_test_loss=0.48,
    )

    obs = await analyst.observe(AgentContext(), history)
    assert obs.pattern != ""
    assert obs.severity in ("low", "medium", "high", "critical")
    assert len(obs.observation) > 0


@pytest.mark.asyncio
async def test_critic_observe_round1():
    """Critic generates Round 1 observation text (no proposals)."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.hyperparameter_critic import HyperparameterCriticAgent, TrainingHistory

    critic = HyperparameterCriticAgent()
    provider = MockLLMProvider(scenario="overfitting")
    critic.set_llm_provider(provider)

    history = TrainingHistory(
        train_losses=[0.3, 0.1, 0.03, 0.01],
        test_losses=[0.3, 0.25, 0.30, 0.35],
        epochs_completed=40,
        final_train_loss=0.01,
        final_test_loss=0.35,
        best_test_loss=0.25,
    )

    obs_text = await critic.observe(AgentContext(), history, {})
    assert "[CRITIC" in obs_text
    assert len(obs_text) > 20


@pytest.mark.asyncio
async def test_architect_analyze_round2():
    """Architect generates Round 2 analysis text (no proposals)."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.architect import ArchitectAgent
    from piano.surrogate.base import TransolverConfig

    architect = ArchitectAgent()
    provider = MockLLMProvider(scenario="underfitting")
    architect.set_llm_provider(provider)

    config = TransolverConfig()
    r1_context = "[ANALYST — Round 1]\nPattern: underfitting. Train=0.45, Test=0.48."

    analysis = await architect.analyze(AgentContext(), config, r1_context)
    assert "[ARCHITECT" in analysis
    assert len(analysis) > 20


@pytest.mark.asyncio
async def test_physicist_analyze_round2():
    """Physicist generates Round 2 analysis text (no proposals)."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.agents.roles.hyperparameter_critic import TrainingHistory

    physicist = PhysicistAgent()
    provider = MockLLMProvider(scenario="underfitting")
    physicist.set_llm_provider(provider)

    history = TrainingHistory(
        train_losses=[0.5, 0.45, 0.45],
        test_losses=[0.5, 0.48, 0.48],
        epochs_completed=80,
        final_train_loss=0.45,
        final_test_loss=0.48,
        best_test_loss=0.48,
    )
    current_config = {"equilibrium": 0.0, "energy": 0.0, "traction_free": 0.0,
                      "stress_intensity": 0.0, "near_tip": 0.0, "j_integral": 0.0}
    r1_context = "[ANALYST — Round 1]\nNo physics terms active."

    analysis = await physicist.analyze(AgentContext(), current_config, history, r1_context)
    assert "[PHYSICIST" in analysis
    assert len(analysis) > 20


@pytest.mark.asyncio
async def test_debate_orchestrator_full_run():
    """4-round debate runs all rounds and returns architecture + physics proposals."""
    from piano.agents.roles.result_analyst import ResultAnalystAgent
    from piano.agents.roles.hyperparameter_critic import HyperparameterCriticAgent, TrainingHistory
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.orchestration.debate import DebateOrchestrator
    from piano.surrogate.base import TransolverConfig

    provider = MockLLMProvider(scenario="underfitting")

    analyst = ResultAnalystAgent()
    critic = HyperparameterCriticAgent()
    architect = ArchitectAgent()
    physicist = PhysicistAgent()
    for agent in (analyst, critic, architect, physicist):
        agent.set_llm_provider(provider)

    debate = DebateOrchestrator(analyst, critic, architect, physicist)

    history = TrainingHistory(
        train_losses=[0.5, 0.48, 0.46, 0.45, 0.45],
        test_losses=[0.52, 0.49, 0.48, 0.48, 0.48],
        pino_losses=[],
        epochs_completed=80,
        final_train_loss=0.45,
        final_test_loss=0.48,
        best_test_loss=0.48,
        has_nan=False,
    )
    config = TransolverConfig()

    results = await debate._run_debate(
        history=history,
        current_config=config,
        dataset_size=30,
        config_history=[],
        problem_type="crack",
        has_singularity=True,
        n_candidates=1,
    )

    # _run_debate always returns a list; single candidate when n_candidates=1
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]

    # Should produce an architecture proposal and physics changes
    assert result.arch_proposal is not None
    assert isinstance(result.physics_changes, dict)
    # Debate log should have messages from all 4 rounds
    assert len(result.debate_log) >= 4
    # Validation text should be present
    assert len(result.validation_text) > 0
    # All 4 rounds should be logged (analyst, critic-obs, arch-analysis, phys-analysis, ...)
    log_text = "\n".join(result.debate_log)
    assert "ANALYST" in log_text
    assert "CRITIC" in log_text
    assert "ARCHITECT" in log_text
    assert "PHYSICIST" in log_text


def test_debate_debate_context_passed_to_proposals():
    """Architect and Physicist build_user_prompt includes debate context when provided."""
    from piano.agents.base import AgentContext
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.agents.roles.hyperparameter_critic import CritiqueResult, TrainingIssue, TrainingHistory
    from piano.surrogate.base import TransolverConfig

    debate_ctx = "DEBATE CONTEXT SENTINEL"

    architect = ArchitectAgent()
    architect_prompt = architect.build_user_prompt(
        AgentContext(),
        current_config=TransolverConfig(),
        critique=CritiqueResult(primary_issue=TrainingIssue.UNDERFITTING, severity="high"),
        dataset_size=30,
        previous_configs=[],
        debate_context=debate_ctx,
    )
    assert debate_ctx in architect_prompt

    physicist = PhysicistAgent()
    history = TrainingHistory(train_losses=[0.5], test_losses=[0.5], final_train_loss=0.5,
                              final_test_loss=0.5, epochs_completed=10)
    physicist_prompt = physicist.build_user_prompt(
        AgentContext(),
        current_config={"equilibrium": 0.0, "energy": 0.0, "traction_free": 0.0,
                        "stress_intensity": 0.0, "near_tip": 0.0, "j_integral": 0.0},
        critique=CritiqueResult(primary_issue=TrainingIssue.LOSS_PLATEAU, severity="medium"),
        training_history=history,
        debate_context=debate_ctx,
    )
    assert debate_ctx in physicist_prompt


# =============================================================================
# 8. Visualization Demo: Agentic Loop Progress with V-Notch FEM
# =============================================================================

def _generate_vnotch_fem_data(
    n_samples: int,
    notch_depth: float = 0.3,
    notch_angle: float = 60.0,
    resolution: int = 20,
    seed: int = 42,
    output_field: str = "von_mises",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Generate V-notch FEM dataset using MFEM linear elasticity solver.
    Falls back to analytical Williams + far-field blending if PyMFEM is unavailable.

    Returns:
        params: (n_samples, 4) - [E, nu, traction, K_I]
        coords: (n_nodes, 2) - mesh coordinates
        triangles: (n_elements, 3) - element connectivity
        outputs: List of output arrays (displacement)
    """
    from piano.data.fem_generator import generate_vnotch_fem_sample, VNotchFEMConfig, compute_ki

    rng = np.random.default_rng(seed)

    config = VNotchFEMConfig(
        notch_depth=notch_depth,
        notch_angle=notch_angle,
        resolution=resolution,
    )

    params_list = []
    outputs = []
    coords = None
    triangles = None

    for i in range(n_samples):
        E = float(rng.uniform(150e9, 250e9))
        nu = float(rng.uniform(0.25, 0.35))
        traction = float(rng.uniform(50e6, 150e6))

        sample = generate_vnotch_fem_sample(E, nu, traction, config)

        if sample is not None:
            # Use coordinates from the FEM sample (MFEM-compacted node list)
            if coords is None:
                coords = sample.coordinates
                # Use mesh element connectivity directly — already excludes notch interior
                triangles = sample.elements

            K_I_val = compute_ki(traction, notch_depth=config.notch_depth,
                                 width=config.width, angle=config.notch_angle)
            params_list.append([E, nu, traction, K_I_val])
            if output_field == "von_mises":
                # Compute nodal von Mises from displacement (element-centered
                # sample.von_mises has 714 values vs 409 nodes — misaligned)
                vm_nodal = _compute_von_mises_nodal(
                    sample.displacement, coords, triangles, E, nu
                )
                # Log-transform: σ ∝ 1/√r near tip → log(σ) ≈ log(K_I) - 0.5·log(r)
                # This converts the power-law singularity to a smooth log-linear function
                # that a small MLP trunk can represent. Inverse: np.expm1(pred).
                outputs.append(np.log1p(vm_nodal)[:, np.newaxis])
            else:
                # Displacement field -> shape (N, 2)
                outputs.append(sample.displacement)

    params = np.array(params_list, dtype=np.float32)
    if coords is None:
        raise RuntimeError("No valid FEM samples generated")

    return params, coords.astype(np.float32), triangles, outputs


def _compute_von_mises_nodal(
    disp: np.ndarray,
    coords: np.ndarray,
    triangles: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """
    Compute nodal von Mises stress from nodal displacement using CST B-matrices.

    Each element contributes a constant stress; values are averaged at shared nodes.

    Args:
        disp:      (N, 2) nodal displacement [u_x, u_y]
        coords:    (N, 2) node coordinates
        triangles: (M, 3) element connectivity
        E, nu:     material properties (plane stress)

    Returns:
        (N,) nodal von Mises stress
    """
    E_fac = E / (1.0 - nu**2)      # plane-stress modulus factor
    mu = E / (2.0 * (1.0 + nu))

    n_nodes = len(coords)
    vm_sum = np.zeros(n_nodes)
    count = np.zeros(n_nodes)

    for tri in triangles:
        n0, n1, n2 = int(tri[0]), int(tri[1]), int(tri[2])
        x0, y0 = coords[n0]
        x1, y1 = coords[n1]
        x2, y2 = coords[n2]

        A2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(A2) < 1e-14:
            continue

        b0 = y1 - y2;  b1 = y2 - y0;  b2 = y0 - y1
        c0 = x2 - x1;  c1 = x0 - x2;  c2 = x1 - x0

        # CST B-matrix: B @ u_e = [eps_xx, eps_yy, gamma_xy]
        u_e = np.array([disp[n0, 0], disp[n0, 1],
                        disp[n1, 0], disp[n1, 1],
                        disp[n2, 0], disp[n2, 1]])
        eps_xx = (b0 * u_e[0] + b1 * u_e[2] + b2 * u_e[4]) / A2
        eps_yy = (c0 * u_e[1] + c1 * u_e[3] + c2 * u_e[5]) / A2
        gam_xy = (c0 * u_e[0] + b0 * u_e[1] + c1 * u_e[2] +
                  b1 * u_e[3] + c2 * u_e[4] + b2 * u_e[5]) / A2

        # Plane-stress constitutive
        sig_xx = E_fac * (eps_xx + nu * eps_yy)
        sig_yy = E_fac * (eps_yy + nu * eps_xx)
        sig_xy = mu * gam_xy

        vm = np.sqrt(sig_xx**2 - sig_xx * sig_yy + sig_yy**2 + 3.0 * sig_xy**2)

        for nid in (n0, n1, n2):
            vm_sum[nid] += vm
            count[nid] += 1

    count = np.maximum(count, 1)
    return vm_sum / count


def _generate_phase_field_data(
    n_samples: int,
    crack_length: float = 0.3,
    resolution: int = 20,
    n_load_steps: int = 15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Generate edge-crack phase field FEM dataset using FEniCS.

    Returns:
        params: (n, 5)  — [E, nu, traction, G_c, crack_length]
        coords: (N, 2)  — mesh nodes (shared topology for fixed crack_length)
        triangles: (M, 3)
        outputs: list of (N, 3) arrays — [u_x, u_y, log1p(σ_vm)]
    """
    import math
    from piano.data.phase_field_generator import PhaseFieldFEMConfig, generate_phase_field_sample

    rng = np.random.default_rng(seed)

    config = PhaseFieldFEMConfig(
        crack_length=crack_length,
        resolution=resolution,
        n_load_steps=n_load_steps,
        l_0=max(0.03, 1.0 / resolution * 1.5),
    )

    params_list: List[List[float]] = []
    outputs: List[np.ndarray] = []
    coords: Optional[np.ndarray] = None
    triangles: Optional[np.ndarray] = None

    collected = 0
    attempts = 0
    max_attempts = n_samples * 3

    while collected < n_samples and attempts < max_attempts:
        attempts += 1
        E   = float(rng.uniform(150e9, 250e9))
        nu  = float(rng.uniform(0.25, 0.35))
        G_c = float(rng.uniform(1e3, 5e3))
        K_Ic = math.sqrt(E * G_c)
        Y    = 1.12
        traction = K_Ic / (Y * math.sqrt(math.pi * crack_length)) * float(rng.uniform(0.6, 1.4))

        sample = generate_phase_field_sample(E, nu, traction, G_c, config)
        if sample is None or not sample.is_valid or sample.displacement is None or sample.von_mises is None:
            continue

        if coords is None:
            coords    = sample.coordinates.astype(np.float32)
            triangles = sample.elements

        u   = sample.displacement.astype(np.float32)                   # (N, 2)
        lvm = np.log1p(sample.von_mises).astype(np.float32)[:, None]   # (N, 1)
        outputs.append(np.hstack([u, lvm]))                            # (N, 3)
        params_list.append([E, nu, traction, G_c, crack_length])
        collected += 1

    if coords is None or len(params_list) == 0:
        raise RuntimeError("No valid FEniCS phase field samples generated")

    params = np.array(params_list, dtype=np.float32)
    print(f"   Generated {collected}/{attempts} valid samples (mesh: {len(coords)} nodes, {len(triangles)} elems)")
    return params, coords, triangles, outputs


def run_agentic_loop_demo(
    n_samples: int = 30,
    epochs_per_round: int = 80,
    max_hpo_rounds: int = 8,
    output_file: str = "tests/test_outputs/agentic_vnotch_demo.png",
    use_real_llm: bool = False,
    use_claude_code: bool = False,
):
    """
    Demonstration of agentic SciML loop for V-notch problem.

    Shows the iterative HPO process:
    - Multiple HPO rounds with agent interventions
    - Loss progression across rounds
    - Error reduction over iterations
    - Final prediction vs FEM ground truth (von Mises stress)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.tri as mtri

    print("=" * 70)
    print("PIANO: Agentic SciML Loop - Edge Crack Phase Field Prediction")
    print("=" * 70)

    # Configuration
    crack_length = 0.3   # fixed crack geometry; E/nu/traction/G_c vary across samples

    print(f"\n1. Generating {n_samples} edge-crack phase field samples via FEniCS...")
    params, coords, triangles, outputs = _generate_phase_field_data(
        n_samples=n_samples,
        crack_length=crack_length,
        resolution=20,
        n_load_steps=15,
    )
    print(f"   Params: [E, nu, traction, G_c, crack_length=0.3]  output_dim=3 (u_x, u_y, log1p(σ_vm))")

    # Williams coordinate enrichment relative to crack tip.
    # (N,2) → (N,8): [x, y, r, log_r, sinθ, cosθ, sin(θ/2), cos(θ/2)]
    tip_xy = np.array([crack_length, 0.5], dtype=np.float32)
    dxy = coords - tip_xy                                           # (N, 2)
    r_feat = np.linalg.norm(dxy, axis=1, keepdims=True).clip(1e-8) # (N, 1)
    log_r_feat = np.log(r_feat)                                     # (N, 1)
    theta = np.arctan2(dxy[:, 1:2], dxy[:, 0:1])                   # (N, 1) ∈ (-π, π]
    # Full-angle: far-field angular variation (no branch cut at crack faces)
    sin_theta = np.sin(theta)                                       # (N, 1)
    cos_theta = np.cos(theta)                                       # (N, 1)
    # Half-angle: Williams mode I basis — discontinuous across θ=±π (crack faces),
    # which is exactly where crack opening displacement jumps
    sin_half = np.sin(theta / 2)                                    # (N, 1)
    cos_half = np.cos(theta / 2)                                    # (N, 1)
    coords_enriched = np.concatenate(
        [coords, r_feat, log_r_feat, sin_theta, cos_theta, sin_half, cos_half], axis=1
    ).astype(np.float32)                                            # (N, 8)

    # Import training components
    from piano.surrogate.base import TransolverConfig, CrackConfig
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory,
    )
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.agents.base import AgentContext
    import asyncio

    critic = HyperparameterCriticAgent()
    architect = ArchitectAgent()
    physicist = PhysicistAgent()
    if use_claude_code:
        from piano.agents.llm.claude_code_provider import ClaudeCodeProvider
        provider = ClaudeCodeProvider(model="sonnet", max_turns=3, allowed_tools=[])
        print("   Using ClaudeCodeProvider (claude CLI session) for all agents")
    elif use_real_llm:
        from piano.agents.llm.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider()
        print("   Using real LLM (AnthropicProvider) for all agents")
    else:
        provider = MockLLMProvider(scenario="underfitting")
    critic.set_llm_provider(provider)
    architect.set_llm_provider(provider)
    physicist.set_llm_provider(provider)

    # Ensemble settings
    n_ensemble_candidates = 3  # number of proposals generated per HPO round
    eval_epochs = 15           # brief-training epochs for candidate selection

    # Track progress across HPO rounds
    round_results = []
    all_train_losses = []
    all_test_losses = []
    agent_actions = []
    ensemble_log = []    # [(round_idx, [brief_losses...], winner_idx)]
    attempt_history = []  # failure memory fed back to Architect each round

    # Crack tip position and tip weighting
    tip = np.array([crack_length, 0.5], dtype=np.float32)
    tip_weight_fixed = 5.0

    # Phase field: params = [E, nu, traction, G_c, crack_length].
    # No K_I in the parameter vector, so K_I-based PINO terms are disabled.
    # Only the force-balance (equilibrium) term is used — it only needs E and nu.
    crack_cfg = CrackConfig(
        tip_x=crack_length,
        tip_y=0.5,
        e_param_idx=0,
        nu_param_idx=1,
        ki_param_idx=2,   # points to traction; singularity weights are all 0 so it's unused
    )

    # Initial config — 3-output model (u_x, u_y, log1p(σ_vm)).
    # equilibrium=1e-3: force-balance regulariser on displacement outputs.
    # Singularity PINO terms start at 0 (not applicable without explicit K_I param).
    current_config = TransolverConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        slice_num=8,
        dropout=0.0,
        learning_rate=1e-3,
        optimizer_type="adamw",
        scheduler_type="cosine",
        epochs=epochs_per_round,
        patience=epochs_per_round,
        batch_size=4,
        output_dim=3,
        equilibrium=1e-3,
        energy=0.0,
        tip_weight=2.0,
        stress_intensity=0.0,
        traction_free=0.0,
        near_tip=0.0,
        j_integral=0.0,
    )

    context = AgentContext()
    loop = asyncio.new_event_loop()

    # Convergence criteria
    min_improvement_pct = 2.0   # stop when round-over-round improvement < 2 %
    min_rounds = 2              # always run at least this many rounds

    print(f"\n2. Agentic loop — runs until convergence "
          f"(max {max_hpo_rounds} rounds, stop when <{min_improvement_pct}% improvement)...")

    round_idx = 0
    converged = False
    best_test_loss = float('inf')
    best_trainer = None
    best_config = current_config   # Gap 2: always propose from the best known config
    no_improve_streak = 0
    max_no_improve = 2          # stop after this many consecutive non-improving rounds

    while round_idx < max_hpo_rounds and not converged:
        print(f"\n   --- Round {round_idx + 1} ---")

        # Train with current config; patience = epochs so we never early-stop within a round
        current_config.patience = current_config.epochs
        trainer = SurrogateTrainer(TrainingConfig(
            surrogate_config=current_config,
            use_ensemble=True,
            n_ensemble=3,
            train_test_split=0.2,
            tip_coords=tip,
            crack_config=None,  # phase field: no K_I param → disable CrackFractureLoss
        ))
        result = trainer.train(params, [coords_enriched] * len(params), outputs)
        if not result.success:
            print(f"   [ERROR] Training failed: {result.error_message}")

        # Store results
        round_results.append({
            "round": round_idx + 1,
            "train_loss": result.train_loss,
            "test_loss": result.test_loss,
            "config": current_config.to_dict().copy(),
            "trainer": trainer,
            "history": result.history,
        })

        # Accumulate loss history
        round_train = result.history.get("train_loss", [])
        round_test  = result.history.get("test_loss", [])
        all_train_losses.extend(round_train)
        all_test_losses.extend(round_test)

        curr_loss = result.test_loss if result.success else float('inf')
        print(f"   Train: {result.train_loss:.6f}, Test: {curr_loss:.6f}")

        # Track best model across all rounds
        if result.success and curr_loss < best_test_loss:
            improvement_vs_best = (best_test_loss - curr_loss) / best_test_loss * 100.0 if best_test_loss < float('inf') else 100.0
            best_test_loss = curr_loss
            best_trainer = trainer
            best_config = current_config   # Gap 2: track config that achieved best loss
            no_improve_streak = 0
            if round_idx > 0:
                print(f"   New best! Improvement: {improvement_vs_best:.1f}%")
        else:
            no_improve_streak += 1
            regress_pct = (curr_loss - best_test_loss) / max(best_test_loss, 1e-12) * 100.0
            print(f"   No improvement ({regress_pct:+.1f}% vs best). Streak: {no_improve_streak}/{max_no_improve}")

        # Convergence check (after minimum rounds)
        if round_idx >= min_rounds - 1:
            if no_improve_streak >= max_no_improve:
                print(f"   Converged — no improvement for {max_no_improve} consecutive rounds.")
                converged = True
            elif round_idx >= 1:
                # Also stop if marginal gain vs best
                if best_test_loss < float('inf'):
                    prev_best = round_results[-2]["test_loss"] if len(round_results) >= 2 else float('inf')
                    if prev_best > 0 and prev_best < float('inf'):
                        round_improvement = (prev_best - curr_loss) / prev_best * 100.0
                        if 0 < round_improvement < min_improvement_pct:
                            print(f"   Converged — improvement {round_improvement:.1f}% < {min_improvement_pct}% threshold.")
                            converged = True

        # Critic analysis — pass full cross-round history so it sees regression across rounds
        round_hist = result.history or {}
        pino_term_losses = {}
        for key in ("elasticity_loss", "crack_loss"):
            vals = round_hist.get(key, [])
            if vals:
                pino_term_losses[key.replace("_loss", "")] = vals
        history = TrainingHistory(
            train_losses=all_train_losses,
            test_losses=all_test_losses,
            pino_term_losses=pino_term_losses,
            ensemble_std=round_hist.get("ensemble_std", 0.0),
            epochs_completed=len(all_train_losses),
            best_test_loss=best_test_loss,
            final_train_loss=result.train_loss,
            final_test_loss=result.test_loss,
        )

        critique = loop.run_until_complete(
            critic.analyze_training(context, history, current_config.to_dict())
        )

        action = {
            "round": round_idx + 1,
            "issue": critique.primary_issue.name,
            "severity": critique.severity,
            "arch_changes": {},
            "phys_changes": {},
        }

        if not converged and critique.should_retrain:
            # Record failure memory
            attempt_history.append({
                "round": round_idx + 1,
                "changes": action["arch_changes"],
                "result": (f"train={result.train_loss:.4f}, test={result.test_loss:.4f} "
                           f"({critique.primary_issue.name}, {critique.severity})"),
                "summary": (f"Round {round_idx+1}: {critique.primary_issue.name} — "
                            f"test={result.test_loss:.4f}"),
            })

            # ── Ensemble candidate generation ─────────────────────────────
            # Generate n_ensemble_candidates proposals, brief-train each,
            # pick the winner by lowest brief test loss.
            from piano.agents.roles.physicist import PhysicsProposal
            from piano.surrogate.trainer import TrainingConfig
            import copy

            candidate_configs = []
            for cand_idx in range(n_ensemble_candidates):
                arch_p = loop.run_until_complete(
                    architect.propose_config(
                        context, best_config, critique, len(params),
                        previous_configs=attempt_history,
                    )
                )

                current_physics = {
                    k: getattr(best_config, k, 0.0)
                    for k in ("equilibrium", "energy", "traction_free",
                              "stress_intensity", "near_tip", "j_integral")
                }
                if physicist.should_consult(history, current_physics):
                    phys_p = loop.run_until_complete(
                        physicist.propose_physics_config(
                            context, best_config.to_dict(), critique, history,
                            len(params), problem_type="notch", has_singularity=True
                        )
                    )
                else:
                    phys_p = PhysicsProposal()

                # Merge architecture + physics into one config
                merged = copy.deepcopy(arch_p.config)
                merged.output_dim = 3
                if hasattr(merged, "tip_weight"):
                    merged.tip_weight = tip_weight_fixed
                for k, v in phys_p.changes.items():
                    if hasattr(merged, k):
                        setattr(merged, k, v)

                candidate_configs.append((merged, arch_p, phys_p))

            # Brief-train each candidate
            brief_losses = []
            for merged, _, _ in candidate_configs:
                brief_cfg = copy.deepcopy(merged)
                brief_cfg.epochs = eval_epochs
                brief_cfg.patience = eval_epochs
                brief_trainer = SurrogateTrainer(TrainingConfig(
                    surrogate_config=brief_cfg,
                    use_ensemble=False,
                    train_test_split=0.2,
                    tip_coords=tip,
                    crack_config=None,
                ))
                brief_result = brief_trainer.train(params, [coords_enriched] * len(params), outputs)
                brief_losses.append(brief_result.test_loss if brief_result.success else float('inf'))

            winner_idx = int(np.argmin(brief_losses))
            current_config, arch_proposal, phys_proposal = candidate_configs[winner_idx]

            ensemble_log.append((round_idx + 1, brief_losses, winner_idx))
            print(f"   Ensemble: brief losses={[f'{l:.4f}' for l in brief_losses]}, "
                  f"winner=C{winner_idx+1}")

            action["arch_changes"] = arch_proposal.changes

            print(f"   Critic: {critique.primary_issue.name}")
            print(f"   Architect: {list(arch_proposal.changes.keys())}")
            print(f"   Physicist: {list(phys_proposal.changes.keys())}")

        agent_actions.append(action)
        round_idx += 1

    loop.close()
    n_rounds_run = round_idx

    # Generate test prediction with final model
    print("\n3. Generating test prediction...")
    test_E, test_nu, test_G_c = 200e9, 0.3, 2.7e3
    import math as _math
    _K_Ic = _math.sqrt(test_E * test_G_c)
    test_traction = _K_Ic / (1.12 * _math.sqrt(_math.pi * crack_length))  # at exactly K_Ic
    test_arr = np.array([[test_E, test_nu, test_traction, test_G_c, crack_length]], dtype=np.float32)

    # Get ground truth von Mises from a fresh FEniCS phase field solve
    from piano.data.phase_field_generator import PhaseFieldFEMConfig, generate_phase_field_sample
    gt_config = PhaseFieldFEMConfig(crack_length=crack_length, resolution=20, n_load_steps=15,
                                    l_0=max(0.03, 1.0 / 20 * 1.5))
    gt_sample = generate_phase_field_sample(test_E, test_nu, test_traction, test_G_c, gt_config)

    if gt_sample is not None and gt_sample.von_mises is not None:
        vm_gt = gt_sample.von_mises.astype(np.float32)
    else:
        # Fall back to first training sample's von Mises
        vm_gt = np.expm1(outputs[0][:, 2])

    # Use the best model across all rounds for the final prediction
    # pred_raw: (N, 3) — columns [u_x, u_y, log1p(σ_vm)]
    pred_raw, _ = best_trainer.predict_with_uncertainty(test_arr, coords_enriched)
    if pred_raw.ndim == 3:
        pred_raw = pred_raw[0]
    vm_pred = np.expm1(pred_raw[:, 2]).clip(0)   # invert log1p → σ_vm [Pa]

    gt_ref = np.percentile(np.abs(vm_gt), 95) + 1e-12
    error = np.abs(vm_pred - vm_gt) / gt_ref * 100.0

    # Use all triangles (edge crack mesh has no interior void)
    valid_triangles = triangles

    # =========================================================================
    # VISUALIZATION: 2x3 Grid showing agentic loop progress
    # =========================================================================
    print("\n4. Creating visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Crack tip position
    tip_x, tip_y = crack_length, 0.5

    # --- Panel 1: Loss Evolution Across All Rounds ---
    ax1 = fig.add_subplot(gs[0, 0])

    epochs_total = len(all_test_losses)
    epochs_arr = np.arange(1, epochs_total + 1)

    ax1.semilogy(epochs_arr, all_train_losses, 'b-', lw=1.5, alpha=0.7, label='Train')
    ax1.semilogy(epochs_arr, all_test_losses, 'r-', lw=2, label='Test')

    # Mark round boundaries
    epoch_offset = 0
    colors = ['green', 'orange', 'purple', 'brown']
    for i, rr in enumerate(round_results):
        n_epochs = len(rr["history"].get("train_loss", []))
        if i > 0:
            ax1.axvline(epoch_offset, color=colors[i % len(colors)], ls='--', lw=1.5, alpha=0.7)
            ax1.text(epoch_offset + 1, ax1.get_ylim()[1] * 0.8, f'R{i+1}',
                     fontsize=9, color=colors[i % len(colors)], fontweight='bold')
        epoch_offset += n_epochs

    ax1.set_xlabel('Epoch (cumulative)')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Loss Evolution Across HPO Rounds', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Test Error vs HPO Round ---
    ax2 = fig.add_subplot(gs[0, 1])

    rounds = [rr["round"] for rr in round_results]
    test_losses = [rr["test_loss"] for rr in round_results]

    ax2.plot(rounds, test_losses, 'ro-', lw=2, markersize=10, markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(rounds, test_losses, alpha=0.3, color='red')

    for i, (r, loss) in enumerate(zip(rounds, test_losses)):
        ax2.annotate(f'{loss:.4f}', (r, loss), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # Improvement = first round loss → best-across-all-rounds loss
    improvement = (test_losses[0] - best_test_loss) / test_losses[0] * 100
    # Highlight best round
    best_round_idx = min(range(len(test_losses)), key=lambda i: test_losses[i])
    ax2.plot(rounds[best_round_idx], test_losses[best_round_idx], 'g*', ms=14,
             zorder=5, label=f'Best (R{rounds[best_round_idx]})')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_xlabel('HPO Round')
    ax2.set_ylabel('Test Loss')
    ax2.set_title(f'Convergence: {improvement:.1f}% improvement (best vs R1)', fontweight='bold')
    ax2.set_xticks(rounds)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Ensemble candidate selection per HPO round ---
    ax3 = fig.add_subplot(gs[0, 2])
    if ensemble_log:
        n_cands = max(len(entry[1]) for entry in ensemble_log)
        ens_rounds = [entry[0] for entry in ensemble_log]
        n_ens_rounds = len(ens_rounds)
        bar_width = 0.8 / n_cands
        x_base = np.arange(n_ens_rounds)
        cand_colors = ["#5599dd", "#dd9955", "#55bb77"]
        for ci in range(n_cands):
            losses_ci = []
            for entry in ensemble_log:
                bl = entry[1]
                losses_ci.append(bl[ci] if ci < len(bl) else float('nan'))
            x_pos = x_base + (ci - (n_cands - 1) / 2) * bar_width
            bars = ax3.bar(
                x_pos, losses_ci, width=bar_width,
                color=cand_colors[ci % len(cand_colors)],
                alpha=0.75, label=f"C{ci+1}",
            )
            # Highlight winner bars
            for j, (bar, entry) in enumerate(zip(bars, ensemble_log)):
                if entry[2] == ci:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2.0)
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        "★", ha="center", va="bottom", fontsize=9, color="black",
                    )
        ax3.set_xticks(x_base)
        ax3.set_xticklabels([f"R{r}" for r in ens_rounds], fontsize=9)
        ax3.legend(fontsize=8, loc="upper right")
    else:
        ax3.text(0.5, 0.5, "No ensemble\nrounds run",
                 ha="center", va="center", transform=ax3.transAxes, color="gray")
    ax3.set_xlabel("HPO round", fontsize=9)
    ax3.set_ylabel("Brief-eval test loss", fontsize=9)
    ax3.set_title(f"Ensemble: Candidate Selection ({n_ensemble_candidates} per round)", fontweight='bold')
    ax3.grid(True, alpha=0.3, axis="y")

    # Triangulation for edge-crack mesh (no interior void to mask)
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1], valid_triangles)

    def _add_crack_overlay(ax):
        """Draw crack line on an axis."""
        ax.plot([0, crack_length], [0.5, 0.5], 'w-', lw=1.5, zorder=4)

    # Von Mises stress common colour scale (clip singularity spike at 95th pct)
    vm_all = np.concatenate([vm_pred, vm_gt])
    vmin = 0.0
    vmax = np.percentile(vm_all, 95)
    levels = np.linspace(vmin, vmax, 25)

    # --- Panel 4: Surrogate Von Mises ---
    ax4 = fig.add_subplot(gs[1, 0])

    cf4 = ax4.tricontourf(triang, np.clip(vm_pred, vmin, vmax),
                          levels=levels, cmap='plasma', extend='max')
    ax4.triplot(triang, 'w-', lw=0.08, alpha=0.10)
    _add_crack_overlay(ax4)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_aspect('equal')
    ax4.set_title('Surrogate: Von Mises Stress', fontweight='bold')
    fig.colorbar(cf4, ax=ax4, shrink=0.7, label=r'$\sigma_{VM}$ [Pa]', format='%.1e')

    # --- Panel 5: Ground Truth Von Mises (FEniCS phase field) ---
    ax5 = fig.add_subplot(gs[1, 1])

    # If gt_sample has different node count, interpolate onto training mesh
    if gt_sample is not None and len(vm_gt) != len(coords):
        from scipy.interpolate import LinearNDInterpolator
        interp = LinearNDInterpolator(gt_sample.coordinates[:, :2], vm_gt)
        vm_gt_plot = interp(coords).clip(0)
        vm_gt_plot = np.where(np.isnan(vm_gt_plot), 0.0, vm_gt_plot)
    else:
        vm_gt_plot = vm_gt

    cf5 = ax5.tricontourf(triang, np.clip(vm_gt_plot, vmin, vmax),
                          levels=levels, cmap='plasma', extend='max')
    ax5.triplot(triang, 'w-', lw=0.08, alpha=0.10)
    _add_crack_overlay(ax5)
    ax5.set_xlim(-0.05, 1.05)
    ax5.set_ylim(-0.05, 1.05)
    ax5.set_aspect('equal')
    ax5.set_title('Ground Truth: Von Mises (FEniCS)', fontweight='bold')
    fig.colorbar(cf5, ax=ax5, shrink=0.7, label=r'$\sigma_{VM}$ [Pa]', format='%.1e')

    # --- Panel 6: Peak stress comparison ---
    ax6 = fig.add_subplot(gs[1, 2])
    max_pred = vm_pred.max()
    max_gt   = vm_gt.max()
    peak_err_pct = abs(max_pred - max_gt) / (max_gt + 1e-12) * 100.0
    bars = ax6.bar(["Surrogate", "Ground Truth"], [max_pred, max_gt],
                   color=["tomato", "steelblue"], edgecolor="black", width=0.5)
    for bar, val in zip(bars, [max_pred, max_gt]):
        ax6.text(bar.get_x() + bar.get_width() / 2, val * 1.01,
                 f"{val:.2e}", ha="center", va="bottom", fontsize=9)
    ax6.set_ylabel("Max von Mises stress [Pa]")
    ax6.set_title(f"Peak Stress Comparison\nError: {peak_err_pct:.1f}%", fontweight='bold')
    ax6.set_ylim(0, max(max_pred, max_gt) * 1.15)
    ax6.grid(True, alpha=0.3, axis="y")

    stop_reason = "converged" if converged else "max rounds reached"
    # Main title
    fig.suptitle(
        f'PIANO: Edge Crack Phase Field — Von Mises ({n_rounds_run} rounds, {improvement:.1f}% improvement, {stop_reason})',
        fontsize=13, fontweight='bold'
    )

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved visualization to: {output_path}")
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Edge crack: length={crack_length}")
    print(f"  Output: [u_x, u_y, log1p(σ_vm)] (output_dim=3), von Mises predicted directly")
    print(f"  Training samples: {n_samples}")
    print(f"  HPO rounds run: {n_rounds_run} ({stop_reason})")
    print(f"  Initial test loss: {round_results[0]['test_loss']:.6f}")
    print(f"  Final test loss: {round_results[-1]['test_loss']:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Mean von Mises relative error: {error.mean():.1f}%")
    print("=" * 70)


def _run_active_learning_demo_REMOVED(
    n_initial: int = 30,
    n_al_rounds: int = 2,
    n_per_al_round: int = 10,
    n_candidates: int = 60,
    epochs_per_round: int = 100,
    output_file: str = "tests/test_outputs/agentic_al_demo.png",
    use_real_llm: bool = False,
    use_claude_code: bool = False,
):
    """
    Active-learning demo for V-notch displacement surrogate.

    Loop:
      1. Train surrogate on current dataset (HPO: Critic + Architect + Physicist)
      2. Score n_candidates random parameter combos by ensemble uncertainty
      3. Run FEA only for the top n_per_al_round most-uncertain combos
      4. Add new samples to dataset and repeat

    With fewer FEA samples, physics weights carry more of the learning signal.
    Initial config starts with equilibrium=0.01 so the Physicist has something
    to calibrate from round 1 onwards.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import asyncio

    from piano.data.fem_generator import generate_vnotch_fem_sample, VNotchFEMConfig, compute_ki
    from piano.surrogate.base import TransolverConfig, CrackConfig
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.agents.roles.hyperparameter_critic import HyperparameterCriticAgent, TrainingHistory
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.roles.physicist import PhysicistAgent
    from piano.agents.base import AgentContext

    print("=" * 70)
    print("PIANO: Active Learning Demo — V-Notch Stress (physics-heavy, few FEM)")
    print("=" * 70)

    notch_depth = 0.3
    notch_angle = 60.0
    fem_config = VNotchFEMConfig(notch_depth=notch_depth, notch_angle=notch_angle, resolution=20)
    tip_xy = np.array([notch_depth, 0.5], dtype=np.float32)
    tip_weight_fixed = 5.0

    # --- helpers ---------------------------------------------------------

    def enrich_coords(raw_coords):
        dxy = raw_coords - tip_xy
        r = np.linalg.norm(dxy, axis=1, keepdims=True).clip(1e-6)
        log_r = np.log(r)
        sin_t = np.sin(np.arctan2(dxy[:, 1:2], dxy[:, 0:1]))
        cos_t = np.cos(np.arctan2(dxy[:, 1:2], dxy[:, 0:1]))
        return np.concatenate([raw_coords, r, log_r, sin_t, cos_t], axis=1).astype(np.float32)

    def run_fem(E, nu, traction):
        sample = generate_vnotch_fem_sample(E, nu, traction, fem_config)
        if sample is None:
            return None, None
        K_I = compute_ki(traction, notch_depth=fem_config.notch_depth,
                         width=fem_config.width, angle=fem_config.notch_angle)
        return np.array([E, nu, traction, K_I], dtype=np.float32), sample.displacement, sample.coordinates, sample.elements

    def score_by_uncertainty(trainer, candidate_params, coords_enriched):
        """Return mean spatial uncertainty for each candidate param set."""
        scores = []
        for p in candidate_params:
            try:
                _, unc = trainer.predict_with_uncertainty(p.reshape(1, -1), coords_enriched)
                scores.append(float(np.mean(unc)) if unc is not None else 0.0)
            except Exception:
                scores.append(0.0)
        return np.array(scores)

    # E and ν are fixed to structural steel — only traction varies.
    E_fixed  = 200e9   # Pa (structural steel)
    nu_fixed = 0.3

    # --- initial FEM dataset ---------------------------------------------
    print(f"\n1. Generating {n_initial} initial V-notch FEM samples (LHS)...")
    print(f"   Fixed: E={E_fixed/1e9:.0f} GPa, nu={nu_fixed} | Varying: traction∈[50,150] MPa")
    rng = np.random.default_rng(42)
    params_list, outputs_list = [], []
    coords, triangles = None, None

    for _ in range(n_initial):
        traction = float(rng.uniform(50e6, 150e6))
        pv, disp, c, tri = run_fem(E_fixed, nu_fixed, traction)
        if pv is not None:
            params_list.append(pv)
            outputs_list.append(disp)
            if coords is None:
                coords = c.astype(np.float32)
                triangles = tri

    params = np.array(params_list, dtype=np.float32)
    coords_enriched = enrich_coords(coords)
    print(f"   {len(params)} samples, mesh: {len(coords)} nodes")

    crack_cfg = CrackConfig(tip_x=notch_depth, tip_y=0.5,
                            e_param_idx=0, nu_param_idx=1, ki_param_idx=3)

    # --- agents ----------------------------------------------------------
    critic = HyperparameterCriticAgent()
    architect = ArchitectAgent()
    physicist = PhysicistAgent()

    if use_claude_code:
        from piano.agents.llm.claude_code_provider import ClaudeCodeProvider
        provider = ClaudeCodeProvider(model="sonnet", max_turns=3, allowed_tools=[])
        print("   Using ClaudeCodeProvider for all agents")
    elif use_real_llm:
        from piano.agents.llm.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider()
        print("   Using AnthropicProvider for all agents")
    else:
        provider = MockLLMProvider(scenario="underfitting")

    critic.set_llm_provider(provider)
    architect.set_llm_provider(provider)
    physicist.set_llm_provider(provider)

    context = AgentContext()
    loop = asyncio.new_event_loop()

    # --- initial config: non-zero equilibrium so physics works from round 1 ---
    current_config = TransolverConfig(
        d_model=32, n_layers=2, n_heads=2, slice_num=8,
        dropout=0.0, learning_rate=1e-3,
        optimizer_type="adamw", scheduler_type="cosine",
        epochs=epochs_per_round, patience=epochs_per_round,
        batch_size=4, output_dim=2,
        equilibrium=0.01,   # higher physics weight for small dataset
        energy=0.0,
        tip_weight=tip_weight_fixed,
        stress_intensity=0.0, traction_free=0.0, near_tip=0.0, j_integral=0.0,
    )

    al_log = []          # [{n_samples, test_loss}]
    hpo_log = []         # [(al_idx, hpo_idx, train_loss, test_loss)]
    physics_log = []     # [(hpo_call_global, {physics weights})]
    all_train_losses, all_test_losses, all_pino_losses = [], [], []
    best_test_loss = float('inf')
    best_trainer = None
    hpo_call_global = 0  # monotonically increasing HPO call index

    # --- active learning outer loop --------------------------------------
    print(f"\n2. Active learning: {n_al_rounds} rounds × {n_per_al_round} new FEM samples each")

    for al_idx in range(n_al_rounds + 1):  # +1 for final training after last AL round
        is_final = al_idx == n_al_rounds
        n_hpo = 3 if not is_final else 4
        print(f"\n   === AL round {al_idx}: {len(params)} samples ===")

        # Reset architect call counter and dedup set each AL round so configs
        # are proposed fresh rather than replaying the same exhausted sequence.
        tried_configs: set = set()
        if hasattr(provider, '_arch_call_count'):
            provider._arch_call_count = 0

        # --- HPO sub-loop ------------------------------------------------
        for hpo_idx in range(n_hpo):
            # Skip training if the current config was already tried this AL round.
            cfg_d = current_config.to_dict()
            cfg_key = (
                cfg_d.get('arch_type', 'transolver'),
                cfg_d.get('d_model', cfg_d.get('hidden_dim', 0)),
                cfg_d.get('n_layers', 0),
                round(cfg_d.get('dropout', 0.0), 3),
                round(cfg_d.get('learning_rate', 1e-3), 8),
                cfg_d.get('n_basis', cfg_d.get('slice_num', 0)),
            )
            if cfg_key in tried_configs:
                print(f"     HPO {hpo_idx+1}: duplicate config, stopping HPO early")
                break
            tried_configs.add(cfg_key)
            current_config.patience = current_config.epochs
            trainer = SurrogateTrainer(TrainingConfig(
                surrogate_config=current_config,
                use_ensemble=True, n_ensemble=3,
                train_test_split=0.2 if len(params) > 5 else 0.1,
                tip_coords=tip_xy, crack_config=crack_cfg,
            ))
            result = trainer.train(params, [coords_enriched] * len(params), outputs_list)
            if not result.success:
                break

            round_train = result.history.get("train_loss", [])
            round_test = result.history.get("test_loss", [])
            round_pino = result.history.get("pino_loss", [])
            all_train_losses.extend(round_train)
            all_test_losses.extend(round_test)
            all_pino_losses.extend(round_pino)

            if result.test_loss < best_test_loss:
                best_test_loss = result.test_loss
                best_trainer = trainer

            hpo_log.append((al_idx, hpo_idx, result.train_loss, result.test_loss))
            hpo_call_global += 1
            print(f"     HPO {hpo_idx+1}: train={result.train_loss:.4f}  test={result.test_loss:.4f}")

            history = TrainingHistory(
                train_losses=all_train_losses, test_losses=all_test_losses,
                pino_losses=all_pino_losses,
                epochs_completed=len(all_train_losses),
                best_test_loss=best_test_loss,
                final_train_loss=result.train_loss, final_test_loss=result.test_loss,
            )
            critique = loop.run_until_complete(
                critic.analyze_training(context, history, current_config.to_dict())
            )
            if not critique.should_retrain:
                break

            arch_proposal = loop.run_until_complete(
                architect.propose_config(context, current_config, critique, len(params))
            )
            current_config = arch_proposal.config
            current_config.output_dim = 2
            current_config.tip_weight = tip_weight_fixed

            current_physics = {k: getattr(current_config, k, 0.0)
                               for k in ("equilibrium", "energy", "traction_free",
                                         "stress_intensity", "near_tip", "j_integral")}
            if physicist.should_consult(history, current_physics):
                phys = loop.run_until_complete(
                    physicist.propose_physics_config(
                        context, current_config.to_dict(), critique, history,
                        len(params), problem_type="notch", has_singularity=True
                    )
                )
                for k, v in phys.changes.items():
                    if hasattr(current_config, k):
                        setattr(current_config, k, v)
                print(f"     Physicist: {phys.changes}")
                # Log physics weights after Physicist update
                pw = {k: getattr(current_config, k, 0.0)
                      for k in ("equilibrium", "energy", "traction_free",
                                "stress_intensity", "near_tip")}
                physics_log.append((hpo_call_global, pw))

        al_log.append({"n_samples": len(params), "test_loss": best_test_loss})

        if is_final:
            break

        # --- score candidates by uncertainty -----------------------------
        print(f"   Scoring {n_candidates} candidates by ensemble uncertainty...")
        cand_tr = rng.uniform(50e6, 150e6, n_candidates)
        cand_ki = np.array([compute_ki(t, notch_depth=fem_config.notch_depth,
                                       width=fem_config.width, angle=fem_config.notch_angle)
                            for t in cand_tr], dtype=np.float32)
        cand_params = np.column_stack([
            np.full(n_candidates, E_fixed),
            np.full(n_candidates, nu_fixed),
            cand_tr,
            cand_ki,
        ]).astype(np.float32)

        scores = score_by_uncertainty(best_trainer, cand_params, coords_enriched)
        top_idx = np.argsort(scores)[::-1][:n_per_al_round]
        selected = cand_params[top_idx]
        print(f"   Selected {n_per_al_round} most-uncertain parameter sets (scores: {scores[top_idx].round(4)})")

        # --- run FEA for selected candidates ----------------------------
        print(f"   Running FEA for {n_per_al_round} new samples...")
        new_p, new_o = [], []
        for p in selected:
            pv, disp, _, _ = run_fem(E_fixed, nu_fixed, float(p[2]))  # p[2] = traction
            if pv is not None:
                new_p.append(pv)
                new_o.append(disp)

        params = np.vstack([params, np.array(new_p)])
        outputs_list = outputs_list + new_o

        al_log[-1]["selected_params"] = selected
        al_log[-1]["scores"] = scores[top_idx]

    loop.close()

    # --- visualization ---------------------------------------------------
    print("\n3. Generating visualization...")
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    title_samples = f"{al_log[0]['n_samples']}→{al_log[-1]['n_samples']}"
    best_loss = al_log[-1]["test_loss"]
    fig.suptitle(
        f"PIANO: Active Learning V-Notch  |  FEM samples: {title_samples}  |  "
        f"Best test loss: {best_loss:.3f}",
        fontsize=13, fontweight="bold"
    )

    # --- panel 1: cumulative loss curves ---
    ax1 = fig.add_subplot(gs[0, 0])
    if all_train_losses:
        ax1.semilogy(all_train_losses, label="train", color="steelblue", lw=1.2)
        ax1.semilogy(all_test_losses,  label="test",  color="tomato",    lw=1.2)
    ax1.set_xlabel("Epoch (cumulative)")
    ax1.set_ylabel("Loss (log)")
    ax1.set_title("Loss Across All AL+HPO Rounds")
    ax1.legend(fontsize=8)

    # --- panel 2: HPO convergence per AL round ---
    ax2 = fig.add_subplot(gs[0, 1])
    al_colors_hpo = ["steelblue", "darkorange", "green", "red", "purple"]
    al_round_losses = {}  # al_idx -> list of test losses per HPO call
    for (a_idx, h_idx, tr, te) in hpo_log:
        al_round_losses.setdefault(a_idx, []).append(te)
    x_offset = 1
    for a_idx in sorted(al_round_losses):
        losses = al_round_losses[a_idx]
        xs = list(range(x_offset, x_offset + len(losses)))
        n_s = al_log[a_idx]["n_samples"] if a_idx < len(al_log) else "?"
        ax2.plot(xs, losses, "o-", color=al_colors_hpo[a_idx % len(al_colors_hpo)],
                 lw=2, ms=6, label=f"AL r{a_idx} ({n_s} FEM)")
        if x_offset > 1:
            ax2.axvline(x_offset - 0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
        x_offset += len(losses)
    ax2.set_xlabel("HPO call (cumulative)")
    ax2.set_ylabel("Test loss")
    ax2.set_title("HPO Convergence per AL Round")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- panel 3: physics weight progression ---
    ax3 = fig.add_subplot(gs[0, 2])
    phys_terms = ["equilibrium", "energy", "traction_free", "stress_intensity", "near_tip"]
    phys_colors = ["royalblue", "darkorange", "green", "red", "purple"]
    if physics_log:
        xs_ph = [entry[0] for entry in physics_log]
        for term, col in zip(phys_terms, phys_colors):
            ys = [entry[1].get(term, 0.0) for entry in physics_log]
            if any(v > 0 for v in ys):
                ax3.semilogy([x + 0.5 for x in xs_ph], [max(v, 1e-8) for v in ys],
                             "o-", color=col, lw=1.5, ms=5, label=term)
    ax3.set_xlabel("HPO call (cumulative)")
    ax3.set_ylabel("Physics weight (log)")
    ax3.set_title("Physicist: Sequential Weight Enabling")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.3)

    # --- bottom row: surrogate vs GT vs error ---
    # Use params[0] / outputs_list[0]: first initial sample, always in training set
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    if best_trainer is not None and triangles is not None:
        # Pick the median sample index to avoid the very first or last
        vis_idx = len(params) // 2
        test_p = params[vis_idx:vis_idx+1].copy()
        test_o = outputs_list[vis_idx]
        pred = best_trainer.predict(test_p, coords_enriched)
        if pred.ndim == 3:
            pred = pred[0]

        E_v, nu_v = E_fixed, nu_fixed
        vm_pred = _compute_von_mises_nodal(pred, coords, triangles, E_v, nu_v)
        vm_gt   = _compute_von_mises_nodal(test_o, coords, triangles, E_v, nu_v)
        gt_ref = np.percentile(np.abs(vm_gt), 95) + 1e-12  # robust ref avoids near-zero denominator
        error_pct = np.abs(vm_pred - vm_gt) / gt_ref * 100.0  # relative error [% of 95th-pct GT]

        # Share color scale between surrogate and GT so comparison is honest
        vm_max = max(vm_gt.max(), vm_pred.max())

        def _tripcolor(ax, vals, title, cmap="plasma", vmax=None, label="Pa"):
            tc = ax.tripcolor(coords[:, 0], coords[:, 1],
                              triangles, vals, cmap=cmap, shading="gouraud",
                              vmin=0, vmax=vmax)
            plt.colorbar(tc, ax=ax, label=label)
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_xlabel("x"); ax.set_ylabel("y")

        _tripcolor(ax4, vm_pred, "Surrogate: Von Mises Stress", vmax=vm_max)
        _tripcolor(ax5, vm_gt,   "Ground Truth: Von Mises Stress", vmax=vm_max)

        # Panel 6: max stress bar chart
        max_pred = vm_pred.max()
        max_gt   = vm_gt.max()
        max_err_pct = abs(max_pred - max_gt) / (max_gt + 1e-12) * 100.0
        bars = ax6.bar(["Surrogate", "Ground Truth"], [max_pred, max_gt],
                       color=["tomato", "steelblue"], edgecolor="black", width=0.5)
        for bar, val in zip(bars, [max_pred, max_gt]):
            ax6.text(bar.get_x() + bar.get_width() / 2, val * 1.01,
                     f"{val:.2e}", ha="center", va="bottom", fontsize=9)
        ax6.set_ylabel("Max von Mises stress [Pa]")
        ax6.set_title(f"Peak Stress Comparison\nError: {max_err_pct:.1f}%")
        ax6.set_ylim(0, max(max_pred, max_gt) * 1.15)
        ax6.grid(True, alpha=0.3, axis="y")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved to: {output_path}")
    print(f"  FEM samples: {al_log[0]['n_samples']} → {al_log[-1]['n_samples']}")
    print(f"  AL rounds: {n_al_rounds}  ({n_per_al_round} new FEM per round)")
    print(f"  Best test loss: {al_log[-1]['test_loss']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PIANO: Agentic HPO SciML — Edge Crack Phase Field Demo")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Number of FEM samples (default: 30)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Epochs per HPO round (default: 80)")
    parser.add_argument("--rounds", type=int, default=8,
                        help="Max HPO rounds (default: 8)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")
    parser.add_argument("--use-real-llm", action="store_true",
                        help="Use AnthropicProvider (ANTHROPIC_API_KEY required)")
    parser.add_argument("--use-claude-code", action="store_true",
                        help="Use ClaudeCodeProvider (claude CLI session)")
    args = parser.parse_args()

    output = args.output or str(PROJECT_ROOT / "tests" / "test_outputs" / "agentic_vnotch_demo.png")

    run_agentic_loop_demo(
        n_samples=args.n_samples,
        epochs_per_round=args.epochs,
        max_hpo_rounds=args.rounds,
        output_file=output,
        use_real_llm=args.use_real_llm,
        use_claude_code=args.use_claude_code,
    )
