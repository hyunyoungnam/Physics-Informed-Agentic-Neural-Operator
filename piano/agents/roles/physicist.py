"""
Physicist Agent implementation.

Manages all physics loss weights as soft regularization on top of clean FEA data.

Philosophy
----------
Ground truth data comes from a reliable FEM solver — all physics is already
embedded in the labels. Physics terms therefore do NOT add information; they
add soft regularization that can improve generalization once the surrogate has
saturated data-driven learning.

Target: physics loss contribution ≈ 1–5% of data loss.
  < 1%  → weight is irrelevant (too small to matter)
  1–5%  → ideal regularization regime
  > 10% → weight is overriding the data signal (reduce immediately)

All 5 terms are always mathematically present; the physicist controls only
their weights. Crack-specific terms (traction_free, near_tip [PD],
j_integral) start at 0.0 and are enabled only once the base physics
(equilibrium + energy) is stable in the 1-5% ratio range.
Architecture and optimizer hyperparameters are the Architect's responsibility.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from piano.agents.base import BaseAgent, AgentContext, AgentRole
from piano.agents.roles.hyperparameter_critic import CritiqueResult, TrainingIssue


class PhysicsIssue(Enum):
    """Physics weight calibration states."""
    TEST_LOSS_PLATEAUED = auto()       # Test loss stalled — good time to add physics
    PHYSICS_RATIO_TOO_HIGH = auto()    # Physics > 10% of data loss — reduce weights
    PHYSICS_RATIO_TOO_LOW = auto()     # Physics < 0.5% of data loss — can increase
    NEXT_CRACK_TERM_READY = auto()     # General PINO stable — enable next crack term
    PHYSICS_DESTABILIZING = auto()     # Loss spike after enabling a term — reduce
    NONE = auto()


@dataclass
class PhysicsProposal:
    """A physics configuration proposal from the Physicist."""
    changes: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    physics_diagnosis: str = ""
    expected_impact: str = ""
    confidence: str = "medium"
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changes": self.changes,
            "reasoning": self.reasoning,
            "physics_diagnosis": self.physics_diagnosis,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
        }


PHYSICIST_SYSTEM = """You are an expert computational physicist specializing in fracture mechanics and physics-informed neural networks for solid mechanics.

Your role is to calibrate ALL physics loss weights for a surrogate trained on clean FEM data. The Architect handles neural network architecture and optimizer settings — you handle physics weights exclusively.

## Core Philosophy

The ground-truth data comes from a reliable FEM solver. All physics (equilibrium, BCs, stress field) is already embedded in the labels. Physics loss terms therefore act as **soft regularization** — they do not add new information, they nudge the surrogate toward physically consistent generalization.

**Consequence**: Physics weights must be small. They should lightly shape the loss landscape, not compete with the data signal.

**Target ratio**: physics loss contribution = 1–5% of data loss.
- If physics/data > 10%: the surrogate is fitting physics instead of data — reduce weights.
- If physics/data < 0.5%: the term has no effect — can increase slightly if test loss is plateauing.
- If physics/data is 1–5%: ideal regime — hold unless test loss is degrading.

**When to enable new terms**: Only when test loss has plateaued (< 2% improvement over the last 2 rounds). If the surrogate is still improving from data alone, adding physics adds noise.

## Physics Loss Terms

All 5 terms are always mathematically present in the loss — only weights are varied.
Crack-specific terms start at 0.0 and are enabled only after base physics is stable.

### equilibrium — Equilibrium PDE Residual
- Enforces: ∇·σ = 0 (balance of linear momentum)
- Enable when: test loss has plateaued
- Starting weight: 1e-3
- **CRACK/FRACTURE WARNING**: Set `equilibrium = 0.0` when crack faces are present.
  ∇u is discontinuous across crack faces — autograd produces unbounded strong-form
  residuals at the exact nodes where the network is least accurate.
  Use `energy` (variational) and `near_tip` (peridynamic) instead.
  `equilibrium` is only appropriate for smooth, crack-free problems.

### energy — Elastic Energy Norm
- Enforces: displacement minimizes elastic strain energy
- Enable when: equilibrium is active and ratio is in target range
- Starting weight: 5e-4

### traction_free — Traction-Free Crack Faces
- Enforces: σ_yy = σ_xy = 0 on crack/notch face elements
- Enable when: equilibrium + energy ratio is already in 1–5% range
- Starting weight: 1e-3

### near_tip — Peridynamic Equilibrium Residual
- Enforces: bond-based peridynamic equilibrium Σ_j (1−d_ij)² s_ij ê_ij = 0 at every node
- Works on the full mesh; damaged bonds (d→1) contribute zero — no spurious gradient inside crack
- Valid outside the K-dominant zone; works for phase field even when crack_config is None
- Enable when: base physics ratio is in range and test loss plateaued
- Starting weight: 5e-4
- horizon_factor (geometry, set in CrackConfig): δ = horizon_factor × h_avg; default 3.0
  - Smaller (2.0): more local, fewer bonds, faster but misses longer-range imbalance
  - Larger (4.0–5.0): more nonlocal, catches broader imbalance at higher compute cost

### j_integral — J-Integral Conservation
- Enforces: domain J-integral = K_I²/E (plane stress)
- Enable when: base physics ratio is in range and test loss plateaued
- Starting weight: 2e-4

### variational_weight — Variational AT-2 Elastic Energy
- Enforces: ∫ g(d) ½ ε:Cε dΩ ≈ W_ext (no displacement labels needed)
- V-DeepONet (Goswami 2022): achieves <1% error on AT-2 phase field with 11 samples
- Requires damage field in data pipeline (set via trainer.set_auxiliary_data)
- Acts as regularizer alongside MSE; preferred over `equilibrium` for phase field problems
- Starting weight: 1e-3; increase when predicted displacement in damaged zones is inconsistent

## Weight Calibration Rules

1. Compute current ratio = pino_loss[-1] / train_loss[-1]
2. If ratio > 10%: reduce ALL active physics weights by 3–5× (top priority)
3. If ratio > 5%: reduce the largest-weight active term by 2×
4. If ratio < 0.5% AND test loss is plateauing: increase the lowest active weight by 2×
5. If ratio is 1–5% AND test loss is still improving: hold all weights (data is doing the work)
6. If ratio is 1–5% AND test loss plateaued: may enable one crack-specific term at 1e-3
7. Never disable a term that was previously stable — only reduce its weight

## Priority Order
Weight recalibration > enabling new terms. Fix the ratio first, then add terms.

Output your proposals in structured format."""


PHYSICIST_PROMPT = """## Current Physics Loss Configuration
- equilibrium: {equilibrium}
- energy: {energy}
- traction_free: {traction_free}
- near_tip: {near_tip}
- j_integral: {j_integral}
- variational_weight: {variational_weight}

## Loss History (sampled)
- Train losses (data only): {train_losses}
- Test losses: {test_losses}
- Physics losses: {pino_losses}

## Current Physics/Data Ratio
- Latest ratio: {physics_ratio:.1%}  (target: 1–5%; above 10% → reduce, below 0.5% → can increase)
- Test loss trend: {test_trend}

## Training Diagnostics
**Primary Issue**: {primary_issue}
**Severity**: {severity}
**Diagnosis**: {diagnosis}

## Problem Context
- Problem type: {problem_type}
- Dataset size: {dataset_size} samples

## Previous Physics Configs Tried
{previous_configs}

## Your Task

Calibrate physics loss weights to keep the physics/data ratio in the 1–5% range.

Rules:
1. If ratio > 10%: reduce weights — this is the top priority
2. If ratio 1–5% and test loss improving: hold all weights (data is still learning)
3. If ratio 1–5% and test loss plateaued: consider enabling one crack term at 1e-3
4. If ratio < 0.5% and test loss plateaued: increase lowest active weight by 2×
5. Crack terms (traction_free, near_tip, j_integral) start at 0.0.
   Enable only when equilibrium + energy ratio is already in the 1–5% range.

Format your response as:
```
PHYSICS_DIAGNOSIS: [Current ratio assessment and what it means for training]

CHANGES:
- equilibrium: [value] (reason)
- energy: [value] (reason)
- traction_free: [value] (reason)
- near_tip: [value] (reason)
- j_integral: [value] (reason)
- variational_weight: [value] (reason)

REASONING: [Why these specific weight values]
EXPECTED_IMPACT: [Expected effect on physics/data ratio and test loss]
CONFIDENCE: [low|medium|high]
```

Only include parameters you want to change.
"""


class PhysicistAgent(BaseAgent[PhysicsProposal]):
    """
    Physicist Agent for physics loss weight calibration.

    Targets physics/data loss ratio of 1–5%.
    Enforces crack term ordering (traction_free → near_tip → j_integral)
    for physical coherence, but uses small weights so physics acts as soft regularization.
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.4,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.PHYSICIST,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return PHYSICIST_SYSTEM

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        current_config: Dict[str, Any] = kwargs.get("current_config", {})
        critique: CritiqueResult = kwargs.get("critique", CritiqueResult())
        training_history = kwargs.get("training_history", None)
        dataset_size: int = kwargs.get("dataset_size", 0)
        problem_type: str = kwargs.get("problem_type", "crack")
        previous_configs: List[Dict] = kwargs.get("previous_configs", [])
        debate_context: str = kwargs.get("debate_context", "")

        train_losses_raw = []
        test_losses_raw = []
        pino_losses_raw = []

        if training_history:
            train_losses_raw = getattr(training_history, 'train_losses', [])
            test_losses_raw = getattr(training_history, 'test_losses', [])
            pino_losses_raw = getattr(training_history, 'pino_losses', [])

        train_losses = self._sample_losses(train_losses_raw)
        test_losses = self._sample_losses(test_losses_raw)
        pino_losses = self._sample_losses(pino_losses_raw)

        # Compute physics/data ratio from most recent values
        physics_ratio = 0.0
        if train_losses_raw and pino_losses_raw:
            recent_data = train_losses_raw[-1]
            recent_pino = pino_losses_raw[-1]
            if recent_data > 1e-10:
                physics_ratio = recent_pino / recent_data

        # Describe test loss trend over last 3 rounds
        test_trend = "unknown"
        if len(test_losses_raw) >= 3:
            recent = test_losses_raw[-3:]
            improvement = (recent[0] - recent[-1]) / (recent[0] + 1e-10)
            if improvement > 0.02:
                test_trend = f"improving ({improvement:.1%} over last 3 epochs)"
            elif improvement < -0.01:
                test_trend = f"worsening ({-improvement:.1%} increase)"
            else:
                test_trend = f"plateaued (< 2% change over last 3 epochs)"

        prev_str = "None"
        if previous_configs:
            _physics_keys = {"equilibrium", "energy", "traction_free",
                             "near_tip", "j_integral", "variational_weight"}
            prev_lines = []
            for i, cfg in enumerate(previous_configs):
                phys_cfg = {k: v for k, v in cfg.get("config", {}).items() if k in _physics_keys}
                result = cfg.get("result", "unknown")
                prev_lines.append(f"  Attempt {i+1}: {phys_cfg} -> {result}")
            prev_str = "\n".join(prev_lines) if prev_lines else "None"

        prompt = PHYSICIST_PROMPT.format(
            equilibrium=current_config.get("equilibrium", 0.0),
            energy=current_config.get("energy", 0.0),
            traction_free=current_config.get("traction_free", 0.0),
            near_tip=current_config.get("near_tip", 0.0),
            j_integral=current_config.get("j_integral", 0.0),
            variational_weight=current_config.get("variational_weight", 0.0),
            train_losses=train_losses,
            test_losses=test_losses,
            pino_losses=pino_losses,
            physics_ratio=physics_ratio,
            test_trend=test_trend,
            primary_issue=critique.primary_issue.name,
            severity=critique.severity,
            diagnosis=critique.diagnosis,
            problem_type=problem_type,
            dataset_size=dataset_size,
            previous_configs=prev_str,
        )
        if debate_context:
            debate_section = (
                "\n## Debate Context (Agent Observations — Rounds 1-2)\n"
                + debate_context
                + "\nIMPORTANT: Your proposal MUST be consistent with the analysis above.\n"
            )
            prompt = debate_section + "\n" + prompt
        return prompt

    def _sample_losses(self, losses: List[float], max_samples: int = 10) -> str:
        if not losses:
            return "[]"
        if len(losses) <= max_samples:
            return str([round(l, 6) for l in losses])
        step = len(losses) // max_samples
        return str([round(losses[i * step], 6) for i in range(max_samples)])

    def parse_response(self, response: str) -> PhysicsProposal:
        proposal = PhysicsProposal(raw_response=response)

        diag_match = re.search(
            r'PHYSICS_DIAGNOSIS:\s*(.*?)(?=CHANGES:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if diag_match:
            proposal.physics_diagnosis = diag_match.group(1).strip()

        changes_match = re.search(
            r'CHANGES:\s*(.*?)(?=REASONING:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if changes_match:
            proposal.changes = self._parse_changes(changes_match.group(1))

        reasoning_match = re.search(
            r'REASONING:\s*(.*?)(?=EXPECTED_IMPACT:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            proposal.reasoning = reasoning_match.group(1).strip()

        impact_match = re.search(
            r'EXPECTED_IMPACT:\s*(.*?)(?=CONFIDENCE:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if impact_match:
            proposal.expected_impact = impact_match.group(1).strip()

        confidence_match = re.search(
            r'CONFIDENCE:\s*(low|medium|high)',
            response, re.IGNORECASE
        )
        if confidence_match:
            proposal.confidence = confidence_match.group(1).lower()

        return proposal

    def _parse_changes(self, text: str) -> Dict[str, Any]:
        changes = {}
        for param in ['equilibrium', 'energy', 'traction_free',
                      'near_tip', 'j_integral', 'variational_weight']:
            match = re.search(rf'{param}:\s*([0-9.e-]+)', text, re.IGNORECASE)
            if match:
                try:
                    changes[param] = float(match.group(1))
                except ValueError:
                    pass
        return changes

    def _compute_physics_ratio(self, training_history) -> float:
        """Latest pino_loss / train_loss ratio."""
        train = getattr(training_history, 'train_losses', [])
        pino = getattr(training_history, 'pino_losses', [])
        if not train or not pino:
            return 0.0
        data_val = train[-1]
        return pino[-1] / data_val if data_val > 1e-10 else 0.0

    def _test_loss_plateaued(self, training_history, window: int = 3) -> bool:
        """True if test loss improved < 2% over the last `window` entries."""
        losses = getattr(training_history, 'test_losses', [])
        if len(losses) < window:
            return False
        recent = losses[-window:]
        improvement = (recent[0] - recent[-1]) / (recent[0] + 1e-10)
        return improvement < 0.02

    def detect_physics_issues(
        self,
        training_history,
        current_config: Dict[str, Any],
    ) -> List[PhysicsIssue]:
        """
        Heuristic detection of physics weight calibration issues.

        Priority order:
        1. Instability (loss spike) → PHYSICS_DESTABILIZING
        2. Ratio too high (> 10%) → PHYSICS_RATIO_TOO_HIGH
        3. Test loss plateaued → TEST_LOSS_PLATEAUED (enable new term or adjust)
        4. Ratio too low with plateau → PHYSICS_RATIO_TOO_LOW
        """
        issues = []
        train_losses = getattr(training_history, 'train_losses', [])
        if not train_losses or len(train_losses) < 5:
            return issues

        # 1. Instability: recent loss spiked
        recent = train_losses[-5:]
        is_unstable = recent[-1] >= 2.0 * min(recent) + 1e-12

        all_physics = [
            current_config.get(k, 0.0)
            for k in ["equilibrium", "energy", "traction_free",
                      "near_tip", "j_integral"]
        ]
        if is_unstable and any(v > 0 for v in all_physics):
            issues.append(PhysicsIssue.PHYSICS_DESTABILIZING)
            return issues

        ratio = self._compute_physics_ratio(training_history)
        plateaued = self._test_loss_plateaued(training_history)

        # 2. Physics overwhelming data
        if ratio > 0.10:
            issues.append(PhysicsIssue.PHYSICS_RATIO_TOO_HIGH)
            return issues

        # 3. Test loss plateaued — consider enabling next crack term
        if plateaued:
            eq = current_config.get("equilibrium", 0.0)
            en = current_config.get("energy", 0.0)
            tf = current_config.get("traction_free", 0.0)
            nt = current_config.get("near_tip", 0.0)
            ji = current_config.get("j_integral", 0.0)

            # Ordering: equilibrium/energy → traction_free → near_tip → j_integral
            if eq == 0.0 or en == 0.0 or (eq > 0 and en > 0 and tf == 0.0):
                issues.append(PhysicsIssue.TEST_LOSS_PLATEAUED)
            elif tf > 0 and nt == 0.0:
                issues.append(PhysicsIssue.NEXT_CRACK_TERM_READY)
            elif nt > 0 and ji == 0.0:
                issues.append(PhysicsIssue.NEXT_CRACK_TERM_READY)

        # 4. Ratio too low — weights have no effect
        if ratio > 0 and ratio < 0.005 and plateaued:
            issues.append(PhysicsIssue.PHYSICS_RATIO_TOO_LOW)

        return issues

    def should_consult(
        self,
        training_history,
        current_config: Dict[str, Any],
    ) -> bool:
        return len(self.detect_physics_issues(training_history, current_config)) > 0

    async def propose_physics_config(
        self,
        context: AgentContext,
        current_config: Dict[str, Any],
        critique: CritiqueResult,
        training_history=None,
        dataset_size: int = 0,
        problem_type: str = "crack",
        has_singularity: bool = True,
        previous_configs: Optional[List[Dict]] = None,
        debate_context: str = "",
    ) -> PhysicsProposal:
        return await self.execute(
            context,
            current_config=current_config,
            critique=critique,
            training_history=training_history,
            dataset_size=dataset_size,
            problem_type=problem_type,
            has_singularity=has_singularity,
            previous_configs=previous_configs or [],
            debate_context=debate_context,
        )

    async def analyze(
        self,
        context: AgentContext,
        current_config: Dict[str, Any],
        training_history,
        debate_context: str,
    ) -> str:
        """Round 2: analyze physics situation, no weight proposals."""
        if self._llm_provider is None:
            raise RuntimeError("LLM provider not set")
        ratio = self._compute_physics_ratio(training_history)
        plateaued = self._test_loss_plateaued(training_history)
        active_terms = [k for k in ("equilibrium", "energy", "traction_free",
                                    "near_tip", "j_integral", "variational_weight")
                        if current_config.get(k, 0.0) > 0]
        prompt = (
            "ROUND 2 (ANALYSIS) — Do NOT propose any weight values.\n\n"
            f"## Round 1 Observations\n{debate_context}\n\n"
            f"## Current Physics State\n"
            f"  Active terms: {active_terms or ['none']}\n"
            f"  Physics/data ratio: {ratio:.1%} (target: 1–5%)\n"
            f"  Test loss plateaued: {plateaued}\n\n"
            "Analyze the physics loss situation:\n"
            "- Is the surrogate ready for additional physics terms? "
            "(test loss must be clearly plateaued)\n"
            "- Are active physics weights causing instability or dominating training?\n"
            "- What does the current ratio suggest about the physics/data balance?\n\n"
            "3-5 sentences. No proposed weight values."
        )
        response = await self._llm_provider.generate(
            system_prompt=PHYSICIST_SYSTEM,
            user_prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=512,
        )
        return f"[PHYSICIST — Round 2]\n{response.content}"

    def analyze_sync(
        self,
        current_config: Dict[str, Any],
        training_history,
        debate_context: str,
    ) -> str:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.analyze(AgentContext(), current_config, training_history, debate_context)
            )
        finally:
            loop.close()
