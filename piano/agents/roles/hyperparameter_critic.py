"""
Hyperparameter Critic Agent implementation.

The Hyperparameter Critic analyzes training curves and metrics to diagnose
issues such as overfitting, underfitting, slow convergence, or instability.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from piano.agents.base import BaseAgent, AgentContext, AgentRole


class TrainingIssue(Enum):
    """Types of training issues the critic can diagnose."""
    OVERFITTING = auto()
    UNDERFITTING = auto()
    SLOW_CONVERGENCE = auto()
    UNSTABLE_TRAINING = auto()
    LOSS_PLATEAU = auto()
    GRADIENT_EXPLOSION = auto()
    LEARNING_RATE_TOO_HIGH = auto()
    LEARNING_RATE_TOO_LOW = auto()
    INSUFFICIENT_CAPACITY = auto()
    EXCESSIVE_CAPACITY = auto()
    POOR_INITIALIZATION = auto()
    NONE = auto()


@dataclass
class CritiqueResult:
    """Result of training analysis by the critic."""
    issues: List[TrainingIssue] = field(default_factory=list)
    primary_issue: TrainingIssue = TrainingIssue.NONE
    severity: str = "low"  # "low", "medium", "high", "critical"
    diagnosis: str = ""
    recommendations: List[str] = field(default_factory=list)
    metrics_analysis: Dict[str, str] = field(default_factory=dict)
    should_retrain: bool = False
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issues": [i.name for i in self.issues],
            "primary_issue": self.primary_issue.name,
            "severity": self.severity,
            "diagnosis": self.diagnosis,
            "recommendations": self.recommendations,
            "metrics_analysis": self.metrics_analysis,
            "should_retrain": self.should_retrain,
        }


@dataclass
class TrainingHistory:
    """Training history data for analysis."""
    train_losses: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    pino_losses: List[float] = field(default_factory=list)
    pino_term_losses: Dict[str, List[float]] = field(default_factory=dict)
    ensemble_std: float = 0.0
    epochs_completed: int = 0
    best_test_loss: float = float('inf')
    final_train_loss: float = float('inf')
    final_test_loss: float = float('inf')
    convergence_epoch: Optional[int] = None
    has_nan: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Generate a text summary of training history."""
        lines = [
            f"Epochs completed: {self.epochs_completed}",
            f"Final train loss: {self.final_train_loss:.6f}",
            f"Final test loss: {self.final_test_loss:.6f}",
            f"Best test loss: {self.best_test_loss:.6f}",
        ]

        if self.has_nan:
            lines.append("WARNING: NaN values detected during training")

        if self.convergence_epoch:
            lines.append(f"Convergence epoch: {self.convergence_epoch}")

        # Add loss trend analysis
        if len(self.train_losses) >= 10:
            early_train = sum(self.train_losses[:5]) / 5
            late_train = sum(self.train_losses[-5:]) / 5
            train_reduction = (early_train - late_train) / early_train * 100 if early_train > 0 else 0
            lines.append(f"Train loss reduction: {train_reduction:.1f}%")

        if len(self.test_losses) >= 10:
            early_test = sum(self.test_losses[:5]) / 5
            late_test = sum(self.test_losses[-5:]) / 5
            test_reduction = (early_test - late_test) / early_test * 100 if early_test > 0 else 0
            lines.append(f"Test loss reduction: {test_reduction:.1f}%")

            # Gap analysis (overfitting indicator)
            gap = late_test - late_train if late_train > 0 else 0
            lines.append(f"Train-test gap: {gap:.6f}")

        # Per-term PINO losses
        if self.pino_term_losses:
            lines.append("\nPer-term PINO losses (initial → final epoch):")
            for term, losses in self.pino_term_losses.items():
                if losses:
                    lines.append(f"  {term}: {losses[0]:.6f} → {losses[-1]:.6f}")

        # Ensemble disagreement
        if self.ensemble_std > 0:
            ratio = self.ensemble_std / (self.final_test_loss + 1e-12)
            variance_label = "high" if ratio > 0.5 else "low"
            lines.append(f"\nEnsemble disagreement (mean std): {self.ensemble_std:.6f}")
            lines.append(f"  Uncertainty/test-loss ratio: {ratio:.3f} ({variance_label} model variance)")

        # Add metrics
        if self.metrics:
            lines.append("\nEvaluation metrics:")
            for key, value in self.metrics.items():
                lines.append(f"  {key}: {value:.6f}")

        return "\n".join(lines)


HYPERPARAMETER_CRITIC_SYSTEM = """You are an expert ML training analyst specializing in neural operator models for physics simulations.

Your role is to analyze training curves and metrics to diagnose issues and recommend hyperparameter changes.

## Analysis Framework

1. **Loss Curve Analysis**
   - Compare train vs test loss trends
   - Identify convergence patterns
   - Detect plateaus and instabilities

2. **Per-Term PINO Loss Analysis**
   When per-term PINO losses are provided, analyze them separately:
   - `elasticity_loss`: PDE equilibrium + energy-norm residual. If high and not decreasing,
     the model is not satisfying linear elasticity — consider increasing pino_weight or fixing
     the physics loss formulation.
   - `crack_loss`: Fracture mechanics terms (traction-free BC, peridynamic near-tip, J-integral).
     If high relative to elasticity_loss, the model is violating fracture BCs specifically.
     The Physicist agent owns these weights — flag for physics reconfiguration.
   A term that stays flat (not decreasing) while data loss improves means it's too weak to
   influence training — the weight needs to increase.

3. **Ensemble Disagreement Analysis**
   `ensemble_std` is the mean standard deviation across ensemble members on the test set.
   - High uncertainty/loss ratio (> 0.5): Members disagree substantially — model has high
     variance, likely due to insufficient data or too large a model for the dataset size.
     Reduce capacity or increase regularization.
   - Low uncertainty/loss ratio (< 0.1): Members agree but loss is still high — systematic
     bias (underfitting) rather than variance. Increase capacity.
   - Moderate ratio (0.1–0.5): Healthy ensemble diversity.

4. **Issue Diagnosis**
   - OVERFITTING: Test loss increases while train loss decreases
   - UNDERFITTING: Both losses remain high, model not learning
   - SLOW_CONVERGENCE: Gradual improvement but far from optimal
   - UNSTABLE_TRAINING: Large fluctuations in loss values
   - LOSS_PLATEAU: No improvement for many epochs
   - GRADIENT_EXPLOSION: NaN values or sudden spikes
   - LEARNING_RATE_TOO_HIGH: Oscillating loss, unstable training
   - LEARNING_RATE_TOO_LOW: Very slow improvement
   - INSUFFICIENT_CAPACITY: Underfitting despite long training
   - EXCESSIVE_CAPACITY: Quick overfitting, large train-test gap

5. **Severity Assessment**
   - critical: Training failing (NaN, diverging)
   - high: Significant issue requiring immediate fix
   - medium: Suboptimal performance, worth addressing
   - low: Minor issue or acceptable performance

6. **Recommendations**
   - Be specific about what to change and why
   - Reference per-term PINO losses and ensemble std in your diagnosis
   - Consider trade-offs (capacity vs generalization)
   - Prioritize high-impact changes

Output your analysis in structured format with clear sections."""


HYPERPARAMETER_CRITIC_PROMPT = """## Training History Analysis

**Model Configuration:**
{config_summary}

**Training Summary:**
{training_summary}

**Loss Curves (sampled):**
Train losses: {train_losses_sample}
Test losses: {test_losses_sample}

**Per-Term PINO Losses (sampled, if available):**
{pino_terms_section}

**Ensemble Disagreement:**
{ensemble_section}

**Previous Attempts (if any):**
{previous_attempts}

## Your Task

Analyze this training history and provide:

1. **Diagnosis**: What issues do you observe? Reference per-term PINO losses and ensemble std.
2. **Primary Issue**: The main problem to address
3. **Severity**: critical/high/medium/low
4. **Recommendations**: Specific changes to hyperparameters
5. **Should Retrain**: true/false - is retraining with different config warranted?

Format your response as:
```
DIAGNOSIS: [Your detailed diagnosis, including PINO term analysis and ensemble variance interpretation]
PRIMARY_ISSUE: [OVERFITTING|UNDERFITTING|SLOW_CONVERGENCE|UNSTABLE_TRAINING|LOSS_PLATEAU|GRADIENT_EXPLOSION|LEARNING_RATE_TOO_HIGH|LEARNING_RATE_TOO_LOW|INSUFFICIENT_CAPACITY|EXCESSIVE_CAPACITY|NONE]
SEVERITY: [critical|high|medium|low]
RECOMMENDATIONS:
- [Specific recommendation 1]
- [Specific recommendation 2]
- [...]
SHOULD_RETRAIN: [true|false]
METRICS_ANALYSIS:
- train_test_gap: [analysis]
- convergence_rate: [analysis]
- stability: [analysis]
- pino_effectiveness: [which PINO terms are working / which are too weak]
- ensemble_variance: [whether ensemble spread indicates variance or bias]
```
"""


class HyperparameterCriticAgent(BaseAgent[CritiqueResult]):
    """
    Hyperparameter Critic Agent for training analysis.

    Responsibilities:
    1. Analyze training loss curves and metrics
    2. Diagnose training issues (overfitting, underfitting, etc.)
    3. Assess severity of issues
    4. Recommend specific hyperparameter changes
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            role=AgentRole.HYPERPARAMETER_CRITIC,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_system_prompt(self) -> str:
        return HYPERPARAMETER_CRITIC_SYSTEM

    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        history: TrainingHistory = kwargs.get("training_history", TrainingHistory())
        config: Dict[str, Any] = kwargs.get("config", {})
        previous_attempts: List[Dict] = kwargs.get("previous_attempts", [])

        # Format config summary
        config_summary = "\n".join([f"  {k}: {v}" for k, v in config.items()])

        # Sample losses for prompt (avoid overwhelming the LLM)
        train_sample = self._sample_losses(history.train_losses)
        test_sample = self._sample_losses(history.test_losses)

        # Per-term PINO section
        if history.pino_term_losses:
            pino_lines = []
            for term, losses in history.pino_term_losses.items():
                sampled = self._sample_losses(losses)
                pino_lines.append(f"  {term}: {sampled}")
            pino_terms_section = "\n".join(pino_lines)
        else:
            pino_terms_section = "Not available (PINO not active or terms not tracked)"

        # Ensemble section
        if history.ensemble_std > 0:
            ratio = history.ensemble_std / (history.final_test_loss + 1e-12)
            ensemble_section = (
                f"Mean ensemble std: {history.ensemble_std:.6f} | "
                f"Uncertainty/test-loss ratio: {ratio:.3f}"
            )
        else:
            ensemble_section = "Not available (single model or ensemble std not computed)"

        # Format previous attempts
        prev_str = "None"
        if previous_attempts:
            prev_lines = []
            for i, attempt in enumerate(previous_attempts):
                prev_lines.append(f"Attempt {i+1}: {attempt.get('summary', 'No summary')}")
            prev_str = "\n".join(prev_lines)

        return HYPERPARAMETER_CRITIC_PROMPT.format(
            config_summary=config_summary,
            training_summary=history.to_summary(),
            train_losses_sample=train_sample,
            test_losses_sample=test_sample,
            pino_terms_section=pino_terms_section,
            ensemble_section=ensemble_section,
            previous_attempts=prev_str,
        )

    def _sample_losses(self, losses: List[float], max_samples: int = 20) -> str:
        """Sample losses for display, showing key points."""
        if not losses:
            return "[]"

        if len(losses) <= max_samples:
            return str([round(l, 6) for l in losses])

        # Sample at regular intervals, always include first and last
        indices = [0]
        step = len(losses) // (max_samples - 2)
        for i in range(1, max_samples - 1):
            indices.append(min(i * step, len(losses) - 1))
        indices.append(len(losses) - 1)

        sampled = [round(losses[i], 6) for i in indices]
        return str(sampled)

    def parse_response(self, response: str) -> CritiqueResult:
        """Parse the LLM response into a CritiqueResult."""
        result = CritiqueResult(raw_response=response)

        # Extract diagnosis
        diagnosis_match = re.search(
            r'DIAGNOSIS:\s*(.*?)(?=PRIMARY_ISSUE:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if diagnosis_match:
            result.diagnosis = diagnosis_match.group(1).strip()

        # Extract primary issue
        issue_match = re.search(
            r'PRIMARY_ISSUE:\s*(\w+)',
            response, re.IGNORECASE
        )
        if issue_match:
            issue_str = issue_match.group(1).upper()
            try:
                result.primary_issue = TrainingIssue[issue_str]
                if result.primary_issue != TrainingIssue.NONE:
                    result.issues.append(result.primary_issue)
            except KeyError:
                pass

        # Extract severity
        severity_match = re.search(
            r'SEVERITY:\s*(critical|high|medium|low)',
            response, re.IGNORECASE
        )
        if severity_match:
            result.severity = severity_match.group(1).lower()

        # Extract recommendations
        rec_match = re.search(
            r'RECOMMENDATIONS:\s*(.*?)(?=SHOULD_RETRAIN:|METRICS_ANALYSIS:|$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if rec_match:
            rec_text = rec_match.group(1)
            recommendations = re.findall(r'-\s*(.+?)(?=\n-|\n\n|$)', rec_text, re.DOTALL)
            result.recommendations = [r.strip() for r in recommendations if r.strip()]

        # Extract should_retrain
        retrain_match = re.search(
            r'SHOULD_RETRAIN:\s*(true|false)',
            response, re.IGNORECASE
        )
        if retrain_match:
            result.should_retrain = retrain_match.group(1).lower() == 'true'

        # Semantic fallback: if structured PRIMARY_ISSUE regex failed, scan for keywords
        if result.primary_issue == TrainingIssue.NONE:
            text = response.lower()
            if any(w in text for w in ('overfitting', 'over-fitting', 'memoriz', 'train-test gap')):
                result.primary_issue = TrainingIssue.OVERFITTING
                result.issues.append(TrainingIssue.OVERFITTING)
            elif any(w in text for w in ('underfitting', 'under-fitting', 'insufficient capacity')):
                result.primary_issue = TrainingIssue.UNDERFITTING
                result.issues.append(TrainingIssue.UNDERFITTING)
            elif any(w in text for w in ('plateau', 'stagnant', 'no improvement', 'not improving')):
                result.primary_issue = TrainingIssue.LOSS_PLATEAU
                result.issues.append(TrainingIssue.LOSS_PLATEAU)
            elif any(w in text for w in ('slow convergence', 'converging slowly', 'too slow')):
                result.primary_issue = TrainingIssue.SLOW_CONVERGENCE
                result.issues.append(TrainingIssue.SLOW_CONVERGENCE)

        # If LLM didn't explicitly set SHOULD_RETRAIN but diagnosed a real issue,
        # infer it: any high/critical/medium issue warrants retraining.
        if not retrain_match and result.primary_issue not in (TrainingIssue.NONE,):
            result.should_retrain = result.severity in ('high', 'critical', 'medium')

        # Extract metrics analysis
        metrics_match = re.search(
            r'METRICS_ANALYSIS:\s*(.*?)(?=$)',
            response, re.DOTALL | re.IGNORECASE
        )
        if metrics_match:
            metrics_text = metrics_match.group(1)
            metric_pairs = re.findall(r'-\s*(\w+):\s*(.+?)(?=\n-|\n\n|$)', metrics_text, re.DOTALL)
            for key, value in metric_pairs:
                result.metrics_analysis[key.strip()] = value.strip()

        return result

    async def analyze_training(
        self,
        context: AgentContext,
        training_history: TrainingHistory,
        config: Dict[str, Any],
        previous_attempts: Optional[List[Dict]] = None,
    ) -> CritiqueResult:
        """
        Analyze training history and diagnose issues.

        Args:
            context: Agent context
            training_history: Training metrics and losses
            config: Current model configuration
            previous_attempts: Previous HPO attempts (for context)

        Returns:
            CritiqueResult with diagnosis and recommendations
        """
        if self._llm_provider is None:
            raise RuntimeError(
                "HyperparameterCriticAgent requires an LLM provider. "
                "Call set_llm_provider() before analyze_training()."
            )
        return await self.execute(
            context,
            training_history=training_history,
            config=config,
            previous_attempts=previous_attempts or [],
        )

    def detect_issues_heuristic(self, history: TrainingHistory) -> List[TrainingIssue]:
        """
        Fast heuristic detection of training issues (no LLM call).

        Used as a pre-filter to decide if full LLM analysis is needed.
        """
        issues = []

        # Check for NaN
        if history.has_nan:
            issues.append(TrainingIssue.GRADIENT_EXPLOSION)
            return issues  # Critical issue, no point checking others

        if len(history.train_losses) < 5 or len(history.test_losses) < 5:
            return issues  # Not enough data

        # Check overfitting: ratio check catches it even when epoch-by-epoch
        # trends are noisy (common with small datasets and short training runs)
        train_trend = self._compute_trend(history.train_losses[-20:])
        test_trend = self._compute_trend(history.test_losses[-20:])

        ratio = history.final_test_loss / (history.final_train_loss + 1e-12)
        if ratio > 5.0 or (train_trend < -0.01 and test_trend > 0.01):
            issues.append(TrainingIssue.OVERFITTING)

        # Check underfitting (both losses high and not improving)
        if history.final_train_loss > 0.1 and history.final_test_loss > 0.1:
            if abs(train_trend) < 0.001 and abs(test_trend) < 0.001:
                issues.append(TrainingIssue.UNDERFITTING)

        # Check plateau
        recent_test = history.test_losses[-10:] if len(history.test_losses) >= 10 else history.test_losses
        if len(recent_test) >= 5:
            std = sum((x - sum(recent_test)/len(recent_test))**2 for x in recent_test) / len(recent_test)
            if std < 1e-8:  # Very small variance
                issues.append(TrainingIssue.LOSS_PLATEAU)

        # Check instability
        if len(history.train_losses) >= 10:
            diffs = [abs(history.train_losses[i] - history.train_losses[i-1])
                     for i in range(1, len(history.train_losses))]
            avg_diff = sum(diffs) / len(diffs)
            if avg_diff > 0.1 * history.final_train_loss:
                issues.append(TrainingIssue.UNSTABLE_TRAINING)

        return issues

    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend of values (positive = increasing)."""
        if len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def should_trigger_hpo(self, history: TrainingHistory, threshold: float = 0.1) -> bool:
        """
        Determine if HPO should be triggered based on training history.

        Args:
            history: Training history
            threshold: Error threshold for triggering. If set very high (>10),
                       only critical issues (NaN) will trigger HPO.

        Returns:
            True if HPO should be triggered
        """
        # Always trigger if NaN detected (critical issue)
        if history.has_nan:
            return True

        # If threshold is very high, user doesn't want HPO
        # Only trigger for critical issues (NaN above)
        if threshold > 10.0:
            return False

        # Trigger if test loss is above threshold
        if history.final_test_loss > threshold:
            return True

        # Only check overfitting/plateau if we have reasonable loss values
        # (avoid triggering on noise when losses are already very low)
        if history.final_test_loss < 0.001:
            return False

        # Trigger if significant overfitting
        if len(history.train_losses) >= 10 and len(history.test_losses) >= 10:
            gap = history.final_test_loss - history.final_train_loss
            if gap > 0.5 * history.final_train_loss:
                return True

        # Trigger if loss plateau detected
        issues = self.detect_issues_heuristic(history)
        if TrainingIssue.LOSS_PLATEAU in issues:
            return True

        return False

    async def review_proposal(
        self,
        context: AgentContext,
        proposal_changes: Dict[str, Any],
        proposal_reasoning: str,
        critique: "CritiqueResult",
    ) -> Dict[str, Any]:
        """
        Review an Architect proposal for feasibility (debate round 2).

        Returns dict with keys: feasible (bool), concerns (str), suggestion (str).
        """
        review_prompt = (
            f"You previously diagnosed: {critique.primary_issue.name} (severity={critique.severity}).\n\n"
            f"The Architect proposes these changes:\n"
            + "\n".join(f"  - {k}: {v}" for k, v in proposal_changes.items())
            + f"\n\nReasoning: {proposal_reasoning}\n\n"
            "Does this proposal directly address the diagnosed issue?\n"
            "Reply strictly in this format:\n"
            "FEASIBLE: yes|no\n"
            "CONCERNS: <one sentence — risks or gaps, or 'none'>\n"
            "SUGGESTION: <one concrete improvement if needed, else 'none'>\n"
        )
        response = await self._llm_provider.generate(
            system_prompt=HYPERPARAMETER_CRITIC_SYSTEM + "\nYou are now reviewing a proposed configuration for feasibility.",
            user_prompt=review_prompt,
        )
        text = response.content

        feasible = True
        concerns = "none"
        suggestion = "none"

        fm = re.search(r'FEASIBLE:\s*(yes|no)', text, re.IGNORECASE)
        if fm:
            feasible = fm.group(1).lower() == "yes"
        cm = re.search(r'CONCERNS:\s*(.*?)(?=SUGGESTION:|$)', text, re.DOTALL | re.IGNORECASE)
        if cm:
            concerns = cm.group(1).strip()
        sm = re.search(r'SUGGESTION:\s*(.*?)$', text, re.DOTALL | re.IGNORECASE)
        if sm:
            suggestion = sm.group(1).strip()

        return {"feasible": feasible, "concerns": concerns, "suggestion": suggestion}

    # ── Debate Round 1: Observation ──────────────────────────────────────────

    async def observe(
        self,
        context: AgentContext,
        history: TrainingHistory,
        config: Dict[str, Any],
    ) -> str:
        """Round 1: describe training state factually, no proposals."""
        if self._llm_provider is None:
            raise RuntimeError("LLM provider not set")
        prompt = (
            "ROUND 1 (OBSERVATION) — Describe only what you see. Do NOT recommend any changes.\n\n"
            f"{history.to_summary()}\n\n"
            "Describe ONLY what you observe:\n"
            "- Loss trajectory: is it improving, stagnating, or worsening?\n"
            "- Train/test gap magnitude?\n"
            "- Any anomalies (NaN, sudden spikes, early plateau)?\n"
            "- Severity of any performance issues?\n\n"
            "3-4 sentences with specific numbers. Do not say 'you should' or suggest changes."
        )
        response = await self._llm_provider.generate(
            system_prompt=HYPERPARAMETER_CRITIC_SYSTEM,
            user_prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=512,
        )
        return f"[CRITIC — Round 1]\n{response.content}"

    def observe_sync(self, history: TrainingHistory, config: Dict[str, Any]) -> str:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.observe(AgentContext(), history, config))
        finally:
            loop.close()

    # ── Debate Round 4: Finalization ─────────────────────────────────────────

    async def validate_proposals(
        self,
        arch_summary: str,
        phys_summary: str,
        debate_context: str,
    ) -> str:
        """Round 4: validate architect and physicist proposals before they are applied."""
        if self._llm_provider is None:
            raise RuntimeError("LLM provider not set")
        prompt = (
            "ROUND 4 (FINALIZATION) — Validate the proposals from the debate.\n\n"
            f"## Full Debate History\n{debate_context}\n\n"
            f"## Architect's Proposal\n{arch_summary}\n\n"
            f"## Physicist's Proposal\n{phys_summary}\n\n"
            "Check these proposals:\n"
            "1. Capacity sanity: n_layers ≥ 2, d_model ≥ 32 (DeepONet: hidden_dim ≥ 32, n_basis ≥ 16)\n"
            "2. Physics ordering: no crack terms (traction_free/near_tip/j_integral)\n"
            "   enabled before equilibrium has converged (test loss still decreasing).\n"
            "3. Consistency: do the proposals address the actual issue identified in Rounds 1-2?\n\n"
            "VALIDATION_STATUS: approved | concerns | rejected\n"
            "CONCERNS: <specific issues with exact parameter names, or 'none'>"
        )
        response = await self._llm_provider.generate(
            system_prompt=HYPERPARAMETER_CRITIC_SYSTEM,
            user_prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=512,
        )
        return f"[CRITIC — Round 4 Validation]\n{response.content}"

    def validate_proposals_sync(
        self,
        arch_summary: str,
        phys_summary: str,
        debate_context: str,
    ) -> str:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.validate_proposals(arch_summary, phys_summary, debate_context)
            )
        finally:
            loop.close()
