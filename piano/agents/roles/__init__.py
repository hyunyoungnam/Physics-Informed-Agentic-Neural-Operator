"""
Agent role implementations for agentic SciML.

HPO debate loop (4-round structured debate):
- ResultAnalystAgent:       observes training curves (Round 1, no proposals)
- HyperparameterCriticAgent: diagnoses training issues + validates proposals (Rounds 1 & 4)
- ArchitectAgent:           proposes architecture/optimizer changes (Rounds 2 & 3)
- PhysicistAgent:           proposes physics loss configuration (Rounds 2 & 3)
- EngineerAgent:            implements code changes via Claude Code CLI
- DebuggerAgent:            diagnoses EngineerAgent failures

Knowledge & data agents (injected into debate context):
- KnowledgeRetrieverAgent:  surfaces relevant KB entries before each round
- DataAnalystAgent:         pre-training dataset EDA (Phase 0)

Candidate selection:
- SelectorEnsembleAgent:    3-LLM majority vote (replaces brief-training fallback)

Active learning agents:
- AdaptiveProposerAgent:    targets weak / high-uncertainty regions
- MeshStrategyAgent:        r/h-refinement resolution decisions
- BudgetAgent:              decides when to collect more data vs stop
"""

# ── HPO debate: observation ───────────────────────────────────────────────────
from piano.agents.roles.result_analyst import ResultAnalystAgent, AnalystObservation

# ── HPO debate: diagnosis & validation ───────────────────────────────────────
from piano.agents.roles.hyperparameter_critic import (
    HyperparameterCriticAgent,
    CritiqueResult,
    TrainingHistory,
    TrainingIssue,
)

# ── HPO debate: proposals ─────────────────────────────────────────────────────
from piano.agents.roles.architect import ArchitectAgent, ArchitectureProposal
from piano.agents.roles.physicist import PhysicistAgent, PhysicsProposal, PhysicsIssue

# ── HPO debate: implementation ────────────────────────────────────────────────
from piano.agents.roles.engineer import EngineerAgent, EngineerResult
from piano.agents.roles.debugger import DebuggerAgent, DebugResult

# ── Knowledge & data ──────────────────────────────────────────────────────────
from piano.agents.roles.knowledge_retriever import KnowledgeRetrieverAgent, KBEntry
from piano.agents.roles.data_analyst import DataAnalystAgent, DataAnalysis

# ── Candidate selection ───────────────────────────────────────────────────────
from piano.agents.roles.selector_ensemble import SelectorEnsembleAgent, SelectionResult, VoteResult

# ── Active learning ───────────────────────────────────────────────────────────
from piano.agents.roles.adaptive_proposer import AdaptiveProposerAgent, AdaptiveProposal
from piano.agents.roles.mesh_strategy import MeshStrategyAgent, MeshStrategyDecision
from piano.agents.roles.budget import BudgetAgent, BudgetDecision

__all__ = [
    # Result Analyst
    "ResultAnalystAgent",
    "AnalystObservation",
    # Hyperparameter Critic
    "HyperparameterCriticAgent",
    "CritiqueResult",
    "TrainingHistory",
    "TrainingIssue",
    # Architect
    "ArchitectAgent",
    "ArchitectureProposal",
    # Physicist
    "PhysicistAgent",
    "PhysicsProposal",
    "PhysicsIssue",
    # Adaptive Proposer
    "AdaptiveProposerAgent",
    "AdaptiveProposal",
    # Engineer
    "EngineerAgent",
    "EngineerResult",
    # Debugger
    "DebuggerAgent",
    "DebugResult",
    # Knowledge Retriever
    "KnowledgeRetrieverAgent",
    "KBEntry",
    # Data Analyst
    "DataAnalystAgent",
    "DataAnalysis",
    # Selector Ensemble
    "SelectorEnsembleAgent",
    "SelectionResult",
    "VoteResult",
    # Mesh Strategy
    "MeshStrategyAgent",
    "MeshStrategyDecision",
    # Budget
    "BudgetAgent",
    "BudgetDecision",
]
