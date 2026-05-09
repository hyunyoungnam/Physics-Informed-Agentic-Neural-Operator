# TODO

## Resolved Issues

### ✅ 1. Surrogate overfitting with small datasets
**Status**: Resolved  
**Fix applied**:
- Switched training target from displacement (output_dim=2) to nodal von Mises stress (output_dim=1) to halve the output space
- Test loss improved ~4× once DeepONet was also introduced
- **Superseded by issue #2 resolution**: output is now displacement (N, 2) again to enable PINO, with tip-weighted MSE handling the singularity instead of log-transform

### ✅ 2. Crack PINO physics loss / displacement incompatibility
**Status**: Resolved  
**Root cause**: `CrackFractureLoss` and `PINOElasticityLoss` both require displacement `(N, 2)`, but surrogate was predicting von Mises scalar `(N, 1)`. Additionally, `use_pino` gate in `trainer.py` required `coord_dim == 2`, which failed silently when using 6-feature enriched trunk coordinates.  
**Fix applied**:
- Switched training target back to displacement `(N, 2)` in `_generate_vnotch_fem_data` (`output_field="displacement"`)
- Removed `coord_dim == 2` constraint from `use_pino` gate in `trainer.py`
- Trainer now slices `coords_t[0, :, :2]` before passing to both `PINOElasticityLoss` and `CrackFractureLoss`, so 6-feature enriched coordinates work correctly
- `pino_eq_weight=0.1` (default) activates the label-free equilibrium residual from round 1
- Von Mises derived from predicted displacement at evaluation time via `_compute_von_mises_nodal`

### ✅ 3. Mock critic always diagnoses UNDERFITTING
**Status**: Resolved (approach changed)  
**Original fix**: Added `_analyze_heuristic()` fallback with ratio-based overfitting check  
**Current state**: Heuristic fallback has been **removed entirely**. `HyperparameterCriticAgent.analyze_training()` now raises `RuntimeError` if no LLM provider is set. The mock LLM provider is used for tests and demo; the real `AnthropicProvider` is used in production via `--use-real-llm`.  
**Remaining**: Mock LLM for critic still returns UNDERFITTING every round regardless of actual training curves — acceptable for tests but means the demo agent loop does not switch strategy when overfitting begins. Fixed only when running with `--use-real-llm`.

### ✅ 4. Triangulation includes notch interior (Delaunay artifact)
**Status**: Resolved  
**Fix applied**:
- Added `elements: Optional[np.ndarray] = None` to `FEMSample`
- `generate_vnotch_fem_sample` populates `elements` from `mesh_gen.generate()` — already filters notch interior
- Demo uses `sample.elements` instead of `Delaunay(coords).simplices`
- `FEMDataset.save/load` handles `elements.npy` per sample

### ✅ 5. AnthropicProvider uses outdated Claude 3 models
**Status**: Resolved  
**Fix applied**:
- Updated to `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`
- Default changed to `claude-haiku-4-5-20251001`
- `--use-real-llm` CLI flag wires `AnthropicProvider` to all three agents

### ✅ 6. DeepONet has no uncertainty quantification in ensemble mode
**Status**: Resolved  
**Fix applied**:
- `EnsembleModel.build()` calls `torch.manual_seed(42 + i)` before each member so weights are uncorrelated

### ✅ 7. Surrogate predicts flat/uniform stress field (misses singularity)
**Status**: Resolved  
**Root cause**: MSE loss minimised by predicting the spatial mean when the singularity (σ ∝ 1/√r) dominates the variance.  
**Fix applied**:
- Polar trunk features `(r, log_r, sin_θ, cos_θ)` relative to notch tip give the trunk explicit geometric signal
- Tip-weighted MSE (`tip_weight=2.0`) upweights near-tip nodes so the loss cannot ignore the singularity
- (Log-transform applied to von Mises targets was part of a previous approach; current displacement-based training uses tip weighting instead)

### ✅ 8. Surrogate shows swirling artifacts in far-field
**Status**: Resolved  
**Root cause**: (a) Raw `θ = atan2(dy, dx)` has a ±π branch-cut discontinuity → trunk maps it to a spatial jump in basis functions. (b) Trunk overfits spatially with no dropout.  
**Fix applied**:
- Replaced `arctan2` with `(sin_theta, cos_theta)` in `coords_enriched`
- Added `trunk_dropout: float = 0.1` to `DeepONetConfig` — trunk MLP uses it; branch uses `dropout` (0.0)

### ✅ 9. Heuristic fallback in HyperparameterCriticAgent
**Status**: Resolved  
**Fix applied**:
- Removed `_analyze_heuristic()` from `HyperparameterCriticAgent`
- `analyze_training()` now raises `RuntimeError` if `_llm_provider is None`
- All three agents (Critic, Architect, Physicist) now require `set_llm_provider()` before use
- `detect_issues_heuristic()` kept for lightweight gating in `should_trigger_hpo()` — not an LLM substitute

### ✅ 10. `trunk_dropout` not tunable by Architect agent
**Status**: Resolved  
**Root cause**: Three-layer gap — `trunk_dropout` was in `DeepONetConfig` but (a) absent from the LLM system prompt, (b) not parsed by `_parse_changes()`, and (c) not forwarded by `apply_changes()`.  
**Fix applied**:
- `ARCHITECT_SYSTEM`: added `trunk_dropout` entry with explanation of branch vs trunk regularisation roles
- `_parse_changes()`: added `'trunk_dropout'` to `float` patterns
- `apply_changes()` DeepONet branch: added `trunk_dropout=changes.get("trunk_dropout", base.get("trunk_dropout", 0.1))`
- Output format template updated so the LLM knows to emit the field

### ✅ 11. `pytest-asyncio` not installed
**Status**: Resolved  
**Fix applied**:
- Installed `pytest-asyncio` — was in `requirements.txt` but missing from environment
- All 23 async agent tests now pass

### ✅ 12. Active learning acquisition functions were dead code
**Status**: Resolved  
**Root cause**: Three-layer gap — `_acquisition_fn` was initialized but never used; `_suggest_new_parameters()` called `evaluator.suggest_samples()` (ignores acquisition functions); `evaluator.suggest_samples_active()` existed but was never called.  
**Fix applied** (plan: compiled-beaming-lobster):
- Fix 1 (`adaptive.py:367`): Replaced nonexistent `self.config.acquisition_strategy` with runtime expression using `self._acquisition_fn.name`
- Fix 2 (`adaptive.py:620–631`): `_suggest_new_parameters()` now routes to `evaluator.suggest_samples_active()` when `_coordinates` is set
- Fix 3 (`trainer.py:302–309`): `ref_coords = train_coords[0]` replaced with average over same-topology samples; warns when mixed mesh sizes are detected

### ✅ 13. New agent system: 6 agents added (Knowledge, Data, Debug, Selector, Mesh, Budget)
**Status**: Implemented (2026-05-07)  
**Motivation**: AgenticSciML paper (Jiang & Karniadakis 2026) ablation: KB retrieval gives 2.3×–20× improvement over no-KB baseline. Paper architecture had 10+ agents; this project had 6 — missing KB retrieval, data analysis, debugger, selector ensemble, mesh strategy, and budget reasoning.  
**Agents implemented**:
- `KnowledgeRetrieverAgent` + `knowledge_base/` (6 entries: Williams expansion, XFEM, adaptive collocation, phase-field, J-integral, displacement decomposition) — wired into `DebateOrchestrator` before Round 1
- `DataAnalystAgent` — pre-training EDA with persistent `data_analysis.md` report; wired into `AgentContext.knowledge_context`
- `DebuggerAgent` — called by `EngineerAgent` on failure; diagnoses traceback + provides fix description; one retry attempt
- `SelectorEnsembleAgent` — 3-LLM majority vote (claude-sonnet-4-6, 2×claude-haiku-4-5) replacing brief-training fallback in `AgenticSurrogateTrainer`
- `MeshStrategyAgent` — r/h-refinement strategy for MFEM; activated via `use_mesh_strategy_agent=True` in `AdaptiveConfig`
- `BudgetAgent` — active learning stopping criterion; replaces fixed `max_samples` heuristic when `use_budget_agent=True` in `AdaptiveConfig`

---

---

## Open Issues

### A. Mock LLM critic cannot detect regime shift (overfitting after round 2)
**Status**: Known limitation  
**Symptom**: From round 3 onwards the training loss is ~1e-4 while test loss spikes to 4–14×. The mock LLM for the critic returns UNDERFITTING every round regardless, so the agent keeps increasing model capacity instead of adding regularisation.  
**Impact**: Demo loop does not converge to the best model when using MockLLMProvider.  
**Fix**: Use `--use-real-llm` with `ANTHROPIC_API_KEY` set. The real critic reads the train/test gap and correctly diagnoses OVERFITTING, allowing the Architect to respond with dropout/weight-decay increases.

### ✅ C. Replace Williams near-tip term with peridynamic equilibrium residual
**Status**: Resolved  
**Root cause**: The Williams asymptotic expansion (Term 3 in `CrackFractureLoss`) is a local LEFM approximation valid only in the K-dominant zone — it breaks down for large phase-field damage zones where `d > 0.1` extends well beyond the crack tip, and gives spurious gradients at broken bonds.  
**Fix applied**:
- New module `piano/surrogate/peridynamic_loss.py`: `PeridynamicEquilibriumLoss` implements the bond-based static PD equation Σ_j (1−d_ij)² s_ij ê_ij = 0 at every node; horizon δ = 3h_avg (scipy `cKDTree.query_pairs`); bond list cached per mesh hash; dimensionless normalization via s_var × n_avg
- `crack_pino_loss.py`: removed `_williams_displacement` + `_williams_residual`; `near_tip` weight now controls PD equilibrium; `r_williams` parameter removed; `horizon_factor` added
- `base.py / CrackConfig`: removed `r_williams`, added `horizon_factor: float = 3.0`
- `trainer.py`: fixed `r_williams=cc.r_williams` TypeError; added standalone PD path for `crack_config=None` (phase field) when `near_tip > 0`
- `agentic_trainer.py`: `AgenticTrainingConfig.crack_config` field added; wired into `_train_once()` and `_train_brief()`
- `physicist.py`: system prompt updated — `near_tip` now described as PD equilibrium with `horizon_factor` tuning guidance

### ✅ B. CrackFractureLoss bc_weight scale mismatch
**Status**: Resolved  
**Root cause**: `_crack_face_bc` was dividing by `E²` (O(1e20 Pa²)), making the loss O(2.5e-7) — too small to matter. The other three terms (`ki_weight`, `williams_weight`, `j_weight`) were already dimensionless O(1) relative errors.  
**Fix applied**:
- `_crack_face_bc` now normalises by `K_I²/(2π·r_ki_min)` — characteristic stress² at the inner extraction radius — giving O(0.13), comparable to the data loss
- `forward()` passes `K_I` to `_crack_face_bc`
- Agent responsibility split: Physicist owns `bc_weight`, `ki_weight`, `williams_weight`, `j_weight`; Architect owns `pino_weight`, `pino_eq_weight`
- Physicist enables terms sequentially: bc → ki → williams → j, one per round, only when training is stable
- `detect_physics_issues()` stability heuristic fixed: `recent[-1] < 2.0 * min(recent)` (was `max < 3 * min`, which incorrectly flagged monotonically decreasing loss as unstable)
