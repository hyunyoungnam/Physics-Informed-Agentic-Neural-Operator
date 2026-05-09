"""
Generate agentic_phase_field_demo.png from pre-generated phase field data on disk.

Loads the 50 FEniCS samples in phase_field_data/arrays/ (no FEniCS required),
runs the full agentic HPO loop, and saves the 2×3 visualization panel.

Usage:
    python scripts/generate_phase_field_demo.py
    python scripts/generate_phase_field_demo.py --epochs 120 --rounds 6
    python scripts/generate_phase_field_demo.py --output my_demo.png
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))  # for MockLLMProvider


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phase_field_dataset(
    max_samples: int = 50,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Load pre-generated phase field samples from phase_field_data/.

    Returns:
        params:       (n, 5) float32 — [E, nu, traction, G_c, crack_length]
        all_coords:   list of (N_i, 2) per sample
        all_elements: list of (M_i, 3) per sample
        all_outputs:  list of (N_i, 3) per sample — [u_x, u_y, log1p(σ_vm)]
        all_damage:   list of (N_i,) per sample
    """
    data_path = PROJECT_ROOT / "phase_field_data"
    with open(data_path / "samples.json") as f:
        meta = json.load(f)

    arr_dir = data_path / "arrays"
    order   = meta["sample_order"]
    samples = meta["samples"]

    params_list, all_coords, all_elements, all_outputs, all_damage = [], [], [], [], []

    for sid in order[:max_samples]:
        s    = samples[sid]
        sdir = arr_dir / sid
        if not sdir.exists():
            continue
        try:
            coords = np.load(sdir / "coordinates.npy").astype(np.float32)[:, :2]
            disp   = np.load(sdir / "displacement.npy").astype(np.float32)
            vm     = np.load(sdir / "von_mises.npy").astype(np.float32)
            elems  = np.load(sdir / "elements.npy").astype(np.int64)
            damage = np.load(sdir / "damage.npy").astype(np.float32)
        except Exception as e:
            print(f"  [warn] skipping {sid}: {e}")
            continue

        lvm    = np.log1p(vm)[:, None]
        output = np.hstack([disp, lvm]).astype(np.float32)
        p      = s["parameters"]
        params_list.append([p["E"], p["nu"], p["traction"], p["G_c"], p["crack_length"]])
        all_coords.append(coords)
        all_elements.append(elems)
        all_outputs.append(output)
        all_damage.append(damage)

    if not params_list:
        raise RuntimeError(f"No valid samples found in {data_path}")

    params = np.array(params_list, dtype=np.float32)
    n      = len(params_list)
    print(f"  Loaded {n} samples; "
          f"node range [{min(c.shape[0] for c in all_coords)}, "
          f"{max(c.shape[0] for c in all_coords)}]")
    return params, all_coords, all_elements, all_outputs, all_damage


def enrich_coords(coords: np.ndarray, crack_length: float) -> np.ndarray:
    """Williams enrichment: (N,2) → (N,8)."""
    tip  = np.array([crack_length, 0.5], dtype=np.float32)
    dxy  = coords - tip
    r    = np.linalg.norm(dxy, axis=1, keepdims=True).clip(1e-8)
    logr = np.log(r)
    th   = np.arctan2(dxy[:, 1:2], dxy[:, 0:1])
    return np.concatenate(
        [coords, r, logr, np.sin(th), np.cos(th),
         np.sin(th / 2), np.cos(th / 2)],
        axis=1,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(
    n_samples: int = 40,
    epochs_per_round: int = 80,
    max_hpo_rounds: int = 6,
    output_file: str = "tests/test_outputs/agentic_phase_field_demo.png",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.tri as mtri

    from piano.surrogate.base import TransolverConfig, CrackConfig
    from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig
    from piano.agents.roles.hyperparameter_critic import (
        HyperparameterCriticAgent, TrainingHistory,
    )
    from piano.agents.roles.architect import ArchitectAgent
    from piano.agents.roles.physicist import PhysicistAgent, PhysicsProposal
    from piano.agents.base import AgentContext
    from test_agentic_sciml import MockLLMProvider

    print("=" * 70)
    print("PIANO: Agentic SciML Loop — Edge Crack Phase Field Demo (disk data)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data from disk
    # ------------------------------------------------------------------
    print(f"\n1. Loading phase field dataset ({n_samples} samples from phase_field_data/)...")
    (params, all_coords, all_elements,
     all_outputs, all_damage) = load_phase_field_dataset(max_samples=n_samples)
    n_loaded = len(params)
    print(f"   params: {params.shape}  [E, nu, traction, G_c, crack_length]")

    # Williams enrichment per sample using its own crack tip = (crack_length, 0.5)
    all_coords_enriched = [
        enrich_coords(all_coords[i], float(params[i, 4]))
        for i in range(n_loaded)
    ]

    # ------------------------------------------------------------------
    # 2. Agents (MockLLMProvider — no API key needed)
    # ------------------------------------------------------------------
    provider  = MockLLMProvider(scenario="underfitting")
    critic    = HyperparameterCriticAgent()
    architect = ArchitectAgent()
    physicist = PhysicistAgent()
    for agent in (critic, architect, physicist):
        agent.set_llm_provider(provider)
    print("   Using MockLLMProvider (no real LLM)")

    # ------------------------------------------------------------------
    # 3. Initial config — small model for demo speed
    # ------------------------------------------------------------------
    current_config = TransolverConfig(
        d_model=64,
        n_layers=3,
        n_heads=4,
        slice_num=16,
        dropout=0.05,
        learning_rate=1e-3,
        optimizer_type="adamw",
        scheduler_type="cosine",
        epochs=epochs_per_round,
        patience=epochs_per_round,
        batch_size=4,
        output_dim=3,
        equilibrium=0.0,   # disabled: crack faces make strong-form ∇u unstable
        energy=1e-4,       # variational elastic energy — safe for crack faces
        tip_weight=0.0,    # disabled: variable-topology samples (different crack_length)
        near_tip=0.0,
        j_integral=0.0,
    )
    # CrackConfig: param indices for E(0), nu(1), traction(2)
    crack_cfg = CrackConfig(
        tip_x=0.3, tip_y=0.5,
        e_param_idx=0, nu_param_idx=1,
        traction_param_idx=2, ki_param_idx=2,
    )

    context = AgentContext()
    loop    = asyncio.new_event_loop()

    round_results:    List[dict]  = []
    all_train_losses: List[float] = []
    all_test_losses:  List[float] = []
    ensemble_log:     List[tuple] = []
    attempt_history:  List[dict]  = []

    best_test_loss    = float('inf')
    best_trainer      = None
    best_config       = current_config
    no_improve_streak = 0
    max_no_improve    = 2
    min_rounds        = 2
    n_candidates      = 3
    eval_epochs       = 12
    converged         = False
    round_idx         = 0

    print(f"\n2. Agentic loop (max {max_hpo_rounds} rounds)...")

    while round_idx < max_hpo_rounds and not converged:
        print(f"\n   --- Round {round_idx + 1} ---")
        current_config.patience = current_config.epochs

        trainer = SurrogateTrainer(TrainingConfig(
            surrogate_config=current_config,
            use_ensemble=True,
            n_ensemble=3,
            train_test_split=0.15,
            tip_coords=None,       # per-sample meshes — no shared tip
            crack_config=crack_cfg,
        ))
        trainer.set_auxiliary_data(all_damage)
        result = trainer.train(params, all_coords_enriched, all_outputs)

        if not result.success:
            print(f"   [ERROR] {result.error_message}")

        round_results.append({
            "round":      round_idx + 1,
            "train_loss": result.train_loss,
            "test_loss":  result.test_loss,
            "config":     current_config.to_dict().copy(),
            "trainer":    trainer,
            "history":    result.history,
        })
        all_train_losses.extend(result.history.get("train_loss", []))
        all_test_losses.extend(result.history.get("test_loss",  []))

        curr_loss = result.test_loss if result.success else float('inf')
        print(f"   Train: {result.train_loss:.6f}  Test: {curr_loss:.6f}")

        if result.success and curr_loss < best_test_loss:
            imp = (best_test_loss - curr_loss) / best_test_loss * 100 if best_test_loss < float('inf') else 100.0
            best_test_loss    = curr_loss
            best_trainer      = trainer
            best_config       = current_config
            no_improve_streak = 0
            if round_idx > 0:
                print(f"   New best! +{imp:.1f}%")
        else:
            no_improve_streak += 1
            print(f"   No improvement. Streak: {no_improve_streak}/{max_no_improve}")

        # Convergence
        if round_idx >= min_rounds - 1 and no_improve_streak >= max_no_improve:
            print(f"   Converged — no improvement for {max_no_improve} rounds.")
            converged = True

        # Agent HPO
        if not converged and round_idx < max_hpo_rounds - 1:
            history = TrainingHistory(
                train_losses=all_train_losses[-60:],
                test_losses=all_test_losses[-60:],
                pino_losses=result.history.get("pino_loss", []),
                epochs_completed=len(result.history.get("train_loss", [])),
                best_test_loss=best_test_loss,
                final_train_loss=result.train_loss,
                final_test_loss=result.test_loss,
            )
            critique = loop.run_until_complete(
                critic.analyze_training(context, history, current_config.to_dict())
            )
            print(f"   Critic: {critique.primary_issue.name} ({critique.severity})")

            attempt_history.append({
                "round":   round_idx + 1,
                "changes": {},
                "result":  f"train={result.train_loss:.4f}, test={result.test_loss:.4f} ({critique.primary_issue.name})",
                "summary": f"Round {round_idx+1}: {critique.primary_issue.name} — test={result.test_loss:.4f}",
            })

            candidate_configs = []
            for _ in range(n_candidates):
                arch_p = loop.run_until_complete(
                    architect.propose_config(
                        context, best_config, critique,
                        dataset_size=n_loaded,
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
                            context, best_config.to_dict(), critique,
                            training_history=history,
                            dataset_size=n_loaded,
                            problem_type="phase_field",
                        )
                    )
                else:
                    phys_p = PhysicsProposal()

                merged            = copy.deepcopy(arch_p.config)
                merged.output_dim = 3
                merged.tip_weight = 0.0  # keep disabled
                for k, v in phys_p.changes.items():
                    if hasattr(merged, k):
                        setattr(merged, k, v)
                candidate_configs.append((merged, arch_p, phys_p))

            # Brief-train candidates and pick winner
            brief_losses = []
            for merged, _, _ in candidate_configs:
                brief_cfg         = copy.deepcopy(merged)
                brief_cfg.epochs  = eval_epochs
                brief_cfg.patience = eval_epochs
                bt = SurrogateTrainer(TrainingConfig(
                    surrogate_config=brief_cfg,
                    use_ensemble=False,
                    train_test_split=0.15,
                    tip_coords=None,
                    crack_config=crack_cfg,
                ))
                bt.set_auxiliary_data(all_damage)
                br = bt.train(params, all_coords_enriched, all_outputs)
                brief_losses.append(br.test_loss if br.success else float('inf'))

            winner_idx     = int(np.argmin(brief_losses))
            current_config = candidate_configs[winner_idx][0]
            ensemble_log.append((round_idx + 1, brief_losses, winner_idx))
            print(f"   Ensemble: {[f'{l:.4f}' for l in brief_losses]}  winner=C{winner_idx+1}")
            print(f"   Architect: {list(candidate_configs[winner_idx][1].changes.keys())}")

        round_idx += 1

    loop.close()
    n_rounds_run = round_idx

    # ------------------------------------------------------------------
    # 4. Prediction on reference sample
    # ------------------------------------------------------------------
    print("\n3. Generating test prediction...")
    ref_idx      = 0
    ref_coords   = all_coords[ref_idx]
    ref_elements = all_elements[ref_idx]
    ref_coords_e = all_coords_enriched[ref_idx]
    ref_params   = params[ref_idx:ref_idx+1]
    vm_gt        = np.expm1(all_outputs[ref_idx][:, 2]).clip(0)

    pred_raw, _ = best_trainer.predict_with_uncertainty(ref_params, ref_coords_e)
    if pred_raw.ndim == 3:
        pred_raw = pred_raw[0]
    vm_pred = np.expm1(pred_raw[:, 2]).clip(0)

    l2_err = np.linalg.norm(vm_pred - vm_gt) / (np.linalg.norm(vm_gt) + 1e-12) * 100
    print(f"   Peak GT={vm_gt.max():.3e}  Pred={vm_pred.max():.3e}  L2={l2_err:.1f}%")

    crack_length_ref = float(params[ref_idx, 4])
    improvement      = (round_results[0]["test_loss"] - best_test_loss) / round_results[0]["test_loss"] * 100

    # ------------------------------------------------------------------
    # 5. Visualization (2x3 panel)
    # ------------------------------------------------------------------
    print("\n4. Creating visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Loss evolution
    ax1 = fig.add_subplot(gs[0, 0])
    ep  = np.arange(1, len(all_test_losses) + 1)
    ax1.semilogy(ep, all_train_losses, 'b-', lw=1.5, alpha=0.7, label='Train')
    ax1.semilogy(ep, all_test_losses,  'r-', lw=2,   label='Test')
    offset = 0
    colors = ['green', 'orange', 'purple', 'brown']
    for i, rr in enumerate(round_results):
        n_ep = len(rr["history"].get("train_loss", []))
        if i > 0:
            ax1.axvline(offset, color=colors[i % 4], ls='--', lw=1.5, alpha=0.7)
        offset += n_ep
    ax1.set_xlabel('Epoch (cumulative)'); ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Loss Evolution Across HPO Rounds', fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Panel 2: Test loss per round
    ax2   = fig.add_subplot(gs[0, 1])
    rnds  = [rr["round"] for rr in round_results]
    tloss = [rr["test_loss"] for rr in round_results]
    ax2.plot(rnds, tloss, 'ro-', lw=2, markersize=10,
             markerfacecolor='white', markeredgewidth=2)
    ax2.fill_between(rnds, tloss, alpha=0.3, color='red')
    for r, l in zip(rnds, tloss):
        ax2.annotate(f'{l:.4f}', (r, l), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)
    best_ri = int(np.argmin(tloss))
    ax2.plot(rnds[best_ri], tloss[best_ri], 'g*', ms=14, zorder=5,
             label=f'Best (R{rnds[best_ri]})')
    ax2.legend(fontsize=8); ax2.set_xlabel('HPO Round'); ax2.set_ylabel('Test Loss')
    ax2.set_title(f'Convergence: {improvement:.1f}% improvement', fontweight='bold')
    ax2.set_xticks(rnds); ax2.grid(True, alpha=0.3)

    # Panel 3: Ensemble candidate selection
    ax3 = fig.add_subplot(gs[0, 2])
    if ensemble_log:
        n_c = max(len(e[1]) for e in ensemble_log)
        x0  = np.arange(len(ensemble_log))
        bw  = 0.8 / n_c
        cc  = ["#5599dd", "#dd9955", "#55bb77"]
        for ci in range(n_c):
            vals = [e[1][ci] if ci < len(e[1]) else float('nan') for e in ensemble_log]
            bars = ax3.bar(x0 + (ci - (n_c-1)/2)*bw, vals, width=bw,
                           color=cc[ci % 3], alpha=0.75, label=f"C{ci+1}")
            for bar, entry in zip(bars, ensemble_log):
                if entry[2] == ci:
                    bar.set_edgecolor("black"); bar.set_linewidth(2)
                    ax3.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height()*1.02, "★",
                             ha="center", va="bottom", fontsize=9)
        ax3.set_xticks(x0)
        ax3.set_xticklabels([f"R{e[0]}" for e in ensemble_log], fontsize=9)
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No ensemble\nrounds run",
                 ha="center", va="center", transform=ax3.transAxes, color="gray")
    ax3.set_xlabel("HPO round"); ax3.set_ylabel("Brief-eval test loss")
    ax3.set_title(f"Ensemble: Candidate Selection ({n_candidates} per round)",
                  fontweight='bold')
    ax3.grid(True, alpha=0.3, axis="y")

    # Mesh panels — use reference sample triangulation
    triang = mtri.Triangulation(ref_coords[:, 0], ref_coords[:, 1], ref_elements)

    def _crack(ax):
        ax.plot([0, crack_length_ref], [0.5, 0.5], 'w-', lw=1.5, zorder=4)

    vm_all = np.concatenate([vm_pred, vm_gt])
    vmin, vmax = 0.0, np.percentile(vm_all, 95)
    levels = np.linspace(vmin, vmax, 25)

    # Panel 4: Surrogate von Mises
    ax4 = fig.add_subplot(gs[1, 0])
    cf4 = ax4.tricontourf(triang, np.clip(vm_pred, vmin, vmax),
                          levels=levels, cmap='plasma', extend='max')
    ax4.triplot(triang, 'w-', lw=0.08, alpha=0.10)
    _crack(ax4)
    ax4.set_xlim(-0.05, 1.05); ax4.set_ylim(-0.05, 1.05); ax4.set_aspect('equal')
    ax4.set_title('Surrogate: Von Mises Stress', fontweight='bold')
    fig.colorbar(cf4, ax=ax4, shrink=0.7, label=r'$\sigma_{VM}$ [Pa]', format='%.1e')

    # Panel 5: Ground truth von Mises
    ax5 = fig.add_subplot(gs[1, 1])
    cf5 = ax5.tricontourf(triang, np.clip(vm_gt, vmin, vmax),
                          levels=levels, cmap='plasma', extend='max')
    ax5.triplot(triang, 'w-', lw=0.08, alpha=0.10)
    _crack(ax5)
    ax5.set_xlim(-0.05, 1.05); ax5.set_ylim(-0.05, 1.05); ax5.set_aspect('equal')
    ax5.set_title('Ground Truth: Von Mises (FEniCS)', fontweight='bold')
    fig.colorbar(cf5, ax=ax5, shrink=0.7, label=r'$\sigma_{VM}$ [Pa]', format='%.1e')

    # Panel 6: Peak stress comparison
    ax6 = fig.add_subplot(gs[1, 2])
    mp, mg = vm_pred.max(), vm_gt.max()
    bars = ax6.bar(["Surrogate", "Ground Truth"], [mp, mg],
                   color=["tomato", "steelblue"], edgecolor="black", width=0.5)
    for bar, val in zip(bars, [mp, mg]):
        ax6.text(bar.get_x() + bar.get_width()/2, val*1.01, f"{val:.2e}",
                 ha="center", va="bottom", fontsize=9)
    err_pct = abs(mp - mg) / (mg + 1e-12) * 100
    ax6.set_ylabel("Max von Mises [Pa]")
    ax6.set_title(f"Peak Stress Error: {err_pct:.1f}%", fontweight='bold')
    ax6.set_ylim(0, max(mp, mg) * 1.15)
    ax6.grid(True, alpha=0.3, axis="y")

    stop = "converged" if converged else "max rounds"
    fig.suptitle(
        f'PIANO: Edge Crack Phase Field — Von Mises '
        f'({n_rounds_run} rounds, {improvement:.1f}% improvement, {stop})',
        fontsize=13, fontweight='bold',
    )

    out = PROJECT_ROOT / output_file
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved → {out}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--epochs",    type=int, default=80)
    parser.add_argument("--rounds",    type=int, default=6)
    parser.add_argument("--output",    type=str,
                        default="tests/test_outputs/agentic_phase_field_demo.png")
    args = parser.parse_args()

    run_demo(
        n_samples=args.n_samples,
        epochs_per_round=args.epochs,
        max_hpo_rounds=args.rounds,
        output_file=args.output,
    )
