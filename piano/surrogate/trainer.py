"""
Surrogate model trainer.

Handles the training workflow for surrogate models including
data preparation, training, validation, and model checkpointing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from piano.data.zero_copy import numpy_to_tensor

from .base import TransolverConfig, EnsembleConfig, SurrogateModel, CrackConfig
from .transolver import TransolverModel
from .ensemble import EnsembleModel
from .deeponet import DeepONetConfig, DeepONetModel
from .pino_loss import PINOElasticityLoss
from .crack_pino_loss import CrackFractureLoss
from .peridynamic_loss import PeridynamicEquilibriumLoss
from .variational_loss import VariationalElasticLoss


def _weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE loss with optional per-node weighting for singularity-aware training.

    Args:
        pred:    (1, N, D) predicted field
        target:  (1, N, D) ground-truth field
        weights: (N,) per-node weights, or None for plain MSE

    Returns:
        Scalar loss
    """
    err = (pred - target) ** 2  # (1, N, D)
    if weights is not None:
        err = err * weights.unsqueeze(0).unsqueeze(-1)
    return err.mean()


def create_optimizer(model: nn.Module, config: TransolverConfig) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    opt_type = config.optimizer_type.lower()
    lr = config.learning_rate

    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}. Choose from ['adamw', 'adam', 'sgd']")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TransolverConfig
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Create LR scheduler based on config."""
    sched_type = config.scheduler_type.lower()

    if sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
    elif sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
    elif sched_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}. Choose from ['plateau', 'cosine', 'none']")


@dataclass
class TrainingConfig:
    """
    Configuration for surrogate training workflow.

    Attributes:
        surrogate_config: Configuration for the surrogate model
        use_ensemble: Whether to use ensemble for uncertainty
        n_ensemble: Number of models in ensemble
        normalize_inputs: Whether to normalize input features
        normalize_outputs: Whether to normalize output targets
        train_test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        save_dir: Directory to save trained model
        log_dir: Directory for training logs
    """
    surrogate_config: TransolverConfig = field(default_factory=TransolverConfig)
    use_ensemble: bool = True
    n_ensemble: int = 5
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    train_test_split: float = 0.1
    random_seed: int = 42
    save_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    tip_coords: Optional[np.ndarray] = None    # (2,) crack/notch tip for singularity-weighted MSE
    crack_config: Optional[CrackConfig] = None  # crack geometry + param indices for fracture PINO


@dataclass
class TrainingResult:
    """
    Result of surrogate training.

    Attributes:
        success: Whether training completed successfully
        train_loss: Final training loss
        test_loss: Final test loss
        metrics: Dictionary of evaluation metrics
        history: Training history
        model_path: Path to saved model
        normalization_params: Parameters for input/output normalization
    """
    success: bool
    train_loss: float = 0.0
    test_loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    history: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[Path] = None
    normalization_params: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class SurrogateTrainer:
    """
    Trainer for surrogate models.

    Handles the complete training workflow:
    1. Data preprocessing and normalization
    2. Train/test splitting
    3. Model building and training
    4. Validation and evaluation
    5. Model checkpointing
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self._model: Optional[SurrogateModel] = None
        self._input_normalizer: Optional[Normalizer] = None
        self._output_normalizer: Optional[Normalizer] = None
        self._damage_fields: Optional[List[np.ndarray]] = None

    def set_auxiliary_data(self, damage_fields: Optional[List[np.ndarray]]) -> None:
        """Provide per-sample nodal damage fields for variational_weight loss.

        Must be called before train() if surrogate_config.variational_weight > 0.
        Each array should have shape (N_i,) matching the corresponding sample's
        node count and contain AT-2 damage values in [0, 1].
        """
        self._damage_fields = damage_fields

    def prepare_data(
        self,
        parameters: np.ndarray,
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray], List[np.ndarray], Dict, np.ndarray]:
        """
        Prepare data for training.

        Args:
            parameters:  Input parameters (N_samples, n_params)
            coordinates: Per-sample coordinates, each (N_i, coord_dim)
            outputs:     Per-sample outputs,     each (N_i, output_dim)

        Returns:
            (train_params, train_coords, test_params, test_coords,
             train_outputs, test_outputs, norm_params)
        """
        np.random.seed(self.config.random_seed)

        n_samples = parameters.shape[0]
        n_test = max(1, int(n_samples * self.config.train_test_split))
        n_train = n_samples - n_test

        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_params  = parameters[train_idx]
        test_params   = parameters[test_idx]
        train_coords  = [coordinates[i] for i in train_idx]
        test_coords   = [coordinates[i] for i in test_idx]
        train_outputs = [outputs[i] for i in train_idx]
        test_outputs  = [outputs[i] for i in test_idx]

        # Keep raw (un-normalized) train params for physics losses that need physical units
        raw_train_params = train_params.copy()

        if self.config.normalize_inputs:
            self._input_normalizer = Normalizer()
            train_params = self._input_normalizer.fit_transform(train_params)
            test_params  = self._input_normalizer.transform(test_params)

        if self.config.normalize_outputs:
            self._output_normalizer = Normalizer()
            # Fit on all training output values concatenated
            all_train = np.concatenate([o.reshape(-1, o.shape[-1]) for o in train_outputs], axis=0)
            self._output_normalizer.fit(all_train)
            train_outputs = [
                self._output_normalizer.transform(o.reshape(-1, o.shape[-1])).reshape(o.shape)
                for o in train_outputs
            ]
            test_outputs = [
                self._output_normalizer.transform(o.reshape(-1, o.shape[-1])).reshape(o.shape)
                for o in test_outputs
            ]

        # Pre-cast all arrays to float32 so the training hot loop can use
        # torch.from_numpy() instead of torch.tensor() — eliminates the
        # CPU→CPU copy that happens on every sample every epoch.
        from piano.data.zero_copy import preallocate_float32
        train_params  = train_params.astype(np.float32, copy=False)
        test_params   = test_params.astype(np.float32, copy=False)
        raw_train_params = raw_train_params.astype(np.float32, copy=False)
        preallocate_float32(train_coords)
        preallocate_float32(test_coords)
        preallocate_float32(train_outputs)
        preallocate_float32(test_outputs)

        return (
            train_params,
            train_coords,
            test_params,
            test_coords,
            train_outputs,
            test_outputs,
            self._get_normalization_params(),
            raw_train_params,
        )

    def _get_normalization_params(self) -> Dict[str, Any]:
        """Get normalization parameters for saving/loading."""
        params = {}
        if self._input_normalizer:
            params["input_mean"] = self._input_normalizer.mean.tolist()
            params["input_std"] = self._input_normalizer.std.tolist()
        if self._output_normalizer:
            params["output_mean"] = self._output_normalizer.mean.tolist()
            params["output_std"] = self._output_normalizer.std.tolist()
        return params

    def train(
        self,
        parameters: np.ndarray,
        coordinates: List[np.ndarray],
        outputs: List[np.ndarray],
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> TrainingResult:
        """
        Train surrogate model with per-sample coordinates.

        Each sample may have a different number of mesh nodes (N_i). Gradients
        are accumulated over `batch_size` samples before each optimizer step,
        which is equivalent to mini-batch training without requiring a fixed N.

        Args:
            parameters:  Input parameters (N_samples, n_params)
            coordinates: Per-sample node coords, each (N_i, coord_dim)
            outputs:     Per-sample field values, each (N_i, output_dim)
            callback:    Optional callback(epoch, train_loss)

        Returns:
            TrainingResult with training metrics and model path
        """
        try:
            (
                train_params,
                train_coords,
                test_params,
                test_coords,
                train_outputs,
                test_outputs,
                norm_params,
                raw_train_params,
            ) = self.prepare_data(parameters, coordinates, outputs)

            n_train = len(train_params)
            n_test  = len(test_params)

            # Dimensions from first sample (coord_dim is fixed; num_points is metadata only)
            n_params  = train_params.shape[1]
            coord_dim = train_coords[0].shape[1]
            output_dim = train_outputs[0].shape[-1]
            num_points = train_coords[0].shape[0]

            self.config.surrogate_config.output_dim = output_dim

            if self.config.use_ensemble:
                ensemble_config = EnsembleConfig(
                    n_members=self.config.n_ensemble,
                    member_config=self.config.surrogate_config,
                )
                model = EnsembleModel(ensemble_config)
            else:
                if isinstance(self.config.surrogate_config, DeepONetConfig):
                    model = DeepONetModel(self.config.surrogate_config)
                else:
                    model = TransolverModel(self.config.surrogate_config)

            model.build(n_params, coord_dim, num_points)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            cfg = self.config.surrogate_config

            # Per-node weights for singularity-aware training
            node_weights = None
            if self.config.tip_coords is not None and cfg.tip_weight > 0:
                tip = self.config.tip_coords  # (2,)
                ref_n = train_coords[0].shape[0]
                same_topo = [c for c in train_coords if c.shape[0] == ref_n]
                if len(same_topo) < len(train_coords):
                    logger.warning(
                        f"Mixed mesh sizes: {len(same_topo)}/{len(train_coords)} samples "
                        f"share ref topology (N={ref_n}). Tip weighting uses matching samples only."
                    )
                avg_r = np.mean(
                    [np.linalg.norm(c[:, :2] - tip, axis=1) for c in same_topo], axis=0
                ) + 1e-8
                w = 1.0 + cfg.tip_weight / avg_r
                w = w / w.mean()
                node_weights = torch.tensor(w, dtype=torch.float32, device=device)

            # Crack fracture PINO loss (optional)
            crack_loss_fn = None
            standalone_pd_loss = None
            out_mean_t = out_std_t = None
            cc = self.config.crack_config
            sc = self.config.surrogate_config

            _ki_terms = any(getattr(sc, k, 0.0) > 0
                            for k in ["stress_intensity", "traction_free", "j_integral"])
            if cc is not None and (_ki_terms or sc.near_tip > 0.0):
                # Full crack fracture loss: K_I-dependent terms + PD equilibrium
                crack_loss_fn = CrackFractureLoss(
                    tip_x=cc.tip_x,
                    tip_y=cc.tip_y,
                    r_ki_min=cc.r_ki_min,
                    r_ki_max=cc.r_ki_max,
                    r_j=cc.r_j,
                    crack_face_tol=cc.crack_face_tol,
                    stress_intensity=sc.stress_intensity,
                    traction_free=sc.traction_free,
                    near_tip=sc.near_tip,
                    j_integral=sc.j_integral,
                    horizon_factor=cc.horizon_factor,
                ).to(device)
            elif sc.near_tip > 0.0:
                # Phase field (no crack_config): PD equilibrium only — no K_I needed
                standalone_pd_loss = PeridynamicEquilibriumLoss(
                    horizon_factor=cc.horizon_factor if cc is not None else 3.0
                ).to(device)

            # Per-channel output normalizer stats for denormalization.
            # Always created when normalizer is set — shape (output_dim,) so
            # u_phys * out_std_t.unsqueeze(0) broadcasts correctly to (N, output_dim).
            if self._output_normalizer is not None:
                out_mean_t = torch.tensor(
                    self._output_normalizer.mean, dtype=torch.float32, device=device
                )  # shape (output_dim,)
                out_std_t = torch.tensor(
                    self._output_normalizer.std, dtype=torch.float32, device=device
                )  # shape (output_dim,)

            # Variational AT-2 elastic loss (optional — requires damage fields)
            variational_loss_fn = None
            train_damage_fields: Optional[List[np.ndarray]] = None
            if (getattr(sc, 'variational_weight', 0.0) > 0.0
                    and self._damage_fields is not None
                    and cc is not None):
                variational_loss_fn = VariationalElasticLoss(
                    E=float(raw_train_params[0, cc.e_param_idx]),
                    nu=float(raw_train_params[0, cc.nu_param_idx]),
                ).to(device)
                train_damage_fields = self._damage_fields

            use_pino = (
                output_dim >= 2
                and (cfg.energy > 0 or cfg.equilibrium > 0)
            )
            _nu_col = cc.nu_param_idx if cc is not None else None
            pino_fn = (
                PINOElasticityLoss(
                    nominal_nu=0.3,
                    eq_weight=cfg.equilibrium,
                    energy_weight=cfg.energy,
                ).to(device)
                if use_pino
                else None
            )

            if self.config.use_ensemble:
                # Bootstrap resampling: each member trains on a different random
                # subset so ensemble disagreement reflects data scarcity, not just
                # random-seed noise.
                boot_rng = np.random.default_rng(self.config.random_seed)
                last_history = {'train_loss': [], 'test_loss': [], 'pino_loss': []}
                member_best_losses = []

                for mi in range(self.config.n_ensemble):
                    torch.manual_seed(42 + mi)
                    boot_idx = boot_rng.choice(n_train, size=n_train, replace=True)
                    b_params  = train_params[boot_idx]
                    b_coords  = [train_coords[i] for i in boot_idx]
                    b_outputs = [train_outputs[i] for i in boot_idx]
                    b_raw     = raw_train_params[boot_idx]
                    b_damage  = ([train_damage_fields[i] for i in boot_idx]
                                 if train_damage_fields is not None else None)

                    member = model._models[mi]
                    h, best_m, best_s = self._run_epoch_loop(
                        member, b_params, b_coords, b_outputs, b_raw,
                        test_params, test_coords, test_outputs,
                        cfg, device, node_weights, pino_fn,
                        crack_loss_fn, out_std_t, out_mean_t, cc,
                        callback if mi == 0 else None,
                        standalone_pd_loss=standalone_pd_loss,
                        variational_loss_fn=variational_loss_fn,
                        train_damage_fields=b_damage,
                    )
                    if best_s:
                        member.load_state_dict(best_s)
                        member.to(device)
                    member._is_trained = True
                    last_history = h
                    member_best_losses.append(best_m)

                # Evaluate ensemble mean on shared test set for final test loss
                model.eval()
                ens_test_loss = 0.0
                with torch.no_grad():
                    for idx in range(n_test):
                        pt = numpy_to_tensor(test_params[idx:idx+1], device)
                        ct = numpy_to_tensor(test_coords[idx], device).unsqueeze(0)
                        ot = numpy_to_tensor(test_outputs[idx], device).unsqueeze(0)
                        pred = model.forward(pt, ct)
                        ens_test_loss += _weighted_mse(pred, ot).item() / n_test

                history = last_history
                best_test_loss = ens_test_loss

                # Compute ensemble disagreement (mean std across test set)
                ensemble_stds = []
                model.eval()
                with torch.no_grad():
                    for idx in range(n_test):
                        pt = numpy_to_tensor(test_params[idx:idx+1], device)
                        ct = numpy_to_tensor(test_coords[idx], device).unsqueeze(0)
                        member_preds = np.stack([
                            m.forward(pt, ct).cpu().numpy() for m in model._models
                        ], axis=0)
                        ensemble_stds.append(float(member_preds.std(axis=0).mean()))
                history['ensemble_std'] = float(np.mean(ensemble_stds)) if ensemble_stds else 0.0
            else:
                history, best_test_loss, best_state = self._run_epoch_loop(
                    model, train_params, train_coords, train_outputs, raw_train_params,
                    test_params, test_coords, test_outputs,
                    cfg, device, node_weights, pino_fn,
                    crack_loss_fn, out_std_t, out_mean_t, cc, callback,
                    standalone_pd_loss=standalone_pd_loss,
                    variational_loss_fn=variational_loss_fn,
                    train_damage_fields=train_damage_fields,
                )
                if best_state:
                    model.load_state_dict(best_state)
                    model.to(device)

            model._is_trained = True
            self._model = model

            model_path = None
            if self.config.save_dir:
                model_path = self.config.save_dir / "surrogate_model.pt"
                model.save(model_path)

            # Final metrics on test set (denormalized)
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for idx in range(n_test):
                    params_t = numpy_to_tensor(test_params[idx:idx+1], device)
                    coords_t = numpy_to_tensor(test_coords[idx], device).unsqueeze(0)
                    pred = model.forward(params_t, coords_t)
                    all_preds.append(pred.cpu().numpy().flatten())
                    all_targets.append(test_outputs[idx].flatten())

            metrics = model.compute_error(
                np.concatenate(all_preds),
                np.concatenate(all_targets),
            )

            return TrainingResult(
                success=True,
                train_loss=history['train_loss'][-1],
                test_loss=best_test_loss,
                metrics=metrics,
                history=history,
                model_path=model_path,
                normalization_params=norm_params,
            )

        except Exception as e:
            import traceback
            msg = traceback.format_exc()
            print(f"[SurrogateTrainer] Training failed:\n{msg}")
            return TrainingResult(
                success=False,
                error_message=str(e),
            )

    def _run_epoch_loop(
        self,
        model: nn.Module,
        train_params: np.ndarray,
        train_coords: List[np.ndarray],
        train_outputs: List[np.ndarray],
        raw_train_params: np.ndarray,
        test_params: np.ndarray,
        test_coords: List[np.ndarray],
        test_outputs: List[np.ndarray],
        cfg,
        device,
        node_weights,
        pino_fn,
        crack_loss_fn,
        out_std_t,
        out_mean_t,
        cc,
        callback,
        standalone_pd_loss=None,
        variational_loss_fn=None,
        train_damage_fields=None,
    ):
        """Run the epoch training loop for a single model.

        Returns (history dict, best_test_loss, best_state_dict).
        Used by both the single-model and bootstrap-ensemble paths.
        """
        n_train = len(train_params)
        n_test  = len(test_params)
        _nu_col = cc.nu_param_idx if cc is not None else None

        optimizer = create_optimizer(model, cfg)
        scheduler = create_scheduler(optimizer, cfg)

        history = {
            'train_loss': [], 'test_loss': [], 'pino_loss': [],
            'elasticity_loss': [], 'crack_loss': [],
        }
        best_test_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_pino_loss = 0.0
            epoch_elasticity_loss = 0.0
            epoch_crack_loss = 0.0
            indices = np.random.permutation(n_train)

            optimizer.zero_grad()
            accum = 0

            for idx in indices:
                params_t = numpy_to_tensor(train_params[idx:idx+1], device)
                coords_t = numpy_to_tensor(train_coords[idx], device).unsqueeze(0)
                output_t = numpy_to_tensor(train_outputs[idx], device).unsqueeze(0)

                pred = model.forward(params_t, coords_t)
                data_loss = _weighted_mse(pred, output_t, node_weights) / cfg.batch_size
                coords_xy = coords_t[0, :, :2]

                if pino_fn is not None:
                    sample_nu = float(raw_train_params[idx][_nu_col]) if _nu_col is not None else None
                    elasticity_component = pino_fn(pred[0], output_t[0], coords_xy, nu=sample_nu) / cfg.batch_size
                else:
                    elasticity_component = torch.tensor(0.0, device=device)

                if crack_loss_fn is not None:
                    u_phys = pred[0]
                    if out_std_t is not None:
                        u_phys = u_phys * out_std_t.unsqueeze(0) + out_mean_t.unsqueeze(0)
                    raw_p = raw_train_params[idx]
                    crack_component = crack_loss_fn(
                        u_phys, coords_xy,
                        K_I=float(raw_p[cc.ki_param_idx]),
                        E=float(raw_p[cc.e_param_idx]),
                        nu=float(raw_p[cc.nu_param_idx]),
                    ) / cfg.batch_size
                elif standalone_pd_loss is not None:
                    # Phase field path: PD equilibrium only (no K_I available)
                    u_phys = pred[0]
                    if out_std_t is not None:
                        u_phys = u_phys * out_std_t.unsqueeze(0) + out_mean_t.unsqueeze(0)
                    crack_component = (
                        cfg.near_tip * standalone_pd_loss(u_phys[:, :2], coords_xy)
                    ) / cfg.batch_size
                else:
                    crack_component = torch.tensor(0.0, device=device)

                # Variational AT-2 elastic energy loss (optional)
                var_component = torch.tensor(0.0, device=device)
                if variational_loss_fn is not None and train_damage_fields is not None:
                    u_v = pred[0]
                    if out_std_t is not None:
                        u_v = u_v * out_std_t.unsqueeze(0) + out_mean_t.unsqueeze(0)
                    damage_np = train_damage_fields[idx]
                    damage_t = numpy_to_tensor(damage_np, device)
                    traction_val = float(raw_train_params[idx][cc.traction_param_idx]) if cc is not None else 0.0
                    var_component = (
                        getattr(cfg, 'variational_weight', 0.0) * variational_loss_fn(
                            u_v[:, :2], coords_xy,
                            elements=None,  # will use Delaunay; pass coords_t[0] topology if available
                            traction=traction_val,
                            damage=damage_t,
                        )
                    ) / cfg.batch_size

                physics_loss = elasticity_component + crack_component + var_component
                loss = data_loss + physics_loss
                loss.backward()
                epoch_loss += data_loss.item() * cfg.batch_size
                epoch_pino_loss += physics_loss.item() * cfg.batch_size
                epoch_elasticity_loss += elasticity_component.item() * cfg.batch_size
                epoch_crack_loss += crack_component.item() * cfg.batch_size
                accum += 1

                if accum >= cfg.batch_size:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum = 0

            if accum > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss = epoch_loss / n_train
            history['train_loss'].append(train_loss)

            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for idx in range(n_test):
                    params_t = numpy_to_tensor(test_params[idx:idx+1], device)
                    coords_t = numpy_to_tensor(test_coords[idx], device).unsqueeze(0)
                    output_t = numpy_to_tensor(test_outputs[idx], device).unsqueeze(0)
                    pred = model.forward(params_t, coords_t)
                    test_loss += _weighted_mse(pred, output_t).item() / n_test
            history['test_loss'].append(test_loss)
            history['pino_loss'].append(epoch_pino_loss / n_train)
            history['elasticity_loss'].append(epoch_elasticity_loss / n_train)
            history['crack_loss'].append(epoch_crack_loss / n_train)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                break

            if callback:
                callback(epoch, train_loss)

        return history, best_test_loss, best_state

    @property
    def model(self) -> Optional[SurrogateModel]:
        """Get the trained model."""
        return self._model

    def predict_with_uncertainty(
        self,
        parameters: np.ndarray,
        coordinates: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Return (mean, uncertainty) both in original (denormalized) scale.

        Args:
            parameters:  Input parameters (N_samples, n_params) or (n_params,)
            coordinates: Query coordinates (num_points, coord_dim)

        Returns:
            Tuple of (mean, uncertainty), each (num_points, output_dim)
            uncertainty is None if model does not support it.
        """
        if self._model is None or not self._model.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if self._input_normalizer:
            parameters = self._input_normalizer.transform(parameters)

        result = self._model.predict(parameters, coordinates)
        mean = result.values
        unc = result.uncertainty

        if self._output_normalizer:
            output_dim = mean.shape[-1] if mean.ndim > 1 else 1
            mean = self._output_normalizer.inverse_transform(
                mean.reshape(-1, output_dim)
            ).reshape(mean.shape)
            if unc is not None:
                unc = (
                    unc.reshape(-1, output_dim) * self._output_normalizer.std
                ).reshape(unc.shape)

        return mean, unc

    def predict(
        self,
        parameters: np.ndarray,
        coordinates: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions with trained model.

        Handles normalization/denormalization automatically.

        Args:
            parameters: Input parameters
            coordinates: Query coordinates

        Returns:
            Predicted field values (denormalized)
        """
        if self._model is None or not self._model.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Normalize inputs
        if self._input_normalizer:
            parameters = self._input_normalizer.transform(parameters)

        # Predict
        result = self._model.predict(parameters, coordinates)
        predictions = result.values

        # Denormalize outputs
        if self._output_normalizer:
            output_dim = predictions.shape[-1] if predictions.ndim >= 2 else 1
            predictions = self._output_normalizer.inverse_transform(
                predictions.reshape(-1, output_dim)
            ).reshape(predictions.shape)

        return predictions


class Normalizer:
    """Simple mean-std normalizer."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "Normalizer":
        """Fit normalizer to data."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero
        self.std = np.where(self.std < 1e-10, 1.0, self.std)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform to original scale."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted.")
        return data * self.std + self.mean
