
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing import (
    load_numerical_dataset, split_dataset, fit_scalers, transform_with_scalers,
    FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_STATE,
)

CAPACITY_COLUMNS = ["Initial Capacity", "Final Capacity"]
RESULTS_PATH = Path("results/phase3_pgnn_metrics.json")
CHECKPOINT_PATH = Path("results/pgnn_best.pt")
CONTROL_CHECKPOINT_PATH = Path("results/pgnn_control_best.pt")



def _build_model(input_dim: int, hidden_sizes=(64, 64, 32), activation: str = "silu"):

    import torch.nn as nn

    act_layer = {"silu": nn.SiLU, "softplus": nn.Softplus}[activation]

    class PGNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_sizes:
                layers += [nn.Linear(prev, h), act_layer()]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.fr_head = nn.Linear(prev, 1)         # primary: FR (scaled)
            self.capacity_head = nn.Linear(prev, 1)   # auxiliary: Final Capacity (scaled)

        def forward(self, x):
            z = self.trunk(x)
            fr_hat = self.fr_head(z).squeeze(-1)
            capacity_hat = self.capacity_head(z).squeeze(-1)
            return fr_hat, capacity_hat

    return PGNN()



@dataclass
class LossWeights:
   
    lambda_insulation: float = 0.1
    lambda_kins: float = 0.1
    lambda_load: float = 0.1
    lambda_tg: float = 0.05
    lambda_bound: float = 0.05
    lambda_capacity: float = 0.2
    lambda_maxload: float = 0.1
    lambda_capacity_fit: float = 0.3   
    alpha_safety: float = 0.8 
    maxload_delta_minutes: float = 5.0 


CONTROL_WEIGHTS = LossWeights(
    lambda_insulation=0.0, lambda_kins=0.0, lambda_load=0.0, lambda_tg=0.0,
    lambda_bound=0.0, lambda_capacity=0.0, lambda_maxload=0.0,
    lambda_capacity_fit=0.0,  
    alpha_safety=0.5,        
)


def _asymmetric_safety_loss(y_pred: "torch.Tensor", y_true: "torch.Tensor", alpha: float) -> "torch.Tensor":
    import torch

    diff_sq = (y_pred - y_true) ** 2
    over_mask = (y_pred > y_true).float()
    weight = alpha * over_mask + (1.0 - alpha) * (1.0 - over_mask)
    return (weight * diff_sq).mean()


def compute_physics_losses(model, x_batch: "torch.Tensor", scalers, weights: LossWeights,
                            initial_capacity_scaled: Optional["torch.Tensor"] = None):
    import torch
    import torch.nn.functional as F

    idx = {name: FEATURE_COLUMNS.index(name) for name in
           ["tins", "hi", "kins", "LR", "Ld", "Tg"]}

    x_batch = x_batch.clone().detach().requires_grad_(True)
    fr_hat, capacity_hat = model(x_batch)


    grad = torch.autograd.grad(
        fr_hat.sum(), x_batch, create_graph=True, retain_graph=True
    )[0]

    losses = {}

    losses["insulation"] = (
        F.relu(-grad[:, idx["tins"]]).mean() + F.relu(-grad[:, idx["hi"]]).mean()
    )

    losses["kins"] = F.relu(grad[:, idx["kins"]]).mean()

    losses["load"] = (
        F.relu(grad[:, idx["LR"]]).mean() + F.relu(grad[:, idx["Ld"]]).mean()
    )

    losses["tg"] = F.relu(-grad[:, idx["Tg"]]).mean()

    sigma_y = float(scalers.y_scaler.scale_[0])
    fr_hat_phys = fr_hat * sigma_y + float(scalers.y_scaler.mean_[0])
    losses["bound"] = F.relu(-fr_hat_phys).mean() / sigma_y

    if initial_capacity_scaled is not None:
        losses["capacity"] = F.relu(capacity_hat - initial_capacity_scaled).mean()
    else:
        losses["capacity"] = torch.tensor(0.0, device=x_batch.device)

    return losses, fr_hat, capacity_hat


def compute_maxload_collocation_loss(model, x_batch_scaled_numpy: np.ndarray, scalers,
                                      weights: LossWeights,
                                      rng: Optional[np.random.Generator] = None):
    import torch
    import torch.nn.functional as F

    if rng is None:
        rng = np.random.default_rng()

    idx_lr = FEATURE_COLUMNS.index("LR")

    x_phys = x_batch_scaled_numpy * scalers.x_scaler.scale_ + scalers.x_scaler.mean_
    synthetic_lr = rng.uniform(100.0, 150.0, size=x_phys.shape[0])
    x_phys_synth = x_phys.copy()
    x_phys_synth[:, idx_lr] = synthetic_lr
    x_scaled_synth = (x_phys_synth - scalers.x_scaler.mean_) / scalers.x_scaler.scale_

    x_synth_t = torch.tensor(x_scaled_synth, dtype=torch.float32)
    fr_hat_synth, _ = model(x_synth_t)
    sigma_y = float(scalers.y_scaler.scale_[0])
    fr_hat_synth_phys = fr_hat_synth * sigma_y + float(scalers.y_scaler.mean_[0])

    
    upper = F.relu(fr_hat_synth_phys - weights.maxload_delta_minutes)
    lower = F.relu(-fr_hat_synth_phys)
    return (upper + lower).mean() / sigma_y


def total_loss(model, x_batch, y_batch, scalers, weights: LossWeights,
                initial_capacity_scaled=None, final_capacity_scaled=None,
                physics_ramp: float = 1.0,
                collocation_rng: Optional[np.random.Generator] = None):
    import torch
    import torch.nn.functional as F

    physics_losses, fr_hat, capacity_hat = compute_physics_losses(
        model, x_batch, scalers, weights, initial_capacity_scaled
    )

    l_data = _asymmetric_safety_loss(fr_hat, y_batch, weights.alpha_safety)

    if final_capacity_scaled is not None and weights.lambda_capacity_fit > 0:
        l_capacity_fit = F.mse_loss(capacity_hat, final_capacity_scaled)
    else:
        l_capacity_fit = torch.tensor(0.0)

    if weights.lambda_maxload > 0 and physics_ramp > 0:
        l_maxload = compute_maxload_collocation_loss(
            model, x_batch.detach().cpu().numpy(), scalers, weights,
            rng=collocation_rng,
        )
    else:
        l_maxload = torch.tensor(0.0)

    total = (
        l_data
        + weights.lambda_capacity_fit * l_capacity_fit
        + physics_ramp * (
            weights.lambda_insulation * physics_losses["insulation"]
            + weights.lambda_kins * physics_losses["kins"]
            + weights.lambda_load * physics_losses["load"]
            + weights.lambda_tg * physics_losses["tg"]
            + weights.lambda_bound * physics_losses["bound"]
            + weights.lambda_capacity * physics_losses["capacity"]
            + weights.lambda_maxload * l_maxload
        )
    )

    breakdown = {
        "data": float(l_data.item()),
        "capacity_fit": float(l_capacity_fit.item()),
        "insulation": float(physics_losses["insulation"].item()),
        "kins": float(physics_losses["kins"].item()),
        "load": float(physics_losses["load"].item()),
        "tg": float(physics_losses["tg"].item()),
        "bound": float(physics_losses["bound"].item()),
        "capacity_bound": float(physics_losses["capacity"].item()),
        "maxload": float(l_maxload.item()),
        "physics_ramp": physics_ramp,
        "total": float(total.item()),
    }
    return total, breakdown



def train_pgnn(splits, scalers, capacity_splits, weights: LossWeights,
                hidden_sizes=(64, 64, 32), activation="silu",
                lr=1e-3, batch_size=256, max_epochs=500, patience=40,
                warmup_epochs=30, weight_decay=1e-4, verbose=True,
                seed: int = RANDOM_STATE):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)

    X_train_s, y_train_s = transform_with_scalers(splits.X_train, splits.y_train, scalers)
    X_val_s, y_val_s = transform_with_scalers(splits.X_val, splits.y_val, scalers)

    cap_scaler = StandardScaler().fit(
        capacity_splits["train"]["Final Capacity"].values.reshape(-1, 1)
    )

    def _scale_cap(series):
        return torch.tensor(
            cap_scaler.transform(series.values.reshape(-1, 1)).ravel(),
            dtype=torch.float32,
        )

    cap_train = _scale_cap(capacity_splits["train"]["Final Capacity"])
    init_cap_train = _scale_cap(capacity_splits["train"]["Initial Capacity"])
    cap_val = _scale_cap(capacity_splits["val"]["Final Capacity"])
    init_cap_val = _scale_cap(capacity_splits["val"]["Initial Capacity"])

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_s, dtype=torch.float32),
        cap_train, init_cap_train,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = _build_model(input_dim=X_train_s.shape[1], hidden_sizes=hidden_sizes, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_s, dtype=torch.float32)

    any_physics = any([
        weights.lambda_insulation, weights.lambda_kins, weights.lambda_load,
        weights.lambda_tg, weights.lambda_bound, weights.lambda_capacity,
        weights.lambda_maxload,
    ])

    for epoch in range(max_epochs):
        if any_physics and warmup_epochs > 0:
            physics_ramp = min(1.0, (epoch + 1) / warmup_epochs)
        else:
            physics_ramp = 1.0 if any_physics else 0.0

        model.train()
        for xb, yb, cap_b, init_cap_b in train_loader:
            optimizer.zero_grad()
            loss, _ = total_loss(model, xb, yb, scalers, weights,
                                  initial_capacity_scaled=init_cap_b,
                                  final_capacity_scaled=cap_b,
                                  physics_ramp=physics_ramp)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_pgnn(model, splits.X_val, splits.y_val, scalers)
        val_rmse = val_metrics["rmse"]
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 10 == 0:
            model.eval()
            _, val_breakdown = total_loss(
                model, X_val_t, y_val_t, scalers, weights,
                initial_capacity_scaled=init_cap_val, final_capacity_scaled=cap_val,
                physics_ramp=physics_ramp,
                collocation_rng=np.random.default_rng(RANDOM_STATE),
            )
            print(f"epoch {epoch:4d} | val_RMSE={val_rmse:8.3f} min | "
                  f"data={val_breakdown['data']:.4f} | "
                  f"cap_fit={val_breakdown['capacity_fit']:.4f} | "
                  f"insulation={val_breakdown['insulation']:.4f} | "
                  f"load={val_breakdown['load']:.4f} | "
                  f"bound={val_breakdown['bound']:.4f} | "
                  f"ramp={physics_ramp:.2f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best val RMSE {best_val_rmse:.3f} min)")
            break

    model.load_state_dict(best_state)
    return model



def evaluate_pgnn(model, X: pd.DataFrame, y: pd.Series, scalers,
                   alpha_safety: float = 0.8) -> dict:
    import torch
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    model.eval()
    X_s = scalers.x_scaler.transform(X.values)
    with torch.no_grad():
        fr_hat, _ = model(torch.tensor(X_s, dtype=torch.float32))
    y_pred = fr_hat.numpy() * scalers.y_scaler.scale_[0] + scalers.y_scaler.mean_[0]

    errors = y_pred - y.values
    over = errors[errors > 0]

    rmse = float(np.sqrt(mean_squared_error(y.values, y_pred)))
    bias = float(errors.mean())

   
    weight = np.where(errors > 0, alpha_safety, 1.0 - alpha_safety)
    safety_loss = float((weight * errors ** 2).mean())

    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y.values, y_pred)),
        "r2": float(r2_score(y.values, y_pred)),

        "mean_error_min": bias,
        "rmse_bias_removed": float(np.sqrt(max(rmse ** 2 - bias ** 2, 0.0))),
        "pct_over_predictions": float((errors > 0).mean() * 100.0),
        "mean_over_pred_min": float(over.mean()) if len(over) else 0.0,
        "max_over_pred_min": float(over.max()) if len(over) else 0.0,
        "safety_loss_min2": safety_loss,
    }



def run_optuna_search(splits, scalers, capacity_splits, n_trials=30):
    import optuna

    def objective(trial):
        weights = LossWeights(
            lambda_insulation=trial.suggest_float("lambda_insulation", 0.01, 1.0, log=True),
            lambda_kins=trial.suggest_float("lambda_kins", 0.01, 1.0, log=True),
            lambda_load=trial.suggest_float("lambda_load", 0.01, 1.0, log=True),
            lambda_tg=trial.suggest_float("lambda_tg", 0.01, 0.5, log=True),
            lambda_bound=trial.suggest_float("lambda_bound", 0.01, 0.5, log=True),
            lambda_capacity=trial.suggest_float("lambda_capacity", 0.01, 1.0, log=True),
            lambda_maxload=trial.suggest_float("lambda_maxload", 0.01, 1.0, log=True),
            lambda_capacity_fit=trial.suggest_float("lambda_capacity_fit", 0.05, 1.0, log=True),
            alpha_safety=0.8,
        )
        model = train_pgnn(splits, scalers, capacity_splits, weights,
                            max_epochs=80, patience=15, warmup_epochs=15, verbose=False)
        val_metrics = evaluate_pgnn(model, splits.X_val, splits.y_val, scalers)
        return val_metrics["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value



def main():
    df = load_numerical_dataset(extra_columns=CAPACITY_COLUMNS)
    splits = split_dataset(df)
    scalers = fit_scalers(splits.X_train, splits.y_train)

    capacity_splits = {
        split_name: pd.DataFrame({
            "Initial Capacity": splits.get_aligned_column("Initial Capacity")[split_name],
            "Final Capacity": splits.get_aligned_column("Final Capacity")[split_name],
        })
        for split_name in ["train", "val", "test", "ood"]
    }

    try:
        import torch

        results = {}

        def _evaluate_all(model, tag):
            out = {}
            for split_name, X, y in [
                ("val", splits.X_val, splits.y_val),
                ("test", splits.X_test, splits.y_test),
                ("ood", splits.X_ood, splits.y_ood),
            ]:
                out[split_name] = evaluate_pgnn(model, X, y, scalers)
                m = out[split_name]
                print(f"  [{tag}] {split_name}: RMSE={m['rmse']:.2f} min | "
                      f"MAE={m['mae']:.2f} min | R2={m['r2']:.4f} | "
                      f"bias={m['mean_error_min']:+.2f} min | "
                      f"RMSE(bias-rm)={m['rmse_bias_removed']:.2f} min | "
                      f"over-pred={m['pct_over_predictions']:.1f}% "
                      f"(mean {m['mean_over_pred_min']:.2f} / max {m['max_over_pred_min']:.2f} min) | "
                      f"safety-loss={m['safety_loss_min2']:.1f}")
            return out

        full = LossWeights()

        variants = {
            "control_no_physics": CONTROL_WEIGHTS,
            "physics_only": replace(full, alpha_safety=0.5),
            "asymmetric_only": replace(
                CONTROL_WEIGHTS, alpha_safety=full.alpha_safety
            ),
            "pgnn": full,
        }
        checkpoint_paths = {
            "control_no_physics": CONTROL_CHECKPOINT_PATH,
            "pgnn": CHECKPOINT_PATH,
        }

        for tag, weights in variants.items():
            print(f"\n=== Training {tag} "
                  f"(alpha_safety={weights.alpha_safety}, "
                  f"physics={'on' if weights.lambda_insulation > 0 else 'off'}) ===")
            model = train_pgnn(splits, scalers, capacity_splits, weights)
            results[tag] = _evaluate_all(model, tag)

            ckpt = checkpoint_paths.get(
                tag, CHECKPOINT_PATH.with_name(f"pgnn_{tag}.pt")
            )
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            print(f"  saved weights to {ckpt}")

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to {RESULTS_PATH}")

        for split_name in ["test", "ood"]:
            print(f"\n=== Ablation summary ({split_name.upper()} split) ===")
            header = (f"{'Variant':<22}{'RMSE':>8}{'R2':>9}{'Bias':>9}"
                      f"{'RMSE-b':>9}{'Over%':>8}{'MaxOver':>9}{'SafeLoss':>10}")
            print(header)
            for tag in variants:
                m = results[tag][split_name]
                print(f"{tag:<22}{m['rmse']:>8.2f}{m['r2']:>9.4f}"
                      f"{m['mean_error_min']:>+9.2f}{m['rmse_bias_removed']:>9.2f}"
                      f"{m['pct_over_predictions']:>8.1f}{m['max_over_pred_min']:>9.2f}"
                      f"{m['safety_loss_min2']:>10.1f}")

    except ImportError as e:
        print(f"[SKIPPED] PGNN training requires torch: {e}")
        print("Install with: pip install torch --break-system-packages")


if __name__ == "__main__":
    main()
