
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor

from preprocessing import (
    load_numerical_dataset, split_dataset, fit_scalers, transform_with_scalers,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_PATH = Path("results/phase2_baseline_metrics.json")



@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float

    @classmethod
    def compute(cls, y_true: np.ndarray, y_pred: np.ndarray) -> "Metrics":
        return cls(
            rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
            mae=float(mean_absolute_error(y_true, y_pred)),
            r2=float(r2_score(y_true, y_pred)),
        )


def evaluate_on_all_splits(model, splits, scalers=None, model_uses_scaled_input=False) -> dict:

    out = {}
    for split_name, X, y_true in [
        ("val", splits.X_val, splits.y_val),
        ("test", splits.X_test, splits.y_test),
        ("ood", splits.X_ood, splits.y_ood),
    ]:
        if len(X) == 0:
            out[split_name] = None
            continue

        if model_uses_scaled_input:
            X_input = scalers.x_scaler.transform(X.values)
            y_pred_scaled = model.predict(X_input).reshape(-1, 1)
            y_pred = scalers.y_scaler.inverse_transform(y_pred_scaled).ravel()
        else:
            y_pred = model.predict(X.values)

        out[split_name] = asdict(Metrics.compute(y_true.values, y_pred))
    return out



def train_xgboost(splits):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    model.fit(
        splits.X_train.values, splits.y_train.values,
        eval_set=[(splits.X_val.values, splits.y_val.values)],
        verbose=False,
    )
    return model


def train_lightgbm(splits):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        splits.X_train.values, splits.y_train.values,
        eval_set=[(splits.X_val.values, splits.y_val.values)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def train_random_forest(splits):
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(splits.X_train.values, splits.y_train.values)
    return model


def train_mlp(splits, scalers):
  
    X_train_s, y_train_s = transform_with_scalers(splits.X_train, splits.y_train, scalers)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 64, 32),
        activation="relu",   # standard baseline deliberately uses ReLU;
                              # Phase 3 PGNN will switch to SiLU/Softplus
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_s, y_train_s)
    return model


def main():
    df = load_numerical_dataset()
    splits = split_dataset(df)
    scalers = fit_scalers(splits.X_train, splits.y_train)

    results = {}

    trainers = {
        "XGBoost": (train_xgboost, False),
        "LightGBM": (train_lightgbm, False),
        "RandomForest": (train_random_forest, False),
        "MLP": (train_mlp, True),
    }

    for name, (trainer_fn, uses_scaled_input) in trainers.items():
        print(f"\n=== Training {name} ===")
        try:
            t0 = time.time()
            if name == "MLP":
                model = trainer_fn(splits, scalers)
            else:
                model = trainer_fn(splits)
            train_time = time.time() - t0

            metrics = evaluate_on_all_splits(
                model, splits, scalers=scalers, model_uses_scaled_input=uses_scaled_input
            )
            metrics["train_time_sec"] = round(train_time, 2)
            results[name] = metrics

            print(f"{name} | train time: {train_time:.1f}s")
            for split_name in ["val", "test", "ood"]:
                m = metrics[split_name]
                if m is None:
                    print(f"  {split_name}: (empty split)")
                else:
                    print(f"  {split_name}: RMSE={m['rmse']:.2f} min | "
                          f"MAE={m['mae']:.2f} min | R2={m['r2']:.4f}")

        except ImportError as e:
            print(f"[SKIPPED] {name}: {e}. Install with "
                  f"`pip install {name.lower()} --break-system-packages`")
            results[name] = {"error": str(e)}

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all baseline metrics to {RESULTS_PATH}")

    # Summary table across whatever models actually trained successfully
    print("\n=== Summary (Test Split) ===")
    print(f"{'Model':<15}{'RMSE':>10}{'MAE':>10}{'R2':>10}")
    for name, m in results.items():
        if "test" in m and m["test"] is not None:
            t = m["test"]
            print(f"{name:<15}{t['rmse']:>10.2f}{t['mae']:>10.2f}{t['r2']:>10.4f}")

    print("\n=== Summary (OOD Extrapolation Split) — the key Phase-4 baseline ===")
    print(f"{'Model':<15}{'RMSE':>10}{'MAE':>10}{'R2':>10}")
    for name, m in results.items():
        if "ood" in m and m["ood"] is not None:
            o = m["ood"]
            print(f"{name:<15}{o['rmse']:>10.2f}{o['mae']:>10.2f}{o['r2']:>10.4f}")


if __name__ == "__main__":
    main()
