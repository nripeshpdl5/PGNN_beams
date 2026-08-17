
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import (
    load_numerical_dataset, split_dataset, fit_scalers,
    FEATURE_COLUMNS, RANDOM_STATE,
)
from phase3 import CAPACITY_COLUMNS, CHECKPOINT_PATH, CONTROL_CHECKPOINT_PATH, _build_model

RESULTS_PATH = Path("results/phase4_pvr_metrics.json")
FIG_DIR = Path("results")

CONSTRAINED_FEATURES = {
    "tins": +1, "hi": +1, "Tg": +1,
    "kins": -1, "LR": -1, "Ld": -1,
}
PERTURB_SIGMA_FRACTION = 0.25   
TOLERANCE_MINUTES = 0.5         
N_EVAL_MAX = 3000               


EXTRA_CHECKPOINTS = {
    "physics_only": Path("results/pgnn_physics_only.pt"),
    "asymmetric_only": Path("results/pgnn_asymmetric_only.pt"),
}


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

MODEL_COLORS = {          
    "PGNN": "#2a78d6",    
    "MLP control": "#eb6834",
    "XGBoost": "#1baf7a",      
}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)



def _load_pgnn_predictor(checkpoint: Path, scalers):
    import torch

    model = _build_model(input_dim=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    def predict(X_phys: np.ndarray) -> np.ndarray:
        X_s = scalers.x_scaler.transform(X_phys)
        with torch.no_grad():
            fr_hat, _ = model(torch.tensor(X_s, dtype=torch.float32))
        return fr_hat.numpy() * scalers.y_scaler.scale_[0] + scalers.y_scaler.mean_[0]

    return predict


def _train_xgboost_predictor(splits):
    from phase2 import train_xgboost

    model = train_xgboost(splits)
    return lambda X_phys: model.predict(X_phys)



def compute_pvr(predict, X: pd.DataFrame, train_std: pd.Series) -> dict:
    X_base = X.values.astype(float)
    y_base = predict(X_base)

    out = {}
    violated_any = np.zeros(len(X_base), dtype=bool)
    for feat, sign in CONSTRAINED_FEATURES.items():
        j = FEATURE_COLUMNS.index(feat)
        delta = PERTURB_SIGMA_FRACTION * float(train_std[feat])
        X_pert = X_base.copy()
        X_pert[:, j] += delta
        d_fr = predict(X_pert) - y_base
        # sign=+1: FR must not DROP when feat rises -> violation if d_fr < -tol
        # sign=-1: FR must not RISE when feat rises -> violation if d_fr > +tol
        violated = (sign * d_fr) < -TOLERANCE_MINUTES
        out[feat] = float(violated.mean() * 100.0)
        violated_any |= violated
    out["any"] = float(violated_any.mean() * 100.0)
    return out



def response_curve(predict, X_ref: pd.DataFrame, feature: str, grid: np.ndarray) -> np.ndarray:
    base = X_ref.median().values.astype(float)
    j = FEATURE_COLUMNS.index(feature)
    X_sweep = np.tile(base, (len(grid), 1))
    X_sweep[:, j] = grid
    return predict(X_sweep)



def plot_pvr_bars(pvr: dict[str, dict], out_path: Path):
    import matplotlib.pyplot as plt

    features = list(CONSTRAINED_FEATURES) + ["any"]
    labels = {"tins": "t_ins", "hi": "h_i", "Tg": "T_g",
              "kins": "k_ins", "LR": "LR", "Ld": "L_d", "any": "Any"}
    models = [m for m in MODEL_COLORS if m in pvr]

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    n = len(models)
    group_w = 0.72                       
    bar_w = group_w / n
    x0 = np.arange(len(features))
    for k, name in enumerate(models):
        xs = x0 - group_w / 2 + (k + 0.5) * bar_w
        vals = [pvr[name][f] for f in features]
        ax.bar(xs, vals, width=bar_w * 0.9,   
               color=MODEL_COLORS[name], edgecolor="none", zorder=3, label=name)
        for x, v in zip(xs, vals):            
            ax.text(x, v + 0.6, f"{v:.1f}",   
                    ha="center", va="bottom", fontsize=7.5, color=INK_2)

    ax.set_xticks(x0)
    ax.set_xticklabels([labels[f] for f in features], color=INK_2, fontsize=10)
    ax.set_ylabel("Physics Violation Rate (%)", color=INK_2, fontsize=10)
    ax.set_title("Monotonicity violations under +0.25σ perturbation (test split)",
                 color=INK, fontsize=11, loc="left", pad=12)
    
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_response(curves: dict[str, np.ndarray], grid: np.ndarray, out_path: Path,
                   xlabel: str, title: str, data_max: float | None = None,
                   collapse_target: float | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    if data_max is not None and data_max < grid.max():
        
        ax.axvspan(data_max, grid.max(), color=GRID, alpha=0.45, zorder=1)
        ax.text(data_max + 0.02 * (grid.max() - grid.min()),
                ax.get_ylim()[1], "beyond training data",
                ha="left", va="top", fontsize=8, color=MUTED)

    for name, y in curves.items():
        ax.plot(grid, y, color=MODEL_COLORS[name], linewidth=2,
                solid_capstyle="round", zorder=3, label=name)
        
        ax.plot(grid[-1], y[-1], "o", ms=5, color=MODEL_COLORS[name],
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)

    if collapse_target is not None:
        ax.axhline(collapse_target, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(grid.min(), collapse_target + 1.5, "physical limit: FR → 0 at LR ≥ 100%",
                fontsize=8, color=MUTED, va="bottom")

    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel("Predicted fire resistance (min)", color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")



def main():
    df = load_numerical_dataset(extra_columns=CAPACITY_COLUMNS)
    splits = split_dataset(df)
    scalers = fit_scalers(splits.X_train, splits.y_train)
    train_std = splits.X_train.std()

    predictors = {}
    try:
        predictors["PGNN"] = _load_pgnn_predictor(CHECKPOINT_PATH, scalers)
        predictors["MLP control"] = _load_pgnn_predictor(CONTROL_CHECKPOINT_PATH, scalers)
    except FileNotFoundError as e:
        raise SystemExit(f"Missing Phase-3 checkpoint ({e}). Run phase3.py first.")
    try:
        predictors["XGBoost"] = _train_xgboost_predictor(splits)
    except ImportError as e:
        print(f"[SKIPPED] XGBoost baseline: {e}")

    extra_predictors = {}
    for tag, path in EXTRA_CHECKPOINTS.items():
        if path.exists():
            extra_predictors[tag] = _load_pgnn_predictor(path, scalers)


    rng = np.random.default_rng(RANDOM_STATE)
    results = {"config": {
        "perturbation": f"{PERTURB_SIGMA_FRACTION} * train sigma",
        "tolerance_minutes": TOLERANCE_MINUTES,
        "n_eval_max": N_EVAL_MAX,
    }}
    for split_name, X in [("test", splits.X_test), ("ood", splits.X_ood)]:
        if len(X) > N_EVAL_MAX:
            X = X.iloc[rng.choice(len(X), N_EVAL_MAX, replace=False)]
        results[split_name] = {}
        for name, predict in {**predictors, **extra_predictors}.items():
            results[split_name][name] = compute_pvr(predict, X, train_std)
            print(f"  PVR[{split_name}] {name}: any={results[split_name][name]['any']:.1f}%")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved PVR metrics to {RESULTS_PATH}")


    plot_pvr_bars(results["test"], FIG_DIR / "fig_pvr_bars.png")

    tins_grid = np.linspace(splits.X_train["tins"].min(), splits.X_train["tins"].max(), 60)
    plot_response(
        {n: response_curve(p, splits.X_test, "tins", tins_grid) for n, p in predictors.items()},
        tins_grid, FIG_DIR / "fig_response_tins.png",
        xlabel="Insulation thickness t_ins (mm)",
        title="Median-beam response to insulation thickness",
    )

    lr_max_data = float(splits.X_train["LR"].max())
    lr_grid = np.linspace(splits.X_train["LR"].min(), 150.0, 90)
    plot_response(
        {n: response_curve(p, splits.X_test, "LR", lr_grid) for n, p in predictors.items()},
        lr_grid, FIG_DIR / "fig_response_lr.png",
        xlabel="Load ratio LR (%)",
        title="Median-beam response to load ratio, extended past the data range",
        data_max=lr_max_data, collapse_target=0.0,
    )


    print("\n=== PVR summary (test | ood, 'any constraint violated') ===")
    for name in {**predictors, **extra_predictors}:
        t, o = results["test"][name]["any"], results["ood"][name]["any"]
        print(f"  {name:<18} {t:6.1f}% | {o:6.1f}%")


if __name__ == "__main__":
    main()