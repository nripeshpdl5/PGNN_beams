

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import (
    load_numerical_dataset, load_fire_test_dataset, split_dataset, fit_scalers,
    FEATURE_COLUMNS, TARGET_COLUMN,
)
from phase3 import CAPACITY_COLUMNS, CHECKPOINT_PATH, CONTROL_CHECKPOINT_PATH
from phase4 import (
    EXTRA_CHECKPOINTS, MODEL_COLORS, SURFACE, INK, INK_2, MUTED,
    _load_pgnn_predictor, _train_xgboost_predictor, _style_axes,
)

RESULTS_PATH = Path("results/phase5_experimental_metrics.json")
FIG_PATH = Path("results/fig_parity_experimental.png")

ALPHA_SAFETY = 0.8  
SEVERITY_BINS = {           
    "mild": 1.0,
    "moderate": 3.0,
    "severe": np.inf,
}




def harmonize_real_units(df_real: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df_real.copy()
    report = {}

    m = df["L"] < 100                      
    df.loc[m, "L"] *= 1000.0
    report["L_m_to_mm"] = int(m.sum())

    m = df["Es"] < 1000                    
    df.loc[m, "Es"] *= 1000.0
    report["Es_gpa_to_mpa"] = int(m.sum())

    m = (df["Efrp"] > 0) & (df["Efrp"] < 1000)   
    df.loc[m, "Efrp"] *= 1000.0
    report["Efrp_gpa_to_mpa"] = int(m.sum())

    m = df["LR"] <= 1.0                    
    df.loc[m, "LR"] *= 100.0
    report["LR_ratio_to_pct"] = int(m.sum())

    report["Efrp_suspect_rows"] = int((df["Efrp"] > 700000).sum())
    return df, report




def extrapolation_severity(X_real: pd.DataFrame, X_train: pd.DataFrame):
    
    lo, hi, std = X_train.min(), X_train.max(), X_train.std().replace(0, 1.0)

    below = (lo - X_real).clip(lower=0)
    above = (X_real - hi).clip(lower=0)
    exceed_sigma = (below + above) / std          

    severity = exceed_sigma.max(axis=1).values
    worst_feature = exceed_sigma.idxmax(axis=1).values

    out_counts = (exceed_sigma > 0).sum()
    out_counts = out_counts[out_counts > 0].sort_values(ascending=False)
    return severity, worst_feature, out_counts


def severity_bin(severity: np.ndarray) -> np.ndarray:
    bins = np.full(len(severity), "severe", dtype=object)
    bins[severity <= SEVERITY_BINS["moderate"]] = "moderate"
    bins[severity <= SEVERITY_BINS["mild"]] = "mild"
    return bins




def make_clipped(predict, X_train: pd.DataFrame):
    
    lo = X_train.min().values
    hi = X_train.max().values

    def predict_clipped(X_phys: np.ndarray) -> np.ndarray:
        return predict(np.clip(X_phys, lo, hi))

    return predict_clipped




def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    errors = y_pred - y_true
    over = errors[errors > 0]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(errors.mean())
    weight = np.where(errors > 0, ALPHA_SAFETY, 1.0 - ALPHA_SAFETY)

    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
       
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 3 else None,
        "mean_error_min": bias,
        "rmse_bias_removed": float(np.sqrt(max(rmse ** 2 - bias ** 2, 0.0))),
        "n_over_predictions": int((errors > 0).sum()),
        "pct_over_predictions": float((errors > 0).mean() * 100.0),
        "mean_over_pred_min": float(over.mean()) if len(over) else 0.0,
        "max_over_pred_min": float(over.max()) if len(over) else 0.0,
        "safety_loss_min2": float((weight * errors ** 2).mean()),
    }




def plot_parity(y_true: np.ndarray, preds: dict[str, np.ndarray],
                 severe_mask: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=1) 

    lim_hi = 1.05 * max(y_true.max(), max(p.max() for p in preds.values()))
    lims = (0, lim_hi)

    ax.plot(lims, lims, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.60 * lim_hi, 0.53 * lim_hi, "conservative (safe)", fontsize=8,
            color=MUTED, rotation=45, rotation_mode="anchor", ha="center", va="top")
    ax.text(0.53 * lim_hi, 0.60 * lim_hi, "over-prediction (unsafe)", fontsize=8,
            color=MUTED, rotation=45, rotation_mode="anchor", ha="center", va="bottom")

    for name, y_pred in preds.items():
        c = MODEL_COLORS[name]
        m = ~severe_mask
        ax.scatter(y_true[m], y_pred[m], s=46, color=c,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.scatter(y_true[severe_mask], y_pred[severe_mask], s=46,
                   facecolors="none", edgecolors=c, linewidths=1.6, zorder=3)

    ax.set_xlabel("Measured fire resistance (min)", color=INK_2, fontsize=10)
    ax.set_ylabel("Predicted fire resistance (min), input-clipped", color=INK_2, fontsize=10)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_title("Blind validation on 50 real fire tests (deployment guard on)",
                 color=INK, fontsize=11, loc="left", pad=12)

    handles = [Line2D([], [], marker="o", linestyle="none", ms=7,
                       color=MODEL_COLORS[n], markeredgecolor=SURFACE, label=n)
               for n in preds]
    handles += [
        Line2D([], [], marker="o", linestyle="none", ms=7, color=INK_2,
               markeredgecolor=SURFACE, label="mild/moderate extrapolation"),
        Line2D([], [], marker="o", linestyle="none", ms=7, markerfacecolor="none",
               markeredgecolor=INK_2, label="severe extrapolation (>3σ)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5,
              labelcolor=INK_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")




def main():

    df_sim = load_numerical_dataset(extra_columns=CAPACITY_COLUMNS)
    splits = split_dataset(df_sim)
    scalers = fit_scalers(splits.X_train, splits.y_train)

    df_real = load_fire_test_dataset()
    df_real, unit_report = harmonize_real_units(df_real)
    if any(v for k, v in unit_report.items() if k != "Efrp_suspect_rows"):
        print(f"[WARNING] unit guard made corrections — workbook may have "
              f"regressed: {unit_report}")
    X_real = df_real[FEATURE_COLUMNS]
    y_real = df_real[TARGET_COLUMN].values.astype(float)
    print(f"Real fire-test records: {len(df_real)}")

    severity, worst_feature, out_counts = extrapolation_severity(X_real, splits.X_train)
    bins = severity_bin(severity)
    print("Extrapolation severity (train-sigma units): "
          + ", ".join(f"{b}={int((bins == b).sum())}" for b in SEVERITY_BINS))
    if len(out_counts):
        print("Out-of-range beams per feature:")
        for feat, cnt in out_counts.items():
            print(f"  {feat:<10} {cnt}")

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

    extra_predictors = {
        tag: _load_pgnn_predictor(path, scalers)
        for tag, path in EXTRA_CHECKPOINTS.items() if path.exists()
    }
    all_predictors = {**predictors, **extra_predictors}


    subgroups = {"all": np.ones(len(y_real), dtype=bool)}
    for b in SEVERITY_BINS:
        subgroups[b] = bins == b

    results = {
        "config": {
            "n_real_tests": int(len(y_real)),
            "alpha_safety": ALPHA_SAFETY,
            "unit_harmonization_guard": unit_report,
            "severity_bin_edges_sigma": {"mild": 1.0, "moderate": 3.0},
            "severity_bin_counts": {b: int((bins == b).sum()) for b in SEVERITY_BINS},
            "out_of_range_feature_counts": {k: int(v) for k, v in out_counts.items()},
            "per_beam_severity": [
                {"index": int(i), "severity_sigma": round(float(s), 2),
                 "worst_feature": str(wf), "measured_FR": float(y)}
                for i, (s, wf, y) in enumerate(zip(severity, worst_feature, y_real))
            ],
        }
    }

    clipped_preds_cache = {}
    for name, predict in all_predictors.items():
        y_raw = predict(X_real.values.astype(float))
        y_clip = make_clipped(predict, splits.X_train)(X_real.values.astype(float))
        clipped_preds_cache[name] = y_clip
        results[name] = {}
        for mode, y_pred in [("raw", y_raw), ("clipped", y_clip)]:
            results[name][mode] = {
                sub: compute_metrics(y_real[mask], y_pred[mask])
                for sub, mask in subgroups.items() if mask.sum() >= 3
            }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved experimental-validation metrics to {RESULTS_PATH}")

    # --- Figure: headline three, clipped predictions ---
    plot_parity(y_real, {n: clipped_preds_cache[n] for n in predictors},
                 bins == "severe", FIG_PATH)


    for mode in ["raw", "clipped"]:
        print(f"\n================ {mode.upper()} predictions ================")
        for sub, mask in subgroups.items():
            if mask.sum() < 3:
                continue
            print(f"\n--- {sub} (n={int(mask.sum())}) ---")
            print(f"{'Model':<18}{'RMSE':>8}{'MAE':>8}{'Bias':>9}{'R2':>9}"
                  f"{'#Over':>7}{'MaxOver':>9}{'SafeLoss':>10}")
            for name in all_predictors:
                m = results[name][mode].get(sub)
                if m is None:
                    continue
                r2 = f"{m['r2']:.3f}" if m["r2"] is not None else "  n/a"
                print(f"{name:<18}{m['rmse']:>8.2f}{m['mae']:>8.2f}"
                      f"{m['mean_error_min']:>+9.2f}{r2:>9}"
                      f"{m['n_over_predictions']:>7d}{m['max_over_pred_min']:>9.2f}"
                      f"{m['safety_loss_min2']:>10.1f}")


if __name__ == "__main__":
    main()
