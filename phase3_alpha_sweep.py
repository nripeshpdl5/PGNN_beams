
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import (
    RANDOM_STATE, load_numerical_dataset, split_dataset, fit_scalers,
)
from phase3 import (
    CAPACITY_COLUMNS, CONTROL_WEIGHTS, LossWeights,
    train_pgnn, evaluate_pgnn,
)

OUT_DIR = Path("results/alpha_sweep")
RAW_JSON = OUT_DIR / "alpha_sweep_raw.json"
PUBLISHED_JSON = Path("results/phase3_seed_metrics.json")


EVAL_ALPHA = 0.8

DEFAULT_ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
DEFAULT_SEEDS = [RANDOM_STATE, 7, 123, 2024, 31415]

METRIC_KEYS = [
    "rmse", "mae", "r2", "mean_error_min", "rmse_bias_removed",
    "pct_over_predictions", "mean_over_pred_min", "max_over_pred_min",
    "safety_loss_min2",
]


PUBLISHED_CELLS = {
    (0.5, "off"): "control_no_physics",
    (0.5, "on"): "physics_only",
    (0.8, "off"): "asymmetric_only",
    (0.8, "on"): "pgnn",
}


def weights_for(alpha: float, physics: bool) -> LossWeights:

    base = LossWeights() if physics else CONTROL_WEIGHTS
    return replace(base, alpha_safety=alpha)


def cell_key(alpha: float, seed: int, physics: bool) -> str:
    return f"alpha={alpha:g}|seed={seed}|physics={'on' if physics else 'off'}"




def run_sweep(alphas, seeds, physics_modes, max_epochs, verbose):
    df = load_numerical_dataset(extra_columns=CAPACITY_COLUMNS)
    splits = split_dataset(df)          # split fixed at RANDOM_STATE, as in phase3_seeds
    scalers = fit_scalers(splits.X_train, splits.y_train)

    capacity_splits = {
        name: pd.DataFrame({
            "Initial Capacity": splits.get_aligned_column("Initial Capacity")[name],
            "Final Capacity": splits.get_aligned_column("Final Capacity")[name],
        })
        for name in ["train", "val", "test", "ood"]
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(RAW_JSON.read_text()) if RAW_JSON.exists() else {}
    if cache:
        print(f"[resume] {len(cache)} cell(s) already complete — skipping those.\n")

    total_cells = len(alphas) * len(seeds) * len(physics_modes)
    done = 0
    t_sweep = time.time()

    for physics in physics_modes:
        for alpha in alphas:
            for seed in seeds:
                done += 1
                key = cell_key(alpha, seed, physics)
                if key in cache:
                    print(f"[{done}/{total_cells}] {key} — cached")
                    continue

                print(f"[{done}/{total_cells}] {key} "
                      f"(ratio {alpha / (1 - alpha):.1f}:1) ...", flush=True)
                t0 = time.time()
                model = train_pgnn(
                    splits, scalers, capacity_splits,
                    weights_for(alpha, physics),
                    max_epochs=max_epochs, verbose=verbose, seed=seed,
                )
                elapsed = time.time() - t0

                cell = {
                    "alpha": alpha,
                    "penalty_ratio": alpha / (1 - alpha),
                    "seed": seed,
                    "physics": "on" if physics else "off",
                    "seconds": round(elapsed, 1),
                    "splits": {
                        name: evaluate_pgnn(model, X, y, scalers, alpha_safety=EVAL_ALPHA)
                        for name, X, y in [("val", splits.X_val, splits.y_val),
                                           ("test", splits.X_test, splits.y_test),
                                           ("ood", splits.X_ood, splits.y_ood)]
                    },
                }
                cache[key] = cell
                # Persist after every cell so an interrupted sweep loses at
                # most one training run.
                RAW_JSON.write_text(json.dumps(cache, indent=2))

                t = cell["splits"]["test"]
                print(f"    done in {elapsed:.0f}s | test R2={t['r2']:.4f} "
                      f"| over={t['pct_over_predictions']:.1f}% "
                      f"| Ls={t['safety_loss_min2']:.1f}")

    print(f"\nSweep finished in {(time.time() - t_sweep) / 60:.1f} min.")
    return cache



def tabulate(cache):
    rows = []
    for cell in cache.values():
        for split_name, m in cell["splits"].items():
            rows.append({
                "alpha": cell["alpha"], "penalty_ratio": cell["penalty_ratio"],
                "seed": cell["seed"], "physics": cell["physics"],
                "split": split_name, **{k: m[k] for k in METRIC_KEYS},
            })
    runs = pd.DataFrame(rows).sort_values(["physics", "alpha", "split", "seed"])
    summary = (runs.groupby(["physics", "alpha", "penalty_ratio", "split"])[METRIC_KEYS]
               .agg(["mean", "std"]).round(4).reset_index())
    return runs, summary


def verify_against_published(runs: pd.DataFrame, tol_pct=2.0):
    """The four published ablation cells are a subset of this grid, so they
    must agree. Any drift means the pipeline moved since the tables were
    generated."""
    if not PUBLISHED_JSON.exists():
        print(f"[verify] {PUBLISHED_JSON} not found — skipping.")
        return
    pub = json.loads(PUBLISHED_JSON.read_text())
    print(f"\n=== Verification against {PUBLISHED_JSON} (tolerance {tol_pct}%) ===")
    any_checked = False
    for (alpha, phys), variant in PUBLISHED_CELLS.items():
        d = runs[(runs.alpha == alpha) & (runs.physics == phys) & (runs.split == "test")]
        if d.empty or variant not in pub:
            continue
        any_checked = True
        for key in ["r2", "safety_loss_min2", "pct_over_predictions"]:
            mine = float(d[key].mean())
            theirs = float(pub[variant]["test"][key]["mean"])
            denom = abs(theirs) if abs(theirs) > 1e-9 else 1.0
            drift = abs(mine - theirs) / denom * 100.0
            flag = "OK " if drift <= tol_pct else "DRIFT"
            print(f"  [{flag}] {variant:20s} {key:22s} "
                  f"sweep={mine:9.4f}  published={theirs:9.4f}  ({drift:.2f}%)")
    if not any_checked:
        print("  (no overlapping cells in this grid — run with alpha 0.5 and 0.8)")


def make_plots(runs: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots skipped] matplotlib not installed.")
        return

    agg = (runs.groupby(["physics", "alpha", "split"])
           .agg(r2_m=("r2", "mean"), r2_s=("r2", "std"),
                over_m=("pct_over_predictions", "mean"), over_s=("pct_over_predictions", "std"),
                mo_m=("max_over_pred_min", "mean"), mo_s=("max_over_pred_min", "std"),
                ls_m=("safety_loss_min2", "mean"), ls_s=("safety_loss_min2", "std"))
           .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for phys in sorted(agg.physics.unique()):
        for split in ["test", "ood"]:
            d = agg[(agg.physics == phys) & (agg.split == split)].sort_values("alpha")
            if d.empty:
                continue
            lab = f"{split}, physics {phys}"
            axes[0].errorbar(d.r2_m, d.over_m, xerr=d.r2_s, yerr=d.over_s,
                             marker="o", capsize=3, label=lab)
            for _, r in d.iterrows():
                axes[0].annotate(f"{r.alpha:g}", (r.r2_m, r.over_m),
                                 textcoords="offset points", xytext=(5, 5), fontsize=8)
            axes[1].errorbar(d.r2_m, d.mo_m, xerr=d.r2_s, yerr=d.mo_s,
                             marker="s", capsize=3, label=lab)
    axes[0].set_xlabel("$R^2$"); axes[0].set_ylabel("over-prediction rate (%)")
    axes[0].set_title(r"Conservatism frontier (point labels = $\alpha$)")
    axes[1].set_xlabel("$R^2$"); axes[1].set_ylabel("worst-case over-prediction (min)")
    axes[1].set_title("Worst-case unsafe error vs accuracy")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_alpha_frontier.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for phys in sorted(agg.physics.unique()):
        d = agg[(agg.physics == phys) & (agg.split == "ood")].sort_values("alpha")
        if d.empty:
            continue
        axes[0].plot(d.alpha, d.ls_s, marker="o", label=f"physics {phys}")
        axes[1].plot(d.alpha, d.mo_s, marker="o", label=f"physics {phys}")
    axes[0].set_xlabel(r"$\alpha$"); axes[0].set_ylabel("std of OOD $L_s$ across seeds")
    axes[0].set_title("Seed dispersion of OOD safety loss")
    axes[1].set_xlabel(r"$\alpha$"); axes[1].set_ylabel("std of OOD MaxOver across seeds")
    axes[1].set_title("Seed dispersion of worst-case over-prediction")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig_alpha_dispersion.png", dpi=200)
    plt.close(fig)
    print(f"Plots written to {OUT_DIR}")



def main():
    p = argparse.ArgumentParser(description="PGNN asymmetry (alpha) sweep.")
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--physics", choices=["on", "off", "both"], default="on")
    p.add_argument("--max-epochs", type=int, default=500,
                   help="per-run epoch cap (default 500, matching phase3.py)")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: alphas 0.5/0.8, 1 seed, 25 epochs")
    p.add_argument("--verify", action="store_true",
                   help="cross-check overlapping cells against phase3_seed_metrics.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.alphas, args.seeds, args.max_epochs = [0.5, 0.8], [RANDOM_STATE], 25

    if any(not 0.0 < a < 1.0 for a in args.alphas):
        raise SystemExit("alpha must lie strictly in (0, 1); 0.5 is symmetric, "
                         ">0.5 penalises over-prediction.")

    physics_modes = {"on": [True], "off": [False], "both": [True, False]}[args.physics]
    n = len(args.alphas) * len(args.seeds) * len(physics_modes)
    print(f"Grid: {len(args.alphas)} alpha x {len(args.seeds)} seed(s) "
          f"x {len(physics_modes)} physics mode(s) = {n} training runs")
    print(f"  alphas : {args.alphas}")
    print(f"  ratios : {[round(a / (1 - a), 1) for a in args.alphas]}")
    print(f"  seeds  : {args.seeds}")
    print(f"  eval alpha (fixed for all cells): {EVAL_ALPHA}\n")

    try:
        cache = run_sweep(args.alphas, args.seeds, physics_modes,
                          args.max_epochs, args.verbose)
    except ImportError as e:
        raise SystemExit(f"[ABORTED] needs torch: {e}")

    runs, summary = tabulate(cache)
    runs.to_csv(OUT_DIR / "alpha_sweep_runs.csv", index=False)
    summary.to_csv(OUT_DIR / "alpha_sweep_summary.csv", index=False)

    for split in ["test", "ood"]:
        print(f"\n=== {split.upper()} split, mean over seeds (std shown for dispersion) ===")
        d = runs[runs.split == split].groupby(["physics", "alpha"]).agg(
            r2=("r2", "mean"),
            rmse=("rmse", "mean"),
            bias=("mean_error_min", "mean"),
            over_pct=("pct_over_predictions", "mean"),
            max_over=("max_over_pred_min", "mean"),
            max_over_std=("max_over_pred_min", "std"),
            Ls=("safety_loss_min2", "mean"),
            Ls_std=("safety_loss_min2", "std"),
        ).round(3)
        print(d.to_string())

    print("\nRead the OOD Ls_std / max_over_std columns against alpha: if they"
          "\nfall monotonically at fixed physics, the seed-dispersion collapse"
          "\nreported in Section 3.5 is driven by the asymmetric objective"
          "\nrather than by the physics constraints, and that section needs"
          "\nrewording (see this file's docstring).")

    if args.verify:
        verify_against_published(runs)

    make_plots(runs)
    print(f"\nWrote:\n  {OUT_DIR / 'alpha_sweep_runs.csv'}"
          f"\n  {OUT_DIR / 'alpha_sweep_summary.csv'}\n  {RAW_JSON}")


if __name__ == "__main__":
    main()
