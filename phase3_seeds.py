
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import RANDOM_STATE
from phase3 import (
    CAPACITY_COLUMNS, CONTROL_WEIGHTS, LossWeights,
    evaluate_pgnn, train_pgnn,
)
from preprocessing import load_numerical_dataset, split_dataset, fit_scalers

RESULTS_PATH = Path("results/phase3_seed_metrics.json")


DEFAULT_SEEDS = [RANDOM_STATE, 7, 123, 2024, 31415]

METRIC_KEYS = [
    "rmse", "mae", "r2", "mean_error_min", "rmse_bias_removed",
    "pct_over_predictions", "mean_over_pred_min", "max_over_pred_min",
    "safety_loss_min2",
]
SPLIT_NAMES = ["val", "test", "ood"]


def build_variants() -> dict[str, LossWeights]:
    """Same 2x2 as phase3.main() — keep in sync if that changes."""
    full = LossWeights()
    return {
        "control_no_physics": CONTROL_WEIGHTS,
        "physics_only": replace(full, alpha_safety=0.5),
        "asymmetric_only": replace(CONTROL_WEIGHTS, alpha_safety=full.alpha_safety),
        "pgnn": full,
    }


def aggregate(per_seed: list[dict]) -> dict:
    """[{metric: value} per seed] -> {metric: {mean, std, values}}.
    std is ddof=1 (sample std) since seeds are a sample of possible inits."""
    out = {}
    for key in METRIC_KEYS:
        values = [m[key] for m in per_seed]
        out[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": [float(v) for v in values],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--seeds", type=int, default=5, choices=range(2, len(DEFAULT_SEEDS) + 1),
                        help="number of seeds to run (2-5, default 5)")
    parser.add_argument("--max-epochs", type=int, default=500,
                        help="per-run epoch cap (default 500, matching phase3.py)")
    args = parser.parse_args()
    seeds = DEFAULT_SEEDS[: args.seeds]

    df = load_numerical_dataset(extra_columns=CAPACITY_COLUMNS)
    splits = split_dataset(df)          # split fixed at RANDOM_STATE — see docstring
    scalers = fit_scalers(splits.X_train, splits.y_train)

    capacity_splits = {
        split_name: pd.DataFrame({
            "Initial Capacity": splits.get_aligned_column("Initial Capacity")[split_name],
            "Final Capacity": splits.get_aligned_column("Final Capacity")[split_name],
        })
        for split_name in ["train", "val", "test", "ood"]
    }

    variants = build_variants()
    n_runs = len(variants) * len(seeds)
    print(f"Running {len(variants)} variants x {len(seeds)} seeds = {n_runs} trainings "
          f"(seeds: {seeds})\n")

    # raw[variant][split] = list of metric dicts, one per seed
    raw = {tag: {s: [] for s in SPLIT_NAMES} for tag in variants}

    run_i = 0
    t_start = time.time()
    for seed in seeds:
        for tag, weights in variants.items():
            run_i += 1
            t0 = time.time()
            print(f"[{run_i}/{n_runs}] {tag} | seed={seed} ...", flush=True)
            model = train_pgnn(splits, scalers, capacity_splits, weights,
                                max_epochs=args.max_epochs, verbose=False, seed=seed)
            for split_name, X, y in [
                ("val", splits.X_val, splits.y_val),
                ("test", splits.X_test, splits.y_test),
                ("ood", splits.X_ood, splits.y_ood),
            ]:
                raw[tag][split_name].append(evaluate_pgnn(model, X, y, scalers))
            m = raw[tag]["test"][-1]
            print(f"    done in {time.time() - t0:.0f}s | test R2={m['r2']:.4f} | "
                  f"RMSE={m['rmse']:.2f} | safety-loss={m['safety_loss_min2']:.1f}")


    results = {
        tag: {s: aggregate(raw[tag][s]) for s in SPLIT_NAMES}
        for tag in variants
    }
    results["config"] = {
        "seeds": seeds,
        "max_epochs": args.max_epochs,
        "split_random_state": RANDOM_STATE,
        "total_wallclock_sec": round(time.time() - t_start, 1),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved aggregated metrics to {RESULTS_PATH}")


    for split_name in ["test", "ood"]:
        print(f"\n=== Multi-seed ablation summary ({split_name.upper()}, "
              f"n={len(seeds)} seeds, mean +/- std) ===")
        print(f"{'Variant':<22}{'RMSE':>16}{'R2':>18}{'Bias':>16}"
              f"{'Over%':>14}{'MaxOver':>16}{'SafeLoss':>18}")
        for tag in variants:
            m = results[tag][split_name]
            def fmt(key, prec):
                return f"{m[key]['mean']:.{prec}f} ± {m[key]['std']:.{prec}f}"
            print(f"{tag:<22}{fmt('rmse', 2):>16}{fmt('r2', 4):>18}"
                  f"{fmt('mean_error_min', 2):>16}{fmt('pct_over_predictions', 1):>14}"
                  f"{fmt('max_over_pred_min', 1):>16}{fmt('safety_loss_min2', 1):>18}")


    ctrl = results["control_no_physics"]["test"]["r2"]["values"]
    phys = results["physics_only"]["test"]["r2"]["values"]
    wins = sum(p >= c for p, c in zip(phys, ctrl))
    print(f"\n'Physics is free' check (test R2, paired by seed): "
          f"physics_only >= control in {wins}/{len(seeds)} seeds")
    if wins == len(seeds):
        print("  -> claim upheld: physics constraints cost nothing (or help) on every seed.")
    elif wins >= len(seeds) - 1:
        print("  -> largely upheld; report mean ± std and note the exception.")
    else:
        print("  -> soften the claim: report 'physics costs nothing within noise' "
              "(difference within ± 1 std), not 'physics improves accuracy'.")


if __name__ == "__main__":
    main()
