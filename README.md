[![DOI](https://zenodo.org/badge/1337349688.svg)](https://doi.org/10.5281/zenodo.21982860)
Codes and scripts that have been used in the research are available here in this public repository.

1. Provenance

Both datasets are derived from the thermo-mechanical finite-element simulation campaign and furnace-test compilation published in:

P. P. Bhatt, V. K. R. Kodur, M. Z. Naser, "Dataset on fire resistance analysis of FRP-strengthened concrete beams," Data in Brief, 52 (2024) 110031. https://doi.org/10.1016/j.dib.2024.110031

If you use this code, please cite both the original Bhatt, Kodur & Naser dataset above and this paper. See CITATION.cff for the preferred citation format for the code/paper; the original dataset should be cited separately per its own terms.

This repository does not redistribute the raw source workbook or any processed/split version of the data. It provides only the scripts that clean, harmonize, and split the data from the original source, so that the full provenance chain from raw source to reported result is traceable and reproducible by anyone who obtains the source data independently from the DOI above.

2. Files

The original data obtained from Bhatt, Kodur & Naser (2024) are not available here. Please visit https://doi.org/10.1016/j.dib.2024.110031 to obtain those datasets. Availability of that source data is required for the scripts below to run.

All scripts sit at the repository root (no subfolder) and write their outputs to results/. Each script is self-contained and can be re-run independently once its required inputs exist.

preprocessing.py — splits the raw source data into individual features, removes unparseable entries, and creates the OOD / in-distribution train/val/test splits.
phase2.py — trains four non-physics-informed baseline models (XGBoost, LightGBM, random forest, MLP) on the in-distribution split from preprocessing.py, and evaluates all of them on the OOD extrapolation set. → results/phase2_baseline_metrics.json
phase3.py — PGNN architecture and custom loss.
Architecture: a 3–4 layer MLP with a shared trunk and two output heads — FR_hat (primary target, fire resistance in minutes) and Capacity_hat (auxiliary target, residual flexural capacity at fire failure, kN·m), needed only so the final-capacity-≤-initial-capacity penalty has something to constrain. Initial capacity itself is not predicted — it's a known ambient-condition value already present in the dataset, used as the ground-truth upper bound.
Hidden layers use SiLU (Swish) by default, not ReLU, because ReLU's zero second derivative (and zero gradient in the dead region) starves the physics-loss gradients that autograd needs for the monotonicity penalties. Softplus is offered as an alternative.
Loss:
    L_total = L_data
              + ramp * [  lambda_insulation * L_insulation   (Penalty 1.1: tins, hi)
                        + lambda_kins       * L_kins          (Penalty 1.2: kins)
                        + lambda_load       * L_load          (Penalty 1.3: LR, Ld)
                        + lambda_tg         * L_tg            (Penalty 1.4: Tg)
                        + lambda_bound      * L_bound         (Penalty 2.1: FR >= 0)
                        + lambda_capacity   * L_capacity      (Penalty 2.2)
                        + lambda_maxload    * L_maxload ]     (Penalty 2.3, collocation)
              + lambda_capacity_fit * L_capacity_fit          (auxiliary head data fit)

→ results/phase3_pgnn_metrics.json

phase3_seeds.py — reruns Phase 3 across the 2×2 ablation (physics on/off × symmetric/asymmetric loss) for N random seeds (default 5), reporting mean ± std for every metric on every split. → results/phase3_seed_metrics.json
phase3_alpha_sweep.py — replaces the single, otherwise-unjustified choice α = 0.8 with a measured frontier: for each asymmetry level, what does the conservatism actually cost in accuracy. → results/alpha_sweep_raw.json, results/alpha_sweep_raw.csv, results/alpha_sweep_summary.csv
phase4.py — Physics Violation Rate (PVR) and monotonicity response-curve analysis; also draws the paper's figures.
PVR: for each of the six sign-constrained features, the % of evaluation beams where perturbing that feature moves predicted FR in the physically impossible direction (tins, hi, Tg up ⇒ FR must not decrease; kins, LR, Ld up ⇒ FR must not increase). Measured by finite perturbation (Δ = 0.25σ of the feature), not autograd, so the identical procedure applies to tree ensembles and neural nets — a gradient-based PVR would be unfair to XGBoost (piecewise-constant, zero gradient almost everywhere) and incomparable across model classes. A tolerance band (|ΔFR| ≤ 0.5 min) keeps numerical noise from counting as a violation.
Monotonicity response curves: predicted FR for a median beam as one feature sweeps its range (tins; and LR extended to 150%, past the dataset's ~73.5% maximum, to expose extrapolation behaviour and the PGNN's collocation-trained collapse toward FR ≈ 0 at LR ≥ 100%).

phase5.py — blind experimental validation: evaluates the trained models against the 50 usable real furnace tests. These records were never used in training, scaling, model selection, or hyperparameter tuning at any phase.
3. License and usage
Code in this repository: MIT License.
Documentation (this file and related .md files): CC BY 4.0.
The underlying dataset is derived from Bhatt, Kodur & Naser (2024), Data in Brief — consult that publication's own license terms before redistributing any raw or processed values beyond this repository.
4. Contact

Questions about the data specifically (as opposed to the modelling code) can be directed to the corresponding author.
