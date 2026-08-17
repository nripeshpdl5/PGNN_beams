
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _resolve_data_path() -> Path:
    """Resolve the workbook path regardless of the current working directory."""
    candidates = [
        Path(__file__).resolve().parent / "dataset.xlsx",
        Path.cwd() / "dataset.xlsx",
        Path(__file__).resolve().parent.parent / "dataset.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find dataset.xlsx. Looked in: "
        + ", ".join(str(path) for path in candidates)
    )


DATA_PATH = _resolve_data_path()

SHEET_NUMERICAL = "02_NumericaModelData"
SHEET_FIRE_TEST = "01_FireTestData"

GEOMETRY_FEATURES = ["L", "Ac", "Cc", "As", "Af"]
INSULATION_FEATURES = ["tins", "hi", "kins", "rinscins"]
MATERIAL_FEATURES = ["fc", "fy", "Es", "fu", "Efrp", "Tg"]
LOADING_FEATURES = ["Ld", "LR"]

FEATURE_COLUMNS = (
    GEOMETRY_FEATURES + INSULATION_FEATURES + MATERIAL_FEATURES + LOADING_FEATURES
)

TARGET_COLUMN = "FR"

USE_FIXED_THRESHOLDS = False
OOD_PERCENTILE = 0.95
OOD_TINS_THRESHOLD = 80.0
OOD_LR_THRESHOLD = 80.0

RANDOM_STATE = 42


def _load_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Load a sheet and strip the units row (row 0 after the header),
    which contains strings like '(mm)', '(MPa)', '(kN)' instead of data."""
    df = pd.read_excel(path, sheet_name=sheet_name)

    first_row = df.iloc[0]
    looks_like_units_row = first_row.astype(str).str.contains(r"^\(.*\)$", regex=True).any()
    if looks_like_units_row:
        df = df.iloc[1:].reset_index(drop=True)

    return df


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_numerical_dataset(path: Path = DATA_PATH, extra_columns: list[str] | None = None) -> pd.DataFrame:

    df = _load_sheet(path, SHEET_NUMERICAL)
    extra_columns = extra_columns or []
    needed = FEATURE_COLUMNS + [TARGET_COLUMN] + extra_columns
    df = _coerce_numeric(df, needed)

    n_before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        print(f"[load_numerical_dataset] Dropped {n_before - n_after} rows "
              f"with missing/unparseable values in feature/target/extra columns.")

    return df


def load_fire_test_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the 51-row real experimental fire-test dataset,
    reserved for Phase 5 validation (never used in training)."""
    df = _load_sheet(path, SHEET_FIRE_TEST)
    needed = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = _coerce_numeric(df, needed)
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    X_ood: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    y_ood: pd.Series
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    ood_idx: np.ndarray
    _source_df: pd.DataFrame

    def get_aligned_column(self, column_name: str) -> dict[str, pd.Series]:
       
        col = self._source_df[column_name]
        return {
            "train": col.iloc[self.train_idx].reset_index(drop=True),
            "val": col.iloc[self.val_idx].reset_index(drop=True),
            "test": col.iloc[self.test_idx].reset_index(drop=True),
            "ood": col.iloc[self.ood_idx].reset_index(drop=True),
        }


def split_dataset(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> DataSplits:
  
    if USE_FIXED_THRESHOLDS:
        tins_cutoff = OOD_TINS_THRESHOLD
        lr_cutoff = OOD_LR_THRESHOLD
    else:
        tins_cutoff = df["tins"].quantile(OOD_PERCENTILE)
        lr_cutoff = df["LR"].quantile(OOD_PERCENTILE)
        print(f"[split_dataset] Data-driven OOD cutoffs (top {(1-OOD_PERCENTILE)*100:.0f}%): "
              f"tins > {tins_cutoff:.2f} mm, LR > {lr_cutoff:.2f} %")

    ood_mask = (df["tins"] > tins_cutoff) | (df["LR"] > lr_cutoff)
    id_positions = np.where(~ood_mask.values)[0]
    ood_idx = np.where(ood_mask.values)[0]

    print(f"[split_dataset] In-distribution rows: {len(id_positions)} | "
          f"OOD extrapolation rows held out: {len(ood_idx)}")

    train_idx, temp_idx = train_test_split(
        id_positions, test_size=0.30, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=random_state
    )

    X_train, y_train = df.iloc[train_idx][FEATURE_COLUMNS].reset_index(drop=True), \
        df.iloc[train_idx][TARGET_COLUMN].reset_index(drop=True)
    X_val, y_val = df.iloc[val_idx][FEATURE_COLUMNS].reset_index(drop=True), \
        df.iloc[val_idx][TARGET_COLUMN].reset_index(drop=True)
    X_test, y_test = df.iloc[test_idx][FEATURE_COLUMNS].reset_index(drop=True), \
        df.iloc[test_idx][TARGET_COLUMN].reset_index(drop=True)
    X_ood, y_ood = df.iloc[ood_idx][FEATURE_COLUMNS].reset_index(drop=True), \
        df.iloc[ood_idx][TARGET_COLUMN].reset_index(drop=True)

    return DataSplits(
        X_train=X_train, X_val=X_val, X_test=X_test, X_ood=X_ood,
        y_train=y_train, y_val=y_val, y_test=y_test, y_ood=y_ood,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, ood_idx=ood_idx,
        _source_df=df,
    )


@dataclass
class Scalers:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    feature_names: list[str]

    def unscale_gradient(self, grad_scaled: np.ndarray, feature_name: str) -> np.ndarray:
       
        idx = self.feature_names.index(feature_name)
        sigma_x = self.x_scaler.scale_[idx]
        sigma_y = self.y_scaler.scale_[0]
        return grad_scaled * (sigma_y / sigma_x)


def fit_scalers(X_train: pd.DataFrame, y_train: pd.Series) -> Scalers:

    x_scaler = StandardScaler().fit(X_train.values)
    y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))
    return Scalers(x_scaler=x_scaler, y_scaler=y_scaler, feature_names=list(X_train.columns))


def transform_with_scalers(X: pd.DataFrame, y: pd.Series, scalers: Scalers):
    X_scaled = scalers.x_scaler.transform(X.values)
    y_scaled = scalers.y_scaler.transform(y.values.reshape(-1, 1)).ravel()
    return X_scaled, y_scaled


if __name__ == "__main__":
    df_numerical = load_numerical_dataset()
    df_fire_test = load_fire_test_dataset()

    print("\nNumerical (simulation) dataset:", df_numerical.shape)
    print(df_numerical[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T[["mean", "std", "min", "max"]])

    print("\nReal fire-test dataset (held out for Phase 5):", df_fire_test.shape)

    splits = split_dataset(df_numerical)
    print(f"\nTrain: {len(splits.X_train)} | Val: {len(splits.X_val)} | "
          f"Test: {len(splits.X_test)} | OOD: {len(splits.X_ood)}")

    scalers = fit_scalers(splits.X_train, splits.y_train)
    X_train_s, y_train_s = transform_with_scalers(splits.X_train, splits.y_train, scalers)
    print(f"\nScaled X_train shape: {X_train_s.shape}, mean~0: {X_train_s.mean():.4f}, "
          f"std~1: {X_train_s.std():.4f}")

    dummy_grad_scaled = np.array([0.5])
    physical_grad = scalers.unscale_gradient(dummy_grad_scaled, "tins")
    print(f"\nExample: scaled gradient 0.5 w.r.t. tins' unscales to "
          f"{physical_grad[0]:.4f} (minutes of FR per mm of insulation)")
