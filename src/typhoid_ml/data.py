"""Dataset loading, auditing, target construction and feature selection."""
from __future__ import annotations

import pandas as pd

from .config import (
    BINARY_LABELS,
    DATA_PATH,
    FEATURE_POLICIES,
    LEAKAGE_EXCLUDED,
    NEGATIVE_CLASS,
    TARGET,
)


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {path}")
    return df


def build_target(df: pd.DataFrame, target_mode: str) -> pd.Series:
    """Return the label vector for the requested target mode."""
    if target_mode == "multiclass":
        return df[TARGET].astype(str)
    if target_mode == "binary":
        positive = df[TARGET].astype(str) != NEGATIVE_CLASS
        return pd.Series(
            [BINARY_LABELS[1] if p else BINARY_LABELS[0] for p in positive],
            index=df.index,
            name="Typhoid Diagnosis",
        )
    raise ValueError(f"Unknown target_mode '{target_mode}'")


def feature_frame(df: pd.DataFrame, policy: str = "routine") -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (X, numeric_features, categorical_features) under a feature policy."""
    if policy not in FEATURE_POLICIES:
        raise ValueError(f"Unknown feature policy '{policy}'")
    numeric = list(FEATURE_POLICIES[policy]["numeric"])
    categorical = list(FEATURE_POLICIES[policy]["categorical"])
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {missing}")
    return df[numeric + categorical].copy(), numeric, categorical


def category_levels(df: pd.DataFrame, categorical: list[str]) -> dict[str, list[str]]:
    """Observed levels for each categorical feature, used to drive the UI."""
    return {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in categorical}


def optional_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Features that are absent for some patients in the source data.

    These must remain optional at the point of entry: a clinician who has not
    performed an assessment should be able to say so rather than invent a value.
    The pipeline imputes them, exactly as it did during training.
    """
    return [c for c in features if df[c].isna().any()]


def numeric_ranges(df: pd.DataFrame, numeric: list[str]) -> dict[str, dict]:
    out = {}
    for c in numeric:
        s = pd.to_numeric(df[c], errors="coerce")
        out[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "mean": float(s.mean()),
        }
    return out


def audit_dataset(df: pd.DataFrame) -> dict:
    binary = build_target(df, "binary")
    _, numeric, categorical = feature_frame(df, "routine")
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "duplicates": int(df.duplicated().sum()),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
        "missing_percentage": {
            c: round(float(v) * 100 / len(df), 2) for c, v in df.isna().sum().items() if v
        },
        "target": TARGET,
        "target_distribution_multiclass": {
            str(k): int(v) for k, v in df[TARGET].value_counts(dropna=False).items()
        },
        "target_distribution_binary": {str(k): int(v) for k, v in binary.value_counts().items()},
        "imbalance_ratio_multiclass": round(
            float(df[TARGET].value_counts().max() / df[TARGET].value_counts().min()), 2
        ),
        "imbalance_ratio_binary": round(
            float(binary.value_counts().max() / binary.value_counts().min()), 2
        ),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
        "excluded_features": LEAKAGE_EXCLUDED + [TARGET],
        "model_features": {"numeric": numeric, "categorical": categorical},
        "category_levels": category_levels(df, categorical),
        "numeric_ranges": numeric_ranges(df, numeric),
    }
