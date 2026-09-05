import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typhoid_ml.config import BINARY_LABELS, MULTICLASS_LABELS, TARGET  # noqa: E402
from typhoid_ml.data import (  # noqa: E402
    audit_dataset,
    build_target,
    category_levels,
    feature_frame,
    load_dataset,
)

DATA = ROOT / "data" / "typhoid_dataset.csv"


@pytest.fixture(scope="module")
def df():
    return load_dataset(DATA)


def test_dataset_shape(df):
    assert df.shape == (31087, 23)


def test_no_duplicates(df):
    assert df.duplicated().sum() == 0


def test_target_is_complete(df):
    assert df[TARGET].isna().sum() == 0
    assert set(df[TARGET].unique()) == set(MULTICLASS_LABELS)


def test_binary_target_collapses_correctly(df):
    y = build_target(df, "binary")
    assert set(y.unique()) == set(BINARY_LABELS)
    assert (y == "No Typhoid").sum() == (df[TARGET] == "Normal or No Typhoid").sum()
    assert (y == "Typhoid").sum() == len(df) - (df[TARGET] == "Normal or No Typhoid").sum()


def test_multiclass_target_is_passthrough(df):
    assert build_target(df, "multiclass").equals(df[TARGET].astype(str))


def test_unknown_target_mode_rejected(df):
    with pytest.raises(ValueError):
        build_target(df, "trinary")


@pytest.mark.parametrize("policy", ["routine", "clinical_only"])
def test_feature_policy_excludes_leakage_columns(df, policy):
    X, numeric, categorical = feature_frame(df, policy)
    for banned in ("Typhoid Status", "Blood Culture Result", "Complications"):
        assert banned not in X.columns
    assert list(X.columns) == numeric + categorical


def test_clinical_only_policy_drops_serology(df):
    _, _, categorical = feature_frame(df, "clinical_only")
    assert "Widal Test" not in categorical
    assert "Typhidot Test" not in categorical


def test_category_levels_are_non_empty(df):
    _, _, categorical = feature_frame(df, "routine")
    levels = category_levels(df, categorical)
    assert set(levels) == set(categorical)
    assert all(len(v) >= 2 for v in levels.values())


def test_audit_reports_both_target_views(df):
    audit = audit_dataset(df)
    assert audit["rows"] == 31087
    assert sum(audit["target_distribution_multiclass"].values()) == 31087
    assert sum(audit["target_distribution_binary"].values()) == 31087
    assert audit["imbalance_ratio_binary"] > 1
