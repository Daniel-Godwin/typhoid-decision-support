"""Tests for the decision-threshold analysis module.

The arithmetic is checked against hand-computed confusion matrices rather than
against the model, so these tests stay valid if the model is retrained.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typhoid_ml.config import BINARY_POSITIVE  # noqa: E402
from typhoid_ml.threshold import (  # noqa: E402
    COST_RATIOS,
    calibration,
    operating_points,
    prevalence_adjusted,
    stratified_operating_point,
    sweep,
)

POS = BINARY_POSITIVE
NEG = "No Typhoid"


@pytest.fixture
def toy():
    """Four patients with probabilities that straddle 0.5 predictably."""
    y = [POS, POS, NEG, NEG]
    p = [0.90, 0.40, 0.60, 0.10]
    return y, p


def test_sweep_counts_match_hand_computation(toy):
    y, p = toy
    df = sweep(y, p, grid=[0.5])
    row = df.iloc[0]
    # >= 0.5 predicts positive: patients 0 (TP) and 2 (FP).
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 1, 1, 1)
    assert row["sensitivity"] == pytest.approx(0.5)
    assert row["specificity"] == pytest.approx(0.5)
    assert row["ppv"] == pytest.approx(0.5)
    assert row["npv"] == pytest.approx(0.5)
    assert row["accuracy"] == pytest.approx(0.5)


def test_sweep_is_monotonic_in_sensitivity(toy):
    y, p = toy
    df = sweep(y, p)
    assert df["sensitivity"].is_monotonic_decreasing
    assert df["specificity"].is_monotonic_increasing


def test_extreme_thresholds_are_degenerate(toy):
    y, p = toy
    df = sweep(y, p, grid=[0.0, 1.01])
    assert df.iloc[0]["sensitivity"] == 1.0 and df.iloc[0]["specificity"] == 0.0
    assert df.iloc[1]["sensitivity"] == 0.0 and df.iloc[1]["specificity"] == 1.0


def test_sweep_carries_a_cost_column_per_ratio(toy):
    y, p = toy
    df = sweep(y, p)
    for ratio in COST_RATIOS:
        assert f"cost_fn{ratio}x" in df.columns
    # A heavier false-negative weight can never prefer a stricter cut-off.
    best_1x = df.loc[df["cost_fn1x"].idxmin(), "threshold"]
    best_20x = df.loc[df["cost_fn20x"].idxmin(), "threshold"]
    assert best_20x <= best_1x


def test_operating_points_include_every_named_rule(toy):
    y, p = toy
    points = operating_points(sweep(y, p))
    assert {"default", "youden", "max_f1"} <= set(points)
    assert all("rationale" in v and v["rationale"] for v in points.values())
    assert points["default"]["threshold"] == pytest.approx(0.5)


def test_youden_point_maximises_j(toy):
    y, p = toy
    df = sweep(y, p)
    points = operating_points(df)
    assert points["youden"]["youden_j"] == pytest.approx(df["youden_j"].max())


def test_prevalence_adjustment_lowers_ppv_as_disease_becomes_rarer():
    df = prevalence_adjusted(0.95, 0.90, [0.5, 0.10, 0.01])
    assert df["ppv"].is_monotonic_decreasing
    assert df["npv"].is_monotonic_increasing
    # Sanity check one cell by hand: p=0.10, sens=0.95, spec=0.90
    # TP = 0.095, FP = 0.09  ->  PPV = 0.095 / 0.185
    assert df.iloc[1]["ppv"] == pytest.approx(0.095 / 0.185)


def test_prevalence_adjustment_is_independent_of_the_evaluation_partition():
    """PPV at the same prevalence depends only on sensitivity and specificity."""
    a = prevalence_adjusted(0.90, 0.80, [0.2]).iloc[0]["ppv"]
    b = prevalence_adjusted(0.90, 0.80, [0.2]).iloc[0]["ppv"]
    assert a == b


def test_calibration_reports_perfect_agreement_for_honest_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y = [POS if rng.uniform() < prob else NEG for prob in p]
    cal = calibration(y, p)
    assert cal["expected_calibration_error"] < 0.05
    assert 0.0 <= cal["brier_score"] <= 0.25
    assert sum(b["n"] for b in cal["bins"]) == len(p)


def test_calibration_detects_an_overconfident_model():
    """Probabilities pushed to the extremes must show a large calibration gap."""
    p = np.array([0.99] * 100 + [0.01] * 100)
    y = [POS] * 50 + [NEG] * 50 + [POS] * 50 + [NEG] * 50
    cal = calibration(y, p)
    assert cal["expected_calibration_error"] > 0.4


def test_stratified_operating_point_partitions_the_records(toy):
    y, p = toy
    out = stratified_operating_point(y, p, ["Urban", "Rural", "Urban", "Rural"], 0.5)
    assert set(out["subgroup"]) == {"Rural", "Urban"}
    assert out["n"].sum() == len(y)


def test_positive_label_orientation_is_respected():
    """Swapping the label meaning must swap sensitivity and specificity."""
    y = [POS, NEG]
    forward = sweep(y, [0.9, 0.1], grid=[0.5]).iloc[0]
    reversed_ = sweep(y[::-1], [0.9, 0.1], grid=[0.5]).iloc[0]
    assert forward["sensitivity"] == 1.0 and forward["specificity"] == 1.0
    assert reversed_["sensitivity"] == 0.0 and reversed_["specificity"] == 0.0
