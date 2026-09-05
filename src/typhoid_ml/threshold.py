"""Decision-threshold analysis for the binary diagnostic model.

Accuracy at the default 0.5 cut-off is the wrong summary for a triage tool. A
screening instrument used before confirmatory testing should be tuned so that
the cost of a missed case is weighted against the cost of an unnecessary
referral, and the resulting operating point should be stated explicitly rather
than inherited from a library default.

This module answers four questions:

1.  How do sensitivity, specificity, PPV and NPV move as the cut-off moves?
2.  Which cut-off should the deployed system use, and under what assumption
    about the relative cost of a false negative?
3.  Are the reported predictive values transportable to a setting whose
    typhoid prevalence differs from that of the evaluation partition?
4.  Are the calibrated probabilities trustworthy enough to threshold at all?

Nothing here retrains or modifies a model. It consumes predicted probabilities
from an already-fitted estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import BINARY_POSITIVE

# Cost of one missed typhoid case expressed as a multiple of the cost of one
# unnecessary confirmatory test. 1 treats the two as equivalent (the implicit
# assumption behind plain accuracy); the larger values reflect that untreated
# typhoid carries a case-fatality risk while an unnecessary Widal or culture
# carries a modest financial and logistical cost.
COST_RATIOS = (1, 5, 10, 20)

# Minimum sensitivity targets used to derive rule-out operating points.
SENSITIVITY_TARGETS = (0.90, 0.95, 0.99)


def positive_probabilities(model, X) -> np.ndarray:
    """Probability of the positive (typhoid) class, in the model's own class order."""
    classes = list(model.classes_)
    if BINARY_POSITIVE not in classes:
        raise ValueError(
            f"Positive label '{BINARY_POSITIVE}' not among model classes {classes}"
        )
    return model.predict_proba(X)[:, classes.index(BINARY_POSITIVE)]


def sweep(y_true, proba, grid=None) -> pd.DataFrame:
    """Confusion counts and derived rates at every candidate cut-off."""
    y = np.asarray([1 if v == BINARY_POSITIVE else 0 for v in y_true])
    p = np.asarray(proba, dtype=float)
    if grid is None:
        grid = np.round(np.arange(0.01, 1.00, 0.01), 2)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        sens = tp / n_pos if n_pos else 0.0
        spec = tn / n_neg if n_neg else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        npv = tn / (tn + fn) if (tn + fn) else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "sensitivity": sens,
                "specificity": spec,
                "ppv": ppv,
                "npv": npv,
                "f1_positive": f1,
                "accuracy": (tp + tn) / len(y),
                "balanced_accuracy": (sens + spec) / 2,
                "youden_j": sens + spec - 1,
                "referral_rate": (tp + fp) / len(y),
            }
        )
    df = pd.DataFrame(rows)
    for ratio in COST_RATIOS:
        # Expected cost per 100 patients assessed, with a false negative
        # counted as `ratio` times as damaging as a false positive.
        df[f"cost_fn{ratio}x"] = (df["fn"] * ratio + df["fp"]) * 100 / len(y)
    return df


def prevalence_adjusted(sensitivity: float, specificity: float, prevalences) -> pd.DataFrame:
    """PPV and NPV at prevalences other than that of the evaluation partition.

    Sensitivity and specificity are properties of the classifier; PPV and NPV
    are not — they depend on how common the disease is where the tool is
    deployed. A model validated on a near-balanced research dataset will report
    a far lower PPV in a clinic where most febrile patients do not have typhoid.
    """
    rows = []
    for pi in prevalences:
        tp = sensitivity * pi
        fn = (1 - sensitivity) * pi
        tn = specificity * (1 - pi)
        fp = (1 - specificity) * (1 - pi)
        rows.append(
            {
                "prevalence": pi,
                "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
                "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
                "positives_per_1000": round((tp + fp) * 1000, 1),
                "missed_per_1000": round(fn * 1000, 1),
            }
        )
    return pd.DataFrame(rows)


def operating_points(df: pd.DataFrame) -> dict:
    """Named candidate cut-offs, each with the rationale that selects it."""
    points = {}

    def record(key, row, rationale):
        points[key] = {
            "threshold": float(row["threshold"]),
            "sensitivity": float(row["sensitivity"]),
            "specificity": float(row["specificity"]),
            "ppv": float(row["ppv"]) if pd.notna(row["ppv"]) else None,
            "npv": float(row["npv"]) if pd.notna(row["npv"]) else None,
            "accuracy": float(row["accuracy"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "youden_j": float(row["youden_j"]),
            "false_negatives": int(row["fn"]),
            "false_positives": int(row["fp"]),
            "referral_rate": float(row["referral_rate"]),
            "rationale": rationale,
        }

    default = df.iloc[(df["threshold"] - 0.5).abs().argmin()]
    record("default", default, "Library default; treats the two error types as equal.")

    record(
        "youden",
        df.loc[df["youden_j"].idxmax()],
        "Maximises Youden's J (sensitivity + specificity - 1); the standard "
        "choice when the two error types are weighted equally.",
    )
    record(
        "max_f1",
        df.loc[df["f1_positive"].idxmax()],
        "Maximises F1 on the positive class; favours precision-recall balance "
        "under class imbalance.",
    )

    for target in SENSITIVITY_TARGETS:
        eligible = df[df["sensitivity"] >= target]
        if eligible.empty:
            continue
        # Among cut-offs meeting the sensitivity floor, take the most specific.
        row = eligible.loc[eligible["specificity"].idxmax()]
        record(
            f"sens_{int(target * 100)}",
            row,
            f"Highest specificity among cut-offs achieving at least "
            f"{target:.0%} sensitivity; a rule-out operating point.",
        )

    for ratio in COST_RATIOS:
        col = f"cost_fn{ratio}x"
        row = df.loc[df[col].idxmin()]
        record(
            f"cost_fn{ratio}x",
            row,
            f"Minimises expected cost when a missed case is treated as "
            f"{ratio}x as costly as an unnecessary confirmatory test.",
        )

    return points


def calibration(y_true, proba, bins: int = 10) -> dict:
    """Reliability of the calibrated probabilities.

    A threshold is only meaningful if the probability it is applied to means
    what it says. Brier score and expected calibration error quantify that.
    """
    y = np.asarray([1 if v == BINARY_POSITIVE else 0 for v in y_true])
    p = np.asarray(proba, dtype=float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if not n:
            continue
        predicted = float(p[mask].mean())
        observed = float(y[mask].mean())
        ece += (n / len(y)) * abs(predicted - observed)
        table.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": n,
                "mean_predicted": round(predicted, 4),
                "observed_frequency": round(observed, 4),
                "gap": round(observed - predicted, 4),
            }
        )
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": round(float(ece), 4),
        "roc_auc": float(roc_auc_score(y, p)),
        "bins": table,
    }


def stratified_operating_point(y_true, proba, strata, threshold: float) -> pd.DataFrame:
    """Per-subgroup performance at one chosen cut-off (Objective 4).

    A single global threshold is only defensible if it behaves comparably
    across the settings in which the tool will be used.
    """
    y = np.asarray([1 if v == BINARY_POSITIVE else 0 for v in y_true])
    p = np.asarray(proba, dtype=float)
    s = pd.Series(list(strata)).astype(str).reset_index(drop=True)

    rows = []
    for name, mask in s.groupby(s).groups.items():
        m = np.zeros(len(y), dtype=bool)
        m[list(mask)] = True
        yy, pp = y[m], (p[m] >= threshold).astype(int)
        tp = int(((pp == 1) & (yy == 1)).sum())
        fp = int(((pp == 1) & (yy == 0)).sum())
        fn = int(((pp == 0) & (yy == 1)).sum())
        tn = int(((pp == 0) & (yy == 0)).sum())
        rows.append(
            {
                "subgroup": name,
                "n": int(m.sum()),
                "prevalence": float(yy.mean()),
                "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
                "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
                "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
                "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
                "accuracy": (tp + tn) / m.sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("subgroup").reset_index(drop=True)
