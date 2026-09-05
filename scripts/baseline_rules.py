"""Trivial baselines, evaluated on the same held-out test partition.

A machine-learning result is only meaningful relative to what a rule of thumb
already achieves. This script scores three reference classifiers on the exact
same stratified 20% test partition used for the SVM, so Chapter 4 can state the
SVM's marginal contribution rather than its raw accuracy alone.

  1. Majority class      - always predict the most frequent class.
  2. Fever-duration rule - predict Typhoid when fever duration >= 1 day.
  3. Single decision stump on each individual feature.

It also reports the association between every feature and the target, which is
what exposes how much of the signal sits in a single column.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typhoid_ml.config import (  # noqa: E402
    BINARY_POSITIVE,
    DATA_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET,
    TEST_SIZE,
)
from typhoid_ml.data import build_target, feature_frame, load_dataset  # noqa: E402

FEVER = "Fever Duration (Days)"


def _score(name, y_true, y_pred, labels):
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "macro_recall": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def associations(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cramer's V for categorical columns, point-biserial r for numeric ones."""
    pos = (y == BINARY_POSITIVE).astype(float)
    rows = []
    for c in df.columns:
        if c == TARGET:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            stat = float(np.corrcoef(pos, pd.to_numeric(df[c], errors="coerce").fillna(0))[0, 1])
            rows.append({"feature": c, "statistic": "point-biserial r", "value": round(stat, 4)})
        else:
            table = pd.crosstab(df[c].fillna("__missing__"), y)
            chi2 = chi2_contingency(table)[0]
            v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
            rows.append({"feature": c, "statistic": "Cramer's V", "value": round(float(v), 4)})
    out = pd.DataFrame(rows)
    return out.reindex(out["value"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_PATH)
    y = build_target(df, "binary")
    labels = ["No Typhoid", BINARY_POSITIVE]

    # Identical split to train.py.
    idx_train, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    df_test = df.loc[idx_test]

    rows = []

    majority = y_train.value_counts().idxmax()
    rows.append(_score(f"Majority class ('{majority}')", y_test, [majority] * len(y_test), labels))

    rule = np.where(df_test[FEVER] >= 1, BINARY_POSITIVE, "No Typhoid")
    rows.append(_score(f"Rule: {FEVER} >= 1", y_test, rule, labels))
    rule_cm = confusion_matrix(y_test, rule, labels=labels).tolist()

    # One-feature decision stumps, to show how the signal is distributed.
    X, numeric, categorical = feature_frame(df, "routine")
    stumps = []
    for col in numeric + categorical:
        enc = (
            pd.to_numeric(X[col], errors="coerce").fillna(-1)
            if col in numeric
            else X[col].astype("category").cat.codes
        )
        stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE).fit(
            enc.loc[idx_train].to_frame(), y_train
        )
        pred = stump.predict(enc.loc[idx_test].to_frame())
        stumps.append(
            {
                "feature": col,
                "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
                "macro_f1": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
            }
        )
    stump_df = pd.DataFrame(stumps).sort_values("macro_f1", ascending=False).reset_index(drop=True)

    baseline_df = pd.DataFrame(rows)

    # Bring in the trained SVM result if it exists, for a like-for-like row.
    svm_path = REPORT_DIR / "final_evaluation_binary.json"
    if svm_path.exists():
        d = json.loads(svm_path.read_text(encoding="utf-8"))
        baseline_df.loc[len(baseline_df)] = {
            "model": "Optimised SVM (routine policy)",
            "accuracy": round(d["accuracy"], 4),
            "balanced_accuracy": round(d["balanced_accuracy"], 4),
            "macro_precision": round(d["macro_precision"], 4),
            "macro_recall": round(d["macro_recall"], 4),
            "macro_f1": round(d["macro_f1"], 4),
        }
    ablation_path = REPORT_DIR / "final_evaluation_binary_no_fever_duration.json"
    if ablation_path.exists():
        d = json.loads(ablation_path.read_text(encoding="utf-8"))
        baseline_df.loc[len(baseline_df)] = {
            "model": "Optimised SVM (fever duration removed)",
            "accuracy": round(d["accuracy"], 4),
            "balanced_accuracy": round(d["balanced_accuracy"], 4),
            "macro_precision": round(d["macro_precision"], 4),
            "macro_recall": round(d["macro_recall"], 4),
            "macro_f1": round(d["macro_f1"], 4),
        }

    assoc_df = associations(df.drop(columns=[TARGET]), y)

    # ---- four-class rule baseline -------------------------------------------
    # Two deterministic rules are present in the data:
    #   WBC > 11,000              <=> Complicated Typhoid
    #   Fever duration >= 1 day    => some form of typhoid
    # Acute and Relapsing are not separable by any attribute.
    y4 = df[TARGET].astype(str)
    y4_test = y4.loc[idx_test]
    wbc = pd.to_numeric(df.loc[idx_test, "White Blood Cell Count"], errors="coerce")
    fev4 = pd.to_numeric(df.loc[idx_test, FEVER], errors="coerce")
    rule4 = np.where(
        wbc > 11000,
        "Complicated Typhoid",
        np.where(fev4 >= 1, "Acute Typhoid Fever", "Normal or No Typhoid"),
    )
    labels4 = [
        "Normal or No Typhoid",
        "Acute Typhoid Fever",
        "Relapsing Typhoid",
        "Complicated Typhoid",
    ]
    rows4 = [
        _score("Majority class", y4_test, [y4.loc[idx_train].value_counts().idxmax()] * len(y4_test), labels4),
        _score("Rules: WBC > 11,000 -> Complicated; fever >= 1 -> Acute", y4_test, rule4, labels4),
    ]
    m4 = REPORT_DIR / "final_evaluation_multiclass.json"
    if m4.exists():
        d = json.loads(m4.read_text(encoding="utf-8"))
        rows4.append(
            {
                "model": "Optimised SVM (four-class)",
                "accuracy": round(d["accuracy"], 4),
                "balanced_accuracy": round(d["balanced_accuracy"], 4),
                "macro_precision": round(d["macro_precision"], 4),
                "macro_recall": round(d["macro_recall"], 4),
                "macro_f1": round(d["macro_f1"], 4),
            }
        )
    baseline4_df = pd.DataFrame(rows4)
    baseline4_df.to_csv(REPORT_DIR / "table_4_0b_baselines_multiclass.csv", index=False)

    wbc_all = pd.to_numeric(df["White Blood Cell Count"], errors="coerce")
    wbc_contingency = pd.crosstab(
        np.where(wbc_all > 11000, "> 11,000", "<= 11,000"), y4
    )
    wbc_contingency.to_csv(REPORT_DIR / "wbc_contingency.csv")

    baseline_df.to_csv(REPORT_DIR / "table_4_0_baselines.csv", index=False)
    stump_df.to_csv(REPORT_DIR / "single_feature_stumps.csv", index=False)
    assoc_df.to_csv(REPORT_DIR / "feature_association.csv", index=False)

    # Fever-duration contingency, the key diagnostic evidence.
    fever_flag = np.where(df[FEVER] >= 1, ">= 1 day", "0 days")
    contingency = pd.crosstab(fever_flag, y)
    contingency.to_csv(REPORT_DIR / "fever_duration_contingency.csv")

    summary = {
        "test_records": int(len(y_test)),
        "baselines": baseline_df.to_dict("records"),
        "baselines_multiclass": baseline4_df.to_dict("records"),
        "wbc_contingency": wbc_contingency.to_dict(),
        "complicated_wbc_range": [
            float(wbc_all[y4 == "Complicated Typhoid"].min()),
            float(wbc_all[y4 == "Complicated Typhoid"].max()),
        ],
        "other_class_wbc_range": [
            float(wbc_all[y4 != "Complicated Typhoid"].min()),
            float(wbc_all[y4 != "Complicated Typhoid"].max()),
        ],
        "fever_rule_confusion_matrix": rule_cm,
        "fever_rule_labels": labels,
        "fever_duration_contingency": contingency.to_dict(),
        "records_with_fever_ge_1": int((df[FEVER] >= 1).sum()),
        "typhoid_among_fever_ge_1": int(((df[FEVER] >= 1) & (y == BINARY_POSITIVE)).sum()),
        "typhoid_with_zero_fever": int(((df[FEVER] == 0) & (y == BINARY_POSITIVE)).sum()),
        "top_associations": assoc_df.head(8).to_dict("records"),
    }
    (REPORT_DIR / "baseline_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Baselines on the held-out test partition ===")
    print(baseline_df.to_string(index=False))
    print("\n=== Fever duration vs diagnosis (whole dataset) ===")
    print(contingency.to_string())
    print("\n=== Feature association with the target ===")
    print(assoc_df.to_string(index=False))
    print("\n=== Four-class baselines ===")
    print(baseline4_df.to_string(index=False))
    print("\n=== White blood cell count vs class ===")
    print(wbc_contingency.to_string())
    print("\n=== Best single-feature decision stumps ===")
    print(stump_df.head(8).to_string(index=False))
    print(f"\nWritten to {REPORT_DIR}")


if __name__ == "__main__":
    main()
