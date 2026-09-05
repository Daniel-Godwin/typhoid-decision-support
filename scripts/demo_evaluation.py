"""Small-sample demonstration: run real patient records through the live application.

Draws balanced samples of real records, submits each one to the running Flask
application through `POST /api/predict` (so the whole deployed stack is
exercised, not just the model object), and reports what the system predicted
against the recorded diagnosis.

Four sets are evaluated:

  test-set-A    10 records from the held-out test partition (5 typhoid, 5 not)
  test-set-B    10 further records from the held-out test partition, disjoint
                from A (5 typhoid, 5 not)
  train-sample  10 records the model was trained on (5 typhoid, 5 not).
                Comparing this against the held-out sets is an overfitting
                check: a model that memorised will score far higher here.
  hard-cases    10 records chosen to be difficult (5 typhoid, 5 not).
                The typhoid cases are ones with a recorded fever duration of
                zero days, which the dataset's dominant rule cannot catch.
                This set is the informative one.

Usage:
    python scripts/run_app.py          # in one terminal
    python scripts/demo_evaluation.py  # in another
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typhoid_ml.config import (  # noqa: E402
    DATA_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET,
    TEST_SIZE,
)
from typhoid_ml.data import build_target, feature_frame, load_dataset  # noqa: E402

FEVER = "Fever Duration (Days)"
POSITIVE = "Typhoid"
NEGATIVE = "No Typhoid"


def call_api(base_url: str, record: dict, target_mode: str = "binary") -> dict:
    payload = json.dumps({"record": record, "target_mode": target_mode, "explain": True})
    req = urllib.request.Request(
        f"{base_url}/api/predict",
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def balanced_sample(X, y, index_pool, n_per_class, seed, exclude=frozenset()):
    """n_per_class records of each class, drawn from index_pool."""
    pool = [i for i in index_pool if i not in exclude]
    chosen = []
    for label in (POSITIVE, NEGATIVE):
        candidates = [i for i in pool if y.loc[i] == label]
        picked = pd.Series(candidates).sample(n=n_per_class, random_state=seed).tolist()
        chosen.extend(picked)
    return chosen


def hard_sample(df, X, y, index_pool, n_per_class, seed, exclude=frozenset()):
    """Typhoid cases with zero recorded fever duration, and the negatives most
    similar to positives, i.e. the records a fever-duration rule gets wrong."""
    pool = [i for i in index_pool if i not in exclude]
    positives = [i for i in pool if y.loc[i] == POSITIVE and df.loc[i, FEVER] == 0]
    negatives = [i for i in pool if y.loc[i] == NEGATIVE and df.loc[i, FEVER] == 0]
    chosen = []
    if positives:
        k = min(n_per_class, len(positives))
        chosen.extend(pd.Series(positives).sample(n=k, random_state=seed).tolist())
    if negatives:
        chosen.extend(pd.Series(negatives).sample(n=n_per_class, random_state=seed).tolist())
    return chosen


def run_set(name, indices, df, X, y, base_url, target_mode="binary"):
    rows = []
    for n, idx in enumerate(indices, start=1):
        record = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                  for k, v in X.loc[idx].to_dict().items()}
        try:
            out = call_api(base_url, record, target_mode)
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"Could not reach the application at {base_url}. "
                f"Start it first with: python scripts/run_app.py\n  ({exc})"
            )
        truth = y.loc[idx]
        pred = out["prediction"]
        drivers = out.get("explanation", {}).get("contributions", [])
        top = next((d for d in drivers if abs(d.get("contribution", 0)) > 0.001), None)
        rows.append(
            {
                "set": name,
                "#": n,
                "record_id": int(idx),
                "age": int(record["Age"]),
                "fever_days": int(record[FEVER]),
                "widal": record["Widal Test"],
                "typhidot": record["Typhidot Test"],
                "wbc": int(record["White Blood Cell Count"]),
                "actual": truth,
                "predicted": pred,
                "confidence": out.get("confidence"),
                "correct": "yes" if pred == truth else "NO",
                "top_driver": top["feature"] if top else "(none above 0.1 pt)",
                "four_class_actual": df.loc[idx, TARGET],
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--n-per-class", type=int, default=5)
    args = ap.parse_args()

    df = load_dataset(DATA_PATH)
    X, _, _ = feature_frame(df, "routine")
    y = build_target(df, "binary")
    idx_train, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    n = args.n_per_class
    set_a = balanced_sample(X, y, idx_test, n, seed=1)
    set_b = balanced_sample(X, y, idx_test, n, seed=2, exclude=set(set_a))
    set_train = balanced_sample(X, y, idx_train, n, seed=3)
    set_hard = hard_sample(df, X, y, idx_test, n, seed=4, exclude=set(set_a) | set(set_b))

    all_rows = []
    for name, indices in (
        ("A. Held-out test sample", set_a),
        ("B. Held-out test sample 2", set_b),
        ("C. Training sample (seen)", set_train),
        ("D. Hard cases (fever = 0)", set_hard),
    ):
        all_rows.extend(run_set(name, indices, df, X, y, args.url))

    results = pd.DataFrame(all_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORT_DIR / "demo_evaluation.csv"
    results.to_csv(out_csv, index=False)

    display_cols = [
        "#", "age", "fever_days", "widal", "typhidot", "wbc",
        "actual", "predicted", "confidence", "correct", "top_driver",
    ]
    lines = []
    for name in results["set"].unique():
        block = results[results["set"] == name]
        correct = int((block["correct"] == "yes").sum())
        lines.append(f"\n{'=' * 100}\n{name}   —   {correct}/{len(block)} correct\n{'=' * 100}")
        lines.append(block[display_cols].to_string(index=False))
    summary = (
        results.groupby("set")
        .apply(lambda b: pd.Series({
            "n": len(b),
            "correct": int((b["correct"] == "yes").sum()),
            "accuracy": round(float((b["correct"] == "yes").mean()), 3),
            "mean_confidence": round(float(pd.to_numeric(b["confidence"]).mean()), 3),
        }), include_groups=False)
        .reset_index()
    )
    lines.append(f"\n{'=' * 100}\nSUMMARY\n{'=' * 100}")
    lines.append(summary.to_string(index=False))

    report = "\n".join(lines)
    print(report)
    (REPORT_DIR / "demo_evaluation.txt").write_text(report, encoding="utf-8")
    summary.to_csv(REPORT_DIR / "demo_evaluation_summary.csv", index=False)
    print(f"\nWritten to {out_csv} and {REPORT_DIR / 'demo_evaluation.txt'}")


if __name__ == "__main__":
    main()
