"""Re-evaluate an already-saved model without retraining it.

Useful after changing reporting code, or to regenerate evaluation artefacts.

    python scripts/evaluate_saved.py --target binary --policy routine
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typhoid_ml.config import (  # noqa: E402
    DATA_PATH,
    DEFAULT_POLICY,
    FEATURE_POLICIES,
    FIGURE_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    SUBGROUP_COLUMN,
    TEST_SIZE,
)
from typhoid_ml.data import build_target, feature_frame, load_dataset  # noqa: E402
from typhoid_ml.evaluate import evaluate_model  # noqa: E402
from typhoid_ml.predict import model_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--policy", choices=list(FEATURE_POLICIES), default=DEFAULT_POLICY)
    args = ap.parse_args()

    path = model_path(args.target, args.policy)
    if not path.exists():
        raise SystemExit(f"No saved model at {path}")

    df = load_dataset(DATA_PATH)
    X, _, _ = feature_frame(df, args.policy)
    y = build_target(df, args.target)
    _, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    model = joblib.load(path)
    result = evaluate_model(
        model,
        X.loc[idx_test],
        y.loc[idx_test],
        args.target,
        REPORT_DIR,
        FIGURE_DIR,
        subgroups=df.loc[idx_test, SUBGROUP_COLUMN],
        tag="" if args.policy == DEFAULT_POLICY else f"_{args.policy}",
    )
    print(
        f"{args.target}/{args.policy}: accuracy {result['accuracy']:.4f} | "
        f"balanced {result['balanced_accuracy']:.4f} | macro-F1 {result['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
