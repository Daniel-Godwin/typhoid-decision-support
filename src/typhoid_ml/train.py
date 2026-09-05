"""End-to-end experiment orchestration.

Modes
-----
audit     : dataset audit only.
baseline  : compare linear / polynomial / RBF kernels under the default settings.
grid      : cross-validated hyperparameter search (Grid Search + StratifiedKFold).
final     : refit the winning configuration on the full training partition,
            enable probability estimates, evaluate on the held-out test set and
            persist the deployable artefact.
all       : baseline -> grid -> final.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CV_FOLDS,
    DATA_PATH,
    FEATURE_POLICIES,
    DEFAULT_POLICY,
    FIGURE_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    SEARCH_SUBSAMPLE,
    SUBGROUP_COLUMN,
    TEST_SIZE,
)
from .data import audit_dataset, build_target, feature_frame, load_dataset
from .evaluate import evaluate_model
from .model import build_pipeline, build_search, n_candidates


def _split(df, target_mode, policy):
    X, numeric, categorical = feature_frame(df, policy)
    y = build_target(df, target_mode)
    idx_train, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return (
        X.loc[idx_train],
        X.loc[idx_test],
        y.loc[idx_train],
        y.loc[idx_test],
        df.loc[idx_test, SUBGROUP_COLUMN],
        numeric,
        categorical,
    )


def _suffix(target_mode, policy):
    return f"_{target_mode}" + ("" if policy == DEFAULT_POLICY else f"_{policy}")


def run_audit(df):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(df)
    (REPORT_DIR / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Audit written: rows={audit['rows']} columns={audit['columns']}")
    return audit


def run_baseline(X_train, X_test, y_train, y_test, numeric, categorical, target_mode, policy):
    rows = []
    for kernel in ["linear", "poly", "rbf"]:
        t = time.time()
        model = build_pipeline(numeric, categorical, kernel=kernel)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        from sklearn.metrics import balanced_accuracy_score, f1_score

        rows.append(
            {
                "kernel": kernel,
                "accuracy": round(float((pred == y_test).mean()), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
                "macro_f1": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
                "weighted_f1": round(
                    float(f1_score(y_test, pred, average="weighted", zero_division=0)), 4
                ),
                "fit_predict_seconds": round(time.time() - t, 1),
            }
        )
        print(f"  {kernel:7s} {rows[-1]}")
    out = REPORT_DIR / f"kernel_comparison{_suffix(target_mode, policy)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Kernel comparison written to {out}")
    return rows


def run_grid(X_train, y_train, numeric, categorical, target_mode, policy, n_jobs):
    subsample = SEARCH_SUBSAMPLE.get(target_mode)
    Xs, ys = X_train, y_train
    if subsample and subsample < len(X_train):
        Xs, _, ys, _ = train_test_split(
            X_train, y_train, train_size=subsample, stratify=y_train, random_state=RANDOM_STATE
        )
        print(
            f"Search executed on a stratified subsample of {len(Xs):,} of "
            f"{len(X_train):,} training records (tractability); the winning "
            f"configuration is refitted on the full training partition."
        )

    search = build_search(numeric, categorical, cv=CV_FOLDS, n_jobs=n_jobs)
    print(f"Grid: {n_candidates()} candidate configurations x {CV_FOLDS} folds")
    t = time.time()
    search.fit(Xs, ys)
    elapsed = time.time() - t

    cv_df = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]
    ].sort_values("rank_test_score")
    cv_df["params"] = cv_df["params"].astype(str)
    cv_path = REPORT_DIR / f"grid_search_results{_suffix(target_mode, policy)}.csv"
    cv_df.to_csv(cv_path, index=False)

    summary = {
        "target_mode": target_mode,
        "feature_policy": policy,
        "scoring": "f1_macro",
        "cv_folds": CV_FOLDS,
        "n_candidates": int(n_candidates()),
        "search_records": int(len(Xs)),
        "search_subsampled": bool(subsample and subsample < len(X_train)),
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "search_seconds": round(elapsed, 1),
        "top_5": cv_df.head(5).to_dict("records"),
    }
    (REPORT_DIR / f"grid_search_summary{_suffix(target_mode, policy)}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Best params: {search.best_params_}  (CV macro-F1 {search.best_score_:.4f}, {elapsed:.0f}s)")
    return summary


def run_final(
    X_train, X_test, y_train, y_test, subgroups, numeric, categorical, target_mode, policy, best_params
):
    params = {k.replace("svc__", ""): v for k, v in best_params.items()}
    print(f"Refitting on full training partition ({len(X_train):,} records) with {params} ...")
    t = time.time()
    model = build_pipeline(numeric, categorical, probability=True, **params)
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t
    print(f"Refit complete in {fit_seconds:.0f}s")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _suffix(target_mode, policy)
    model_path = MODEL_DIR / f"svm{suffix}.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "target_mode": target_mode,
        "feature_policy": policy,
        "best_params": params,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "classes": list(model.classes_),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fit_seconds": round(fit_seconds, 1),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "random_state": RANDOM_STATE,
    }
    (MODEL_DIR / f"svm{suffix}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # `tag` carries the feature policy so a sensitivity run can never overwrite
    # the artefacts of the primary run.
    result = evaluate_model(
        model,
        X_test,
        y_test,
        target_mode,
        REPORT_DIR,
        FIGURE_DIR,
        subgroups=subgroups,
        tag="" if policy == DEFAULT_POLICY else f"_{policy}",
    )
    print(
        f"Test accuracy {result['accuracy']:.4f} | balanced accuracy "
        f"{result['balanced_accuracy']:.4f} | macro-F1 {result['macro_f1']:.4f}"
    )
    print(f"Model saved to {model_path}")
    return model, result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Typhoid SVM training pipeline")
    ap.add_argument("--mode", choices=["audit", "baseline", "grid", "final", "all"], default="all")
    ap.add_argument("--target", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--policy", choices=list(FEATURE_POLICIES), default=DEFAULT_POLICY)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument(
        "--params",
        type=str,
        default=None,
        help="JSON of SVC params to skip the search, e.g. '{\"kernel\":\"rbf\",\"C\":10}'",
    )
    args = ap.parse_args(argv)

    df = load_dataset(DATA_PATH)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "audit":
        run_audit(df)
        return

    run_audit(df)
    X_train, X_test, y_train, y_test, subgroups, numeric, categorical = _split(
        df, args.target, args.policy
    )
    print(
        f"\nTarget={args.target} policy={args.policy} | train={len(X_train):,} test={len(X_test):,} "
        f"| features: {len(numeric)} numeric + {len(categorical)} categorical"
    )
    print(f"Training class balance: {y_train.value_counts().to_dict()}\n")

    best_params = None
    if args.params:
        best_params = {f"svc__{k}": v for k, v in json.loads(args.params).items()}

    if args.mode in ("baseline", "all"):
        print("--- Kernel comparison ---")
        run_baseline(X_train, X_test, y_train, y_test, numeric, categorical, args.target, args.policy)

    if args.mode in ("grid", "all") and best_params is None:
        print("\n--- Grid search ---")
        summary = run_grid(
            X_train, y_train, numeric, categorical, args.target, args.policy, args.n_jobs
        )
        best_params = summary["best_params"]

    if args.mode in ("final", "all"):
        if best_params is None:
            path = REPORT_DIR / f"grid_search_summary{_suffix(args.target, args.policy)}.json"
            if not path.exists():
                raise SystemExit(f"No grid search summary at {path}; run --mode grid first.")
            best_params = json.loads(path.read_text())["best_params"]
        print("\n--- Final model ---")
        run_final(
            X_train,
            X_test,
            y_train,
            y_test,
            subgroups,
            numeric,
            categorical,
            args.target,
            args.policy,
            best_params,
        )


if __name__ == "__main__":
    main()
