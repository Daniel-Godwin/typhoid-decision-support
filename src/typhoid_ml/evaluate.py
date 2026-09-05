"""Evaluation: headline metrics, confusion matrices, subgroup scalability, curves."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import BINARY_POSITIVE, SUBGROUP_COLUMN, labels_for


def core_metrics(y_true, y_pred, labels) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(labels),
    }
    if len(labels) == 2:
        pos = labels.index(BINARY_POSITIVE) if BINARY_POSITIVE in labels else 1
        neg = 1 - pos
        tp = int(cm[pos, pos])
        fn = int(cm[pos, neg])
        fp = int(cm[neg, pos])
        tn = int(cm[neg, neg])
        out.update(
            {
                "true_positive": tp,
                "false_negative": fn,
                "false_positive": fp,
                "true_negative": tn,
                "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
                "precision_positive": tp / (tp + fp) if (tp + fp) else 0.0,
                "f1_positive": float(f1_score(y_true, y_pred, pos_label=labels[pos], zero_division=0)),
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
            }
        )
    return out


def measure_latency(model, X_sample, repeats: int = 3) -> dict:
    """Single-record and batch inference latency (Table 3.1, 'Processing Time')."""
    single = X_sample.iloc[[0]]
    t = time.perf_counter()
    for _ in range(repeats):
        model.predict(single)
    single_ms = (time.perf_counter() - t) * 1000 / repeats

    t = time.perf_counter()
    model.predict(X_sample)
    batch_s = time.perf_counter() - t
    return {
        "single_record_ms": round(single_ms, 2),
        "batch_records": int(len(X_sample)),
        "batch_seconds": round(batch_s, 3),
        "records_per_second": round(len(X_sample) / batch_s, 1) if batch_s else None,
    }


def subgroup_report(model, X_test, y_test, subgroup_series, labels) -> list[dict]:
    """Objective 4: performance across geographic/health-environment subgroups."""
    pred = model.predict(X_test)
    rows = []
    for group in sorted(subgroup_series.unique()):
        mask = (subgroup_series == group).to_numpy()
        if mask.sum() == 0:
            continue
        yt = np.asarray(y_test)[mask]
        yp = np.asarray(pred)[mask]
        m = core_metrics(yt, yp, labels)
        rows.append(
            {
                "subgroup": str(group),
                "n": int(mask.sum()),
                "accuracy": round(m["accuracy"], 4),
                "balanced_accuracy": round(m["balanced_accuracy"], 4),
                "macro_precision": round(m["macro_precision"], 4),
                "macro_recall": round(m["macro_recall"], 4),
                "macro_f1": round(m["macro_f1"], 4),
                **(
                    {
                        "sensitivity": round(m["sensitivity"], 4),
                        "specificity": round(m["specificity"], 4),
                    }
                    if "sensitivity" in m
                    else {}
                ),
            }
        )
    return rows


def plot_confusion_matrix(cm, labels, path: Path, title: str, normalise: bool = False):
    data = np.array(cm, dtype=float)
    if normalise:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)
    size = 6.0 + 0.9 * len(labels)
    fig, ax = plt.subplots(figsize=(size, size * 0.82))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() else 1)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    thresh = data.max() / 2 if data.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            raw = np.array(cm)[i, j]
            txt = f"{data[i, j]:.2f}" if normalise else f"{raw:,}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if data[i, j] > thresh else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_binary_curves(y_true, scores, positive_label, path_roc: Path, path_pr: Path):
    y_bin = (np.asarray(y_true) == positive_label).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, scores)
    auc = roc_auc_score(y_bin, scores)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot(fpr, tpr, lw=2, label=f"SVM (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="grey", label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve - typhoid diagnosis")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path_roc.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_roc, dpi=180)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_bin, scores)
    ap = average_precision_score(y_bin, scores)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot(rec, prec, lw=2, label=f"SVM (AP = {ap:.4f})")
    ax.axhline(y_bin.mean(), ls="--", lw=1, color="grey", label=f"Prevalence = {y_bin.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve - typhoid diagnosis")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_pr, dpi=180)
    plt.close(fig)
    return {"roc_auc": float(auc), "average_precision": float(ap)}


def evaluate_model(
    model,
    X_test,
    y_test,
    target_mode: str,
    report_dir: Path,
    figure_dir: Path,
    subgroups: pd.Series | None = None,
    tag: str = "",
) -> dict:
    labels = labels_for(target_mode)
    pred = model.predict(X_test)
    result = core_metrics(y_test, pred, labels)
    result["target_mode"] = target_mode
    result["n_test"] = int(len(y_test))
    result["classification_report"] = classification_report(
        y_test, pred, labels=labels, output_dict=True, zero_division=0
    )
    result["latency"] = measure_latency(model, X_test.head(2000))

    suffix = f"_{target_mode}{tag}"
    plot_confusion_matrix(
        result["confusion_matrix"],
        labels,
        figure_dir / f"confusion_matrix{suffix}.png",
        f"SVM confusion matrix ({target_mode})",
    )
    plot_confusion_matrix(
        result["confusion_matrix"],
        labels,
        figure_dir / f"confusion_matrix_normalised{suffix}.png",
        f"SVM confusion matrix, row-normalised ({target_mode})",
        normalise=True,
    )

    if target_mode == "binary":
        try:
            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)
                scores = model.predict_proba(X_test)[:, classes.index(BINARY_POSITIVE)]
            else:
                scores = model.decision_function(X_test)
                if getattr(model, "classes_", [None, None])[1] != BINARY_POSITIVE:
                    scores = -scores
            result.update(
                plot_binary_curves(
                    y_test,
                    scores,
                    BINARY_POSITIVE,
                    figure_dir / f"roc_curve{suffix}.png",
                    figure_dir / f"pr_curve{suffix}.png",
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            result["curve_error"] = str(exc)

    if subgroups is not None:
        result["subgroup_evaluation"] = {
            "column": SUBGROUP_COLUMN,
            "rows": subgroup_report(model, X_test, y_test, subgroups, labels),
        }
        df = pd.DataFrame(result["subgroup_evaluation"]["rows"])
        report_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_dir / f"subgroup_evaluation{suffix}.csv", index=False)

    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / f"final_evaluation{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
