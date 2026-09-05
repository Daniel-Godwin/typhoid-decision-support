"""Assemble every artefact into Chapter 4 ready tables and a results summary.

Reads whatever exists under reports/ and writes:
  reports/RESULTS.md              - narrative summary with all tables
  reports/table_4_1_kernels.csv   - kernel comparison
  reports/table_4_2_metrics.csv   - final model performance
  reports/table_4_3_perclass.csv  - per-class precision/recall/F1
  reports/table_4_4_subgroups.csv - scalability across locations (Objective 4)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(x, digits=4):
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "-"


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def kernel_table() -> pd.DataFrame | None:
    frames = []
    for mode, label in (("binary", "Binary"), ("multiclass", "Four-class")):
        p = REPORTS / f"kernel_comparison_{mode}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.insert(0, "model", label)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def metrics_table() -> pd.DataFrame | None:
    rows = []
    for mode, label in (("binary", "Binary diagnosis"), ("multiclass", "Severity stratification")):
        data = _load(REPORTS / f"final_evaluation_{mode}.json")
        meta = _load(MODELS / f"svm_{mode}_metadata.json")
        if not data:
            continue
        row = {
            "model": label,
            "kernel": (meta or {}).get("best_params", {}).get("kernel", "-"),
            "C": (meta or {}).get("best_params", {}).get("C", "-"),
            "gamma": (meta or {}).get("best_params", {}).get("gamma", "-"),
            "accuracy": _fmt(data.get("accuracy")),
            "balanced_accuracy": _fmt(data.get("balanced_accuracy")),
            "macro_precision": _fmt(data.get("macro_precision")),
            "macro_recall": _fmt(data.get("macro_recall")),
            "macro_f1": _fmt(data.get("macro_f1")),
            "weighted_f1": _fmt(data.get("weighted_f1")),
        }
        if "sensitivity" in data:
            row.update(
                {
                    "sensitivity": _fmt(data["sensitivity"]),
                    "specificity": _fmt(data["specificity"]),
                    "fpr": _fmt(data["false_positive_rate"]),
                    "fnr": _fmt(data["false_negative_rate"]),
                    "roc_auc": _fmt(data.get("roc_auc")),
                }
            )
        row["inference_ms_per_record"] = data.get("latency", {}).get("single_record_ms", "-")
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def per_class_table() -> pd.DataFrame | None:
    rows = []
    for mode, label in (("binary", "Binary"), ("multiclass", "Four-class")):
        data = _load(REPORTS / f"final_evaluation_{mode}.json")
        if not data:
            continue
        for cls, m in data["classification_report"].items():
            if not isinstance(m, dict) or cls in ("macro avg", "weighted avg"):
                continue
            rows.append(
                {
                    "model": label,
                    "class": cls,
                    "precision": _fmt(m["precision"]),
                    "recall": _fmt(m["recall"]),
                    "f1_score": _fmt(m["f1-score"]),
                    "support": int(m["support"]),
                }
            )
    return pd.DataFrame(rows) if rows else None


def subgroup_table() -> pd.DataFrame | None:
    frames = []
    for mode, label in (("binary", "Binary"), ("multiclass", "Four-class")):
        p = REPORTS / f"subgroup_evaluation_{mode}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.insert(0, "model", label)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


POLICY_LABELS = [
    ("", "Routine — all permitted attributes"),
    ("_clinical_only", "Pre-laboratory — Widal and Typhidot removed"),
    ("_no_fever_duration", "Ablation — Fever Duration (Days) removed"),
]


def sensitivity_table() -> pd.DataFrame | None:
    rows = []
    for suffix, label in POLICY_LABELS:
        d = _load(REPORTS / f"final_evaluation_binary{suffix}.json")
        if not d:
            continue
        rows.append(
            {
                "feature policy": label,
                "accuracy": _fmt(d.get("accuracy")),
                "balanced_accuracy": _fmt(d.get("balanced_accuracy")),
                "macro_f1": _fmt(d.get("macro_f1")),
                "sensitivity": _fmt(d.get("sensitivity")),
                "specificity": _fmt(d.get("specificity")),
                "roc_auc": _fmt(d.get("roc_auc")),
            }
        )
    return pd.DataFrame(rows) if len(rows) > 1 else None


def baseline_table() -> pd.DataFrame | None:
    p = REPORTS / "table_4_0_baselines.csv"
    return pd.read_csv(p) if p.exists() else None


def association_table() -> pd.DataFrame | None:
    p = REPORTS / "feature_association.csv"
    return pd.read_csv(p) if p.exists() else None


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    audit = _load(REPORTS / "dataset_audit.json") or {}
    parts = ["# Results\n"]

    if audit:
        parts.append("## Dataset\n")
        parts.append(
            f"The dataset contains **{audit['rows']:,} records** and **{audit['columns']} attributes**, "
            f"with **{audit['duplicates']} duplicate rows**.\n"
        )
        dist = pd.DataFrame(
            [
                {"class": k, "count": f"{v:,}", "percentage": f"{v * 100 / audit['rows']:.2f}%"}
                for k, v in audit["target_distribution_multiclass"].items()
            ]
        )
        parts.append("### Target distribution (four-class)\n")
        parts.append(_md_table(dist) + "\n")
        distb = pd.DataFrame(
            [
                {"class": k, "count": f"{v:,}", "percentage": f"{v * 100 / audit['rows']:.2f}%"}
                for k, v in audit["target_distribution_binary"].items()
            ]
        )
        parts.append("### Target distribution (binary)\n")
        parts.append(_md_table(distb) + "\n")

    sections = [
        (
            "Table 4.0 - Reference baselines on the same held-out test partition",
            baseline_table(),
            "table_4_0_baselines.csv",
        ),
        ("Table 4.1 - Kernel comparison", kernel_table(), "table_4_1_kernels.csv"),
        ("Table 4.2 - Optimised model performance", metrics_table(), "table_4_2_metrics.csv"),
        ("Table 4.3 - Per-class performance", per_class_table(), "table_4_3_perclass.csv"),
        (
            "Table 4.4 - Scalability across geographic settings (Objective 4)",
            subgroup_table(),
            "table_4_4_subgroups.csv",
        ),
        (
            "Table 4.5 - Feature policy sensitivity and ablation analysis",
            sensitivity_table(),
            "table_4_5_sensitivity.csv",
        ),
        (
            "Table 4.6 - Association between each attribute and the diagnosis",
            association_table(),
            "table_4_6_association.csv",
        ),
    ]
    for title, df, filename in sections:
        if df is None or df.empty:
            continue
        df.to_csv(REPORTS / filename, index=False)
        parts.append(f"## {title}\n")
        parts.append(_md_table(df) + "\n")

    for mode in ("binary", "multiclass"):
        summary = _load(REPORTS / f"grid_search_summary_{mode}.json")
        if summary:
            parts.append(f"## Hyperparameter search - {mode}\n")
            parts.append(
                f"- Candidates evaluated: **{summary['n_candidates']}** "
                f"over **{summary['cv_folds']}-fold** stratified cross-validation\n"
                f"- Search records: **{summary['search_records']:,}**"
                f"{' (stratified subsample)' if summary['search_subsampled'] else ''}\n"
                f"- Selection metric: **macro F1**\n"
                f"- Best configuration: `{summary['best_params']}`\n"
                f"- Best cross-validated macro F1: **{summary['best_cv_macro_f1']:.4f}**\n"
                f"- Search wall-clock: **{summary['search_seconds']:.0f}s**\n"
            )

    analysis = _load(REPORTS / "baseline_analysis.json")
    if analysis:
        parts.append("## Signal audit\n")
        parts.append(
            f"Of the {analysis['records_with_fever_ge_1']:,} records with a fever duration of one "
            f"day or more, **{analysis['typhoid_among_fever_ge_1']:,}** are labelled typhoid "
            f"({analysis['typhoid_among_fever_ge_1'] * 100 / max(analysis['records_with_fever_ge_1'], 1):.1f}%). "
            f"A further {analysis['typhoid_with_zero_fever']:,} typhoid cases have a recorded fever "
            f"duration of zero days.\n\n"
            "`Fever Duration (Days)` therefore acts as a near-deterministic proxy for the target, "
            "and the reference baselines in Table 4.0 must be read alongside the model results. "
            "See `docs/DATASET_AUDIT.md` for the full analysis.\n"
        )

    figures = sorted((REPORTS / "figures").glob("*.png")) if (REPORTS / "figures").exists() else []
    if figures:
        parts.append("## Figures\n")
        for f in figures:
            parts.append(f"- `reports/figures/{f.name}`")
        parts.append("")

    out = REPORTS / "RESULTS.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")
    for _, df, filename in sections:
        if df is not None and not df.empty:
            print(f"Wrote {REPORTS / filename}")


if __name__ == "__main__":
    main()
