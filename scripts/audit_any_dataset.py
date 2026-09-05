"""Dataset integrity audit — checks whether a dataset can support a modelling claim.

Reported accuracy is only meaningful if the data can actually support it. This
script runs the checks that expose a dataset whose apparent predictability comes
from how it was constructed rather than from clinical signal:

  1. Shape, class balance, missingness, duplicate rows (and which class they sit in).
  2. Association between every attribute and the target
     (Cramer's V for categorical, point-biserial r for numeric).
  3. Class-conditional ranges — non-overlapping ranges mean a hard threshold
     separates the classes.
  4. Distribution width by class — a class whose values are far tighter than the
     other is a sign of generated rather than observed data.
  5. Deterministic categories — any category level that maps to one class with
     no exceptions.
  6. Trivial baselines: majority class, best single-attribute decision stump,
     and shallow decision trees. If a depth-2 or depth-3 tree reaches near-perfect
     accuracy, the dataset is separable by a handful of thresholds and no
     sophisticated model is being tested.
  7. Post-deduplication cross-validated performance.

Usage:
    python scripts/audit_any_dataset.py --file data/Typhoid_Dataset.xlsx --target typhoid
    python scripts/audit_any_dataset.py --file data/typhoid_dataset.csv \
        --target "Typhoid Status" --positive-not "Normal or No Typhoid" \
        --drop "Blood Culture Result" "Complications"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

SEED = 42


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise SystemExit(f"Unsupported file type: {path.suffix}")


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype("category").cat.codes
    return out


def associations(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            v = pd.to_numeric(X[c], errors="coerce")
            mask = v.notna()
            stat = float(np.corrcoef(y[mask].astype(float), v[mask])[0, 1]) if mask.sum() > 2 else 0.0
            rows.append({"attribute": c, "statistic": "point-biserial r", "value": round(stat, 4)})
        else:
            table = pd.crosstab(X[c].fillna("__missing__"), y)
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            chi2 = chi2_contingency(table)[0]
            v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
            rows.append({"attribute": c, "statistic": "Cramer's V", "value": round(float(v), 4)})
    out = pd.DataFrame(rows)
    return out.reindex(out["value"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def range_overlap(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            continue
        a, b = X.loc[y == 1, c].dropna(), X.loc[y == 0, c].dropna()
        if a.empty or b.empty:
            continue
        overlaps = max(a.min(), b.min()) <= min(a.max(), b.max())
        sd_ratio = (a.std() / b.std()) if b.std() else float("inf")
        rows.append(
            {
                "attribute": c,
                "positive_range": f"[{a.min():g}, {a.max():g}]",
                "negative_range": f"[{b.min():g}, {b.max():g}]",
                "ranges_overlap": overlaps,
                "sd_positive": round(float(a.std()), 3),
                "sd_negative": round(float(b.std()), 3),
                "sd_ratio": round(float(sd_ratio), 2),
            }
        )
    return pd.DataFrame(rows)


def deterministic_categories(X: pd.DataFrame, y: pd.Series, min_count=10) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            continue
        for level, grp in y.groupby(X[c].fillna("__missing__")):
            if len(grp) >= min_count and grp.nunique() == 1:
                rows.append(
                    {
                        "attribute": c,
                        "level": str(level),
                        "n": int(len(grp)),
                        "always_class": int(grp.iloc[0]),
                    }
                )
    return pd.DataFrame(rows)


def baselines(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    Xe = encode(X)
    Xtr, Xte, ytr, yte = train_test_split(Xe, y, test_size=0.2, stratify=y, random_state=SEED)

    def score(name, pred):
        return {
            "classifier": name,
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
            "macro_f1": round(float(f1_score(yte, pred, average="macro", zero_division=0)), 4),
        }

    rows = [score("Majority class", np.full(len(yte), ytr.mode().iloc[0]))]

    stumps = []
    for c in Xe.columns:
        m = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(Xtr[[c]], ytr)
        p = m.predict(Xte[[c]])
        stumps.append(
            {
                "attribute": c,
                "accuracy": round(float(accuracy_score(yte, p)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(yte, p)), 4),
                "macro_f1": round(float(f1_score(yte, p, average="macro", zero_division=0)), 4),
            }
        )
    stump_df = pd.DataFrame(stumps).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    best = stump_df.iloc[0]
    rows.append(
        {
            "classifier": f"Best single attribute ({best['attribute']})",
            "accuracy": best["accuracy"],
            "balanced_accuracy": best["balanced_accuracy"],
            "macro_f1": best["macro_f1"],
        }
    )

    tree_text = ""
    for depth in (2, 3, 5):
        m = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(Xtr, ytr)
        rows.append(score(f"Decision tree, depth {depth}", m.predict(Xte)))
        if depth == 3:
            tree_text = export_text(m, feature_names=list(Xe.columns), decimals=1)

    return pd.DataFrame(rows), stump_df, tree_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument(
        "--positive-not",
        default=None,
        help="Target value meaning NEGATIVE; everything else becomes positive.",
    )
    ap.add_argument("--drop", nargs="*", default=[], help="Attributes to exclude")
    ap.add_argument("--out", default=None, help="Output name stem (default: file stem)")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    df = load_any(path)
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found. Columns: {list(df.columns)}")

    raw_target = df[args.target]
    if args.positive_not is not None:
        y = (raw_target.astype(str) != args.positive_not).astype(int)
    else:
        vals = sorted(raw_target.dropna().unique())
        if len(vals) != 2:
            raise SystemExit(
                f"Target has {len(vals)} values {vals}; pass --positive-not to binarise."
            )
        y = (raw_target == vals[-1]).astype(int)

    X = df.drop(columns=[args.target] + [c for c in args.drop if c in df.columns])

    stem = args.out or path.stem
    report_dir = Path(__file__).resolve().parents[1] / "reports" / "audits"
    report_dir.mkdir(parents=True, exist_ok=True)

    dup_mask = df.duplicated(keep=False)
    dup_by_class = y[dup_mask].value_counts().to_dict() if dup_mask.any() else {}
    deduped = df.drop_duplicates()

    assoc = associations(X, y)
    ranges = range_overlap(X, y)
    determ = deterministic_categories(X, y)
    base, stumps, tree_text = baselines(X, y)

    Xd = encode(deduped.drop(columns=[args.target] + [c for c in args.drop if c in deduped.columns]))
    yd = (
        (deduped[args.target].astype(str) != args.positive_not).astype(int)
        if args.positive_not is not None
        else (deduped[args.target] == sorted(deduped[args.target].dropna().unique())[-1]).astype(int)
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    dedup_scores = {
        f"tree_depth_{d}": round(
            float(
                cross_val_score(
                    DecisionTreeClassifier(max_depth=d, random_state=SEED),
                    Xd,
                    yd,
                    cv=cv,
                    scoring="balanced_accuracy",
                ).mean()
            ),
            4,
        )
        for d in (1, 2, 3)
    }

    lines = [f"DATASET INTEGRITY AUDIT — {path.name}", "=" * 78, ""]
    lines.append(f"Rows: {len(df):,}   Attributes: {df.shape[1]}   Target: {args.target}")
    lines.append(f"Class balance (1 = positive): {y.value_counts().to_dict()}")
    lines.append(f"Positive rate: {y.mean() * 100:.1f}%")
    lines.append(f"Duplicate rows: {int(dup_mask.sum())} ({dup_mask.sum() * 100 / len(df):.1f}%)")
    if dup_by_class:
        lines.append(f"  duplicated rows by class: {dup_by_class}")
    miss = {c: int(v) for c, v in df.isna().sum().items() if v}
    lines.append(f"Missing values: {miss if miss else 'none'}")
    lines.append("")
    lines.append("ASSOCIATION WITH TARGET")
    lines.append("-" * 78)
    lines.append(assoc.to_string(index=False))
    lines.append("")
    if not ranges.empty:
        lines.append("CLASS-CONDITIONAL RANGES  (overlap=False means a hard threshold separates)")
        lines.append("-" * 78)
        lines.append(ranges.to_string(index=False))
        lines.append("")
    if not determ.empty:
        lines.append("DETERMINISTIC CATEGORY LEVELS  (level always maps to one class)")
        lines.append("-" * 78)
        lines.append(determ.to_string(index=False))
        lines.append("")
    lines.append("TRIVIAL BASELINES  (held-out 20%)")
    lines.append("-" * 78)
    lines.append(base.to_string(index=False))
    lines.append("")
    lines.append("BEST SINGLE ATTRIBUTES")
    lines.append("-" * 78)
    lines.append(stumps.head(6).to_string(index=False))
    lines.append("")
    lines.append("DEPTH-3 DECISION TREE")
    lines.append("-" * 78)
    lines.append(tree_text)
    lines.append("AFTER DEDUPLICATION  (5-fold cross-validated balanced accuracy)")
    lines.append("-" * 78)
    lines.append(f"Rows: {len(df):,} -> {len(deduped):,}   class balance {yd.value_counts().to_dict()}")
    for k, v in dedup_scores.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    verdicts = []
    d3 = float(base.loc[base["classifier"] == "Decision tree, depth 3", "balanced_accuracy"].iloc[0])
    if d3 >= 0.95:
        verdicts.append(
            f"A depth-3 decision tree reaches {d3:.4f} balanced accuracy. The classes are "
            "separable by three thresholds, so no complex model is being meaningfully tested."
        )
    hard = ranges[~ranges["ranges_overlap"]] if not ranges.empty else pd.DataFrame()
    for _, r in hard.iterrows():
        verdicts.append(
            f"'{r['attribute']}' has non-overlapping class ranges "
            f"({r['positive_range']} vs {r['negative_range']}): a hard threshold, not a tendency."
        )
    wide = ranges[ranges["sd_ratio"] > 5] if not ranges.empty else pd.DataFrame()
    for _, r in wide.iterrows():
        verdicts.append(
            f"'{r['attribute']}' varies {r['sd_ratio']:.0f}x more in one class than the other "
            f"(sd {r['sd_positive']} vs {r['sd_negative']}): consistent with generated data."
        )
    for _, r in determ.iterrows():
        verdicts.append(
            f"'{r['attribute']}' = '{r['level']}' maps to class {r['always_class']} in all "
            f"{r['n']} records, without exception."
        )
    if dup_by_class and len(dup_by_class) == 1:
        verdicts.append(
            f"All {int(dup_mask.sum())} duplicated rows belong to a single class "
            f"({list(dup_by_class)[0]}): consistent with a small sample padded by copying."
        )

    lines.append("FINDINGS")
    lines.append("=" * 78)
    if verdicts:
        for i, v in enumerate(verdicts, 1):
            lines.append(f"{i}. {v}")
        lines.append("")
        lines.append(
            "Treat reported accuracy on this dataset with caution: report trivial baselines "
            "alongside any model result."
        )
    else:
        lines.append("No structural artefacts detected by these checks.")

    report = "\n".join(lines)
    print(report)
    (report_dir / f"audit_{stem}.txt").write_text(report, encoding="utf-8")
    assoc.to_csv(report_dir / f"association_{stem}.csv", index=False)
    base.to_csv(report_dir / f"baselines_{stem}.csv", index=False)
    if not ranges.empty:
        ranges.to_csv(report_dir / f"ranges_{stem}.csv", index=False)
    summary = {
        "file": path.name,
        "rows": int(len(df)),
        "attributes": int(df.shape[1]),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().items()},
        "duplicate_rows": int(dup_mask.sum()),
        "duplicates_by_class": {str(k): int(v) for k, v in dup_by_class.items()},
        "baselines": base.to_dict("records"),
        "deduplicated_cv": dedup_scores,
        "findings": verdicts,
    }
    (report_dir / f"audit_{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten to {report_dir}")


if __name__ == "__main__":
    main()
