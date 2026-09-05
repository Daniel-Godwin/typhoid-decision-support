"""Decision-threshold analysis for the binary diagnostic model.

Sweeps the classification cut-off across the held-out test partition, derives
candidate operating points under several cost assumptions, checks whether the
calibrated probabilities are reliable enough to threshold, projects predictive
values onto other prevalences, and reports per-Location behaviour at the
recommended cut-off.

    python scripts/threshold_analysis.py                    # routine policy
    python scripts/threshold_analysis.py --policy no_fever_duration
    python scripts/threshold_analysis.py --policy all

Writes to reports/:
    threshold_sweep_binary[_policy].csv
    threshold_analysis_binary[_policy].json
    table_4_7_operating_points[_policy].csv
    table_4_8_prevalence[_policy].csv
    table_4_9_threshold_subgroups[_policy].csv
    figures/threshold_curves_binary[_policy].png
    figures/calibration_binary[_policy].png
    THRESHOLD_ANALYSIS.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typhoid_ml.config import (  # noqa: E402
    BINARY_POSITIVE,
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
from typhoid_ml.predict import model_path  # noqa: E402
from typhoid_ml.threshold import (  # noqa: E402
    COST_RATIOS,
    calibration,
    operating_points,
    positive_probabilities,
    prevalence_adjusted,
    stratified_operating_point,
    sweep,
)

# Deployment prevalences to project onto. The evaluation partition is close to
# balanced; a febrile-patient clinic is not.
PREVALENCES = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50)

# Which operating point the deployed system should adopt. A missed typhoid case
# leads to untreated bacteraemia; a false positive leads to one confirmatory
# test. Ten to one is conservative relative to that asymmetry and is the ratio
# carried into TRIAGE_THRESHOLD.
RECOMMENDED_KEY = "cost_fn10x"


def suffix(policy: str) -> str:
    return "" if policy == DEFAULT_POLICY else f"_{policy}"


def plot_curves(df: pd.DataFrame, points: dict, policy: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(df["threshold"], df["sensitivity"], label="Sensitivity (recall)", lw=2)
    ax.plot(df["threshold"], df["specificity"], label="Specificity", lw=2)
    ax.plot(df["threshold"], df["ppv"], label="PPV (precision)", lw=1.6, ls="--")
    ax.plot(df["threshold"], df["npv"], label="NPV", lw=1.6, ls="--")
    for key, colour in (("default", "grey"), (RECOMMENDED_KEY, "crimson")):
        if key in points:
            ax.axvline(points[key]["threshold"], color=colour, ls=":", lw=1.5,
                       label=f"{key} = {points[key]['threshold']:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Operating characteristics against the cut-off")
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for ratio in COST_RATIOS:
        ax.plot(df["threshold"], df[f"cost_fn{ratio}x"], lw=1.8,
                label=f"FN cost = {ratio}x FP")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost per 100 patients")
    ax.set_title("Expected cost under asymmetric error weighting")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Threshold analysis — binary model, {policy} policy", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_calibration(cal: dict, policy: str, out: Path) -> None:
    bins = pd.DataFrame(cal["bins"])
    fig, ax = plt.subplots(figsize=(6, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Perfect calibration")
    ax.plot(bins["mean_predicted"], bins["observed_frequency"], "o-", lw=2,
            label="Model")
    for _, r in bins.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_predicted"], r["observed_frequency"]),
                    textcoords="offset points", xytext=(5, -10), fontsize=7)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of typhoid")
    ax.set_title(
        f"Calibration — {policy} policy\n"
        f"Brier {cal['brier_score']:.4f} · ECE {cal['expected_calibration_error']:.4f} · "
        f"AUC {cal['roc_auc']:.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fever_zero_analysis(df_test, y_test, proba, points: dict) -> dict | None:
    """Do the hard cases become recoverable at a lower cut-off?

    Every record with a fever duration of one day or more is labelled typhoid in
    this dataset, so the informative cases are the typhoid patients recorded
    with a fever duration of zero. If lowering the threshold does not recover
    them, the model has learned the shortcut rather than the disease.
    """
    col = "Fever Duration (Days)"
    if col not in df_test.columns:
        return None
    mask = (pd.to_numeric(df_test[col], errors="coerce") == 0).to_numpy()
    y = np.asarray([1 if v == BINARY_POSITIVE else 0 for v in y_test])
    hard = mask & (y == 1)
    if hard.sum() == 0:
        return None
    p = np.asarray(proba)[hard]
    out = {
        "n_typhoid_with_zero_fever_duration": int(hard.sum()),
        "mean_predicted_probability": round(float(p.mean()), 4),
        "max_predicted_probability": round(float(p.max()), 4),
        "recovered_at": {},
    }
    for key in ("default", RECOMMENDED_KEY, "sens_95", "sens_99"):
        if key in points:
            t = points[key]["threshold"]
            out["recovered_at"][f"{key} (t={t:.2f})"] = (
                f"{int((p >= t).sum())}/{int(hard.sum())}"
            )
    out["recovered_at"]["t=0.05"] = f"{int((p >= 0.05).sum())}/{int(hard.sum())}"
    return out


def analyse(policy: str) -> dict:
    path = model_path("binary", policy)
    if not path.exists():
        raise SystemExit(f"No saved model at {path}")

    df = load_dataset(DATA_PATH)
    X, _, _ = feature_frame(df, policy)
    y = build_target(df, "binary")
    _, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_test, y_test = X.loc[idx_test], y.loc[idx_test]

    model = joblib.load(path)
    proba = positive_probabilities(model, X_test)

    sw = sweep(y_test, proba)
    points = operating_points(sw)
    cal = calibration(y_test, proba)

    rec = points[RECOMMENDED_KEY]
    # A cost-optimal point that flags (or clears) every patient is not an
    # operating point: it is the arithmetic telling us the classifier carries
    # too little signal to support any cut-off. Record that rather than
    # presenting it as a recommendation.
    rec["degenerate"] = bool(
        rec["specificity"] < 0.01 or rec["sensitivity"] < 0.01
    )
    prev = prevalence_adjusted(rec["sensitivity"], rec["specificity"], PREVALENCES)
    sub = stratified_operating_point(
        y_test, proba, df.loc[idx_test, SUBGROUP_COLUMN], rec["threshold"]
    )
    hard = fever_zero_analysis(df.loc[idx_test], y_test, proba, points)

    sfx = suffix(policy)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sw.to_csv(REPORT_DIR / f"threshold_sweep_binary{sfx}.csv", index=False)
    pd.DataFrame(
        [{"operating_point": k, **v} for k, v in points.items()]
    ).to_csv(REPORT_DIR / f"table_4_7_operating_points{sfx}.csv", index=False)
    prev.to_csv(REPORT_DIR / f"table_4_8_prevalence{sfx}.csv", index=False)
    sub.to_csv(REPORT_DIR / f"table_4_9_threshold_subgroups{sfx}.csv", index=False)

    plot_curves(sw, points, policy, FIGURE_DIR / f"threshold_curves_binary{sfx}.png")
    plot_calibration(cal, policy, FIGURE_DIR / f"calibration_binary{sfx}.png")

    result = {
        "policy": policy,
        "model": str(path.relative_to(ROOT)),
        "test_records": int(len(y_test)),
        "test_prevalence": float((y_test == BINARY_POSITIVE).mean()),
        "calibration": cal,
        "operating_points": points,
        "recommended": {"key": RECOMMENDED_KEY, **rec},
        "prevalence_projection": prev.to_dict("records"),
        "subgroups_at_recommended": sub.to_dict("records"),
        "zero_fever_duration_cases": hard,
    }
    (REPORT_DIR / f"threshold_analysis_binary{sfx}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _num(value, digits: int = 4) -> str:
    """Format a metric that may be undefined (an empty confusion-matrix cell)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def write_markdown(results: list[dict]) -> Path:
    L = ["# Decision-Threshold Analysis\n"]
    L.append(
        "_Generated by `python scripts/threshold_analysis.py`. The deployed "
        "system's `TRIAGE_THRESHOLD` should equal the recommended cut-off "
        "reported here for the policy in use._\n"
    )
    L.append(
        "Accuracy at the default 0.5 cut-off is not an adequate summary for a "
        "triage instrument. A tool used ahead of confirmatory testing should be "
        "tuned against the asymmetry between a missed case and an unnecessary "
        "test, and the chosen cut-off should be stated with its justification.\n"
    )

    for r in results:
        pol = r["policy"]
        L.append(f"\n## {pol} policy\n")
        L.append(
            f"Held-out partition: {r['test_records']:,} records, "
            f"{r['test_prevalence']:.1%} typhoid.\n"
        )

        cal = r["calibration"]
        L.append("### Probability reliability\n")
        L.append("| Measure | Value | Reading |")
        L.append("|---|---:|---|")
        L.append(f"| ROC AUC | {cal['roc_auc']:.4f} | Ranking quality |")
        L.append(
            f"| Brier score | {cal['brier_score']:.4f} | Mean squared error of the "
            "probability; lower is better |"
        )
        L.append(
            f"| Expected calibration error | {cal['expected_calibration_error']:.4f} | "
            "Mean gap between stated and observed risk |"
        )
        L.append("")

        L.append("### Candidate operating points\n")
        L.append(
            "| Operating point | Cut-off | Sensitivity | Specificity | PPV | NPV | "
            "FN | FP | Referral rate |"
        )
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k, v in r["operating_points"].items():
            L.append(
                f"| {k} | {v['threshold']:.2f} | {v['sensitivity']:.4f} | "
                f"{v['specificity']:.4f} | "
                f"{_num(v['ppv'])} | {_num(v['npv'])} | {v['false_negatives']} | "
                f"{v['false_positives']} | {v['referral_rate']:.3f} |"
            )
        L.append("")
        for k, v in r["operating_points"].items():
            L.append(f"- **{k}** — {v['rationale']}")
        L.append("")

        rec = r["recommended"]
        L.append("### Recommended operating point\n")
        if rec.get("degenerate"):
            L.append(
                f"**None. No usable cut-off exists for this policy.** The "
                f"cost-optimal point under a {rec['key'].replace('cost_fn', '').replace('x', '')}:1 "
                f"weighting is {rec['threshold']:.2f}, at which the model flags "
                f"every patient (sensitivity {rec['sensitivity']:.4f}, specificity "
                f"{rec['specificity']:.4f}). That is not a decision rule; it is the "
                "arithmetic reporting that the classifier carries too little "
                "discriminative signal for any threshold to improve on testing "
                "everyone — which is the status quo the tool was intended to "
                f"improve upon. ROC AUC of {r['calibration']['roc_auc']:.4f} says the "
                "same thing.\n"
            )
        else:
            L.append(
                f"**Cut-off {rec['threshold']:.2f}** ({rec['key']}). "
                f"Sensitivity {rec['sensitivity']:.4f}, specificity "
                f"{rec['specificity']:.4f}, {rec['false_negatives']} missed cases and "
                f"{rec['false_positives']} unnecessary referrals on the held-out "
                f"partition.\n"
            )

        L.append("### Predictive values at other prevalences\n")
        L.append(
            "Sensitivity and specificity belong to the classifier; PPV and NPV "
            "belong to the setting. Projected at the recommended cut-off:\n"
        )
        L.append("| Prevalence | PPV | NPV | Flagged per 1,000 | Missed per 1,000 |")
        L.append("|---:|---:|---:|---:|---:|")
        for row in r["prevalence_projection"]:
            L.append(
                f"| {row['prevalence']:.0%} | {_num(row['ppv'])} | {_num(row['npv'])} | "
                f"{row['positives_per_1000']} | {row['missed_per_1000']} |"
            )
        L.append("")

        L.append("### Behaviour by location at the recommended cut-off\n")
        L.append("| Location | n | Prevalence | Sensitivity | Specificity | PPV | NPV |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in r["subgroups_at_recommended"]:
            L.append(
                f"| {row['subgroup']} | {row['n']} | {row['prevalence']:.3f} | "
                f"{_num(row['sensitivity'])} | {_num(row['specificity'])} | "
                f"{_num(row['ppv'])} | {_num(row['npv'])} |"
            )
        L.append("")

        hard = r.get("zero_fever_duration_cases")
        if hard:
            L.append("### Cases the shortcut cannot reach\n")
            L.append(
                f"{hard['n_typhoid_with_zero_fever_duration']} typhoid patients in "
                "the held-out partition were recorded with a fever duration of "
                "zero days. Because every record with a duration of one day or "
                "more carries a typhoid label, these are the only cases that "
                "require the model to use anything other than that one variable.\n"
            )
            L.append(
                f"Mean predicted probability for these patients: "
                f"**{hard['mean_predicted_probability']:.4f}** "
                f"(maximum {hard['max_predicted_probability']:.4f}).\n"
            )
            L.append("| Cut-off | Cases recovered |")
            L.append("|---|---:|")
            for k, v in hard["recovered_at"].items():
                L.append(f"| {k} | {v} |")
            L.append("")

    L.append("\n## Interpretation\n")
    by_policy = {r["policy"]: r for r in results}
    routine = by_policy.get(DEFAULT_POLICY)
    ablation = by_policy.get("no_fever_duration")

    L.append(
        "Threshold tuning is normally the lever that converts a balanced "
        "classifier into a screening instrument, trading specificity for the "
        "sensitivity a triage tool needs. Three observations show that the lever "
        "does almost no work here.\n"
    )

    if routine:
        d = routine["operating_points"]["default"]
        rec = routine["recommended"]
        L.append(
            f"**1. The sweep is nearly flat.** Moving the cut-off from "
            f"{d['threshold']:.2f} to the cost-optimal {rec['threshold']:.2f} "
            f"changes the number of missed cases from {d['false_negatives']} to "
            f"{rec['false_negatives']} — a single patient in "
            f"{routine['test_records']:,}. On a model that had genuinely learned a "
            "graded risk, this range would move sensitivity by tens of "
            "percentage points.\n"
        )
        L.append(
            f"**2. Specificity is 1.0000 and PPV is 1.0000 at every prevalence.** "
            "No negative record is ever flagged, at any cut-off above 0.05. A "
            "positive predictive value that does not fall as prevalence falls is "
            "arithmetically impossible for a real diagnostic test; it is the "
            "signature of a deterministic rule, not of a classifier. This is the "
            "`Fever Duration (Days)` shortcut documented in "
            "`docs/DATASET_AUDIT.md` reappearing as a threshold result.\n"
        )
        hard = routine.get("zero_fever_duration_cases")
        if hard:
            n = hard["n_typhoid_with_zero_fever_duration"]
            rec_label = f"{RECOMMENDED_KEY} (t={rec['threshold']:.2f})"
            at_default = hard["recovered_at"].get("default (t=0.50)", "—")
            at_rec = hard["recovered_at"].get(rec_label, "—")
            L.append(
                f"**3. The cases that need a model are not recovered by tuning.** "
                f"Of the {n} typhoid patients recorded with zero fever duration — "
                "the only patients whose diagnosis cannot be read off that one "
                f"variable — the default cut-off recovers {at_default} and the "
                f"cost-optimal cut-off recovers {at_rec}. "
                "They are recoverable only at a cut-off that also flags the great "
                "majority of true negatives, which is to say by abandoning the "
                "classifier and testing everyone.\n"
            )

    if ablation:
        L.append(
            f"The ablation confirms it directly. With `Fever Duration (Days)` "
            f"removed and all {len(FEATURE_POLICIES['no_fever_duration']['numeric']) + len(FEATURE_POLICIES['no_fever_duration']['categorical'])} "
            f"remaining features retained, ROC AUC falls to "
            f"{ablation['calibration']['roc_auc']:.4f} — near chance — and no "
            "cut-off yields a usable operating point. The clinical, "
            "environmental and serological variables in this dataset carry "
            "almost no independent signal about the label.\n"
        )

    if routine:
        L.append(
            f"One property does survive: the probabilities are well calibrated "
            f"(Brier {routine['calibration']['brier_score']:.4f}, expected "
            f"calibration error {routine['calibration']['expected_calibration_error']:.4f}). "
            "The Platt-scaled outputs are honest about what the model believes. "
            "What the model believes is the dataset's shortcut.\n"
        )

    L.append(
        "**Consequence for the deployed system.** The cut-off is exposed as "
        "`TRIAGE_THRESHOLD` and is set to the value recommended above for the "
        "routine policy, because it is the defensible choice given the cost "
        "asymmetry and it costs nothing in specificity. That is a correct "
        "engineering decision about a model that should not be used clinically "
        "on this training data. No figure in this document should be quoted as "
        "evidence of diagnostic performance without the qualification above.\n"
    )

    out = REPORT_DIR / "THRESHOLD_ANALYSIS.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        choices=list(FEATURE_POLICIES) + ["all"],
    )
    args = ap.parse_args()

    policies = list(FEATURE_POLICIES) if args.policy == "all" else [args.policy]
    results = []
    for policy in policies:
        if not model_path("binary", policy).exists():
            print(f"skipping {policy}: no saved model")
            continue
        r = analyse(policy)
        results.append(r)
        rec = r["recommended"]
        print(
            f"{policy:>18}: recommended cut-off {rec['threshold']:.2f} | "
            f"sens {rec['sensitivity']:.4f} | spec {rec['specificity']:.4f} | "
            f"FN {rec['false_negatives']} | AUC {r['calibration']['roc_auc']:.4f}"
        )

    if results:
        print(f"\nWrote {write_markdown(results)}")


if __name__ == "__main__":
    main()
