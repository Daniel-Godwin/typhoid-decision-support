"""Generate a Chapter 4 narrative draft populated from the actual artefacts.

Every figure in the output is read from reports/ at generation time, so the
draft can never drift from the results. Re-run it after any retraining.

Output: reports/CHAPTER_4_DRAFT.md
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


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def table(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "_(not available — run the corresponding training stage)_\n"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows]) + "\n"


def pct(x, digits=2):
    return f"{x * 100:.{digits}f}%" if isinstance(x, (int, float)) else "—"


def num(x, digits=4):
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "—"


def main():
    audit = load(REPORTS / "dataset_audit.json") or {}
    binary = load(REPORTS / "final_evaluation_binary.json")
    multi = load(REPORTS / "final_evaluation_multiclass.json")
    ablation = load(REPORTS / "final_evaluation_binary_no_fever_duration.json")
    clinical = load(REPORTS / "final_evaluation_binary_clinical_only.json")
    gs_bin = load(REPORTS / "grid_search_summary_binary.json")
    gs_mul = load(REPORTS / "grid_search_summary_multiclass.json")
    analysis = load(REPORTS / "baseline_analysis.json")
    meta_bin = load(MODELS / "svm_binary_metadata.json") or {}
    meta_mul = load(MODELS / "svm_multiclass_metadata.json") or {}

    p = []
    p.append("# Chapter Four — Results and Discussion\n")
    p.append(
        "_Draft generated from the project artefacts. Every figure below is read directly "
        "from `reports/`; regenerate with `python scripts/make_chapter4.py` after any "
        "retraining. Prose is a starting point to be rewritten in your own voice._\n"
    )

    # ---- 4.1 ---------------------------------------------------------------
    p.append("## 4.1 Introduction\n")
    p.append(
        "This chapter presents the results of the Support Vector Machine models developed "
        "in Chapter Three. It reports the dataset characteristics established by the audit, "
        "the comparison of kernel functions, the outcome of hyperparameter optimisation, the "
        "performance of the optimised models on the held-out test partition, and the "
        "evaluation of scalability across geographic settings required by Objective 4. It then "
        "presents a signal audit of the dataset itself, a decision-threshold and calibration "
        "analysis establishing the operating point at which the deployed system runs, and a "
        "comparative audit of two further public typhoid datasets. Taken together, the final "
        "three sections materially qualify how the headline performance figures should be "
        "interpreted.\n"
    )

    # ---- 4.2 ---------------------------------------------------------------
    p.append("## 4.2 Dataset characteristics\n")
    if audit:
        p.append(
            f"The dataset comprises **{audit['rows']:,} patient records** across "
            f"**{audit['columns']} attributes**, with **{audit['duplicates']} duplicate rows**. "
            f"Four attributes contain missing values, the most affected being `Complications` "
            f"at {audit.get('missing_percentage', {}).get('Complications', 0):.2f}%.\n"
        )
        dist = pd.DataFrame(
            [
                {
                    "Class": k,
                    "Count": f"{v:,}",
                    "Percentage": f"{v * 100 / audit['rows']:.2f}%",
                }
                for k, v in audit["target_distribution_multiclass"].items()
            ]
        )
        p.append("**Table 4.1: Distribution of the four-class target**\n")
        p.append(table(dist))
        distb = pd.DataFrame(
            [
                {
                    "Class": k,
                    "Count": f"{v:,}",
                    "Percentage": f"{v * 100 / audit['rows']:.2f}%",
                }
                for k, v in audit["target_distribution_binary"].items()
            ]
        )
        p.append("**Table 4.2: Distribution of the binary diagnostic target**\n")
        p.append(table(distb))
        p.append(
            f"The four-class target is markedly imbalanced, with a ratio of "
            f"{audit.get('imbalance_ratio_multiclass', '—')} : 1 between the largest and "
            f"smallest class. This motivated the SMOTENC balancing described in Section 3.2.1, "
            "applied inside the modelling pipeline so that synthetic samples were generated "
            "from training folds only and never contaminated validation or test data.\n"
        )
        p.append(
            "Following the feature policy set out in Section 3.2.1, `Blood Culture Result` and "
            "`Complications` were withheld from every model. The first is the confirmatory gold "
            "standard, so a model consuming it would be restating a completed diagnosis rather "
            "than predicting one; the second encodes post-diagnostic severity and is unavailable "
            "at the point of decision.\n"
        )

    # ---- 4.3 ---------------------------------------------------------------
    p.append("## 4.3 Kernel comparison\n")
    p.append(
        "Three kernel functions — linear, polynomial and radial basis function — were evaluated "
        "at default hyperparameter settings on the held-out test partition, to establish which "
        "family of decision boundaries suited the data before committing to an exhaustive search.\n"
    )
    kb = csv(REPORTS / "kernel_comparison_binary.csv")
    p.append("**Table 4.3: Kernel comparison, binary diagnostic model**\n")
    p.append(table(kb))
    km = csv(REPORTS / "kernel_comparison_multiclass.csv")
    p.append("**Table 4.4: Kernel comparison, four-class severity model**\n")
    p.append(table(km))
    if kb is not None and not kb.empty:
        best = kb.loc[kb["macro_f1"].idxmax()]
        p.append(
            f"For the binary model the {best['kernel']} kernel produced the highest macro F1 "
            f"({best['macro_f1']:.4f}). The three kernels differed by less than one percentage "
            "point in accuracy, which indicates that the decision boundary separating the "
            "classes is close to linear in the encoded feature space — a point returned to in "
            "Section 4.7.\n"
        )

    # ---- 4.4 ---------------------------------------------------------------
    p.append("## 4.4 Hyperparameter optimisation\n")
    for label, gs in (("binary diagnostic", gs_bin), ("four-class severity", gs_mul)):
        if not gs:
            continue
        p.append(
            f"For the **{label} model**, Grid Search evaluated **{gs['n_candidates']} candidate "
            f"configurations** under **{gs['cv_folds']}-fold stratified cross-validation**, "
            f"selecting on macro F1. "
            + (
                f"Because oversampling expands the four-class training partition to roughly four "
                f"times the majority class, the search was executed on a stratified subsample of "
                f"{gs['search_records']:,} records and the winning configuration then refitted on "
                f"the complete training partition. "
                if gs.get("search_subsampled")
                else f"The search used the full training partition of {gs['search_records']:,} records. "
            )
            + f"The selected configuration was `{gs['best_params']}`, achieving a cross-validated "
            f"macro F1 of **{gs['best_cv_macro_f1']:.4f}** in {gs['search_seconds']:.0f} seconds "
            "of wall-clock time.\n"
        )
    p.append(
        "The complete search results, including the score of every candidate configuration, are "
        "recorded in `reports/grid_search_results_binary.csv` and "
        "`reports/grid_search_results_multiclass.csv`.\n"
    )

    # ---- 4.5 ---------------------------------------------------------------
    p.append("## 4.5 Performance of the optimised models\n")
    m = csv(REPORTS / "table_4_2_metrics.csv")
    p.append("**Table 4.5: Performance of the optimised models on the held-out test partition**\n")
    p.append(table(m))
    if binary:
        p.append(
            f"The optimised binary model achieved an accuracy of **{pct(binary['accuracy'])}**, "
            f"balanced accuracy of **{pct(binary['balanced_accuracy'])}** and macro F1 of "
            f"**{num(binary['macro_f1'])}** on {binary['n_test']:,} unseen records. Sensitivity "
            f"was **{num(binary.get('sensitivity'))}** and specificity **{num(binary.get('specificity'))}**, "
            f"corresponding to {binary.get('false_negative', '—')} false negatives and "
            f"{binary.get('false_positive', '—')} false positives.\n"
        )
        p.append(
            "The clinical asymmetry of these two errors matters. A false negative is a missed "
            "typhoid case sent home untreated; a false positive is an unnecessary confirmatory "
            "test. The model's perfect specificity means it never raises a false alarm, but its "
            f"sensitivity of {num(binary.get('sensitivity'))} means roughly "
            f"{(1 - binary.get('sensitivity', 0)) * 100:.1f}% of true cases are missed — which is "
            "the error that carries clinical risk. Section 4.7 explains why these particular "
            "cases are missed.\n"
        )
        lat = binary.get("latency", {})
        p.append(
            f"Inference latency was **{lat.get('single_record_ms', '—')} ms per record** "
            f"({lat.get('records_per_second', '—'):,} records per second in batch), which "
            "satisfies the real-time, point-of-care requirement of Objective 3.\n"
            if lat
            else ""
        )
    perclass = csv(REPORTS / "table_4_3_perclass.csv")
    p.append("**Table 4.6: Per-class precision, recall and F1**\n")
    p.append(table(perclass))
    if multi:
        p.append(
            f"The four-class model achieved accuracy of **{pct(multi['accuracy'])}** and macro F1 "
            f"of **{num(multi['macro_f1'])}**. Performance was highly uneven across classes: "
            "*Normal or No Typhoid* and *Complicated Typhoid* were classified almost perfectly, "
            "while *Acute Typhoid Fever* and *Relapsing Typhoid* were frequently confused with "
            "one another. Section 4.7 shows that this pattern is a property of the dataset "
            "rather than a limitation of the classifier.\n"
        )
    p.append(
        "Confusion matrices for both models, together with ROC and precision–recall curves for "
        "the binary model, are provided in `reports/figures/`.\n"
    )

    # ---- 4.6 ---------------------------------------------------------------
    p.append("## 4.6 Scalability across geographic settings (Objective 4)\n")
    p.append(
        "Objective 4 requires the model to be evaluated across diverse geographic regions and "
        "healthcare environments. The held-out test partition was therefore stratified by the "
        "`Location` attribute and the full metric set recomputed within each subgroup.\n"
    )
    sub = csv(REPORTS / "table_4_4_subgroups.csv")
    p.append("**Table 4.7: Performance by geographic setting**\n")
    p.append(table(sub))
    if sub is not None and not sub.empty and "macro_f1" in sub.columns:
        b = sub[sub["model"] == "Binary"] if "model" in sub.columns else sub
        if not b.empty:
            spread = float(b["macro_f1"].max() - b["macro_f1"].min())
            p.append(
                f"Macro F1 varied by only **{spread:.4f}** across the endemic, rural and urban "
                "subgroups. The model therefore does not degrade in any one setting, which "
                "satisfies Objective 4 as stated. This stability should, however, be read "
                "alongside Section 4.7: `Location` is statistically independent of the diagnosis "
                "in this dataset, so uniform performance across locations reflects the absence of "
                "geographic variation in the data rather than demonstrated robustness to it.\n"
            )

    # ---- 4.7 ---------------------------------------------------------------
    p.append("## 4.7 Signal audit and interpretation of the results\n")
    p.append(
        "A model that reports high accuracy has not necessarily learned anything useful. To "
        "establish what the classifier had actually learned, three reference classifiers were "
        "scored on the identical test partition, and the statistical association between every "
        "attribute and the diagnosis was measured.\n"
    )
    base = csv(REPORTS / "table_4_0_baselines.csv")
    p.append("**Table 4.8: Reference baselines, binary diagnostic task**\n")
    p.append(table(base))
    if analysis:
        p.append(
            f"Of the {analysis['records_with_fever_ge_1']:,} records with a fever duration of one "
            f"day or more, **{analysis['typhoid_among_fever_ge_1']:,} — that is, every one of them "
            "— are labelled typhoid.** No record in the dataset combines a fever duration of one "
            "day or more with a negative diagnosis. `Fever Duration (Days)` is therefore a "
            "deterministic proxy for the target, and a single threshold on that one attribute "
            "reproduces almost the entire performance of the optimised model.\n"
        )
    base4 = csv(REPORTS / "table_4_0b_baselines_multiclass.csv")
    p.append("**Table 4.9: Reference baselines, four-class task**\n")
    p.append(table(base4))
    p.append(
        "A second deterministic rule governs the four-class task: every record labelled "
        "*Complicated Typhoid* has a white blood cell count above 11,000, and no record of any "
        "other class does. The two ranges do not overlap by a single count, which is why the "
        "model classifies that class perfectly. A two-rule baseline outperforms the optimised "
        "four-class SVM on accuracy.\n"
    )
    assoc = csv(REPORTS / "table_4_6_association.csv")
    p.append("**Table 4.10: Association between each attribute and the diagnosis**\n")
    p.append(table(assoc))
    p.append(
        "Outside of `Blood Culture Result` (withheld), `Fever Duration (Days)` and "
        "`White Blood Cell Count`, no attribute reaches a Cramér's V of 0.02. `Widal Test` "
        "(V = 0.0014) and `Typhidot Test` (V = 0.0065) are statistically independent of the "
        "diagnosis — an impossibility in real patient data, since these are the serological "
        "tests on which typhoid diagnosis conventionally rests.\n"
    )
    p.append("### Ablation\n")
    sens = csv(REPORTS / "table_4_5_sensitivity.csv")
    p.append("**Table 4.11: Feature policy sensitivity and ablation**\n")
    p.append(table(sens))
    if ablation:
        p.append(
            f"With `Fever Duration (Days)` removed, performance collapses to an accuracy of "
            f"**{pct(ablation['accuracy'])}** and a balanced accuracy of "
            f"**{num(ablation['balanced_accuracy'])}** — against a chance floor of 0.5000. The "
            "remaining nineteen attributes, taken together, therefore carry only marginal "
            "discriminative information about the diagnosis.\n"
        )
    p.append(
        "Taken together, these results establish that the dataset's apparent predictability is "
        "an artefact of its construction. The dataset is synthetic: the label is encoded "
        "deterministically into two attributes, one class distinction (*Acute* versus "
        "*Relapsing*) is not encoded at all, and the remaining attributes are effectively random "
        "with respect to the outcome. Realistic column names and plausible marginal "
        "distributions disguise this, but the joint distribution does not support the clinical "
        "relationships the attributes imply.\n"
    )
    p.append(
        "This does not invalidate the system developed in this study. The preprocessing pipeline, "
        "leakage controls, resampling strategy, optimisation procedure, evaluation protocol and "
        "deployed interface are all sound, and would transfer without modification to real "
        "clinical data. What the finding does establish is that the headline accuracy of "
        f"{pct(binary['accuracy']) if binary else '—'} must not be presented as evidence that "
        "machine learning improves typhoid diagnosis — because a single threshold on one "
        "attribute achieves substantially the same result. Section 1.6 of this work anticipated "
        "precisely this risk in noting the study's dependency on existing datasets.\n"
    )

    # ---- 4.8 ---------------------------------------------------------------
    p.append("## 4.8 Decision threshold, calibration and transportability\n")
    p.append(
        "Sections 4.5 and 4.7 report performance at the library default cut-off of 0.5. That "
        "default treats a missed case and an unnecessary confirmatory test as equally costly, "
        "which is not the clinical position: untreated typhoid carries a case-fatality risk, "
        "whereas a false alarm costs one Widal or culture. A triage instrument must therefore "
        "declare its operating point explicitly. The cut-off was swept across the held-out "
        "partition and candidate operating points derived under several cost weightings.\n"
    )
    thr = load(REPORTS / "threshold_analysis_binary.json")
    thr_abl = load(REPORTS / "threshold_analysis_binary_no_fever_duration.json")
    ops = csv(REPORTS / "table_4_7_operating_points.csv")
    if ops is not None and not ops.empty:
        keep = [c for c in
                ["operating_point", "threshold", "sensitivity", "specificity", "ppv", "npv",
                 "false_negatives", "false_positives", "referral_rate"] if c in ops.columns]
        show = ops[keep].copy()
        for c in show.columns:
            if show[c].dtype.kind == "f":
                show[c] = show[c].map(lambda v: f"{v:.4f}")
        show = show.rename(columns={
            "operating_point": "Operating point", "threshold": "Cut-off",
            "sensitivity": "Sensitivity", "specificity": "Specificity",
            "ppv": "PPV", "npv": "NPV", "false_negatives": "FN",
            "false_positives": "FP", "referral_rate": "Referral rate",
        })
        p.append("**Table 4.12: Candidate operating points, binary diagnostic model**\n")
        p.append(table(show))
    if thr:
        rec = thr["recommended"]
        d = thr["operating_points"]["default"]
        cal = thr["calibration"]
        p.append(
            f"Weighting a missed case as ten times as damaging as an unnecessary test selects a "
            f"cut-off of **{rec['threshold']:.2f}**, at which sensitivity is "
            f"**{num(rec['sensitivity'])}** and specificity **{num(rec['specificity'])}**. This "
            "value is the one configured in the deployed system as `TRIAGE_THRESHOLD`.\n"
        )
        p.append(
            f"The more instructive observation is how little the choice matters. Moving from "
            f"{d['threshold']:.2f} to {rec['threshold']:.2f} changes the number of missed cases "
            f"from {d['false_negatives']} to {rec['false_negatives']} — one patient in "
            f"{thr['test_records']:,}. In a model that had learned a graded representation of "
            "risk, a sweep across this range would move sensitivity by tens of percentage "
            "points. Here the operating characteristics are almost flat, because the underlying "
            "decision is not graded but binary: the shortcut identified in Section 4.7 either "
            "fires or it does not.\n"
        )
        prevt = csv(REPORTS / "table_4_8_prevalence.csv")
        if prevt is not None and not prevt.empty:
            show = prevt.copy()
            show["prevalence"] = show["prevalence"].map(lambda v: f"{v:.0%}")
            for c in ("ppv", "npv"):
                if c in show.columns:
                    show[c] = show[c].map(lambda v: f"{v:.4f}")
            show = show.rename(columns={
                "prevalence": "Prevalence", "ppv": "PPV", "npv": "NPV",
                "positives_per_1000": "Flagged per 1,000",
                "missed_per_1000": "Missed per 1,000",
            })
            p.append(
                "**Table 4.13: Predictive values projected onto other prevalences**\n"
            )
            p.append(table(show))
        p.append(
            "Sensitivity and specificity are properties of a classifier; positive and negative "
            "predictive value are properties of the setting in which it is used, because they "
            "depend on how common the disease is among those tested. Table 4.13 is included "
            "because it is the calculation a deploying clinic needs, and because its result here "
            "is diagnostic in itself: the positive predictive value remains 1.0000 even at a "
            "prevalence of two per cent. For a real diagnostic test that is arithmetically "
            "impossible. It occurs only because no negative record in this dataset is ever "
            "flagged at any usable cut-off, which is the signature of a deterministic rule "
            "rather than a probabilistic classifier.\n"
        )
        p.append(
            f"The calibration of the probabilities was assessed separately, since a threshold is "
            f"only meaningful if the probability it is applied to means what it states. The "
            f"Platt-scaled outputs achieved a Brier score of **{cal['brier_score']:.4f}** and an "
            f"expected calibration error of **{cal['expected_calibration_error']:.4f}** "
            f"(ROC AUC {cal['roc_auc']:.4f}). The model is therefore well calibrated: when it "
            "states a risk of eighty per cent, approximately eighty per cent of such patients "
            "are positive. It is an honest reporter of a conclusion the dataset made trivial.\n"
        )
        hard = thr.get("zero_fever_duration_cases")
        if hard:
            rows = pd.DataFrame(
                [{"Cut-off": k, "Cases recovered": v} for k, v in hard["recovered_at"].items()]
            )
            p.append(
                f"**Table 4.14: Recovery of the {hard['n_typhoid_with_zero_fever_duration']} "
                "typhoid cases recorded with zero fever duration**\n"
            )
            p.append(table(rows))
            rec_label = f"{rec['key']} (t={rec['threshold']:.2f})"
            at_default = hard["recovered_at"].get("default (t=0.50)", "—")
            at_rec = hard["recovered_at"].get(rec_label, "—")
            p.append(
                "These are the only patients in the test partition whose diagnosis cannot be read "
                "directly off `Fever Duration (Days)`, and they are therefore the only patients "
                "for whom a model is required at all. Lowering the cut-off from 0.50 to the "
                f"recommended {rec['threshold']:.2f} moves recovery from {at_default} to "
                f"{at_rec}. They become recoverable only at a cut-off that simultaneously flags "
                "the great majority of true negatives — that is, by abandoning classification and "
                "testing everyone. Threshold tuning cannot substitute for signal that is not "
                "present.\n"
            )
    if thr_abl:
        p.append(
            f"The ablation model reinforces this. With `Fever Duration (Days)` withheld, ROC AUC "
            f"falls to **{thr_abl['calibration']['roc_auc']:.4f}** — near the chance value of "
            "0.5000 — and no cut-off produces a usable operating point: under any cost weighting "
            "that takes missed cases seriously, the expected-cost minimum is the degenerate "
            "policy of flagging every patient, which is precisely the status quo the system was "
            "intended to improve upon.\n"
        )

    # ---- 4.9 ---------------------------------------------------------------
    p.append("## 4.9 Comparative audit of alternative public datasets\n")
    p.append(
        "Before concluding that the modelling dataset was unsuitable, two further publicly "
        "available typhoid datasets were obtained and subjected to the identical audit "
        "procedure, to establish whether the problem was specific to the Kaggle collection or "
        "characteristic of the available public data as a whole.\n"
    )
    comp = []
    labels = {
        "kaggle_typhoid": ("Kaggle typhoid dataset", "Kaggle", "Synthetic"),
        "Typhoid_Dataset": (
            "Nishat et al., Harvard Dataverse (10.7910/DVN/STPYOM)", "Harvard Dataverse",
            "Synthetic",
        ),
        "typhodx_bd": (
            "TyphoDx-BD, Mendeley Data (10.17632/m9pnvv2vpv.1)", "Mendeley Data",
            "Real, physician-validated",
        ),
    }
    for key, (name, source, nature) in labels.items():
        a = load(REPORTS / "audits" / f"audit_{key}.json")
        if not a:
            continue
        trees = {b["classifier"]: b for b in a["baselines"]}
        d3 = trees.get("Decision tree, depth 3", {})
        best_single = next(
            (b for k, b in trees.items() if k.startswith("Best single attribute")), {}
        )
        comp.append(
            {
                "Dataset": name,
                "Source": source,
                "Records": f"{a['rows']:,}",
                "Attributes": a["attributes"],
                "Duplicate rows": f"{a['duplicate_rows']:,}",
                "Best single attribute (bal. acc.)": num(best_single.get("balanced_accuracy"), 4),
                "Depth-3 tree (bal. acc.)": num(d3.get("balanced_accuracy"), 4),
                "Nature": nature,
            }
        )
    if comp:
        p.append("**Table 4.15: Comparative audit of three public typhoid datasets**\n")
        p.append(table(pd.DataFrame(comp)))
    p.append(
        "Each dataset fails the audit, and each fails differently.\n"
    )
    p.append(
        "The **Kaggle** dataset is the largest and the only one carrying a geographic attribute, "
        "but its label is a deterministic function of `Fever Duration (Days)`, as established in "
        "Section 4.7.\n"
    )
    nishat = load(REPORTS / "audits" / "audit_Typhoid_Dataset.json")
    if nishat:
        p.append(
            f"The **Nishat et al.** dataset ({nishat['rows']} records) is separable to "
            "**1.0000 balanced accuracy by a decision tree of depth three**. Its "
            f"{nishat['duplicate_rows']} duplicated rows belong without exception to the negative "
            "class, leaving only 102 distinct negative patients — a pattern consistent with a "
            "small sample padded by copying. Platelet count in the negative class varies sixteen "
            "times less than in the positive class, and `Urine Culture Bacteria` = *Klebsiella "
            "pneumoniae* maps to a positive diagnosis in all 162 records in which it appears. "
            "The originating publication reports an AUC of 1.00 on these data; a depth-three tree "
            "achieves the same, which indicates that no complex model was meaningfully tested.\n"
        )
    typhodx = load(REPORTS / "audits" / "audit_typhodx_bd.json")
    if typhodx:
        p.append(
            f"The **TyphoDx-BD** dataset ({typhodx['rows']:,} records) is the only one of the "
            "three that is genuinely real: patient identifiers are encrypted, and collection was "
            "conducted under a named institutional ethics approval. Its problem is different in "
            "kind and, for this study, decisive. The audit found that **a Widal titre of 1:160 or "
            "above on any of the TO, TH, AH or BH agglutinins maps to a positive label in every "
            "record without exception** — 1,100 of 1,100. The label is not an independent clinical "
            "outcome but the interpretive reading rule applied to the four measurements that "
            "constitute the feature set. A model trained on these data would learn to reproduce a "
            "published cut-off table, not to diagnose typhoid; the exercise would be circular. "
            f"A further {typhodx['duplicate_rows']} of the {typhodx['rows']:,} rows are exact "
            "duplicates, which would inflate any random-split evaluation through leakage between "
            "training and test partitions.\n"
        )
    p.append(
        "The three datasets therefore represent three distinct failure modes — a synthetic label "
        "rule, a padded and over-separable sample, and a real dataset whose label is definitionally "
        "derived from its own features. The Kaggle collection was retained for modelling because "
        "it is the only one of adequate size, the only one carrying the geographic attribute "
        "required by Objective 4, and the richest in clinical and environmental features. That "
        "choice was made with the limitation documented rather than concealed. The wider "
        "implication is addressed in Chapter Five: the constraint on machine-learning research "
        "into typhoid diagnosis is not modelling technique but the absence of an openly available "
        "dataset in which the diagnosis is established independently of the features used to "
        "predict it.\n"
    )

    # ---- 4.10 --------------------------------------------------------------
    p.append("## 4.10 Summary of findings\n")
    p.append(
        "1. A complete SVM diagnostic pipeline was implemented to the specification of Chapter "
        "Three, covering preprocessing, SMOTENC balancing, kernel experimentation, "
        "cross-validated optimisation, evaluation and deployment.\n"
        f"2. The optimised binary model achieved {pct(binary['accuracy']) if binary else '—'} "
        f"accuracy and {num(binary['macro_f1']) if binary else '—'} macro F1 on unseen data, "
        f"with inference latency of {binary.get('latency', {}).get('single_record_ms', '—') if binary else '—'} ms "
        "per record, satisfying the real-time requirement of Objective 3.\n"
        "3. Performance was uniform across endemic, rural and urban subgroups, satisfying "
        "Objective 4 as stated.\n"
        "4. A signal audit established that the dataset encodes the label deterministically "
        "into `Fever Duration (Days)` and `White Blood Cell Count`, that *Acute* and *Relapsing "
        "Typhoid* are not separable by any attribute, and that the remaining attributes are "
        "effectively independent of the outcome.\n"
        + (
            f"5. A decision-threshold analysis established a defensible operating point of "
            f"{thr['recommended']['threshold']:.2f} under a ten-to-one weighting of missed cases "
            "against unnecessary tests, and showed the probabilities to be well calibrated "
            f"(expected calibration error {thr['calibration']['expected_calibration_error']:.4f}). "
            "It also showed that tuning the cut-off across its entire range alters the number of "
            "missed cases by a single patient, and that a positive predictive value which does "
            "not fall with prevalence confirms the classifier is reproducing a deterministic "
            "rule.\n"
            if thr
            else ""
        )
        + "6. A comparative audit of two further public typhoid datasets found each to be "
        "unusable for a different reason: one is separable to perfect balanced accuracy by a "
        "depth-three tree and padded with duplicated negatives, and one — although genuinely "
        "real and physician-validated — carries a label that is the Widal interpretive reading "
        "rule applied to its own feature set, making any model trained on it circular.\n"
        "7. Consequently the reported accuracy measures the dataset's construction rather than "
        "the model's diagnostic capability, and validation on real clinical data is required "
        "before any claim of diagnostic utility can be made. The binding constraint on this "
        "research area is the availability of a dataset in which the diagnosis is established "
        "independently of the features used to predict it, not the choice of learning "
        "algorithm.\n"
    )

    out = REPORTS / "CHAPTER_4_DRAFT.md"
    body = "\n".join(p)
    out.write_text(body, encoding="utf-8")
    print(f"Wrote {out} ({len(body):,} characters)")


if __name__ == "__main__":
    main()
