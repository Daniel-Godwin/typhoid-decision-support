# Chapter Four — Results and Discussion

_Draft generated from the project artefacts. Every figure below is read directly from `reports/`; regenerate with `python scripts/make_chapter4.py` after any retraining. Prose is a starting point to be rewritten in your own voice._

## 4.1 Introduction

This chapter presents the results of the Support Vector Machine models developed in Chapter Three. It reports the dataset characteristics established by the audit, the comparison of kernel functions, the outcome of hyperparameter optimisation, the performance of the optimised models on the held-out test partition, and the evaluation of scalability across geographic settings required by Objective 4. It then presents a signal audit of the dataset itself, a decision-threshold and calibration analysis establishing the operating point at which the deployed system runs, and a comparative audit of two further public typhoid datasets. Taken together, the final three sections materially qualify how the headline performance figures should be interpreted.

## 4.2 Dataset characteristics

The dataset comprises **31,087 patient records** across **24 attributes**, with **0 duplicate rows**. Four attributes contain missing values, the most affected being `Complications` at 96.98%.

**Table 4.1: Distribution of the four-class target**

| Class | Count | Percentage |
|---|---|---|
| Normal or No Typhoid | 21,701 | 69.81% |
| Acute Typhoid Fever | 5,649 | 18.17% |
| Relapsing Typhoid | 2,486 | 8.00% |
| Complicated Typhoid | 1,251 | 4.02% |

**Table 4.2: Distribution of the binary diagnostic target**

| Class | Count | Percentage |
|---|---|---|
| No Typhoid | 21,701 | 69.81% |
| Typhoid | 9,386 | 30.19% |

The four-class target is markedly imbalanced, with a ratio of 17.35 : 1 between the largest and smallest class. This motivated the SMOTENC balancing described in Section 3.2.1, applied inside the modelling pipeline so that synthetic samples were generated from training folds only and never contaminated validation or test data.

Following the feature policy set out in Section 3.2.1, `Blood Culture Result` and `Complications` were withheld from every model. The first is the confirmatory gold standard, so a model consuming it would be restating a completed diagnosis rather than predicting one; the second encodes post-diagnostic severity and is unavailable at the point of decision.

### 4.2.1 Attributes retained and dropped during preprocessing

The stated aim of the study is prediction and diagnosis at the point of care in resource-limited settings, where the clinician has the patient's presenting symptoms, exposure history and prior illness history, but may have no laboratory available. An attribute requiring a laboratory investigation cannot be supplied at the moment the prediction is needed, and a model depending on one would not be usable in the setting for which it is intended. Two further considerations apply: an attribute recorded for only some patients would oblige the point-of-care interface to accept an incomplete entry, and an attribute describing the wider community rather than the patient is not a clinical sign of that patient's illness.

On this basis a further seven attributes were dropped during preprocessing, in addition to the two withheld as diagnostic leakage. Table 4.3 states each exclusion with its reason.

**Table 4.3: Attributes excluded during preprocessing**

| Attribute | Ground for exclusion | Reason |
|---|---|---|
| Blood Culture Result | Diagnostic leakage | The confirmatory gold standard; a model consuming it would restate a completed diagnosis rather than predict one |
| Complications | Diagnostic leakage | Post-diagnostic severity, unavailable at the point of decision, absent for 96.98% of records |
| White Blood Cell Count | Laboratory investigation | Requires a haematology laboratory; not obtainable at the point of care |
| Platelet Count | Laboratory investigation | Requires a haematology laboratory; not obtainable at the point of care |
| Widal Test | Laboratory investigation | Serological assay requiring laboratory support; also independent of the label (V = 0.0014) |
| Typhidot Test | Laboratory investigation | Serological assay requiring laboratory support; also independent of the label (V = 0.0065) |
| Typhoid Vaccination Status | Not a presenting sign | History of prophylaxis rather than of disease or presentation |
| Gastrointestinal Symptoms | Incomplete recording | Absent for 24.96% of records, which would require an optional field at entry |
| Ongoing Infection in Society | Not a patient attribute | Describes community transmission rather than the individual patient; absent for 24.70% of records |

In addition, `Neurological Symptoms` was recoded as a binary `Headache` indicator, so that the attribute carries the symptom named in Objective 1 rather than a mixture of neurological presentations. A recorded value of *Headache* maps to *Yes*; *Confusion*, *Delirium* and an absent value map to *No*. No discriminative information is lost in the recoding, because there was none to lose: the probability of typhoid is 0.2972 given *Confusion*, 0.3023 given *Delirium*, 0.3085 given *Headache* and 0.2998 where the field is blank, against an overall rate of 0.3019.

Thirteen attributes therefore form the modelling feature space — two numeric (`Age` and `Fever Duration (Days)`) and eleven categorical (`Gender`, `Location`, `Socioeconomic Status`, `Water Source Type`, `Sanitation Facilities`, `Hand Hygiene`, `Consumption of Street Food`, `Weather Condition`, `Skin Manifestations`, `Headache` and `Previous History of Typhoid`). Every one is obtainable by history and examination without laboratory support, and every one is complete for all 31,087 records, so the deployed data entry form carries no optional field and no entry the clinician skips can move the result.

The pre-review feature space of twenty attributes was retained as a comparator, and is reported alongside the deployed policy throughout this chapter so that the cost of the reduction can be read directly rather than assumed.

## 4.3 Kernel comparison

Three kernel functions — linear, polynomial and radial basis function — were evaluated at default hyperparameter settings on the held-out test partition, to establish which family of decision boundaries suited the data before committing to an exhaustive search.

**Table 4.4: Kernel comparison, binary diagnostic model**

| kernel | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | fit_predict_seconds |
|---|---|---|---|---|---|
| linear | 0.9875 | 0.9792 | 0.9849 | 0.9874 | 11.3 |
| poly | 0.9875 | 0.9792 | 0.9849 | 0.9874 | 25.3 |
| rbf | 0.9833 | 0.9723 | 0.9798 | 0.9831 | 24.9 |

**Table 4.5: Kernel comparison, four-class severity model**

| kernel | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | fit_predict_seconds |
|---|---|---|---|---|---|
| linear | 0.7774 | 0.4862 | 0.4408 | 0.7731 | 520.4 |
| poly | 0.804 | 0.4727 | 0.4701 | 0.8082 | 332.6 |
| rbf | 0.8168 | 0.4888 | 0.4898 | 0.8167 | 301.5 |

For the binary model the linear kernel produced the highest macro F1 (0.9849). The three kernels differed by less than one percentage point in accuracy, which indicates that the decision boundary separating the classes is close to linear in the encoded feature space — a point returned to in Section 4.7.

## 4.4 Hyperparameter optimisation

For the **binary diagnostic model**, Grid Search evaluated **24 candidate configurations** under **3-fold stratified cross-validation**, selecting on macro F1. The search used the full training partition of 24,869 records. The selected configuration was `{'svc__C': 0.1, 'svc__kernel': 'linear'}`, achieving a cross-validated macro F1 of **0.9842** in 501 seconds of wall-clock time.

For the **four-class severity model**, Grid Search evaluated **24 candidate configurations** under **3-fold stratified cross-validation**, selecting on macro F1. Because oversampling expands the four-class training partition to roughly four times the majority class, the search was executed on a stratified subsample of 10,000 records and the winning configuration then refitted on the complete training partition. The selected configuration was `{'svc__C': 10.0, 'svc__degree': 2, 'svc__gamma': 'scale', 'svc__kernel': 'poly'}`, achieving a cross-validated macro F1 of **0.4959** in 951 seconds of wall-clock time.

The complete search results, including the score of every candidate configuration, are recorded in `reports/grid_search_results_binary.csv` and `reports/grid_search_results_multiclass.csv`.

## 4.5 Performance of the optimised models

**Table 4.6: Performance of the optimised models on the held-out test partition**

| model | kernel | C | gamma | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | sensitivity | specificity | fpr | fnr | roc_auc | inference_ms_per_record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Binary diagnosis | linear | 0.1 | - | 0.9875 | 0.9792 | 0.9912 | 0.9792 | 0.9849 | 0.9874 | 0.9584 | 1.0 | 0.0 | 0.0416 | 0.9804 | 7.78 |
| Severity stratification | poly | 10.0 | scale | 0.7951 | 0.4921 | 0.4952 | 0.4921 | 0.4722 | 0.7996 | nan | nan | nan | nan | nan | 18.08 |

The optimised binary model achieved an accuracy of **98.75%**, balanced accuracy of **97.92%** and macro F1 of **0.9849** on 6,218 unseen records. Sensitivity was **0.9584** and specificity **1.0000**, corresponding to 78 false negatives and 0 false positives.

The clinical asymmetry of these two errors matters. A false negative is a missed typhoid case sent home untreated; a false positive is an unnecessary confirmatory test. The model's perfect specificity means it never raises a false alarm, but its sensitivity of 0.9584 means roughly 4.2% of true cases are missed — which is the error that carries clinical risk. Section 4.7 explains why these particular cases are missed.

Inference latency was **7.78 ms per record** (4,379.5 records per second in batch), which satisfies the real-time, point-of-care requirement of Objective 3.

**Table 4.7: Per-class precision, recall and F1**

| model | class | precision | recall | f1_score | support |
|---|---|---|---|---|---|
| Binary | No Typhoid | 0.9823 | 1.0 | 0.9911 | 4341 |
| Binary | Typhoid | 1.0 | 0.9584 | 0.9788 | 1877 |
| Four-class | Normal or No Typhoid | 0.9848 | 1.0 | 0.9923 | 4341 |
| Four-class | Acute Typhoid Fever | 0.6049 | 0.3265 | 0.4241 | 1130 |
| Four-class | Relapsing Typhoid | 0.2492 | 0.2978 | 0.2713 | 497 |
| Four-class | Complicated Typhoid | 0.1419 | 0.344 | 0.2009 | 250 |

The four-class severity model achieved accuracy of **79.51%** and macro F1 of **0.4722**. The gap between the two figures is the result of note: a high accuracy on a heavily imbalanced target is carried by the majority class, and a macro F1 well below it shows that the minority severity classes are not being recovered.

This is a direct and expected consequence of the feature reduction described in Section 4.2.1, and it is worth stating plainly rather than leaving to be inferred. Severity in this dataset is encoded almost entirely in the white blood cell count: every record labelled *Complicated Typhoid* has a count above 11,000 cells per microlitre and no record of any other class does, so the two ranges do not overlap by a single count (Section 4.7). A model given that attribute separates the class perfectly; a model restricted to what a clinician can observe without a laboratory cannot, because the distinguishing information is a laboratory measurement. The distinction between *Acute* and *Relapsing Typhoid* is not encoded in any attribute at all, and neither feature space recovers it.

The practical reading is that severity stratification is not a point-of-care task on this data. The binary diagnostic model, which is the model the objectives specify and the deployed system serves by default, is unaffected: it loses 0.03 accuracy points to the reduction, as Table 4.12 records. Severity stratification is retained as a secondary output, with the limitation above stated in the interface, and Chapter Five recommends that it be revisited where haematology is available.

Confusion matrices for both models, together with ROC and precision–recall curves for the binary model, are provided in `reports/figures/`.

## 4.6 Scalability across geographic settings (Objective 4)

Objective 4 requires the model to be evaluated across diverse geographic regions and healthcare environments. The held-out test partition was therefore stratified by the `Location` attribute and the full metric set recomputed within each subgroup.

**Table 4.8: Performance by geographic setting**

| model | subgroup | n | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|---|
| Binary | Endemic | 2084 | 0.9856 | 0.9773 | 0.9897 | 0.9773 | 0.9832 | 0.9547 | 1.0 |
| Binary | Rural | 2046 | 0.9912 | 0.9847 | 0.9939 | 0.9847 | 0.9892 | 0.9694 | 1.0 |
| Binary | Urban | 2088 | 0.9856 | 0.9761 | 0.9899 | 0.9761 | 0.9827 | 0.9522 | 1.0 |
| Four-class | Endemic | 2063 | 0.7843 | 0.4918 | 0.4957 | 0.4918 | 0.4689 | nan | nan |
| Four-class | Rural | 2090 | 0.7861 | 0.4732 | 0.4852 | 0.4732 | 0.4499 | nan | nan |
| Four-class | Urban | 2065 | 0.815 | 0.5121 | 0.5099 | 0.5121 | 0.4997 | nan | nan |

Macro F1 varied by only **0.0065** across the endemic, rural and urban subgroups. The model therefore does not degrade in any one setting, which satisfies Objective 4 as stated. This stability should, however, be read alongside Section 4.7: `Location` is statistically independent of the diagnosis in this dataset, so uniform performance across locations reflects the absence of geographic variation in the data rather than demonstrated robustness to it.

## 4.7 Signal audit and interpretation of the results

A model that reports high accuracy has not necessarily learned anything useful. To establish what the classifier had actually learned, three reference classifiers were scored on the identical test partition, and the statistical association between every attribute and the diagnosis was measured.

**Table 4.9: Reference baselines, binary diagnostic task**

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 |
|---|---|---|---|---|---|
| Majority class ('No Typhoid') | 0.6981 | 0.5 | 0.3491 | 0.5 | 0.4111 |
| Rule: Fever Duration (Days) >= 1 | 0.9875 | 0.9792 | 0.9912 | 0.9792 | 0.9849 |
| Optimised SVM (deployed policy) | 0.9875 | 0.9792 | 0.9912 | 0.9792 | 0.9849 |
| Optimised SVM (fever duration removed) | 0.4241 | 0.4874 | 0.4881 | 0.4874 | 0.4235 |

Of the 8,981 records with a fever duration of one day or more, **8,981 — that is, every one of them — are labelled typhoid.** No record in the dataset combines a fever duration of one day or more with a negative diagnosis. `Fever Duration (Days)` is therefore a deterministic proxy for the target, and a single threshold on that one attribute reproduces almost the entire performance of the optimised model.

**Table 4.10: Reference baselines, four-class task**

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 |
|---|---|---|---|---|---|
| Majority class | 0.6981 | 0.25 | 0.1745 | 0.25 | 0.2056 |
| Rules: WBC > 11,000 -> Complicated; fever >= 1 -> Acute | 0.9133 | 0.7396 | 0.6709 | 0.7396 | 0.7002 |
| Optimised SVM (four-class) | 0.7951 | 0.4921 | 0.4952 | 0.4921 | 0.4722 |

A second deterministic rule governs the four-class task: every record labelled *Complicated Typhoid* has a white blood cell count above 11,000, and no record of any other class does. The two ranges do not overlap by a single count. A two-rule baseline on these two attributes alone outperforms the optimised four-class SVM on accuracy.

This rule is also the explanation for the severity model's macro F1 reported in Section 4.5. The white blood cell count was dropped at supervisory review because it requires a haematology laboratory, and it is the only attribute in which severity is encoded. The deployed severity model therefore cannot recover *Complicated Typhoid*, and the drop in macro F1 relative to the pre-review comparator measures exactly the size of that single deterministic rule. The two findings are the same finding seen from two directions: what the reduction removes is not clinical signal but the dataset's construction.

**Table 4.11: Association between each attribute and the diagnosis**

| feature | statistic | value |
|---|---|---|
| Blood Culture Result | Cramer's V | 0.9999 |
| Fever Duration (Days) | point-biserial r | 0.8138 |
| Complications | Cramer's V | 0.2685 |
| White Blood Cell Count | point-biserial r | 0.1948 |
| Water Source Type | Cramer's V | 0.0132 |
| Location | Cramer's V | 0.012 |
| Neurological Symptoms | Cramer's V | 0.0091 |
| Headache | Cramer's V | 0.0081 |
| Ongoing Infection in Society | Cramer's V | 0.0073 |
| Sanitation Facilities | Cramer's V | 0.0069 |
| Socioeconomic Status | Cramer's V | 0.0066 |
| Typhidot Test | Cramer's V | 0.0065 |
| Gastrointestinal Symptoms | Cramer's V | 0.0052 |
| Previous History of Typhoid | Cramer's V | 0.0045 |
| Hand Hygiene | Cramer's V | 0.0042 |
| Consumption of Street Food | Cramer's V | 0.0035 |
| Platelet Count | point-biserial r | -0.0026 |
| Weather Condition | Cramer's V | 0.0023 |
| Skin Manifestations | Cramer's V | 0.0022 |
| Typhoid Vaccination Status | Cramer's V | 0.0021 |
| Age | point-biserial r | 0.0014 |
| Widal Test | Cramer's V | 0.0014 |
| Gender | Cramer's V | 0.0008 |

Outside of `Blood Culture Result` (withheld), `Fever Duration (Days)` and `White Blood Cell Count`, no attribute reaches a Cramér's V of 0.02. `Widal Test` (V = 0.0014) and `Typhidot Test` (V = 0.0065) are statistically independent of the diagnosis — an impossibility in real patient data, since these are the serological tests on which typhoid diagnosis conventionally rests.

### Ablation

**Table 4.12: Feature policy sensitivity and ablation**

| feature policy | accuracy | balanced_accuracy | macro_f1 | sensitivity | specificity | roc_auc |
|---|---|---|---|---|---|---|
| Deployed — 13 post-review attributes | 0.9875 | 0.9792 | 0.9849 | 0.9584 | 1.0 | 0.9804 |
| Pre-review — all 20 permitted attributes | 0.9881 | 0.9803 | 0.9857 | 0.9606 | 1.0 | 0.9797 |
| Pre-laboratory — Widal and Typhidot removed | 0.9875 | 0.9792 | 0.9849 | 0.9584 | 1.0 | 0.9802 |
| Ablation — Fever Duration (Days) removed | 0.4241 | 0.4874 | 0.4235 | 0.6473 | 0.3276 | 0.4852 |

With `Fever Duration (Days)` removed, performance collapses to an accuracy of **42.41%** and a balanced accuracy of **0.4874** — against a chance floor of 0.5000. The remaining twelve attributes, taken together, therefore carry only marginal discriminative information about the diagnosis.

One methodological point should be recorded about this ablation. The exhaustive grid search of Section 4.4 was attempted on the ablated feature space and did not terminate: with `Fever Duration (Days)` removed, the classes are very nearly inseparable, and the sequential minimal optimisation solver does not converge at the larger regularisation values in the search grid. The ablation model was therefore fitted with the kernel identified by the comparison in Table 4.4 at the library default regularisation, rather than by search. The failure to converge is itself evidence for the conclusion drawn here — a search that cannot separate the classes is a search over data that does not separate — and the quantity of interest is the level of performance, not its tuning.

Taken together, these results establish that the dataset's apparent predictability is an artefact of its construction. The dataset is synthetic: the label is encoded deterministically into two attributes, one class distinction (*Acute* versus *Relapsing*) is not encoded at all, and the remaining attributes are effectively random with respect to the outcome. Realistic column names and plausible marginal distributions disguise this, but the joint distribution does not support the clinical relationships the attributes imply.

This does not invalidate the system developed in this study. The preprocessing pipeline, leakage controls, resampling strategy, optimisation procedure, evaluation protocol and deployed interface are all sound, and would transfer without modification to real clinical data. What the finding does establish is that the headline accuracy of 98.75% must not be presented as evidence that machine learning improves typhoid diagnosis — because a single threshold on one attribute achieves substantially the same result. Section 1.6 of this work anticipated precisely this risk in noting the study's dependency on existing datasets.

## 4.8 Decision threshold, calibration and transportability

Sections 4.5 and 4.7 report performance at the library default cut-off of 0.5. That default treats a missed case and an unnecessary confirmatory test as equally costly, which is not the clinical position: untreated typhoid carries a case-fatality risk, whereas a false alarm costs one Widal or culture. A triage instrument must therefore declare its operating point explicitly. The cut-off was swept across the held-out partition and candidate operating points derived under several cost weightings.

**Table 4.13: Candidate operating points, binary diagnostic model**

| Operating point | Cut-off | Sensitivity | Specificity | PPV | NPV | FN | FP | Referral rate |
|---|---|---|---|---|---|---|---|---|
| default | 0.5000 | 0.9584 | 1.0000 | 1.0000 | 0.9823 | 78 | 0 | 0.2893 |
| youden | 0.0300 | 0.9584 | 1.0000 | 1.0000 | 0.9823 | 78 | 0 | 0.2893 |
| max_f1 | 0.0300 | 0.9584 | 1.0000 | 1.0000 | 0.9823 | 78 | 0 | 0.2893 |
| sens_90 | 0.0300 | 0.9584 | 1.0000 | 1.0000 | 0.9823 | 78 | 0 | 0.2893 |
| sens_95 | 0.0300 | 0.9584 | 1.0000 | 1.0000 | 0.9823 | 78 | 0 | 0.2893 |
| sens_99 | 0.0100 | 1.0000 | 0.0000 | 0.3019 | nan | 0 | 4341 | 1.0000 |
| cost_fn1x | 0.0300 | 0.9584 | 1.0000 | 1.0000 | 0.9823 | 78 | 0 | 0.2893 |
| cost_fn5x | 0.0300 | 0.9584 | 1.0000 | 1.0000 | 0.9823 | 78 | 0 | 0.2893 |
| cost_fn10x | 0.0300 | 0.9584 | 1.0000 | 1.0000 | 0.9823 | 78 | 0 | 0.2893 |
| cost_fn20x | 0.0300 | 0.9584 | 1.0000 | 1.0000 | 0.9823 | 78 | 0 | 0.2893 |

Weighting a missed case as ten times as damaging as an unnecessary test selects a cut-off of **0.03**, at which sensitivity is **0.9584** and specificity **1.0000**. This value is the one configured in the deployed system as `TRIAGE_THRESHOLD`.

The more instructive observation is how little the choice matters. Moving from 0.50 to 0.03 changes the number of missed cases from 78 to 78 — one patient in 6,218. In a model that had learned a graded representation of risk, a sweep across this range would move sensitivity by tens of percentage points. Here the operating characteristics are almost flat, because the underlying decision is not graded but binary: the shortcut identified in Section 4.7 either fires or it does not.

**Table 4.14: Predictive values projected onto other prevalences**

| Prevalence | PPV | NPV | Flagged per 1,000 | Missed per 1,000 |
|---|---|---|---|---|
| 2% | 1.0000 | 0.9992 | 19.2 | 0.8 |
| 5% | 1.0000 | 0.9978 | 47.9 | 2.1 |
| 10% | 1.0000 | 0.9954 | 95.8 | 4.2 |
| 20% | 1.0000 | 0.9897 | 191.7 | 8.3 |
| 35% | 1.0000 | 0.9781 | 335.5 | 14.5 |
| 50% | 1.0000 | 0.9601 | 479.2 | 20.8 |

Sensitivity and specificity are properties of a classifier; positive and negative predictive value are properties of the setting in which it is used, because they depend on how common the disease is among those tested. Table 4.14 is included because it is the calculation a deploying clinic needs, and because its result here is diagnostic in itself: the positive predictive value remains 1.0000 even at a prevalence of two per cent. For a real diagnostic test that is arithmetically impossible. It occurs only because no negative record in this dataset is ever flagged at any usable cut-off, which is the signature of a deterministic rule rather than a probabilistic classifier.

The calibration of the probabilities was assessed separately, since a threshold is only meaningful if the probability it is applied to means what it states. The Platt-scaled outputs achieved a Brier score of **0.0124** and an expected calibration error of **0.0046** (ROC AUC 0.9804). The model is therefore well calibrated: when it states a risk of eighty per cent, approximately eighty per cent of such patients are positive. It is an honest reporter of a conclusion the dataset made trivial.

**Table 4.15: Recovery of the 78 typhoid cases recorded with zero fever duration**

| Cut-off | Cases recovered |
|---|---|
| default (t=0.50) | 0/78 |
| cost_fn10x (t=0.03) | 0/78 |
| sens_95 (t=0.03) | 0/78 |
| sens_99 (t=0.01) | 78/78 |
| t=0.05 | 0/78 |

These are the only patients in the test partition whose diagnosis cannot be read directly off `Fever Duration (Days)`, and they are therefore the only patients for whom a model is required at all. Lowering the cut-off from 0.50 to the recommended 0.03 moves recovery from 0/78 to 0/78. They become recoverable only at a cut-off that simultaneously flags the great majority of true negatives — that is, by abandoning classification and testing everyone. Threshold tuning cannot substitute for signal that is not present.

The ablation model reinforces this. With `Fever Duration (Days)` withheld, ROC AUC falls to **0.4852** — near the chance value of 0.5000 — and no cut-off produces a usable operating point: under any cost weighting that takes missed cases seriously, the expected-cost minimum is the degenerate policy of flagging every patient, which is precisely the status quo the system was intended to improve upon.

## 4.9 Comparative audit of alternative public datasets

Before concluding that the modelling dataset was unsuitable, two further publicly available typhoid datasets were obtained and subjected to the identical audit procedure, to establish whether the problem was specific to the Kaggle collection or characteristic of the available public data as a whole.

**Table 4.16: Comparative audit of three public typhoid datasets**

| Dataset | Source | Records | Attributes | Duplicate rows | Best single attribute (bal. acc.) | Depth-3 tree (bal. acc.) | Nature |
|---|---|---|---|---|---|---|---|
| Kaggle typhoid dataset | Kaggle | 31,087 | 23 | 0 | 0.9792 | 0.9806 | Synthetic |
| Nishat et al., Harvard Dataverse (10.7910/DVN/STPYOM) | Harvard Dataverse | 659 | 14 | 86 | 0.8900 | 1.0000 | Synthetic |
| TyphoDx-BD, Mendeley Data (10.17632/m9pnvv2vpv.1) | Mendeley Data | 1,100 | 11 | 528 | 0.7930 | 1.0000 | Real, physician-validated |

Each dataset fails the audit, and each fails differently.

The **Kaggle** dataset is the largest and the only one carrying a geographic attribute, but its label is a deterministic function of `Fever Duration (Days)`, as established in Section 4.7.

The **Nishat et al.** dataset (659 records) is separable to **1.0000 balanced accuracy by a decision tree of depth three**. Its 86 duplicated rows belong without exception to the negative class, leaving only 102 distinct negative patients — a pattern consistent with a small sample padded by copying. Platelet count in the negative class varies sixteen times less than in the positive class, and `Urine Culture Bacteria` = *Klebsiella pneumoniae* maps to a positive diagnosis in all 162 records in which it appears. The originating publication reports an AUC of 1.00 on these data; a depth-three tree achieves the same, which indicates that no complex model was meaningfully tested.

The **TyphoDx-BD** dataset (1,100 records) is the only one of the three that is genuinely real: patient identifiers are encrypted, and collection was conducted under a named institutional ethics approval. Its problem is different in kind and, for this study, decisive. The audit found that **a Widal titre of 1:160 or above on any of the TO, TH, AH or BH agglutinins maps to a positive label in every record without exception** — 1,100 of 1,100. The label is not an independent clinical outcome but the interpretive reading rule applied to the four measurements that constitute the feature set. A model trained on these data would learn to reproduce a published cut-off table, not to diagnose typhoid; the exercise would be circular. A further 528 of the 1,100 rows are exact duplicates, which would inflate any random-split evaluation through leakage between training and test partitions.

The three datasets therefore represent three distinct failure modes — a synthetic label rule, a padded and over-separable sample, and a real dataset whose label is definitionally derived from its own features. The Kaggle collection was retained for modelling because it is the only one of adequate size, the only one carrying the geographic attribute required by Objective 4, and the richest in clinical and environmental features. That choice was made with the limitation documented rather than concealed. The wider implication is addressed in Chapter Five: the constraint on machine-learning research into typhoid diagnosis is not modelling technique but the absence of an openly available dataset in which the diagnosis is established independently of the features used to predict it.

## 4.10 Summary of findings

1. A complete SVM diagnostic pipeline was implemented to the specification of Chapter Three, covering preprocessing, SMOTENC balancing, kernel experimentation, cross-validated optimisation, evaluation and deployment.
2. The optimised binary model achieved 98.75% accuracy and 0.9849 macro F1 on unseen data, with inference latency of 7.78 ms per record, satisfying the real-time requirement of Objective 3.
3. Performance was uniform across endemic, rural and urban subgroups, satisfying Objective 4 as stated.
4. A signal audit established that the dataset encodes the label deterministically into `Fever Duration (Days)` and `White Blood Cell Count`, that *Acute* and *Relapsing Typhoid* are not separable by any attribute, and that the remaining attributes are effectively independent of the outcome.
5. A decision-threshold analysis established a defensible operating point of 0.03 under a ten-to-one weighting of missed cases against unnecessary tests, and showed the probabilities to be well calibrated (expected calibration error 0.0046). It also showed that tuning the cut-off across its entire range alters the number of missed cases by a single patient, and that a positive predictive value which does not fall with prevalence confirms the classifier is reproducing a deterministic rule.
6. A comparative audit of two further public typhoid datasets found each to be unusable for a different reason: one is separable to perfect balanced accuracy by a depth-three tree and padded with duplicated negatives, and one — although genuinely real and physician-validated — carries a label that is the Widal interpretive reading rule applied to its own feature set, making any model trained on it circular.
7. Consequently the reported accuracy measures the dataset's construction rather than the model's diagnostic capability, and validation on real clinical data is required before any claim of diagnostic utility can be made. The binding constraint on this research area is the availability of a dataset in which the diagnosis is established independently of the features used to predict it, not the choice of learning algorithm.
