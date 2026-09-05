# Chapter Four — Results and Discussion

_Draft generated from the project artefacts. Every figure below is read directly from `reports/`; regenerate with `python scripts/make_chapter4.py` after any retraining. Prose is a starting point to be rewritten in your own voice._

## 4.1 Introduction

This chapter presents the results of the Support Vector Machine models developed in Chapter Three. It reports the dataset characteristics established by the audit, the comparison of kernel functions, the outcome of hyperparameter optimisation, the performance of the optimised models on the held-out test partition, and the evaluation of scalability across geographic settings required by Objective 4. It then presents a signal audit of the dataset itself, a decision-threshold and calibration analysis establishing the operating point at which the deployed system runs, and a comparative audit of two further public typhoid datasets. Taken together, the final three sections materially qualify how the headline performance figures should be interpreted.

## 4.2 Dataset characteristics

The dataset comprises **31,087 patient records** across **23 attributes**, with **0 duplicate rows**. Four attributes contain missing values, the most affected being `Complications` at 96.98%.

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

## 4.3 Kernel comparison

Three kernel functions — linear, polynomial and radial basis function — were evaluated at default hyperparameter settings on the held-out test partition, to establish which family of decision boundaries suited the data before committing to an exhaustive search.

**Table 4.3: Kernel comparison, binary diagnostic model**

| kernel | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | fit_predict_seconds |
|---|---|---|---|---|---|
| linear | 0.9875 | 0.9792 | 0.9849 | 0.9874 | 35.0 |
| poly | 0.983 | 0.9718 | 0.9794 | 0.9828 | 42.4 |
| rbf | 0.9773 | 0.9624 | 0.9725 | 0.9771 | 68.2 |

**Table 4.4: Kernel comparison, four-class severity model**

| kernel | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | fit_predict_seconds |
|---|---|---|---|---|---|
| linear | 0.8681 | 0.7363 | 0.7347 | 0.8701 | 492.0 |
| poly | 0.8731 | 0.7301 | 0.7341 | 0.8728 | 240.4 |
| rbf | 0.8705 | 0.7199 | 0.7266 | 0.8677 | 261.3 |

For the binary model the linear kernel produced the highest macro F1 (0.9849). The three kernels differed by less than one percentage point in accuracy, which indicates that the decision boundary separating the classes is close to linear in the encoded feature space — a point returned to in Section 4.7.

## 4.4 Hyperparameter optimisation

For the **binary diagnostic model**, Grid Search evaluated **24 candidate configurations** under **3-fold stratified cross-validation**, selecting on macro F1. The search used the full training partition of 24,869 records. The selected configuration was `{'svc__C': 10.0, 'svc__degree': 2, 'svc__gamma': 0.1, 'svc__kernel': 'poly'}`, achieving a cross-validated macro F1 of **0.9847** in 935 seconds of wall-clock time.

For the **four-class severity model**, Grid Search evaluated **24 candidate configurations** under **3-fold stratified cross-validation**, selecting on macro F1. Because oversampling expands the four-class training partition to roughly four times the majority class, the search was executed on a stratified subsample of 10,000 records and the winning configuration then refitted on the complete training partition. The selected configuration was `{'svc__C': 10.0, 'svc__degree': 2, 'svc__gamma': 'scale', 'svc__kernel': 'poly'}`, achieving a cross-validated macro F1 of **0.7435** in 824 seconds of wall-clock time.

The complete search results, including the score of every candidate configuration, are recorded in `reports/grid_search_results_binary.csv` and `reports/grid_search_results_multiclass.csv`.

## 4.5 Performance of the optimised models

**Table 4.5: Performance of the optimised models on the held-out test partition**

| model | kernel | C | gamma | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | sensitivity | specificity | fpr | fnr | roc_auc | inference_ms_per_record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Binary diagnosis | poly | 10.0 | 0.1 | 0.9881 | 0.9803 | 0.9916 | 0.9803 | 0.9857 | 0.988 | 0.9606 | 1.0 | 0.0 | 0.0394 | 0.9797 | 9.96 |
| Severity stratification | poly | 10.0 | scale | 0.8697 | 0.7394 | 0.7455 | 0.7394 | 0.7377 | 0.8717 | nan | nan | nan | nan | nan | 19.04 |

The optimised binary model achieved an accuracy of **98.81%**, balanced accuracy of **98.03%** and macro F1 of **0.9857** on 6,218 unseen records. Sensitivity was **0.9606** and specificity **1.0000**, corresponding to 74 false negatives and 0 false positives.

The clinical asymmetry of these two errors matters. A false negative is a missed typhoid case sent home untreated; a false positive is an unnecessary confirmatory test. The model's perfect specificity means it never raises a false alarm, but its sensitivity of 0.9606 means roughly 3.9% of true cases are missed — which is the error that carries clinical risk. Section 4.7 explains why these particular cases are missed.

Inference latency was **9.96 ms per record** (6,329.7 records per second in batch), which satisfies the real-time, point-of-care requirement of Objective 3.

**Table 4.6: Per-class precision, recall and F1**

| model | class | precision | recall | f1_score | support |
|---|---|---|---|---|---|
| Binary | No Typhoid | 0.9832 | 1.0 | 0.9915 | 4341 |
| Binary | Typhoid | 1.0 | 0.9606 | 0.9799 | 1877 |
| Four-class | Normal or No Typhoid | 0.9859 | 1.0 | 0.9929 | 4341 |
| Four-class | Acute Typhoid Fever | 0.6928 | 0.5389 | 0.6063 | 1130 |
| Four-class | Relapsing Typhoid | 0.3032 | 0.4185 | 0.3516 | 497 |
| Four-class | Complicated Typhoid | 1.0 | 1.0 | 1.0 | 250 |

The four-class model achieved accuracy of **86.97%** and macro F1 of **0.7377**. Performance was highly uneven across classes: *Normal or No Typhoid* and *Complicated Typhoid* were classified almost perfectly, while *Acute Typhoid Fever* and *Relapsing Typhoid* were frequently confused with one another. Section 4.7 shows that this pattern is a property of the dataset rather than a limitation of the classifier.

Confusion matrices for both models, together with ROC and precision–recall curves for the binary model, are provided in `reports/figures/`.

## 4.6 Scalability across geographic settings (Objective 4)

Objective 4 requires the model to be evaluated across diverse geographic regions and healthcare environments. The held-out test partition was therefore stratified by the `Location` attribute and the full metric set recomputed within each subgroup.

**Table 4.7: Performance by geographic setting**

| model | subgroup | n | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|---|
| Binary | Endemic | 2084 | 0.9861 | 0.9781 | 0.99 | 0.9781 | 0.9838 | 0.9562 | 1.0 |
| Binary | Rural | 2046 | 0.9922 | 0.9864 | 0.9946 | 0.9864 | 0.9904 | 0.9728 | 1.0 |
| Binary | Urban | 2088 | 0.9861 | 0.9769 | 0.9903 | 0.9769 | 0.9832 | 0.9537 | 1.0 |
| Four-class | Endemic | 2063 | 0.8682 | 0.7391 | 0.7451 | 0.7391 | 0.7396 | nan | nan |
| Four-class | Rural | 2090 | 0.8651 | 0.7484 | 0.7509 | 0.7484 | 0.7414 | nan | nan |
| Four-class | Urban | 2065 | 0.876 | 0.7279 | 0.738 | 0.7279 | 0.7288 | nan | nan |

Macro F1 varied by only **0.0072** across the endemic, rural and urban subgroups. The model therefore does not degrade in any one setting, which satisfies Objective 4 as stated. This stability should, however, be read alongside Section 4.7: `Location` is statistically independent of the diagnosis in this dataset, so uniform performance across locations reflects the absence of geographic variation in the data rather than demonstrated robustness to it.

## 4.7 Signal audit and interpretation of the results

A model that reports high accuracy has not necessarily learned anything useful. To establish what the classifier had actually learned, three reference classifiers were scored on the identical test partition, and the statistical association between every attribute and the diagnosis was measured.

**Table 4.8: Reference baselines, binary diagnostic task**

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 |
|---|---|---|---|---|---|
| Majority class ('No Typhoid') | 0.6981 | 0.5 | 0.3491 | 0.5 | 0.4111 |
| Rule: Fever Duration (Days) >= 1 | 0.9875 | 0.9792 | 0.9912 | 0.9792 | 0.9849 |
| Optimised SVM (routine policy) | 0.9881 | 0.9803 | 0.9916 | 0.9803 | 0.9857 |
| Optimised SVM (fever duration removed) | 0.6012 | 0.5485 | 0.545 | 0.5485 | 0.5454 |

Of the 8,981 records with a fever duration of one day or more, **8,981 — that is, every one of them — are labelled typhoid.** No record in the dataset combines a fever duration of one day or more with a negative diagnosis. `Fever Duration (Days)` is therefore a deterministic proxy for the target, and a single threshold on that one attribute reproduces almost the entire performance of the optimised model.

**Table 4.9: Reference baselines, four-class task**

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 |
|---|---|---|---|---|---|
| Majority class | 0.6981 | 0.25 | 0.1745 | 0.25 | 0.2056 |
| Rules: WBC > 11,000 -> Complicated; fever >= 1 -> Acute | 0.9133 | 0.7396 | 0.6709 | 0.7396 | 0.7002 |
| Optimised SVM (four-class) | 0.8697 | 0.7394 | 0.7455 | 0.7394 | 0.7377 |

A second deterministic rule governs the four-class task: every record labelled *Complicated Typhoid* has a white blood cell count above 11,000, and no record of any other class does. The two ranges do not overlap by a single count, which is why the model classifies that class perfectly. A two-rule baseline outperforms the optimised four-class SVM on accuracy.

**Table 4.10: Association between each attribute and the diagnosis**

| feature | statistic | value |
|---|---|---|
| Blood Culture Result | Cramer's V | 0.9999 |
| Fever Duration (Days) | point-biserial r | 0.8138 |
| Complications | Cramer's V | 0.2685 |
| White Blood Cell Count | point-biserial r | 0.1948 |
| Water Source Type | Cramer's V | 0.0132 |
| Location | Cramer's V | 0.012 |
| Neurological Symptoms | Cramer's V | 0.0091 |
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

**Table 4.11: Feature policy sensitivity and ablation**

| feature policy | accuracy | balanced_accuracy | macro_f1 | sensitivity | specificity | roc_auc |
|---|---|---|---|---|---|---|
| Routine — all permitted attributes | 0.9881 | 0.9803 | 0.9857 | 0.9606 | 1.0 | 0.9797 |
| Pre-laboratory — Widal and Typhidot removed | 0.9875 | 0.9792 | 0.9849 | 0.9584 | 1.0 | 0.9802 |
| Ablation — Fever Duration (Days) removed | 0.6012 | 0.5485 | 0.5454 | 0.4156 | 0.6814 | 0.5729 |

With `Fever Duration (Days)` removed, performance collapses to an accuracy of **60.12%** and a balanced accuracy of **0.5485** — against a chance floor of 0.5000. The remaining nineteen attributes, taken together, therefore carry only marginal discriminative information about the diagnosis.

Taken together, these results establish that the dataset's apparent predictability is an artefact of its construction. The dataset is synthetic: the label is encoded deterministically into two attributes, one class distinction (*Acute* versus *Relapsing*) is not encoded at all, and the remaining attributes are effectively random with respect to the outcome. Realistic column names and plausible marginal distributions disguise this, but the joint distribution does not support the clinical relationships the attributes imply.

This does not invalidate the system developed in this study. The preprocessing pipeline, leakage controls, resampling strategy, optimisation procedure, evaluation protocol and deployed interface are all sound, and would transfer without modification to real clinical data. What the finding does establish is that the headline accuracy of 98.81% must not be presented as evidence that machine learning improves typhoid diagnosis — because a single threshold on one attribute achieves substantially the same result. Section 1.6 of this work anticipated precisely this risk in noting the study's dependency on existing datasets.

## 4.8 Decision threshold, calibration and transportability

Sections 4.5 and 4.7 report performance at the library default cut-off of 0.5. That default treats a missed case and an unnecessary confirmatory test as equally costly, which is not the clinical position: untreated typhoid carries a case-fatality risk, whereas a false alarm costs one Widal or culture. A triage instrument must therefore declare its operating point explicitly. The cut-off was swept across the held-out partition and candidate operating points derived under several cost weightings.

**Table 4.12: Candidate operating points, binary diagnostic model**

| Operating point | Cut-off | Sensitivity | Specificity | PPV | NPV | FN | FP | Referral rate |
|---|---|---|---|---|---|---|---|---|
| default | 0.5000 | 0.9606 | 1.0000 | 1.0000 | 0.9832 | 74 | 0 | 0.2900 |
| youden | 0.0800 | 0.9611 | 1.0000 | 1.0000 | 0.9835 | 73 | 0 | 0.2901 |
| max_f1 | 0.0800 | 0.9611 | 1.0000 | 1.0000 | 0.9835 | 73 | 0 | 0.2901 |
| sens_90 | 0.0800 | 0.9611 | 1.0000 | 1.0000 | 0.9835 | 73 | 0 | 0.2901 |
| sens_95 | 0.0800 | 0.9611 | 1.0000 | 1.0000 | 0.9835 | 73 | 0 | 0.2901 |
| sens_99 | 0.0100 | 0.9957 | 0.0885 | 0.3208 | 0.9796 | 8 | 3957 | 0.9370 |
| cost_fn1x | 0.0800 | 0.9611 | 1.0000 | 1.0000 | 0.9835 | 73 | 0 | 0.2901 |
| cost_fn5x | 0.0800 | 0.9611 | 1.0000 | 1.0000 | 0.9835 | 73 | 0 | 0.2901 |
| cost_fn10x | 0.0800 | 0.9611 | 1.0000 | 1.0000 | 0.9835 | 73 | 0 | 0.2901 |
| cost_fn20x | 0.0500 | 0.9622 | 0.9938 | 0.9853 | 0.9838 | 71 | 27 | 0.2948 |

Weighting a missed case as ten times as damaging as an unnecessary test selects a cut-off of **0.08**, at which sensitivity is **0.9611** and specificity **1.0000**. This value is the one configured in the deployed system as `TRIAGE_THRESHOLD`.

The more instructive observation is how little the choice matters. Moving from 0.50 to 0.08 changes the number of missed cases from 74 to 73 — one patient in 6,218. In a model that had learned a graded representation of risk, a sweep across this range would move sensitivity by tens of percentage points. Here the operating characteristics are almost flat, because the underlying decision is not graded but binary: the shortcut identified in Section 4.7 either fires or it does not.

**Table 4.13: Predictive values projected onto other prevalences**

| Prevalence | PPV | NPV | Flagged per 1,000 | Missed per 1,000 |
|---|---|---|---|---|
| 2% | 1.0000 | 0.9992 | 19.2 | 0.8 |
| 5% | 1.0000 | 0.9980 | 48.1 | 1.9 |
| 10% | 1.0000 | 0.9957 | 96.1 | 3.9 |
| 20% | 1.0000 | 0.9904 | 192.2 | 7.8 |
| 35% | 1.0000 | 0.9795 | 336.4 | 13.6 |
| 50% | 1.0000 | 0.9626 | 480.6 | 19.4 |

Sensitivity and specificity are properties of a classifier; positive and negative predictive value are properties of the setting in which it is used, because they depend on how common the disease is among those tested. Table 4.13 is included because it is the calculation a deploying clinic needs, and because its result here is diagnostic in itself: the positive predictive value remains 1.0000 even at a prevalence of two per cent. For a real diagnostic test that is arithmetically impossible. It occurs only because no negative record in this dataset is ever flagged at any usable cut-off, which is the signature of a deterministic rule rather than a probabilistic classifier.

The calibration of the probabilities was assessed separately, since a threshold is only meaningful if the probability it is applied to means what it states. The Platt-scaled outputs achieved a Brier score of **0.0117** and an expected calibration error of **0.0020** (ROC AUC 0.9797). The model is therefore well calibrated: when it states a risk of eighty per cent, approximately eighty per cent of such patients are positive. It is an honest reporter of a conclusion the dataset made trivial.

**Table 4.14: Recovery of the 78 typhoid cases recorded with zero fever duration**

| Cut-off | Cases recovered |
|---|---|
| default (t=0.50) | 4/78 |
| cost_fn10x (t=0.08) | 5/78 |
| sens_95 (t=0.08) | 5/78 |
| sens_99 (t=0.01) | 70/78 |
| t=0.05 | 7/78 |

These are the only patients in the test partition whose diagnosis cannot be read directly off `Fever Duration (Days)`, and they are therefore the only patients for whom a model is required at all. Lowering the cut-off from 0.50 to the recommended 0.08 moves recovery from 4/78 to 5/78. They become recoverable only at a cut-off that simultaneously flags the great majority of true negatives — that is, by abandoning classification and testing everyone. Threshold tuning cannot substitute for signal that is not present.

The ablation model reinforces this. With `Fever Duration (Days)` withheld, ROC AUC falls to **0.5729** — near the chance value of 0.5000 — and no cut-off produces a usable operating point: under any cost weighting that takes missed cases seriously, the expected-cost minimum is the degenerate policy of flagging every patient, which is precisely the status quo the system was intended to improve upon.

## 4.9 Comparative audit of alternative public datasets

Before concluding that the modelling dataset was unsuitable, two further publicly available typhoid datasets were obtained and subjected to the identical audit procedure, to establish whether the problem was specific to the Kaggle collection or characteristic of the available public data as a whole.

**Table 4.15: Comparative audit of three public typhoid datasets**

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
2. The optimised binary model achieved 98.81% accuracy and 0.9857 macro F1 on unseen data, with inference latency of 9.96 ms per record, satisfying the real-time requirement of Objective 3.
3. Performance was uniform across endemic, rural and urban subgroups, satisfying Objective 4 as stated.
4. A signal audit established that the dataset encodes the label deterministically into `Fever Duration (Days)` and `White Blood Cell Count`, that *Acute* and *Relapsing Typhoid* are not separable by any attribute, and that the remaining attributes are effectively independent of the outcome.
5. A decision-threshold analysis established a defensible operating point of 0.08 under a ten-to-one weighting of missed cases against unnecessary tests, and showed the probabilities to be well calibrated (expected calibration error 0.0020). It also showed that tuning the cut-off across its entire range alters the number of missed cases by a single patient, and that a positive predictive value which does not fall with prevalence confirms the classifier is reproducing a deterministic rule.
6. A comparative audit of two further public typhoid datasets found each to be unusable for a different reason: one is separable to perfect balanced accuracy by a depth-three tree and padded with duplicated negatives, and one — although genuinely real and physician-validated — carries a label that is the Widal interpretive reading rule applied to its own feature set, making any model trained on it circular.
7. Consequently the reported accuracy measures the dataset's construction rather than the model's diagnostic capability, and validation on real clinical data is required before any claim of diagnostic utility can be made. The binding constraint on this research area is the availability of a dataset in which the diagnosis is established independently of the features used to predict it, not the choice of learning algorithm.
