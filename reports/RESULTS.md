# Results

## Dataset

The dataset contains **31,087 records** and **24 attributes**, with **0 duplicate rows**.

### Target distribution (four-class)

| class | count | percentage |
|---|---|---|
| Normal or No Typhoid | 21,701 | 69.81% |
| Acute Typhoid Fever | 5,649 | 18.17% |
| Relapsing Typhoid | 2,486 | 8.00% |
| Complicated Typhoid | 1,251 | 4.02% |

### Target distribution (binary)

| class | count | percentage |
|---|---|---|
| No Typhoid | 21,701 | 69.81% |
| Typhoid | 9,386 | 30.19% |

## Table 4.0 - Reference baselines on the same held-out test partition

| model | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 |
|---|---|---|---|---|---|
| Majority class ('No Typhoid') | 0.6981 | 0.5 | 0.3491 | 0.5 | 0.4111 |
| Rule: Fever Duration (Days) >= 1 | 0.9875 | 0.9792 | 0.9912 | 0.9792 | 0.9849 |
| Optimised SVM (deployed policy) | 0.9875 | 0.9792 | 0.9912 | 0.9792 | 0.9849 |
| Optimised SVM (fever duration removed) | 0.4241 | 0.4874 | 0.4881 | 0.4874 | 0.4235 |

## Table 4.1 - Kernel comparison

| model | kernel | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | fit_predict_seconds |
|---|---|---|---|---|---|---|
| Binary | linear | 0.9875 | 0.9792 | 0.9849 | 0.9874 | 11.3 |
| Binary | poly | 0.9875 | 0.9792 | 0.9849 | 0.9874 | 25.3 |
| Binary | rbf | 0.9833 | 0.9723 | 0.9798 | 0.9831 | 24.9 |
| Four-class | linear | 0.7774 | 0.4862 | 0.4408 | 0.7731 | 520.4 |
| Four-class | poly | 0.804 | 0.4727 | 0.4701 | 0.8082 | 332.6 |
| Four-class | rbf | 0.8168 | 0.4888 | 0.4898 | 0.8167 | 301.5 |

## Table 4.2 - Optimised model performance

| model | kernel | C | gamma | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | sensitivity | specificity | fpr | fnr | roc_auc | inference_ms_per_record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Binary diagnosis | linear | 0.1 | - | 0.9875 | 0.9792 | 0.9912 | 0.9792 | 0.9849 | 0.9874 | 0.9584 | 1.0000 | 0.0000 | 0.0416 | 0.9804 | 7.78 |
| Severity stratification | poly | 10.0 | scale | 0.7951 | 0.4921 | 0.4952 | 0.4921 | 0.4722 | 0.7996 | nan | nan | nan | nan | nan | 18.08 |

## Table 4.3 - Per-class performance

| model | class | precision | recall | f1_score | support |
|---|---|---|---|---|---|
| Binary | No Typhoid | 0.9823 | 1.0000 | 0.9911 | 4341 |
| Binary | Typhoid | 1.0000 | 0.9584 | 0.9788 | 1877 |
| Four-class | Normal or No Typhoid | 0.9848 | 1.0000 | 0.9923 | 4341 |
| Four-class | Acute Typhoid Fever | 0.6049 | 0.3265 | 0.4241 | 1130 |
| Four-class | Relapsing Typhoid | 0.2492 | 0.2978 | 0.2713 | 497 |
| Four-class | Complicated Typhoid | 0.1419 | 0.3440 | 0.2009 | 250 |

## Table 4.4 - Scalability across geographic settings (Objective 4)

| model | subgroup | n | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|---|
| Binary | Endemic | 2084 | 0.9856 | 0.9773 | 0.9897 | 0.9773 | 0.9832 | 0.9547 | 1.0 |
| Binary | Rural | 2046 | 0.9912 | 0.9847 | 0.9939 | 0.9847 | 0.9892 | 0.9694 | 1.0 |
| Binary | Urban | 2088 | 0.9856 | 0.9761 | 0.9899 | 0.9761 | 0.9827 | 0.9522 | 1.0 |
| Four-class | Endemic | 2063 | 0.7843 | 0.4918 | 0.4957 | 0.4918 | 0.4689 | nan | nan |
| Four-class | Rural | 2090 | 0.7861 | 0.4732 | 0.4852 | 0.4732 | 0.4499 | nan | nan |
| Four-class | Urban | 2065 | 0.815 | 0.5121 | 0.5099 | 0.5121 | 0.4997 | nan | nan |

## Table 4.5 - Feature policy sensitivity and ablation analysis

| feature policy | accuracy | balanced_accuracy | macro_f1 | sensitivity | specificity | roc_auc |
|---|---|---|---|---|---|---|
| Deployed — 13 post-review attributes | 0.9875 | 0.9792 | 0.9849 | 0.9584 | 1.0000 | 0.9804 |
| Pre-review — all 20 permitted attributes | 0.9881 | 0.9803 | 0.9857 | 0.9606 | 1.0000 | 0.9797 |
| Pre-laboratory — Widal and Typhidot removed | 0.9875 | 0.9792 | 0.9849 | 0.9584 | 1.0000 | 0.9802 |
| Ablation — Fever Duration (Days) removed | 0.4241 | 0.4874 | 0.4235 | 0.6473 | 0.3276 | 0.4852 |

## Table 4.6 - Association between each attribute and the diagnosis

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

## Hyperparameter search - binary

- Candidates evaluated: **24** over **3-fold** stratified cross-validation
- Search records: **24,869**
- Selection metric: **macro F1**
- Best configuration: `{'svc__C': 0.1, 'svc__kernel': 'linear'}`
- Best cross-validated macro F1: **0.9842**
- Search wall-clock: **501s**

## Hyperparameter search - multiclass

- Candidates evaluated: **24** over **3-fold** stratified cross-validation
- Search records: **10,000** (stratified subsample)
- Selection metric: **macro F1**
- Best configuration: `{'svc__C': 10.0, 'svc__degree': 2, 'svc__gamma': 'scale', 'svc__kernel': 'poly'}`
- Best cross-validated macro F1: **0.4959**
- Search wall-clock: **951s**

## Signal audit

Of the 8,981 records with a fever duration of one day or more, **8,981** are labelled typhoid (100.0%). A further 405 typhoid cases have a recorded fever duration of zero days.

`Fever Duration (Days)` therefore acts as a near-deterministic proxy for the target, and the reference baselines in Table 4.0 must be read alongside the model results. See `docs/DATASET_AUDIT.md` for the full analysis.

## Figures

- `reports/figures/calibration_binary.png`
- `reports/figures/calibration_binary_clinical_only.png`
- `reports/figures/calibration_binary_no_fever_duration.png`
- `reports/figures/calibration_binary_routine.png`
- `reports/figures/confusion_matrix_binary.png`
- `reports/figures/confusion_matrix_binary_clinical_only.png`
- `reports/figures/confusion_matrix_binary_no_fever_duration.png`
- `reports/figures/confusion_matrix_binary_routine.png`
- `reports/figures/confusion_matrix_multiclass.png`
- `reports/figures/confusion_matrix_normalised_binary.png`
- `reports/figures/confusion_matrix_normalised_binary_clinical_only.png`
- `reports/figures/confusion_matrix_normalised_binary_no_fever_duration.png`
- `reports/figures/confusion_matrix_normalised_binary_routine.png`
- `reports/figures/confusion_matrix_normalised_multiclass.png`
- `reports/figures/pr_curve_binary.png`
- `reports/figures/pr_curve_binary_clinical_only.png`
- `reports/figures/pr_curve_binary_no_fever_duration.png`
- `reports/figures/pr_curve_binary_routine.png`
- `reports/figures/roc_curve_binary.png`
- `reports/figures/roc_curve_binary_clinical_only.png`
- `reports/figures/roc_curve_binary_no_fever_duration.png`
- `reports/figures/roc_curve_binary_routine.png`
- `reports/figures/threshold_curves_binary.png`
- `reports/figures/threshold_curves_binary_clinical_only.png`
- `reports/figures/threshold_curves_binary_no_fever_duration.png`
- `reports/figures/threshold_curves_binary_routine.png`
