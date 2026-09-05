# Results

## Dataset

The dataset contains **31,087 records** and **23 attributes**, with **0 duplicate rows**.

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
| Optimised SVM (routine policy) | 0.9881 | 0.9803 | 0.9916 | 0.9803 | 0.9857 |
| Optimised SVM (fever duration removed) | 0.6012 | 0.5485 | 0.545 | 0.5485 | 0.5454 |

## Table 4.1 - Kernel comparison

| model | kernel | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | fit_predict_seconds |
|---|---|---|---|---|---|---|
| Binary | linear | 0.9875 | 0.9792 | 0.9849 | 0.9874 | 35.0 |
| Binary | poly | 0.983 | 0.9718 | 0.9794 | 0.9828 | 42.4 |
| Binary | rbf | 0.9773 | 0.9624 | 0.9725 | 0.9771 | 68.2 |
| Four-class | linear | 0.8681 | 0.7363 | 0.7347 | 0.8701 | 492.0 |
| Four-class | poly | 0.8731 | 0.7301 | 0.7341 | 0.8728 | 240.4 |
| Four-class | rbf | 0.8705 | 0.7199 | 0.7266 | 0.8677 | 261.3 |

## Table 4.2 - Optimised model performance

| model | kernel | C | gamma | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 | sensitivity | specificity | fpr | fnr | roc_auc | inference_ms_per_record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Binary diagnosis | poly | 10.0 | 0.1 | 0.9881 | 0.9803 | 0.9916 | 0.9803 | 0.9857 | 0.9880 | 0.9606 | 1.0000 | 0.0000 | 0.0394 | 0.9797 | 9.96 |
| Severity stratification | poly | 10.0 | scale | 0.8697 | 0.7394 | 0.7455 | 0.7394 | 0.7377 | 0.8717 | nan | nan | nan | nan | nan | 19.04 |

## Table 4.3 - Per-class performance

| model | class | precision | recall | f1_score | support |
|---|---|---|---|---|---|
| Binary | No Typhoid | 0.9832 | 1.0000 | 0.9915 | 4341 |
| Binary | Typhoid | 1.0000 | 0.9606 | 0.9799 | 1877 |
| Four-class | Normal or No Typhoid | 0.9859 | 1.0000 | 0.9929 | 4341 |
| Four-class | Acute Typhoid Fever | 0.6928 | 0.5389 | 0.6063 | 1130 |
| Four-class | Relapsing Typhoid | 0.3032 | 0.4185 | 0.3516 | 497 |
| Four-class | Complicated Typhoid | 1.0000 | 1.0000 | 1.0000 | 250 |

## Table 4.4 - Scalability across geographic settings (Objective 4)

| model | subgroup | n | accuracy | balanced_accuracy | macro_precision | macro_recall | macro_f1 | sensitivity | specificity |
|---|---|---|---|---|---|---|---|---|---|
| Binary | Endemic | 2084 | 0.9861 | 0.9781 | 0.99 | 0.9781 | 0.9838 | 0.9562 | 1.0 |
| Binary | Rural | 2046 | 0.9922 | 0.9864 | 0.9946 | 0.9864 | 0.9904 | 0.9728 | 1.0 |
| Binary | Urban | 2088 | 0.9861 | 0.9769 | 0.9903 | 0.9769 | 0.9832 | 0.9537 | 1.0 |
| Four-class | Endemic | 2063 | 0.8682 | 0.7391 | 0.7451 | 0.7391 | 0.7396 | nan | nan |
| Four-class | Rural | 2090 | 0.8651 | 0.7484 | 0.7509 | 0.7484 | 0.7414 | nan | nan |
| Four-class | Urban | 2065 | 0.876 | 0.7279 | 0.738 | 0.7279 | 0.7288 | nan | nan |

## Table 4.5 - Feature policy sensitivity and ablation analysis

| feature policy | accuracy | balanced_accuracy | macro_f1 | sensitivity | specificity | roc_auc |
|---|---|---|---|---|---|---|
| Routine — all permitted attributes | 0.9881 | 0.9803 | 0.9857 | 0.9606 | 1.0000 | 0.9797 |
| Pre-laboratory — Widal and Typhidot removed | 0.9875 | 0.9792 | 0.9849 | 0.9584 | 1.0000 | 0.9802 |
| Ablation — Fever Duration (Days) removed | 0.6012 | 0.5485 | 0.5454 | 0.4156 | 0.6814 | 0.5729 |

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
- Best configuration: `{'svc__C': 10.0, 'svc__degree': 2, 'svc__gamma': 0.1, 'svc__kernel': 'poly'}`
- Best cross-validated macro F1: **0.9847**
- Search wall-clock: **935s**

## Hyperparameter search - multiclass

- Candidates evaluated: **24** over **3-fold** stratified cross-validation
- Search records: **10,000** (stratified subsample)
- Selection metric: **macro F1**
- Best configuration: `{'svc__C': 10.0, 'svc__degree': 2, 'svc__gamma': 'scale', 'svc__kernel': 'poly'}`
- Best cross-validated macro F1: **0.7435**
- Search wall-clock: **824s**

## Signal audit

Of the 8,981 records with a fever duration of one day or more, **8,981** are labelled typhoid (100.0%). A further 405 typhoid cases have a recorded fever duration of zero days.

`Fever Duration (Days)` therefore acts as a near-deterministic proxy for the target, and the reference baselines in Table 4.0 must be read alongside the model results. See `docs/DATASET_AUDIT.md` for the full analysis.

## Figures

- `reports/figures/confusion_matrix_binary.png`
- `reports/figures/confusion_matrix_binary_clinical_only.png`
- `reports/figures/confusion_matrix_binary_no_fever_duration.png`
- `reports/figures/confusion_matrix_multiclass.png`
- `reports/figures/confusion_matrix_normalised_binary.png`
- `reports/figures/confusion_matrix_normalised_binary_clinical_only.png`
- `reports/figures/confusion_matrix_normalised_binary_no_fever_duration.png`
- `reports/figures/confusion_matrix_normalised_multiclass.png`
- `reports/figures/pr_curve_binary.png`
- `reports/figures/pr_curve_binary_clinical_only.png`
- `reports/figures/pr_curve_binary_no_fever_duration.png`
- `reports/figures/roc_curve_binary.png`
- `reports/figures/roc_curve_binary_clinical_only.png`
- `reports/figures/roc_curve_binary_no_fever_duration.png`
