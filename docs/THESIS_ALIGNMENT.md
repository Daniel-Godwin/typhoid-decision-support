# Thesis Alignment and Discrepancies

This note maps the implementation onto Chapters 1–3 of the proposal and records
every point where the code and the written methodology diverge, so that the
write-up can be corrected rather than silently contradicted.

## Objectives coverage

| Objective (Section 1.3) | Where it is implemented | Evidence |
|---|---|---|
| 1. Collect and preprocess the Kaggle typhoid dataset; evaluate features for model input | `data.py`, `preprocessing.py` | `reports/dataset_audit.json` |
| 2. Design an SVM model for typhoid | `model.py` | `docs/ARCHITECTURE.md` |
| 3. Develop and train the SVM on the preprocessed dataset | `train.py` | `reports/kernel_comparison_*.csv`, `reports/grid_search_results_*.csv` |
| 4. Evaluate scalability using accuracy, precision, recall and F1 across regions, by splitting on `Location` | `evaluate.subgroup_report` | `reports/subgroup_evaluation_*.csv`, Table 4.4 |

## The finding that must shape Chapters 4 and 5

`Fever Duration (Days)` is a **deterministic proxy for the target** in this
dataset. Every one of the 8,981 records with a fever duration of one day or more
is labelled typhoid; not a single one is negative. A one-line rule — *predict
typhoid when fever duration ≥ 1 day* — scores 0.9870 accuracy and 0.9844 macro
F1, against the optimised SVM's 0.9881 and 0.9857.

Meanwhile `Widal Test` (Cramér's V = 0.0014) and `Typhidot Test` (V = 0.0065)
are statistically independent of the diagnosis, which cannot be true of real
patient data. The dataset is synthetic, and most attributes appear to have been
drawn at random around a label encoded into fever duration.

**Consequence for the write-up.** The 98.8% figure cannot be presented as
evidence that machine learning improves typhoid diagnosis, because a threshold
on one column achieves the same. Presenting it that way invites the single
question the work cannot survive: *what is the model actually learning?*

**Recommended framing.** Report the baselines (Table 4.0) beside the model,
state the marginal gain explicitly, present the fever-duration ablation
(Table 4.5) as the honest measure of what the remaining attributes contribute,
and make dataset validity a finding of the study in Chapter 5. A dissertation
that identifies a flaw in a widely reused public dataset and quantifies its
effect is stronger than one that reports an unexamined 98.8%. Section 1.6
already anticipates this — "there is a dependency on existing datasets... these
datasets might not capture the global heterogeneity" — so the limitation can be
carried forward from Chapter 1 rather than introduced defensively.

Full evidence: `docs/DATASET_AUDIT.md`, `reports/feature_association.csv`,
`reports/table_4_0_baselines.csv`, `reports/single_feature_stumps.csv`.

## Discrepancies to correct in the write-up

### 1. Number of records

Section 3.1.1 states **31,088** patient records. The supplied file contains
**31,087** rows with zero duplicates. Amend the figure to 31,087.

### 2. Number of target classes

Section 3.1.1 describes the target as having **two outcome classes** ("No
(false) Typhoid, Yes (true) Typhoid") and then calls the resulting model a
"multi-class classification model" in the same sentence. The dataset's
`Typhoid Status` column actually holds **four** classes:

| Class | Count | Percentage |
|---|---:|---:|
| Normal or No Typhoid | 21,701 | 69.81% |
| Acute Typhoid Fever | 5,649 | 18.17% |
| Relapsing Typhoid | 2,486 | 8.00% |
| Complicated Typhoid | 1,251 | 4.02% |

Both readings are now implemented. The recommended correction is to describe the
binary model as the **primary diagnostic model** (matching the stated two-class
target) and the four-class model as a **secondary severity-stratification
model** derived from the same data, and to remove the phrase "multi-class" from
the binary description.

### 3. Disease scope

The title, Statement of the Problem and Objectives address typhoid alone, but
Section 1.3 ("Aim") and parts of Sections 1.6 and 1.7 still refer to "malaria
and typhoid", carried over from an earlier draft. Several entries in the
literature review and Table 2.1 are also malaria-only studies. Restrict the Aim
and the operational definitions to typhoid, and either drop the malaria-only
entries from Table 2.1 or introduce them explicitly as methodological analogues
rather than as typhoid work.

### 4. Feature exclusions are not yet documented in Chapter 3

`Blood Culture Result` and `Complications` are excluded from the feature set.
This is a substantive methodological decision and needs a paragraph in Section
3.2.1:

- `Blood Culture Result` is the confirmatory gold standard. A model that
  consumes it is not predicting a diagnosis; it is restating one. Including it
  would inflate every reported metric while destroying the "early diagnosis,
  before laboratory confirmation" claim that motivates the study.
- `Complications` is missing in **96.98%** of records and encodes
  post-diagnostic severity, so it is unavailable at the moment of decision.

Table 3.0 currently lists both variables without noting that they are held out.

### 5. Widal and Typhidot results are model inputs

Section 3.1.1 lists the Widal and Typhidot tests among the features, and they
are retained in the default (`routine`) policy. Chapter 1, however, argues that
these tests are unreliable and often unavailable in the settings the model
targets. A `clinical_only` policy that removes both is therefore also reported
as a sensitivity analysis (Table 4.5), which lets the discussion state directly
how much diagnostic performance survives when no serology is available.

### 6. Loss function

Section 3.2.1.1 states that the model is trained by minimising cross-entropy
loss. A Support Vector Machine minimises regularised **hinge loss**, not
cross-entropy:

$$\min_{w,b}\ \tfrac{1}{2}\lVert w\rVert^{2} + C\sum_{i=1}^{n}\max\left(0,\ 1 - y_i\left(w^{\top}\phi(x_i) + b\right)\right)$$

Equations 3.1 and 3.2 are otherwise correct; only the parenthetical naming the
loss needs replacing. Cross-entropy would be appropriate for logistic regression
or a neural network, neither of which is the model here.

### 7. "Real-time" claim

Section 3.2.6 and Objective 3 refer to real-time, point-of-care operation.
This is now measurable rather than asserted: `reports/final_evaluation_*.json`
records per-record and batch inference latency, and the deployment exposes a
JSON endpoint (`POST /api/predict`) that a point-of-care client can call.

### 8. Class balancing method

Section 3.2.1 specifies SMOTE. The dataset is predominantly categorical, and
plain SMOTE interpolates between category codes, producing values that
correspond to no real category. **SMOTENC** — the variant designed for mixed
numeric/categorical data — is used instead and should be named as such in the
methodology. It is applied inside the pipeline so that resampling only ever
touches training folds.

### 9. Validation methods

Table 3.3 lists cross-validation and hold-out validation. Both are used:
stratified 3-fold cross-validation drives the hyperparameter search, and a
stratified 20% hold-out partition, untouched throughout training and
optimisation, produces every reported final metric.

## Additions beyond the proposal

These strengthen the work and should be mentioned in Chapter 4:

- **Probability calibration.** An SVM decision function is a signed distance
  from the hyperplane, not a probability. Platt scaling turns it into the
  calibrated confidence the interface reports.
- **Interpretability.** Chapter 2 identifies the "black box" problem as a
  barrier to clinical adoption (Asuquo *et al.*, 2024). Each prediction now
  carries an occlusion-based attribution showing which findings moved it and by
  how many probability points.
- **Balanced accuracy and per-class metrics.** With a 69.8% majority class,
  plain accuracy is a weak summary. Balanced accuracy, macro F1 and per-class
  precision/recall are reported alongside it.
- **Sensitivity and specificity.** Reported explicitly for the binary model,
  because a false negative (a missed typhoid case) and a false positive carry
  very different clinical costs, and clinicians read these two numbers rather
  than macro F1.
