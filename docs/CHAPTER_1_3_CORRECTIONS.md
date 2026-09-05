# Corrections to Chapters One to Three

Nine corrections, each given as replacement text ready to paste into the thesis
document. Original wording is quoted first so the location is unambiguous.

---

## Correction 1 — Number of records (Section 3.1.1)

**Currently reads:**

> The dataset to be used in this study will be obtained from Kaggle and will
> consist of 31,088 patient records collected for the prediction of typhoid fever.

**Replace with:**

> The dataset used in this study was obtained from Kaggle and consists of 31,087
> patient records collected for the prediction of typhoid fever. The dataset was
> verified to contain no duplicate records.

**Reason:** The file contains 31,087 rows. Verified in `reports/dataset_audit.json`.

---

## Correction 2 — Number of target classes (Section 3.1.1)

**Currently reads:**

> The target variable will be Typhoid Status, which will represent the patient's
> clinical diagnosis. The variable comprises two outcome classes: No (false)
> Typhoid, Yes (true) Typhoid, thereby enabling the development of a multi-class
> classification model capable of distinguishing healthy individuals from
> different clinical manifestations of typhoid fever.

This is internally contradictory — it defines two classes and then calls the
model multi-class — and neither reading matches the data, which holds four
classes.

**Replace with:**

> The target variable is Typhoid Status, representing the patient's clinical
> diagnosis. As supplied, the variable comprises four outcome classes: Normal or
> No Typhoid (21,701 records, 69.81%), Acute Typhoid Fever (5,649 records,
> 18.17%), Relapsing Typhoid (2,486 records, 8.00%) and Complicated Typhoid
> (1,251 records, 4.02%).
>
> Two models are therefore developed. The primary model addresses the diagnostic
> question stated in the aim of this study by collapsing the three typhoid
> categories into a single positive class, producing a binary target of Typhoid
> (9,386 records, 30.19%) against No Typhoid (21,701 records, 69.81%). A
> secondary model retains all four classes and addresses severity stratification.
> Reporting both allows the diagnostic performance that the study set out to
> measure to be separated from the harder question of grading severity, and
> allows each to be evaluated on its own terms.

**Reason:** Removes the contradiction and documents what was actually built.

---

## Correction 3 — Disease scope (Section 1.3, and Sections 1.6 and 1.7)

**Section 1.3 currently reads:**

> The aim of this study is to develop and evaluate machine learning-based models
> using Support Vector Machine for the rapid and accurate prediction and
> diagnosis of malaria and typhoid fever, particularly in resource-limited
> settings.

**Replace with:**

> The aim of this study is to develop and evaluate a machine learning-based model
> using the Support Vector Machine algorithm for the rapid and accurate
> prediction and diagnosis of typhoid fever, particularly in resource-limited
> settings.

**Section 1.6 currently reads:**

> ...for the improvement of accuracy, promptness, and diagnostics in malaria and
> typhoid fever prevalent and resource-scarce environments.

**Replace with:**

> ...for the improvement of accuracy, promptness and diagnostic reliability in
> typhoid-endemic, resource-scarce environments.

**Section 1.7** — the definitions of *Diagnosis*, *Prediction*, *Learning* and
*Accuracy* each mention malaria. Delete "malaria and" or "malaria or" from all
four. For example:

> **Diagnosis**: The identification of typhoid fever using patient data,
> laboratory tests and machine learning models.

**Reason:** The title, Statement of the Problem and all four Objectives address
typhoid alone. The malaria references are residue from an earlier draft and are
the first inconsistency an examiner will notice.

---

## Correction 4 — Malaria-only studies in the literature review (Table 2.1)

Eight of the thirty entries in Table 2.1 are malaria studies with no typhoid
component: Sachs *et al.* (2019), Hughes *et al.* (2022), Kumar *et al.* (2023),
Davidson and Chen (2022), Miller and Zhang (2022), Oluwatobi and Ibrahim (2023),
Ramos-Briceño *et al.* (2025, the CNN parasite-detection paper), and
Tusting *et al.* (2021).

Two options, either defensible:

1. **Remove them** and renumber, leaving a table focused on typhoid.
2. **Keep them and label them.** Add a column headed "Disease" and introduce the
   table with:

> Table 2.1 summarises the reviewed literature. Entries marked *malaria* are
> included not as typhoid studies but as methodological analogues: they address
> the same problem structure — early diagnosis of a febrile illness in a
> resource-constrained setting, where the confirmatory test is slow, costly or
> unavailable — and the modelling choices they report informed the design of the
> present study.

Option 2 is recommended: it preserves the breadth of the review while removing
the implication that these are typhoid results.

---

## Correction 5 — Loss function (Section 3.2.1.1)

**Currently reads:**

> where 𝓛 is the loss function (e.g., cross-entropy loss).

and later:

> where 𝓛 is the loss function measuring the difference between the true
> diagnosis and the predicted diagnosis (commonly cross-entropy loss).

**Replace both with:**

> where 𝓛 is the loss function. A Support Vector Machine minimises regularised
> hinge loss rather than cross-entropy, so for the model developed in this study
> Equation (3.2) takes the specific form:
>
> min over w, b of ½‖w‖² + C · Σᵢ max(0, 1 − yᵢ(wᵀφ(xᵢ) + b))
>
> where w and b define the separating hyperplane, φ(·) is the implicit mapping
> induced by the kernel function, C is the regularisation parameter controlling
> the trade-off between margin width and training error, and the summation runs
> over the n training examples. The first term maximises the margin; the second
> penalises examples that fall inside it or on the wrong side of the boundary.

**Reason:** Cross-entropy is the loss of logistic regression and neural networks.
Naming it for an SVM is a conceptual error, and it is the kind of detail a panel
asks about directly.

---

## Correction 6 — Class balancing method (Section 3.2.1 and Section 3.2.4)

**Currently reads:**

> Finally, class imbalance will be mitigated using data augmentation techniques,
> specifically the Synthetic Minority Oversampling Technique (SMOTE).

**Replace with:**

> Class imbalance is mitigated using SMOTE-NC (Synthetic Minority Oversampling
> Technique for Nominal and Continuous features), the variant of SMOTE designed
> for datasets containing both categorical and numerical attributes. Standard
> SMOTE generates synthetic samples by interpolating between neighbouring
> records, which is valid for continuous attributes but produces meaningless
> values for categorical ones — an interpolation between two category codes does
> not correspond to any real category. Since sixteen of the twenty model
> attributes in this study are categorical, SMOTE-NC is the appropriate choice:
> it interpolates the continuous attributes and assigns each categorical
> attribute the most frequent value among the neighbours used to generate the
> sample.
>
> Resampling is applied within the modelling pipeline rather than to the dataset
> as a whole. This ensures that synthetic samples are generated only from
> training data at each stage of cross-validation, and that no synthetic record
> can influence the validation or test partitions. Applying oversampling before
> partitioning is a common source of optimistically biased results and is avoided
> here by construction.

**Reason:** Names the method actually used and states the leakage control, which
is a methodological strength currently going unreported.

---

## Correction 7 — Excluded attributes (new subsection after Table 3.0)

Table 3.0 lists all twenty-two attributes, but three are withheld from the
models and this is not stated anywhere.

**Insert after Table 3.0:**

> ### 3.1.2 Attribute exclusion policy
>
> Three attributes listed in Table 3.0 are withheld from the models.
>
> `Typhoid Status` is the target variable and is therefore never a predictor.
>
> `Blood Culture Result` is excluded because it is the confirmatory gold standard
> for typhoid diagnosis. In the dataset it exhibits a Cramér's V of 0.9999 with
> the target, that is, near-perfect association. A model consuming this attribute
> would not be predicting a diagnosis but restating one that has already been
> established, and its reported accuracy would measure only its ability to copy a
> column. Since the stated purpose of this study is early diagnostic support
> before laboratory confirmation is available, including this attribute would
> invalidate the premise of the work.
>
> `Complications` is excluded on two grounds. It is missing in 96.98% of records,
> leaving too few observations to support reliable imputation, and it encodes
> post-diagnostic severity information that is not available at the point of
> decision.
>
> The remaining twenty attributes — four numerical and sixteen categorical —
> constitute the feature set for the primary model. Two further attribute
> policies are evaluated as sensitivity analyses and are described in Section 3.4.

**Reason:** This is a substantive methodological decision that strengthens the
work. Leaving it undocumented invites the assumption that it was overlooked.

---

## Correction 8 — Attribute policies for sensitivity analysis (new Section 3.4)

**Insert as a new section:**

> ### 3.4 Attribute policies
>
> Three attribute policies are evaluated, so that the contribution of different
> categories of clinical information can be separated.
>
> **Routine.** All twenty permitted attributes, including the Widal and Typhidot
> serological results. This is the primary policy.
>
> **Pre-laboratory.** The Widal and Typhidot results are additionally withheld,
> leaving eighteen attributes. Chapter One establishes that these tests are
> frequently unavailable or unreliable in the settings this study targets, so
> this policy measures the diagnostic performance obtainable where no serological
> testing can be performed at all.
>
> **Ablation.** `Fever Duration (Days)` is withheld, leaving nineteen attributes.
> The rationale for this policy follows from the signal audit reported in Section
> 4.7 and is discussed there.
>
> Each policy is trained, optimised and evaluated independently under the same
> protocol, and the results are compared in Table 4.11.

**Reason:** Introduces the sensitivity analyses in the methodology so that
Chapter Four is not presenting analyses the reader has not been prepared for.

---

## Correction 9 — Validation methods (Table 3.3 and surrounding text)

Table 3.3 lists cross-validation and hold-out validation without stating how
each is used. **Insert before the table:**

> Both validation strategies listed in Table 3.3 are applied, at different stages
> and for different purposes. Stratified k-fold cross-validation with k = 3 is
> used during hyperparameter optimisation, where each candidate configuration is
> scored as the mean macro F1 across the three folds. A stratified hold-out
> partition comprising 20% of the records is set aside before any training,
> optimisation or resampling takes place, and is used exclusively to produce the
> final performance figures reported in Chapter Four. No record in the hold-out
> partition influences model selection at any stage.
>
> Macro F1 is used as the selection criterion in preference to accuracy. With a
> majority class representing 69.81% of records, a classifier that predicted the
> majority class for every patient would achieve 69.81% accuracy while never
> identifying a single case of typhoid. Macro F1 averages the per-class F1 scores
> with equal weight, so performance on the minority classes cannot be concealed
> by performance on the majority class.

**Reason:** Makes the protocol explicit and pre-empts the question of why
accuracy was not the selection metric.

---

## Additional note on Section 3.2.6 (Model deployment)

The section describes deployment in the future tense as a plan. It is now
implemented and can be described in the past tense, with these specifics
available: a Flask-based interface with attribute values constrained to those
present in the training data, input validation, calibrated probability output via
Platt scaling, a per-prediction explanation of which findings moved the result,
and a JSON endpoint (`POST /api/predict`) for point-of-care clients. Measured
inference latency is 9.96 milliseconds per record, which substantiates the
real-time claim in Objective 3 rather than asserting it.
