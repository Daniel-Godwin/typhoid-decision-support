# Dataset Audit

Source: `data/typhoid_dataset.csv` (Kaggle typhoid prediction dataset)

- Rows: **31,087**
- Columns: **23**
- Duplicate rows: **0**
- Target: **Typhoid Status**

> Section 3.1.1 of the proposal states 31,088 records. The file contains 31,087.

## Target distribution

### Four-class (as supplied)

| Class | Count | Percentage |
|---|---:|---:|
| Normal or No Typhoid | 21,701 | 69.81% |
| Acute Typhoid Fever | 5,649 | 18.17% |
| Relapsing Typhoid | 2,486 | 8.00% |
| Complicated Typhoid | 1,251 | 4.02% |

Imbalance ratio 17.3 : 1 between the largest and smallest class.

### Binary (as specified in Chapter 3)

| Class | Count | Percentage |
|---|---:|---:|
| No Typhoid | 21,701 | 69.81% |
| Typhoid | 9,386 | 30.19% |

## Missing values

| Feature | Missing | Percentage |
|---|---:|---:|
| Complications | 30,147 | 96.98% |
| Gastrointestinal Symptoms | 7,760 | 24.96% |
| Neurological Symptoms | 7,744 | 24.91% |
| Ongoing Infection in Society | 7,680 | 24.70% |

All remaining columns are complete.

---

# Signal audit — read this before interpreting any result

Association between each attribute and the binary target (Cramér's V for
categorical attributes, point-biserial *r* for numeric):

| Feature | Statistic | Value |
|---|---|---:|
| Blood Culture Result | Cramér's V | **0.9999** |
| Fever Duration (Days) | point-biserial *r* | **0.8138** |
| Complications | Cramér's V | 0.2685 |
| White Blood Cell Count | point-biserial *r* | 0.1948 |
| Water Source Type | Cramér's V | 0.0132 |
| Location | Cramér's V | 0.0120 |
| Neurological Symptoms | Cramér's V | 0.0091 |
| Typhidot Test | Cramér's V | 0.0065 |
| Gastrointestinal Symptoms | Cramér's V | 0.0052 |
| Hand Hygiene | Cramér's V | 0.0042 |
| Skin Manifestations | Cramér's V | 0.0022 |
| Age | point-biserial *r* | 0.0014 |
| Widal Test | Cramér's V | **0.0014** |
| Gender | Cramér's V | 0.0008 |

Full table: `reports/feature_association.csv`.

## Finding 1 — `Blood Culture Result` is the label

Cramér's V of 0.9999. It is the confirmatory gold standard, so a model that
consumes it is restating a completed diagnosis rather than predicting one. It is
excluded from every feature policy.

## Finding 2 — `Fever Duration (Days)` is a deterministic proxy for the label

| Fever duration | No Typhoid | Typhoid | Typhoid rate |
|---|---:|---:|---:|
| 0 days | 21,701 | 405 | 1.8% |
| ≥ 1 day | **0** | **8,981** | **100.0%** |

Not one record in the dataset has a fever duration of one day or more and a
negative typhoid status. The consequence is that a single rule reaches almost
the same score as the optimised model:

| Classifier | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Majority class | 0.6981 | 0.5000 | 0.4111 |
| Rule: fever duration ≥ 1 day | 0.9870 | 0.9784 | 0.9844 |
| One-level decision stump on fever duration | 0.9875 | 0.9792 | 0.9849 |
| **Optimised SVM (all features)** | **0.9881** | **0.9803** | **0.9857** |

The optimised SVM improves on a one-line rule by **0.11 accuracy points**.

## Finding 3 — `White Blood Cell Count` is a deterministic proxy for *Complicated Typhoid*

| White blood cell count | Normal | Acute | Relapsing | Complicated |
|---|---:|---:|---:|---:|
| ≤ 11,000 | 21,701 | 5,649 | 2,486 | **0** |
| > 11,000 | 0 | 0 | 0 | **1,251** |

Every record labelled *Complicated Typhoid* has a white cell count between
12,013 and 19,994; every other record is at or below 11,000. The two ranges do
not overlap by a single count. This is why the four-class model reaches a
**perfect F1 of 1.0000** on *Complicated Typhoid* — it is reading a threshold,
not a clinical pattern.

Combined with Finding 2, the four-class problem reduces to two rules plus an
unlearnable residue:

| Classifier | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Majority class | 0.6981 | 0.2500 | 0.2056 |
| Rules: WBC > 11,000 → Complicated; fever ≥ 1 → Acute | **0.9133** | 0.7396 | 0.7002 |
| Optimised four-class SVM | 0.8697 | 0.7394 | 0.7377 |

The two-rule baseline **beats the optimised SVM on accuracy** (0.9133 vs
0.8697). The SVM scores higher on macro F1 (0.7377 vs 0.7002) only because
SMOTENC balancing pushes it to attempt the *Relapsing* class, which the rule
baseline ignores entirely.

## Finding 4 — *Acute* and *Relapsing Typhoid* are not distinguishable

Restricted to those two classes, the strongest association of any attribute is
Cramér's V = 0.033 (`Water Source Type`) — indistinguishable from noise. The
trained model reflects this exactly: F1 0.6063 for *Acute* and 0.3516 for
*Relapsing*, with 478 of 1,130 *Acute* cases predicted as *Relapsing*. No model
can separate these classes on this data, because the distinction was not encoded
into the attributes.

## Finding 5 — the serological tests carry no signal

`Widal Test` (V = 0.0014) and `Typhidot Test` (V = 0.0065) are statistically
independent of the diagnosis in this dataset. That is clinically impossible in
real patient data — these are the serological tests the diagnosis rests on. Age,
gender, sanitation, water source and hygiene are likewise independent of the
outcome.

## Interpretation

The dataset is synthetic, and its generative structure is now fully
characterised:

1. `Blood Culture Result` mirrors the label exactly.
2. `Fever Duration (Days) ≥ 1` ⇒ typhoid, without exception.
3. `White Blood Cell Count > 11,000` ⇔ *Complicated Typhoid*, without exception.
4. *Acute* versus *Relapsing* is random with respect to every attribute.
5. Everything else — age, gender, sanitation, water source, hygiene, street
   food, weather, vaccination status, Widal, Typhidot — is independent of the
   outcome.

Realistic column names and plausible marginal distributions disguise this, but
the joint distribution does not support the clinical relationships the
attributes imply.

This does not invalidate the engineering work — the pipeline, optimisation,
evaluation protocol and deployment are all sound and would transfer unchanged to
real data. It does mean the headline accuracy must not be presented as evidence
that machine learning improves typhoid diagnosis.

## What the project does about it

1. **Trivial baselines are reported alongside the model** (`reports/table_4_0_baselines.csv`)
   so the SVM's marginal contribution is stated rather than implied.
2. **An ablation model excludes `Fever Duration (Days)`**
   (`--policy no_fever_duration`), measuring what the remaining clinical,
   environmental and serological attributes actually contribute.
3. **A pre-laboratory policy excludes Widal and Typhidot**
   (`--policy clinical_only`) as a further sensitivity analysis.
4. The limitation is documented for Chapters 3 and 5 in `docs/THESIS_ALIGNMENT.md`.

## Primary feature policy

Excluded from every model:

| Feature | Reason |
|---|---|
| `Typhoid Status` | Target |
| `Blood Culture Result` | Confirmatory gold standard; V = 0.9999 |
| `Complications` | Post-diagnostic severity; 96.98% missing |

Regenerate this audit with `python scripts/audit_dataset.py` and
`python scripts/baseline_rules.py`.
