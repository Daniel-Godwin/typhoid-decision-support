# System Architecture

The implementation follows the flow proposed in Chapter 3 (Figures 3.1–3.3)
while separating concerns into independently testable modules.

```text
                      ┌────────────────────────────┐
                      │  Raw typhoid dataset       │
                      │  31,087 x 23               │
                      └─────────────┬──────────────┘
                                    ↓
                      ┌────────────────────────────┐
                      │  Dataset audit             │
                      │  missingness, balance,     │
                      │  duplicates, category      │
                      │  levels                    │
                      └─────────────┬──────────────┘
                                    ↓
                      ┌────────────────────────────┐
                      │  Feature policy            │
                      │  drop target + leakage-    │
                      │  prone columns             │
                      └─────────────┬──────────────┘
                                    ↓
                      ┌────────────────────────────┐
                      │  Target construction       │
                      │  binary  |  four-class     │
                      └─────────────┬──────────────┘
                                    ↓
                      ┌────────────────────────────┐
                      │  Stratified 80:20 split    │
                      └─────────────┬──────────────┘
                                    ↓
      ══════════ pipeline (fitted on training folds only) ══════════
                                    ↓
                      ┌────────────────────────────┐
                      │  Median / mode imputation  │
                      │  + ordinal encoding        │
                      └─────────────┬──────────────┘
                                    ↓
                      ┌────────────────────────────┐
                      │  SMOTENC oversampling      │
                      └─────────────┬──────────────┘
                                    ↓
                      ┌────────────────────────────┐
                      │  One-hot encoding          │
                      │  + standardisation         │
                      └─────────────┬──────────────┘
                                    ↓
              ┌─────────────────────┼─────────────────────┐
              ↓                     ↓                     ↓
        Linear SVM           Polynomial SVM            RBF SVM
              └─────────────────────┼─────────────────────┘
                                    ↓
                      ┌────────────────────────────┐
                      │  Grid Search               │
                      │  + StratifiedKFold (k=3)   │
                      │  selection: macro F1       │
                      └─────────────┬──────────────┘
                                    ↓
                      ┌────────────────────────────┐
                      │  Refit on full training    │
                      │  set + Platt calibration   │
                      └─────────────┬──────────────┘
                                    ↓
              ┌─────────────────────┼─────────────────────┐
              ↓                     ↓                     ↓
    Held-out evaluation    Subgroup analysis      Latency measurement
    accuracy, precision,   by Location            per-record and batch
    recall, F1, ROC,       (Objective 4)
    confusion matrix
                                    ↓
                      ┌────────────────────────────┐
                      │  models/svm_<target>.joblib│
                      └─────────────┬──────────────┘
                                    ↓
                      ┌────────────────────────────┐
                      │  Flask decision support    │
                      │  + JSON API                │
                      └────────────────────────────┘
```

## Why this ordering

SMOTENC requires a dense numeric matrix in which categorical columns are
identifiable by position, so the raw mixed-type frame is imputed and
ordinal-encoded first. Synthetic samples are generated *inside* the pipeline,
which is what guarantees resampling never sees validation or test data. Only
afterwards are categories one-hot expanded and numeric columns standardised,
because an SVM is sensitive to feature magnitude and would otherwise treat the
arbitrary ordinal category codes as an ordered scale.

Probability calibration sits at the end because an SVM's decision function is a
signed distance from the separating hyperplane, not a probability. Platt scaling
(`CalibratedClassifierCV`, fitted out of fold) is what turns the margin into the
confidence figure the interface reports.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Paths, random state, feature policies, target definitions, search spaces |
| `data.py` | Loading, auditing, target construction, feature selection, category levels |
| `preprocessing.py` | Imputation, ordinal encoding, SMOTENC sampler, one-hot + scaling |
| `model.py` | SVM pipeline assembly, calibration wrapper, GridSearchCV construction |
| `train.py` | Experiment orchestration (audit → baseline → grid → final) |
| `evaluate.py` | Metrics, confusion matrices, ROC/PR curves, subgroup analysis, latency |
| `explain.py` | Occlusion-based per-prediction attribution, linear coefficient view |
| `predict.py` | Model loading, schema, input validation, inference |
| `app.py` | Flask interface and JSON API |

## Search space

Rather than one cartesian product over kernel × C × γ × degree — which wastes
fits, since γ is ignored by the linear kernel and `degree` by both linear and
RBF — the space is expressed per kernel:

| Kernel | C | γ | degree | Candidates |
|---|---|---|---|---|
| linear | 0.1, 1, 10 | — | — | 3 |
| RBF | 0.1, 1, 10 | scale, 0.01, 0.1 | — | 9 |
| polynomial | 0.1, 1, 10 | scale, 0.1 | 2, 3 | 12 |

24 candidates × 3 folds = 72 fits.

## Computational note

The four-class problem oversamples to approximately four times the majority
class (~87,000 training rows), and SVC training cost grows roughly with the
square of the sample count. An exhaustive search at that scale is impractical,
so the four-class search runs on a stratified subsample and the winning
configuration is refitted on the complete training partition. This is recorded
explicitly in `reports/grid_search_summary_multiclass.json`. The binary search
runs on the full training partition.
