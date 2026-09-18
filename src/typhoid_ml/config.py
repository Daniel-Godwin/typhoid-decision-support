"""Central configuration: paths, reproducibility, feature policy and search spaces."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "typhoid_dataset.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TARGET = "Typhoid Status"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3

# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------
# The supplied dataset carries four clinical outcome classes. The thesis
# methodology (Chapter 3) specifies a binary diagnosis target. Both are
# supported: `binary` is the primary diagnostic model, `multiclass` is the
# secondary severity-stratification model.
NEGATIVE_CLASS = "Normal or No Typhoid"
MULTICLASS_LABELS = [
    "Normal or No Typhoid",
    "Acute Typhoid Fever",
    "Relapsing Typhoid",
    "Complicated Typhoid",
]
BINARY_LABELS = ["No Typhoid", "Typhoid"]
BINARY_POSITIVE = "Typhoid"

TARGET_MODES = ("binary", "multiclass")


def labels_for(target_mode: str) -> list[str]:
    return BINARY_LABELS if target_mode == "binary" else MULTICLASS_LABELS


# ---------------------------------------------------------------------------
# Feature policy
# ---------------------------------------------------------------------------
# `Blood Culture Result` is the confirmatory gold standard and `Complications`
# encodes post-diagnostic severity information with 96.98% missingness. Both are
# excluded from every model so that the system is evaluated as a decision-support
# tool rather than a restatement of a completed laboratory workup.
LEAKAGE_EXCLUDED = ["Blood Culture Result", "Complications"]

# ---------------------------------------------------------------------------
# Features dropped at supervisory review
# ---------------------------------------------------------------------------
# The review directed that prediction should rest on the presenting symptoms
# and the history a clinician can obtain at the point of care, rather than on
# laboratory investigations that a resource-limited setting may be unable to
# perform. Seven attributes were therefore dropped during preprocessing.
#
# Each is listed with the reason. The haematology and serology are removed
# because they are laboratory investigations; the two optional symptom fields
# because a quarter of records lack them and an optional field that changes the
# prediction is a liability at the point of care; vaccination status because it
# is history of prophylaxis rather than of disease.
DROPPED_AT_REVIEW = {
    "White Blood Cell Count": "laboratory investigation, not available at the point of care",
    "Platelet Count": "laboratory investigation, not available at the point of care",
    "Widal Test": "serological investigation; also independent of the label (V = 0.0014)",
    "Typhidot Test": "serological investigation; also independent of the label (V = 0.0065)",
    "Typhoid Vaccination Status": "prophylaxis history, not a presenting sign or disease history",
    "Gastrointestinal Symptoms": "absent for 24.96% of records; optional field at entry",
    "Ongoing Infection in Society": "absent for 24.70% of records; community context, not a patient sign",
}

# `Neurological Symptoms` records one of Confusion, Delirium or Headache, or is
# absent. The review directed that it be reduced to headache alone, to match the
# symptom named in Objective 1. It is therefore recoded as a binary indicator:
# Headache -> Yes, and Confusion, Delirium or absent -> No. No discriminative
# information is lost, because there was none: P(typhoid) is 0.2972 to 0.3085
# across the four levels against an overall rate of 0.3019.
HEADACHE_SOURCE = "Neurological Symptoms"
HEADACHE_FEATURE = "Headache"

NUMERIC_FEATURES = [
    "Age",
    "Fever Duration (Days)",
    "White Blood Cell Count",
    "Platelet Count",
]

# Serological / rapid tests.
LAB_TEST_FEATURES = ["Widal Test", "Typhidot Test"]

CATEGORICAL_FEATURES = [
    "Gender",
    "Location",
    "Socioeconomic Status",
    "Water Source Type",
    "Sanitation Facilities",
    "Hand Hygiene",
    "Consumption of Street Food",
    "Gastrointestinal Symptoms",
    "Neurological Symptoms",
    "Skin Manifestations",
    "Typhoid Vaccination Status",
    "Previous History of Typhoid",
    "Weather Condition",
    "Ongoing Infection in Society",
    "Widal Test",
    "Typhidot Test",
]

# --- the feature space after the review -----------------------------------
SYMPTOM_NUMERIC = ["Age", "Fever Duration (Days)"]
SYMPTOM_CATEGORICAL = [
    "Gender",
    "Location",                     # retained: Objective 4 requires it
    "Socioeconomic Status",
    "Water Source Type",
    "Sanitation Facilities",
    "Hand Hygiene",
    "Consumption of Street Food",
    "Weather Condition",
    "Skin Manifestations",
    HEADACHE_FEATURE,
    "Previous History of Typhoid",
]

FEATURE_POLICIES = {
    # Post-review policy. Presenting symptoms, environmental exposure and
    # disease history only. Every field is mandatory at entry, so the
    # application no longer carries an optional field that moves the result.
    "symptom_based": {
        "numeric": SYMPTOM_NUMERIC,
        "categorical": SYMPTOM_CATEGORICAL,
    },
    # The full permitted attribute set, retained as the pre-review comparator
    # so Chapter Four can report what the seven dropped attributes contributed.
    "routine": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    },
    # Pre-laboratory sensitivity analysis: no Widal / Typhidot.
    "clinical_only": {
        "numeric": NUMERIC_FEATURES,
        "categorical": [c for c in CATEGORICAL_FEATURES if c not in LAB_TEST_FEATURES],
    },
    # Ablation: `Fever Duration (Days)` removed from the post-review set. In
    # this dataset every record with a fever duration of one day or more is
    # labelled typhoid, so the variable is a deterministic proxy for the target.
    # Excluding it measures what the remaining attributes actually contribute.
    # See docs/DATASET_AUDIT.md.
    "no_fever_duration": {
        "numeric": [c for c in SYMPTOM_NUMERIC if c != "Fever Duration (Days)"],
        "categorical": SYMPTOM_CATEGORICAL,
    },
}
DEFAULT_POLICY = "symptom_based"

# Subgroup used for the scalability evaluation (Objective 4).
SUBGROUP_COLUMN = "Location"

# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
# Kernel-specific grids. A single cartesian product over kernel x C x gamma x
# degree wastes fits (gamma is ignored by the linear kernel, degree by linear
# and RBF), so the space is expressed per kernel instead.
KERNEL_GRIDS = [
    {"svc__kernel": ["linear"], "svc__C": [0.1, 1.0, 10.0]},
    {"svc__kernel": ["rbf"], "svc__C": [0.1, 1.0, 10.0], "svc__gamma": ["scale", 0.01, 0.1]},
    {
        "svc__kernel": ["poly"],
        "svc__C": [0.1, 1.0, 10.0],
        "svc__degree": [2, 3],
        "svc__gamma": ["scale", 0.1],
    },
]

SCORING = "f1_macro"

# The multiclass problem oversamples to ~4x the majority class, which makes an
# exhaustive search on the full training partition impractical. The search is
# therefore run on a stratified subsample and the winning configuration is
# refitted on the complete training partition.
SEARCH_SUBSAMPLE = {"binary": None, "multiclass": 10000}


def artefact_name(stem: str, target_mode: str, policy: str = DEFAULT_POLICY, ext: str = "json") -> str:
    suffix = "" if policy == DEFAULT_POLICY else f"_{policy}"
    return f"{stem}_{target_mode}{suffix}.{ext}"
