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

NUMERIC_FEATURES = [
    "Age",
    "Fever Duration (Days)",
    "White Blood Cell Count",
    "Platelet Count",
]

# Serological / rapid tests. Present in the routine policy (they are listed in
# the thesis dataset description) but removable for the pre-laboratory
# sensitivity analysis.
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

FEATURE_POLICIES = {
    # Routine policy: demographics + environment + symptoms + haematology + serology.
    "routine": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    },
    # Pre-laboratory policy: no Widal / Typhidot. Used as a sensitivity analysis
    # for settings where no serological testing is available at all.
    "clinical_only": {
        "numeric": NUMERIC_FEATURES,
        "categorical": [c for c in CATEGORICAL_FEATURES if c not in LAB_TEST_FEATURES],
    },
    # Ablation policy: `Fever Duration (Days)` is removed. In this dataset every
    # record with a fever duration of one day or more is labelled typhoid, so the
    # variable behaves as a deterministic proxy for the target. Excluding it
    # measures what the remaining clinical, environmental and serological
    # features actually contribute. See docs/DATASET_AUDIT.md.
    "no_fever_duration": {
        "numeric": [c for c in NUMERIC_FEATURES if c != "Fever Duration (Days)"],
        "categorical": CATEGORICAL_FEATURES,
    },
}
DEFAULT_POLICY = "routine"

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
