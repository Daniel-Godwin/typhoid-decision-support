"""SVM pipeline construction and cross-validated hyperparameter search."""
from __future__ import annotations

from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from .config import CV_FOLDS, KERNEL_GRIDS, RANDOM_STATE, SCORING
from .preprocessing import OrdinalPreprocessor, PostSMOTEEncoder, SmoteNCResampler


def build_pipeline(
    numeric_features,
    categorical_features,
    kernel="rbf",
    C=1.0,
    gamma="scale",
    degree=3,
    probability=False,
    random_state=RANDOM_STATE,
):
    """Raw features -> impute/ordinal -> SMOTENC -> one-hot/scale -> SVC.

    When ``probability`` is set the SVC is wrapped in ``CalibratedClassifierCV``
    (Platt scaling fitted out-of-fold). An SVM is a maximum-margin classifier and
    its decision function is not a probability, so calibration is what turns the
    margin into the confidence estimate the decision-support interface reports.
    """
    n_num, n_cat = len(numeric_features), len(categorical_features)
    svc = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        class_weight="balanced",
        cache_size=500,
        random_state=random_state,
    )
    classifier = (
        CalibratedClassifierCV(svc, method="sigmoid", ensemble=False, cv=3) if probability else svc
    )
    return Pipeline(
        [
            ("preprocess", OrdinalPreprocessor(numeric_features, categorical_features)),
            ("smote", SmoteNCResampler(n_num, n_cat, random_state=random_state)),
            (
                "encode_scale",
                PostSMOTEEncoder(n_num, n_cat, categorical_names=categorical_features),
            ),
            ("svc", classifier),
        ]
    )


def build_search(numeric_features, categorical_features, cv=CV_FOLDS, n_jobs=-1, verbose=1):
    """GridSearchCV over the kernel-specific search spaces."""
    pipe = build_pipeline(numeric_features, categorical_features)
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(
        pipe,
        KERNEL_GRIDS,
        scoring=SCORING,
        cv=cv_obj,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=False,  # refit is performed explicitly on the full training partition
        error_score="raise",
    )


def n_candidates() -> int:
    total = 0
    for grid in KERNEL_GRIDS:
        n = 1
        for values in grid.values():
            n *= len(values)
        total += n
    return total
