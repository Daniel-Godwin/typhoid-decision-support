"""Per-prediction explanation.

An SVM with a non-linear kernel has no directly readable coefficient vector, so
attribution is computed by occlusion: each feature in turn is reset to its
population reference value (median for numeric, mode for categorical) and the
model is re-scored. The resulting change in the predicted probability is the
contribution attributed to that feature's observed value. The method is
model-agnostic, works for every kernel, and answers the question a clinician
actually asks - "which of this patient's findings pushed the prediction?".

For a linear kernel a global coefficient view is also available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reference_record(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    ref = {}
    for c in numeric:
        ref[c] = float(pd.to_numeric(df[c], errors="coerce").median())
    for c in categorical:
        mode = df[c].dropna()
        ref[c] = str(mode.mode().iloc[0]) if len(mode) else ""
    return ref


def _positive_index(model, positive_label):
    classes = list(model.classes_)
    return classes.index(positive_label) if positive_label in classes else len(classes) - 1


def explain_prediction(
    model,
    record: dict,
    reference: dict,
    positive_label: str,
    top_n: int = 8,
) -> dict:
    """Occlusion attribution for a single record.

    Returns contributions in probability points; positive values pushed the
    prediction towards ``positive_label``.
    """
    features = [c for c in reference if c in record]
    base_df = pd.DataFrame([record])
    pos = _positive_index(model, positive_label)
    base_p = float(model.predict_proba(base_df)[0, pos])

    variants, changed = [], []
    for c in features:
        if str(record[c]) == str(reference[c]):
            continue
        perturbed = dict(record)
        perturbed[c] = reference[c]
        variants.append(perturbed)
        changed.append(c)

    def _logit(p, eps=1e-6):
        p = min(max(float(p), eps), 1 - eps)
        return float(np.log(p / (1 - p)))

    base_logit = _logit(base_p)

    contributions = []
    if variants:
        probs = model.predict_proba(pd.DataFrame(variants))[:, pos]
        for c, p in zip(changed, probs):
            contributions.append(
                {
                    "feature": c,
                    "value": record[c],
                    "reference": reference[c],
                    "contribution": round(float(base_p - p), 4),
                    # Log-odds delta stays informative when the probability
                    # saturates at 0 or 1 and a probability delta would read as
                    # zero for every feature.
                    "log_odds_contribution": round(base_logit - _logit(p), 4),
                }
            )
    contributions.sort(
        key=lambda d: (abs(d["log_odds_contribution"]), abs(d["contribution"])), reverse=True
    )
    return {
        "probability": round(base_p, 4),
        "positive_label": positive_label,
        "baseline_probability": round(
            float(model.predict_proba(pd.DataFrame([reference]))[0, pos]), 4
        ),
        "contributions": contributions[:top_n],
        "method": "occlusion (feature reset to population reference)",
    }


def linear_coefficients(model, numeric: list[str], categorical: list[str], top_n: int = 25):
    """Global weights when the fitted SVM uses a linear kernel; else None."""
    step = model.named_steps.get("svc")
    base = step
    if hasattr(step, "calibrated_classifiers_"):
        base = step.calibrated_classifiers_[0].estimator
    if getattr(base, "kernel", None) != "linear" or not hasattr(base, "coef_"):
        return None
    names = model.named_steps["encode_scale"].get_feature_names_out(
        np.array(list(numeric) + list(categorical))
    )
    coef = np.asarray(base.coef_)
    if coef.shape[0] != 1:
        return None
    order = np.argsort(-np.abs(coef[0]))[:top_n]
    return [{"feature": str(names[i]), "weight": round(float(coef[0][i]), 4)} for i in order]
