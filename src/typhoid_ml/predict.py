"""Model loading, input validation and inference for the deployment layer."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

# Saved pipelines reference the top-level package name `typhoid_ml`. If the app
# was started as `src.typhoid_ml.app`, that name is not importable and joblib
# cannot reconstruct the custom transformers, so make `src/` importable here.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from .config import BINARY_POSITIVE, DATA_PATH, DEFAULT_POLICY, MODEL_DIR
from .data import (
    category_levels,
    feature_frame,
    load_dataset,
    numeric_ranges,
    optional_features,
)
from .explain import explain_prediction, reference_record

# Value the interface and API use to mean "this was not assessed / not recorded".
NOT_RECORDED = "Not recorded"


def model_path(target_mode: str = "binary", policy: str = DEFAULT_POLICY) -> Path:
    suffix = f"_{target_mode}" + ("" if policy == DEFAULT_POLICY else f"_{policy}")
    return MODEL_DIR / f"svm{suffix}.joblib"


def available_models() -> dict[str, Path]:
    return {
        p.stem.replace("svm_", ""): p
        for p in sorted(MODEL_DIR.glob("svm_*.joblib"))
    }


@lru_cache(maxsize=4)
def load_model(target_mode: str = "binary", policy: str = DEFAULT_POLICY):
    path = model_path(target_mode, policy)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run: python scripts/train_model.py "
            f"--mode all --target {target_mode}"
        )
    model = joblib.load(path)
    meta_path = path.with_name(path.stem + "_metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return model, meta


@lru_cache(maxsize=2)
def form_schema(policy: str = DEFAULT_POLICY) -> dict:
    """Field definitions driven by the dataset itself, so the UI can never offer
    a category the model has never seen."""
    df = load_dataset(DATA_PATH)
    _, numeric, categorical = feature_frame(df, policy)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "levels": category_levels(df, categorical),
        "ranges": numeric_ranges(df, numeric),
        "reference": reference_record(df, numeric, categorical),
        # Attributes that are missing for some patients in the source data, and
        # may therefore legitimately be left unanswered at the point of entry.
        "optional": optional_features(df, numeric + categorical),
        "not_recorded_value": NOT_RECORDED,
    }


class ValidationError(ValueError):
    pass


def validate_record(raw: dict, policy: str = DEFAULT_POLICY) -> dict:
    """Coerce and validate a submitted record against the dataset schema."""
    schema = form_schema(policy)
    optional = set(schema["optional"])
    record, errors = {}, {}

    def _is_blank(v):
        if v is None:
            return True
        if isinstance(v, float) and v != v:  # NaN
            return True
        return str(v).strip() in ("", NOT_RECORDED)

    for field in schema["numeric"]:
        value = raw.get(field, "")
        if _is_blank(value):
            if field in optional:
                record[field] = float("nan")  # imputed by the pipeline
                continue
            errors[field] = "required"
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[field] = "must be a number"
            continue
        rng = schema["ranges"][field]
        # Values outside the observed range are accepted but flagged, because a
        # real patient can legitimately fall outside the training distribution.
        record[field] = number
        if number < rng["min"] or number > rng["max"]:
            errors.setdefault("_warnings", {})[field] = (
                f"outside the observed range {rng['min']:g}-{rng['max']:g}"
            )

    for field in schema["categorical"]:
        value = raw.get(field, "")
        if _is_blank(value):
            if field in optional:
                # float('nan'), not None: SimpleImputer recognises NaN as
                # missing, whereas None in an object column is treated as a
                # category in its own right and would be encoded as unknown.
                record[field] = float("nan")
                continue
            errors[field] = "required"
            continue
        value = str(value)
        if value not in schema["levels"][field]:
            errors[field] = f"must be one of: {', '.join(schema['levels'][field])}"
            continue
        record[field] = value

    warnings = errors.pop("_warnings", {})
    if errors:
        raise ValidationError(json.dumps(errors))
    return {"record": record, "warnings": warnings}


def predict_record(
    record: dict,
    target_mode: str = "binary",
    policy: str = DEFAULT_POLICY,
    explain: bool = True,
) -> dict:
    model, meta = load_model(target_mode, policy)
    frame = pd.DataFrame([record])
    label = str(model.predict(frame)[0])

    out = {"prediction": label, "target_mode": target_mode, "feature_policy": policy}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame)[0]
        out["probabilities"] = {
            str(c): round(float(p), 4) for c, p in zip(model.classes_, probs)
        }
        out["confidence"] = round(float(max(probs)), 4)

    if explain and hasattr(model, "predict_proba"):
        positive = BINARY_POSITIVE if target_mode == "binary" else label
        schema = form_schema(policy)
        out["explanation"] = explain_prediction(
            model, record, schema["reference"], positive
        )

    out["model"] = {
        "kernel": meta.get("best_params", {}).get("kernel"),
        "C": meta.get("best_params", {}).get("C"),
        "gamma": meta.get("best_params", {}).get("gamma"),
        "trained_at": meta.get("trained_at"),
    }
    return out
