"""The only bridge between the web application and the machine-learning pipeline.

Nothing else in `webapp` imports `typhoid_ml`. Everything the application needs
from the model — the input schema, validation, inference, explanation — is
requested through this module, which delegates to the pipeline unchanged.

That boundary is deliberate. The ML package can be retrained, re-tuned or
replaced without touching a single view, and the web application can be
restructured without any risk to the trained artefacts or their preprocessing
requirements.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

# Make the sibling ML package importable regardless of how the app was started.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from typhoid_ml import predict as ml_predict  # noqa: E402
from typhoid_ml.config import (  # noqa: E402
    BINARY_POSITIVE,
    DEFAULT_POLICY,
    FEATURE_POLICIES,
    TARGET_MODES,
)

ValidationError = ml_predict.ValidationError
NOT_RECORDED = ml_predict.NOT_RECORDED


class ModelUnavailable(RuntimeError):
    """Raised when a requested model artefact is not present on disk."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def form_schema(policy: str = DEFAULT_POLICY) -> dict:
    """Field names, permitted values, ranges and reference record."""
    return ml_predict.form_schema(policy)


def available_target_modes() -> tuple[str, ...]:
    return TARGET_MODES


def available_policies() -> list[str]:
    return list(FEATURE_POLICIES)


def available_models() -> dict:
    return ml_predict.available_models()


def model_metadata(target_mode: str = "binary", policy: str = DEFAULT_POLICY) -> dict:
    try:
        _, meta = ml_predict.load_model(target_mode, policy)
    except FileNotFoundError as exc:
        raise ModelUnavailable(str(exc)) from exc
    return meta


def model_is_available(target_mode: str = "binary", policy: str = DEFAULT_POLICY) -> bool:
    return ml_predict.model_path(target_mode, policy).exists()


def preload(target_modes=("binary",), policy: str = DEFAULT_POLICY) -> dict:
    """Load artefacts at start-up so the first request is not slow.

    Failures are reported, never raised: a missing model must not stop the
    application from starting, it must show up on the dashboard.
    """
    status = {}
    for mode in target_modes:
        try:
            ml_predict.load_model(mode, policy)
            status[mode] = "loaded"
        except FileNotFoundError:
            status[mode] = "missing"
        except Exception as exc:  # pragma: no cover - defensive
            status[mode] = f"error: {exc}"
    return status


# ---------------------------------------------------------------------------
# Validation and inference
# ---------------------------------------------------------------------------
def validate(raw: dict, policy: str = DEFAULT_POLICY) -> dict:
    """Delegates to the pipeline's validator. Raises ValidationError."""
    return ml_predict.validate_record(raw, policy)


def run_prediction(
    record: dict,
    target_mode: str = "binary",
    policy: str = DEFAULT_POLICY,
    explain: bool = True,
    triage_threshold: float = 0.5,
) -> dict:
    """Run one prediction and return a result enriched for the application.

    The ML output is passed through unchanged; the extra keys are presentation
    concerns (reference id, latency, triage flag) that the pipeline should not
    know about.
    """
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unknown target mode '{target_mode}'")
    if policy not in FEATURE_POLICIES:
        raise ValueError(f"Unknown feature policy '{policy}'")

    started = time.perf_counter()
    try:
        result = ml_predict.predict_record(record, target_mode, policy, explain=explain)
    except FileNotFoundError as exc:
        raise ModelUnavailable(str(exc)) from exc
    latency_ms = (time.perf_counter() - started) * 1000

    probabilities = result.get("probabilities") or {}
    if target_mode == "binary":
        probability_positive = probabilities.get(BINARY_POSITIVE)
    else:
        negative = "Normal or No Typhoid"
        probability_positive = (
            round(1 - probabilities[negative], 4) if negative in probabilities else None
        )

    result["reference"] = f"ASM-{uuid.uuid4().hex[:10].upper()}"
    result["latency_ms"] = round(latency_ms, 2)
    result["probability_positive"] = probability_positive
    result["triage_threshold"] = triage_threshold
    result["flagged_for_testing"] = bool(
        probability_positive is not None and probability_positive >= triage_threshold
    )
    result["positive_label"] = BINARY_POSITIVE if target_mode == "binary" else None
    return result


def validate_and_predict(
    raw: dict,
    target_mode: str = "binary",
    policy: str = DEFAULT_POLICY,
    explain: bool = True,
    triage_threshold: float = 0.5,
) -> dict:
    """Validate then predict. Raises ValidationError or ModelUnavailable."""
    validated = validate(raw, policy)
    result = run_prediction(
        validated["record"], target_mode, policy, explain, triage_threshold
    )
    result["warnings"] = validated["warnings"]
    result["record"] = validated["record"]
    return result


def triage_recommendation(result: dict) -> dict:
    """Turn a probability into clinical guidance for a resource-limited setting."""
    p = result.get("probability_positive")
    if p is None:
        return {
            "level": "unknown",
            "headline": "No probability estimate available",
            "detail": "This model does not report calibrated probabilities.",
        }
    if result.get("flagged_for_testing"):
        if p >= 0.9:
            return {
                "level": "high",
                "headline": "Prioritise confirmatory testing",
                "detail": (
                    "The model places this patient well above the referral threshold. "
                    "Blood culture or an equivalent confirmatory test is indicated, and "
                    "empirical treatment may be considered pending results."
                ),
            }
        return {
            "level": "moderate",
            "headline": "Refer for confirmatory testing",
            "detail": (
                "The estimated probability is at or above the referral threshold. "
                "Confirmatory testing is recommended before treatment is committed."
            ),
        }
    if p >= 0.2:
        return {
            "level": "low",
            "headline": "Below referral threshold — review clinically",
            "detail": (
                "The estimate falls below the referral threshold but is not negligible. "
                "Consider other febrile illnesses and re-assess if symptoms persist."
            ),
        }
    return {
        "level": "minimal",
        "headline": "Typhoid unlikely on these findings",
        "detail": (
            "Investigate alternative causes of fever. Re-assess if the clinical "
            "picture changes."
        ),
    }


DISCLAIMER = (
    "Decision-support output only. This system does not replace blood culture, "
    "confirmatory laboratory testing or clinical judgement."
)
