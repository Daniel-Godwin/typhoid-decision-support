"""JSON API for point-of-care and integration clients.

Authenticated by session (a signed-in clinician's browser) — the same role rules
apply as in the interface. CSRF is exempt because the API takes JSON, not forms.
"""
from __future__ import annotations

import json

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required

from ..extensions import db, limiter
from ..models import Assessment, Role
from ..security import record_audit
from ..services import prediction_service as ps

api_bp = Blueprint("api", __name__)


@api_bp.get("/schema")
@login_required
def schema():
    policy = request.args.get("policy", current_app.config["DEFAULT_FEATURE_POLICY"])
    if policy not in ps.available_policies():
        return jsonify({"error": "unknown_policy", "allowed": ps.available_policies()}), 400
    s = ps.form_schema(policy)
    return jsonify({
        "policy": policy,
        "numeric": s["numeric"],
        "categorical": s["categorical"],
        "levels": s["levels"],
        "ranges": s["ranges"],
        "optional": s["optional"],
        "not_recorded_value": s["not_recorded_value"],
    })


@api_bp.get("/models")
@login_required
def models():
    out = {}
    for mode in ps.available_target_modes():
        available = ps.model_is_available(mode)
        out[mode] = {
            "available": available,
            "metadata": ps.model_metadata(mode) if available else None,
        }
    return jsonify(out)


@api_bp.post("/predict")
@login_required
@limiter.limit("120 per hour")
def predict():
    if current_user.role == Role.AUDITOR and not current_user.is_admin:
        return jsonify({"error": "forbidden", "message": "Auditors have read-only access."}), 403

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "invalid_json", "message": "A JSON body is required."}), 400

    record = payload.get("record", payload)
    target_mode = payload.get("target_mode", current_app.config["DEFAULT_TARGET_MODE"])
    policy = payload.get("policy", current_app.config["DEFAULT_FEATURE_POLICY"])
    persist = bool(payload.get("persist", True))

    if target_mode not in ps.available_target_modes():
        return jsonify({
            "error": "unknown_target_mode", "allowed": list(ps.available_target_modes())
        }), 400
    if policy not in ps.available_policies():
        return jsonify({"error": "unknown_policy", "allowed": ps.available_policies()}), 400

    try:
        outcome = ps.validate_and_predict(
            record,
            target_mode=target_mode,
            policy=policy,
            explain=bool(payload.get("explain", True)),
            triage_threshold=current_app.config["TRIAGE_THRESHOLD"],
        )
    except ps.ValidationError as exc:
        record_audit("api_prediction_rejected", category="prediction",
                     detail="validation failed", success=False)
        return jsonify({"error": "validation_failed", "fields": json.loads(str(exc))}), 400
    except ps.ModelUnavailable as exc:
        current_app.logger.error("api model unavailable: %s", exc)
        return jsonify({"error": "model_unavailable", "message": str(exc)}), 503
    except Exception:  # pragma: no cover - defensive
        current_app.logger.exception("api prediction failed")
        return jsonify({"error": "internal_error"}), 500

    outcome["triage"] = ps.triage_recommendation(outcome)
    outcome["disclaimer"] = ps.DISCLAIMER

    if persist:
        assessment = Assessment(
            reference=outcome["reference"],
            clinician_id=current_user.id,
            input_json=json.dumps(outcome.get("record", {}), default=str),
            target_mode=target_mode,
            feature_policy=policy,
            model_kernel=(outcome.get("model") or {}).get("kernel"),
            model_trained_at=(outcome.get("model") or {}).get("trained_at"),
            prediction=outcome["prediction"],
            confidence=outcome.get("confidence"),
            probability_positive=outcome.get("probability_positive"),
            probabilities_json=json.dumps(outcome.get("probabilities", {})),
            explanation_json=json.dumps(outcome.get("explanation", {}), default=str),
            warnings_json=json.dumps(outcome.get("warnings", {})),
            triage_threshold=outcome.get("triage_threshold"),
            flagged_for_testing=outcome.get("flagged_for_testing", False),
            latency_ms=outcome.get("latency_ms"),
        )
        db.session.add(assessment)
        db.session.commit()
        record_audit("api_prediction_created", category="prediction",
                     entity_type="Assessment", entity_id=assessment.reference,
                     detail=f"model={target_mode} prediction={outcome['prediction']}")

    outcome.pop("record", None)
    return jsonify(outcome)


@api_bp.get("/assessments")
@login_required
def assessments():
    limit = min(request.args.get("limit", 25, type=int), 200)
    query = Assessment.query
    if current_user.role == Role.CLINICIAN and not current_user.is_admin:
        query = query.filter(Assessment.clinician_id == current_user.id)
    rows = query.order_by(Assessment.created_at.desc()).limit(limit).all()
    return jsonify({"count": len(rows), "results": [r.to_dict() for r in rows]})


@api_bp.get("/assessments/<reference>")
@login_required
def assessment(reference: str):
    row = Assessment.query.filter_by(reference=reference).first()
    if row is None:
        return jsonify({"error": "not_found"}), 404
    if (
        current_user.role == Role.CLINICIAN
        and not current_user.is_admin
        and row.clinician_id != current_user.id
    ):
        return jsonify({"error": "forbidden"}), 403
    data = row.to_dict()
    data["inputs"] = row.inputs
    data["explanation"] = row.explanation
    return jsonify(data)
