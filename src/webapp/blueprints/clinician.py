"""Clinician workspace: dashboard, patients, assessments."""
from __future__ import annotations

import json
from datetime import timedelta

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import func, or_

from ..extensions import db
from ..forms import PatientForm
from ..models import Assessment, Patient, Role, User, utcnow
from ..security import clinician_required, record_audit, roles_required
from ..services import prediction_service as ps

clinician_bp = Blueprint("clinician", __name__)

FIELD_GROUPS = [
    ("Demographics", ["Age", "Gender", "Location", "Socioeconomic Status"]),
    ("Environmental exposure", [
        "Water Source Type", "Sanitation Facilities", "Hand Hygiene",
        "Consumption of Street Food", "Weather Condition", "Ongoing Infection in Society",
    ]),
    ("Clinical presentation", [
        "Fever Duration (Days)", "Gastrointestinal Symptoms",
        "Neurological Symptoms", "Skin Manifestations",
    ]),
    ("Laboratory and history", [
        "White Blood Cell Count", "Platelet Count", "Widal Test", "Typhidot Test",
        "Typhoid Vaccination Status", "Previous History of Typhoid",
    ]),
]

FIELD_HELP = {
    "Age": "Patient age in years",
    "Fever Duration (Days)": "Days of fever before presentation",
    "White Blood Cell Count": "WBC, cells per microlitre",
    "Platelet Count": "Platelets per microlitre",
}


def _grouped_fields(schema):
    known = set(schema["numeric"]) | set(schema["categorical"])
    optional = set(schema.get("optional", []))
    groups, placed = [], set()
    for title, names in FIELD_GROUPS:
        items = []
        for name in names:
            if name not in known:
                continue
            placed.add(name)
            items.append({
                "name": name,
                "kind": "numeric" if name in schema["numeric"] else "categorical",
                "options": schema["levels"].get(name, []),
                "range": schema["ranges"].get(name),
                "help": FIELD_HELP.get(name, ""),
                "optional": name in optional,
            })
        if items:
            groups.append({"title": title, "fields": items})
    leftovers = sorted(known - placed)
    if leftovers:
        groups.append({"title": "Other", "fields": [{
            "name": n,
            "kind": "numeric" if n in schema["numeric"] else "categorical",
            "options": schema["levels"].get(n, []),
            "range": schema["ranges"].get(n),
            "help": "",
            "optional": n in optional,
        } for n in leftovers]})
    return groups


def _visible_assessments():
    """Clinicians see their own work; admins and auditors see everything."""
    query = Assessment.query
    if current_user.role == Role.CLINICIAN and not current_user.is_admin:
        query = query.filter(Assessment.clinician_id == current_user.id)
    return query


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@clinician_bp.route("/")
@login_required
def dashboard():
    since = utcnow() - timedelta(days=30)
    base = _visible_assessments()

    total = base.count()
    recent = base.filter(Assessment.created_at >= since).count()
    flagged = base.filter(Assessment.flagged_for_testing.is_(True)).count()
    positives = base.filter(
        ~Assessment.prediction.in_(["No Typhoid", "Normal or No Typhoid"])
    ).count()

    latest = base.order_by(Assessment.created_at.desc()).limit(8).all()

    patient_query = Patient.query
    if current_user.role == Role.CLINICIAN and not current_user.is_admin:
        patient_query = patient_query.filter(Patient.created_by_id == current_user.id)

    breakdown = (
        base.with_entities(Assessment.prediction, func.count(Assessment.id))
        .group_by(Assessment.prediction)
        .all()
    )

    model_status = {}
    for mode in ("binary", "multiclass"):
        model_status[mode] = ps.model_is_available(mode)

    return render_template(
        "clinician/dashboard.html",
        stats={
            "total": total,
            "recent": recent,
            "flagged": flagged,
            "positives": positives,
            "patients": patient_query.count(),
        },
        latest=latest,
        breakdown=breakdown,
        model_status=model_status,
        threshold=current_app.config["TRIAGE_THRESHOLD"],
    )


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------
@clinician_bp.route("/assess", methods=["GET", "POST"])
@login_required
@clinician_required
def assess():
    policy = current_app.config["DEFAULT_FEATURE_POLICY"]
    schema = ps.form_schema(policy)
    groups = _grouped_fields(schema)

    patients = Patient.query
    if current_user.role == Role.CLINICIAN and not current_user.is_admin:
        patients = patients.filter(Patient.created_by_id == current_user.id)
    patients = patients.order_by(Patient.full_name).all()

    target_mode = request.values.get("target_mode", current_app.config["DEFAULT_TARGET_MODE"])
    if target_mode not in ps.available_target_modes():
        target_mode = "binary"

    submitted, errors, result, triage = {}, None, None, None

    if request.method == "POST":
        submitted = {
            k: v for k, v in request.form.items()
            if k not in ("csrf_token", "target_mode", "patient_id")
        }
        patient_id = request.form.get("patient_id") or None
        try:
            outcome = ps.validate_and_predict(
                submitted,
                target_mode=target_mode,
                policy=policy,
                explain=True,
                triage_threshold=current_app.config["TRIAGE_THRESHOLD"],
            )
            triage = ps.triage_recommendation(outcome)
            assessment = _persist(outcome, patient_id, policy)
            result = outcome
            record_audit(
                "prediction_created",
                category="prediction",
                entity_type="Assessment",
                entity_id=assessment.reference,
                detail=(
                    f"model={target_mode} prediction={outcome['prediction']} "
                    f"confidence={outcome.get('confidence')}"
                ),
            )
            flash(f"Assessment {assessment.reference} recorded.", "success")
            return redirect(url_for("clinician.assessment_detail", reference=assessment.reference))

        except ps.ValidationError as exc:
            errors = json.loads(str(exc))
            record_audit(
                "prediction_rejected", category="prediction",
                detail=f"validation failed: {list(errors)}", success=False,
            )
            flash("Some fields need attention before the assessment can run.", "error")
        except ps.ModelUnavailable as exc:
            errors = {"_model": str(exc)}
            current_app.logger.error("model unavailable: %s", exc)
            flash("The prediction model is not available. Contact an administrator.", "error")
        except Exception as exc:  # pragma: no cover - defensive
            current_app.logger.exception("assessment failed")
            errors = {"_error": "An unexpected error occurred. It has been logged."}
            flash("The assessment could not be completed.", "error")

    return render_template(
        "clinician/assess.html",
        groups=groups,
        reference_record=schema["reference"],
        submitted=submitted,
        errors=errors,
        result=result,
        triage=triage,
        target_mode=target_mode,
        patients=patients,
        selected_patient=request.values.get("patient_id", ""),
        disclaimer=ps.DISCLAIMER,
    )


def _persist(outcome: dict, patient_id, policy: str) -> Assessment:
    model_info = outcome.get("model") or {}
    assessment = Assessment(
        reference=outcome["reference"],
        patient_id=int(patient_id) if patient_id else None,
        clinician_id=current_user.id,
        input_json=json.dumps(outcome.get("record", {}), default=str),
        target_mode=outcome["target_mode"],
        feature_policy=policy,
        model_kernel=model_info.get("kernel"),
        model_trained_at=model_info.get("trained_at"),
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
    return assessment


# ---------------------------------------------------------------------------
# Assessment history
# ---------------------------------------------------------------------------
@clinician_bp.route("/assessments")
@login_required
def assessments():
    page = request.args.get("page", 1, type=int)
    search = (request.args.get("q") or "").strip()
    outcome = request.args.get("outcome", "")

    query = _visible_assessments()
    if search:
        query = query.outerjoin(Patient).filter(
            or_(
                Assessment.reference.ilike(f"%{search}%"),
                Patient.patient_code.ilike(f"%{search}%"),
                Patient.full_name.ilike(f"%{search}%"),
            )
        )
    if outcome == "flagged":
        query = query.filter(Assessment.flagged_for_testing.is_(True))
    elif outcome == "positive":
        query = query.filter(~Assessment.prediction.in_(["No Typhoid", "Normal or No Typhoid"]))
    elif outcome == "negative":
        query = query.filter(Assessment.prediction.in_(["No Typhoid", "Normal or No Typhoid"]))

    pagination = query.order_by(Assessment.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    return render_template(
        "clinician/assessments.html",
        pagination=pagination,
        search=search,
        outcome=outcome,
    )


@clinician_bp.route("/assessments/<reference>")
@login_required
def assessment_detail(reference: str):
    assessment = Assessment.query.filter_by(reference=reference).first_or_404()
    if (
        current_user.role == Role.CLINICIAN
        and not current_user.is_admin
        and assessment.clinician_id != current_user.id
    ):
        abort(403)
    triage = ps.triage_recommendation({
        "probability_positive": assessment.probability_positive,
        "flagged_for_testing": assessment.flagged_for_testing,
    })
    return render_template(
        "clinician/assessment_detail.html",
        assessment=assessment,
        triage=triage,
        disclaimer=ps.DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------
@clinician_bp.route("/patients")
@login_required
def patients():
    page = request.args.get("page", 1, type=int)
    search = (request.args.get("q") or "").strip()
    query = Patient.query
    if current_user.role == Role.CLINICIAN and not current_user.is_admin:
        query = query.filter(Patient.created_by_id == current_user.id)
    if search:
        query = query.filter(
            or_(
                Patient.patient_code.ilike(f"%{search}%"),
                Patient.full_name.ilike(f"%{search}%"),
                Patient.location.ilike(f"%{search}%"),
            )
        )
    pagination = query.order_by(Patient.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    return render_template("clinician/patients.html", pagination=pagination, search=search)


@clinician_bp.route("/patients/new", methods=["GET", "POST"])
@login_required
@clinician_required
def patient_new():
    form = PatientForm()
    if form.validate_on_submit():
        patient = Patient(
            patient_code=form.patient_code.data.strip(),
            full_name=form.full_name.data.strip(),
            age=form.age.data,
            gender=form.gender.data or None,
            location=form.location.data or None,
            contact=(form.contact.data or "").strip() or None,
            notes=(form.notes.data or "").strip() or None,
            created_by_id=current_user.id,
        )
        db.session.add(patient)
        db.session.commit()
        record_audit("patient_created", category="patient", entity_type="Patient",
                     entity_id=patient.id, detail=patient.patient_code)
        flash(f"Patient {patient.patient_code} added.", "success")
        return redirect(url_for("clinician.patient_detail", patient_id=patient.id))
    return render_template("clinician/patient_form.html", form=form, patient=None)


@clinician_bp.route("/patients/<int:patient_id>")
@login_required
def patient_detail(patient_id: int):
    patient = db.get_or_404(Patient, patient_id)
    if (
        current_user.role == Role.CLINICIAN
        and not current_user.is_admin
        and patient.created_by_id != current_user.id
    ):
        abort(403)
    history = patient.assessments.order_by(Assessment.created_at.desc()).all()
    return render_template("clinician/patient_detail.html", patient=patient, history=history)


@clinician_bp.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@login_required
@clinician_required
def patient_edit(patient_id: int):
    patient = db.get_or_404(Patient, patient_id)
    if not current_user.is_admin and patient.created_by_id != current_user.id:
        abort(403)
    form = PatientForm(obj=patient, editing_patient=patient)
    if form.validate_on_submit():
        form.populate_obj(patient)
        patient.patient_code = form.patient_code.data.strip()
        db.session.commit()
        record_audit("patient_updated", category="patient", entity_type="Patient",
                     entity_id=patient.id, detail=patient.patient_code)
        flash("Patient record updated.", "success")
        return redirect(url_for("clinician.patient_detail", patient_id=patient.id))
    return render_template("clinician/patient_form.html", form=form, patient=patient)
