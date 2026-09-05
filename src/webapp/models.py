"""Database models.

Four tables:

  users        — accounts, roles, credentials, lockout state
  patients     — patient records owned by the clinic
  assessments  — one row per prediction, storing the inputs, the model output
                 and which model version produced it, so any result can be
                 reproduced and audited later
  audit_logs   — append-only record of security- and data-relevant actions
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from flask_login import UserMixin
from sqlalchemy import func
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, login_manager


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role:
    ADMIN = "admin"
    CLINICIAN = "clinician"
    AUDITOR = "auditor"

    ALL = (ADMIN, CLINICIAN, AUDITOR)
    LABELS = {
        ADMIN: "Administrator",
        CLINICIAN: "Clinician",
        AUDITOR: "Auditor",
    }
    DESCRIPTIONS = {
        ADMIN: "Full access: user management, system logs, configuration and clinical features.",
        CLINICIAN: "Clinical access: register patients, run assessments, view own results.",
        AUDITOR: "Read-only access to assessments and audit logs. Cannot run or alter anything.",
    }


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(160), nullable=False)
    role = db.Column(db.String(32), nullable=False, default=Role.CLINICIAN, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    facility = db.Column(db.String(160))
    is_active_account = db.Column(db.Boolean, nullable=False, default=True)
    must_change_password = db.Column(db.Boolean, nullable=False, default=False)

    failed_logins = db.Column(db.Integer, nullable=False, default=0)
    locked_until = db.Column(db.DateTime(timezone=True))
    last_login_at = db.Column(db.DateTime(timezone=True))
    last_login_ip = db.Column(db.String(64))

    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    assessments = db.relationship("Assessment", back_populates="clinician", lazy="dynamic")
    patients = db.relationship("Patient", back_populates="created_by", lazy="dynamic")

    # --- credentials ------------------------------------------------------
    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password, method="pbkdf2:sha256:600000")

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    # --- lockout ----------------------------------------------------------
    @property
    def is_locked(self) -> bool:
        if self.locked_until is None:
            return False
        locked_until = self.locked_until
        if locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=timezone.utc)
        return locked_until > utcnow()

    def register_failed_login(self, max_attempts: int, lockout_minutes: int) -> bool:
        """Returns True if this failure locked the account."""
        self.failed_logins = (self.failed_logins or 0) + 1
        if self.failed_logins >= max_attempts:
            self.locked_until = utcnow() + timedelta(minutes=lockout_minutes)
            self.failed_logins = 0
            return True
        return False

    def register_successful_login(self, ip: str | None) -> None:
        self.failed_logins = 0
        self.locked_until = None
        self.last_login_at = utcnow()
        self.last_login_ip = ip

    # --- flask-login ------------------------------------------------------
    @property
    def is_active(self) -> bool:  # overrides UserMixin
        return bool(self.is_active_account) and not self.is_locked

    # --- roles ------------------------------------------------------------
    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN

    @property
    def is_clinician(self) -> bool:
        return self.role in (Role.CLINICIAN, Role.ADMIN)

    @property
    def role_label(self) -> str:
        return Role.LABELS.get(self.role, self.role.title())

    @property
    def initials(self) -> str:
        parts = [p for p in (self.full_name or "").split() if p]
        return "".join(p[0].upper() for p in parts[:2]) or self.email[:2].upper()

    def __repr__(self) -> str:
        return f"<User {self.email} ({self.role})>"


@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))


class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    patient_code = db.Column(db.String(40), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(160), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    location = db.Column(db.String(80))
    contact = db.Column(db.String(80))
    notes = db.Column(db.Text)

    created_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_by = db.relationship("User", back_populates="patients")

    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    assessments = db.relationship(
        "Assessment",
        back_populates="patient",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )

    @property
    def latest_assessment(self):
        return self.assessments.order_by(Assessment.created_at.desc()).first()

    def __repr__(self) -> str:
        return f"<Patient {self.patient_code}>"


class Assessment(db.Model):
    """One prediction. Stores inputs and outputs so results are reproducible."""

    __tablename__ = "assessments"

    id = db.Column(db.Integer, primary_key=True)
    reference = db.Column(db.String(40), unique=True, nullable=False, index=True)

    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=True, index=True)
    patient = db.relationship("Patient", back_populates="assessments")

    clinician_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    clinician = db.relationship("User", back_populates="assessments")

    # Inputs exactly as submitted to the ML pipeline.
    input_json = db.Column(db.Text, nullable=False)

    # Model identification, so a result can always be traced to what produced it.
    target_mode = db.Column(db.String(32), nullable=False)
    feature_policy = db.Column(db.String(40), nullable=False)
    model_kernel = db.Column(db.String(20))
    model_trained_at = db.Column(db.String(40))

    # Outputs.
    prediction = db.Column(db.String(80), nullable=False, index=True)
    confidence = db.Column(db.Float)
    probability_positive = db.Column(db.Float)
    probabilities_json = db.Column(db.Text)
    explanation_json = db.Column(db.Text)
    warnings_json = db.Column(db.Text)

    triage_threshold = db.Column(db.Float)
    flagged_for_testing = db.Column(db.Boolean, default=False, index=True)

    latency_ms = db.Column(db.Float)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    # --- JSON helpers -----------------------------------------------------
    @staticmethod
    def _load(raw):
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    @property
    def inputs(self) -> dict:
        return self._load(self.input_json) or {}

    @property
    def probabilities(self) -> dict:
        return self._load(self.probabilities_json) or {}

    @property
    def explanation(self) -> dict:
        return self._load(self.explanation_json) or {}

    @property
    def warnings(self) -> dict:
        return self._load(self.warnings_json) or {}

    @property
    def is_positive(self) -> bool:
        return self.prediction not in ("No Typhoid", "Normal or No Typhoid")

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "patient_code": self.patient.patient_code if self.patient else None,
            "clinician": self.clinician.full_name if self.clinician else None,
            "target_mode": self.target_mode,
            "feature_policy": self.feature_policy,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probability_positive": self.probability_positive,
            "probabilities": self.probabilities,
            "flagged_for_testing": self.flagged_for_testing,
            "model": {"kernel": self.model_kernel, "trained_at": self.model_trained_at},
            "latency_ms": self.latency_ms,
        }

    def __repr__(self) -> str:
        return f"<Assessment {self.reference} {self.prediction}>"


class AuditLog(db.Model):
    """Append-only activity trail. Never updated, never deleted by the application."""

    __tablename__ = "audit_logs"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    user_email = db.Column(db.String(255))  # retained even if the account is removed

    action = db.Column(db.String(80), nullable=False, index=True)
    category = db.Column(db.String(40), nullable=False, default="general", index=True)
    entity_type = db.Column(db.String(60))
    entity_id = db.Column(db.String(60))
    detail = db.Column(db.Text)
    success = db.Column(db.Boolean, nullable=False, default=True, index=True)

    ip_address = db.Column(db.String(64))
    user_agent = db.Column(db.String(300))

    user = db.relationship("User", foreign_keys=[user_id])

    CATEGORIES = ("auth", "user_admin", "patient", "prediction", "system", "security", "general")

    @classmethod
    def counts_by_action(cls, limit: int = 10):
        return (
            db.session.query(cls.action, func.count(cls.id).label("n"))
            .group_by(cls.action)
            .order_by(func.count(cls.id).desc())
            .limit(limit)
            .all()
        )

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} by {self.user_email}>"
