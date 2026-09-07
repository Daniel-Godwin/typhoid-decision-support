"""End-to-end tests for the web application.

Covers the twelve areas required before deployment:
  1 authentication   2 authorisation      3 admin functions   4 clinician functions
  5 database ops     6 patient management 7 model loading     8 preprocessing
  9 results          10 logging           11 errors/validation 12 production config
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from webapp import create_app  # noqa: E402
from webapp.extensions import db  # noqa: E402
from webapp.models import Assessment, AuditLog, Patient, Role, User  # noqa: E402
from webapp.services import prediction_service as ps  # noqa: E402

MODEL_PRESENT = (ROOT / "models" / "svm_binary.joblib").exists()
requires_model = pytest.mark.skipif(not MODEL_PRESENT, reason="trained model not present")

PASSWORD = "TestPass2026x"


@pytest.fixture
def app():
    application = create_app("testing")
    with application.app_context():
        db.create_all()
        for email, name, role in [
            ("admin@example.com", "Test Admin", Role.ADMIN),
            ("doc@example.com", "Test Clinician", Role.CLINICIAN),
            ("doc2@example.com", "Other Clinician", Role.CLINICIAN),
            ("audit@example.com", "Test Auditor", Role.AUDITOR),
        ]:
            user = User(full_name=name, email=email, role=role)
            user.set_password(PASSWORD)
            db.session.add(user)
        db.session.commit()
        db.session.add(
            Patient(
                patient_code="PT-TEST-1", full_name="Test Patient", age=30,
                gender="Male", location="Urban",
                created_by_id=User.query.filter_by(email="doc@example.com").first().id,
            )
        )
        db.session.commit()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def login(client, email, password=PASSWORD):
    return client.post(
        "/auth/login", data={"email": email, "password": password}, follow_redirects=True
    )


@pytest.fixture
def valid_record():
    return dict(ps.form_schema("routine")["reference"])


# ===========================================================================
# 1. Authentication
# ===========================================================================
class TestAuthentication:
    def test_login_page_renders(self, client):
        resp = client.get("/auth/login")
        assert resp.status_code == 200
        assert b"Sign in" in resp.data

    def test_valid_login_succeeds(self, client):
        resp = login(client, "doc@example.com")
        assert resp.status_code == 200
        assert b"Test Clinician" in resp.data

    def test_wrong_password_rejected(self, client):
        resp = client.post("/auth/login", data={"email": "doc@example.com", "password": "wrong"})
        assert resp.status_code == 401

    def test_unknown_account_gives_same_message(self, client):
        resp = client.post("/auth/login", data={"email": "nobody@example.com", "password": "x"})
        assert resp.status_code == 401
        assert b"Incorrect email or password" in resp.data

    def test_account_locks_after_repeated_failures(self, client, app):
        for _ in range(app.config["LOGIN_MAX_ATTEMPTS"]):
            client.post("/auth/login", data={"email": "doc@example.com", "password": "wrong"})
        resp = client.post("/auth/login", data={"email": "doc@example.com", "password": PASSWORD})
        assert resp.status_code == 423
        with app.app_context():
            assert User.query.filter_by(email="doc@example.com").first().is_locked

    def test_disabled_account_cannot_sign_in(self, client, app):
        with app.app_context():
            user = User.query.filter_by(email="doc@example.com").first()
            user.is_active_account = False
            db.session.commit()
        assert client.post(
            "/auth/login", data={"email": "doc@example.com", "password": PASSWORD}
        ).status_code == 403

    def test_logout_ends_session(self, client):
        login(client, "doc@example.com")
        client.get("/auth/logout", follow_redirects=True)
        assert client.get("/clinic/", follow_redirects=False).status_code == 302

    def test_password_change_requires_current_password(self, client):
        login(client, "doc@example.com")
        resp = client.post("/auth/password", data={
            "current_password": "wrong", "new_password": "NewPass2026x",
            "confirm_password": "NewPass2026x",
        })
        assert resp.status_code == 400

    def test_password_change_succeeds_and_new_password_works(self, client, app):
        login(client, "doc@example.com")
        client.post("/auth/password", data={
            "current_password": PASSWORD, "new_password": "BrandNew2026x",
            "confirm_password": "BrandNew2026x",
        }, follow_redirects=True)
        client.get("/auth/logout")
        assert login(client, "doc@example.com", "BrandNew2026x").status_code == 200

    def test_weak_password_rejected(self, client):
        login(client, "doc@example.com")
        resp = client.post("/auth/password", data={
            "current_password": PASSWORD, "new_password": "short", "confirm_password": "short",
        })
        assert b"at least 10 characters" in resp.data

    def test_password_is_hashed_not_stored(self, app):
        with app.app_context():
            user = User.query.filter_by(email="doc@example.com").first()
            assert PASSWORD not in user.password_hash
            assert user.check_password(PASSWORD)

    def test_open_redirect_is_blocked(self, client):
        resp = client.post(
            "/auth/login?next=https://evil.example/x",
            data={"email": "doc@example.com", "password": PASSWORD},
        )
        assert "evil.example" not in resp.headers.get("Location", "")


# ===========================================================================
# 2. Authorisation (role-based access control)
# ===========================================================================
class TestAuthorisation:
    @pytest.mark.parametrize("path", [
        "/clinic/", "/clinic/assess", "/clinic/patients", "/admin/", "/admin/users", "/admin/logs",
    ])
    def test_anonymous_is_redirected(self, client, path):
        resp = client.get(path)
        assert resp.status_code == 302
        assert "/auth/login" in resp.headers["Location"]

    def test_clinician_cannot_reach_admin(self, client):
        login(client, "doc@example.com")
        assert client.get("/admin/").status_code == 403
        assert client.get("/admin/users").status_code == 403

    def test_admin_reaches_everything(self, client):
        login(client, "admin@example.com")
        for path in ("/admin/", "/admin/users", "/admin/logs", "/admin/system", "/clinic/", "/clinic/assess"):
            assert client.get(path).status_code == 200, path

    def test_auditor_is_read_only(self, client):
        login(client, "audit@example.com")
        assert client.get("/admin/logs").status_code == 200      # may read the log
        assert client.get("/clinic/assessments").status_code == 200
        assert client.get("/clinic/assess").status_code == 403    # may not run assessments
        assert client.get("/clinic/patients/new").status_code == 403

    def test_clinician_cannot_see_another_clinicians_assessment(self, client, app):
        with app.app_context():
            other = User.query.filter_by(email="doc2@example.com").first()
            db.session.add(Assessment(
                reference="ASM-OTHER", clinician_id=other.id, input_json="{}",
                target_mode="binary", feature_policy="routine", prediction="Typhoid",
            ))
            db.session.commit()
        login(client, "doc@example.com")
        assert client.get("/clinic/assessments/ASM-OTHER").status_code == 403

    def test_admin_cannot_remove_own_admin_role(self, client, app):
        login(client, "admin@example.com")
        with app.app_context():
            admin_id = User.query.filter_by(email="admin@example.com").first().id
        resp = client.post(f"/admin/users/{admin_id}/edit", data={
            "full_name": "Test Admin", "email": "admin@example.com",
            "role": Role.CLINICIAN, "is_active_account": "y",
        })
        assert resp.status_code == 400

    def test_admin_cannot_disable_own_account(self, client, app):
        login(client, "admin@example.com")
        with app.app_context():
            admin_id = User.query.filter_by(email="admin@example.com").first().id
        client.post(f"/admin/users/{admin_id}/toggle", follow_redirects=True)
        with app.app_context():
            assert User.query.filter_by(email="admin@example.com").first().is_active_account


# ===========================================================================
# 3. Admin functionality
# ===========================================================================
class TestAdmin:
    def test_dashboard_shows_statistics(self, client):
        login(client, "admin@example.com")
        resp = client.get("/admin/")
        assert resp.status_code == 200
        assert b"User accounts" in resp.data

    def test_create_user(self, client, app):
        login(client, "admin@example.com")
        resp = client.post("/admin/users/new", data={
            "full_name": "New Nurse", "email": "nurse@example.com", "role": Role.CLINICIAN,
            "password": "NursePass2026", "is_active_account": "y", "must_change_password": "y",
        }, follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            user = User.query.filter_by(email="nurse@example.com").first()
            assert user is not None and user.must_change_password

    def test_duplicate_email_rejected(self, client):
        login(client, "admin@example.com")
        resp = client.post("/admin/users/new", data={
            "full_name": "Clone", "email": "doc@example.com", "role": Role.CLINICIAN,
            "password": "ClonePass2026", "is_active_account": "y",
        })
        assert b"already exists" in resp.data

    def test_toggle_and_unlock(self, client, app):
        login(client, "admin@example.com")
        with app.app_context():
            user = User.query.filter_by(email="doc2@example.com").first()
            user.locked_until = __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ) + __import__("datetime").timedelta(minutes=30)
            db.session.commit()
            uid = user.id
        client.post(f"/admin/users/{uid}/toggle", follow_redirects=True)
        with app.app_context():
            assert not db.session.get(User, uid).is_active_account
        client.post(f"/admin/users/{uid}/unlock", follow_redirects=True)
        with app.app_context():
            assert not db.session.get(User, uid).is_locked

    def test_system_page_reports_models_and_database(self, client):
        login(client, "admin@example.com")
        resp = client.get("/admin/system")
        assert resp.status_code == 200
        assert b"Database" in resp.data and b"Machine-learning models" in resp.data


# ===========================================================================
# 4 & 6. Clinician functionality and patient management
# ===========================================================================
class TestClinician:
    def test_dashboard(self, client):
        login(client, "doc@example.com")
        assert client.get("/clinic/").status_code == 200

    def test_assessment_form_renders_dataset_options(self, client):
        login(client, "doc@example.com")
        resp = client.get("/clinic/assess")
        assert resp.status_code == 200
        assert b"Open Defecation" in resp.data
        for banned in (b'name="Blood Culture Result"', b'name="Complications"', b'name="Typhoid Status"'):
            assert banned not in resp.data

    def test_create_patient(self, client, app):
        login(client, "doc@example.com")
        resp = client.post("/clinic/patients/new", data={
            "patient_code": "PT-NEW-9", "full_name": "New Patient",
            "age": 41, "gender": "Female", "location": "Rural",
        }, follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            assert Patient.query.filter_by(patient_code="PT-NEW-9").first() is not None

    def test_duplicate_patient_code_rejected(self, client):
        login(client, "doc@example.com")
        resp = client.post("/clinic/patients/new", data={
            "patient_code": "PT-TEST-1", "full_name": "Duplicate",
        })
        assert b"already in use" in resp.data

    def test_patient_age_out_of_range_rejected(self, client):
        login(client, "doc@example.com")
        resp = client.post("/clinic/patients/new", data={
            "patient_code": "PT-BAD", "full_name": "Bad Age", "age": 900,
        })
        assert b"0" in resp.data and b"120" in resp.data

    def test_patient_list_and_search(self, client):
        login(client, "doc@example.com")
        assert b"PT-TEST-1" in client.get("/clinic/patients").data
        assert b"PT-TEST-1" in client.get("/clinic/patients?q=PT-TEST").data


# ===========================================================================
# 5. Database operations
# ===========================================================================
class TestDatabase:
    def test_tables_exist(self, app):
        with app.app_context():
            assert User.query.count() == 4
            assert Patient.query.count() == 1
            assert Assessment.query.count() == 0

    def test_cascade_delete_removes_assessments(self, app):
        with app.app_context():
            patient = Patient.query.first()
            db.session.add(Assessment(
                reference="ASM-CASCADE", patient_id=patient.id,
                clinician_id=patient.created_by_id, input_json="{}",
                target_mode="binary", feature_policy="routine", prediction="Typhoid",
            ))
            db.session.commit()
            db.session.delete(patient)
            db.session.commit()
            assert Assessment.query.filter_by(reference="ASM-CASCADE").first() is None

    def test_unique_constraints(self, app):
        from sqlalchemy.exc import IntegrityError

        with app.app_context():
            duplicate = User(full_name="Dup", email="doc@example.com", role=Role.CLINICIAN)
            duplicate.set_password(PASSWORD)
            db.session.add(duplicate)
            with pytest.raises(IntegrityError):
                db.session.commit()
            db.session.rollback()


# ===========================================================================
# 7 & 8. Model loading, preprocessing and feature handling
# ===========================================================================
class TestModelIntegration:
    def test_schema_matches_trained_feature_set(self):
        schema = ps.form_schema("routine")
        assert len(schema["numeric"]) == 4
        assert len(schema["categorical"]) == 16
        assert set(schema["optional"]) == {
            "Gastrointestinal Symptoms", "Neurological Symptoms", "Ongoing Infection in Society"
        }

    def test_policies_and_modes_exposed(self):
        assert "routine" in ps.available_policies()
        assert set(ps.available_target_modes()) == {"binary", "multiclass"}

    def test_validation_rejects_unknown_category(self, valid_record):
        with pytest.raises(ps.ValidationError):
            ps.validate(dict(valid_record, Gender="Unspecified"))

    def test_validation_rejects_missing_required_field(self, valid_record):
        bad = dict(valid_record)
        bad.pop("Age")
        with pytest.raises(ps.ValidationError):
            ps.validate(bad)

    def test_optional_field_imputes_rather_than_failing(self, valid_record):
        out = ps.validate(dict(valid_record, **{"Neurological Symptoms": "Not recorded"}))
        value = out["record"]["Neurological Symptoms"]
        assert isinstance(value, float) and value != value  # NaN

    def test_out_of_range_numeric_warns(self, valid_record):
        assert "Age" in ps.validate(dict(valid_record, Age=250))["warnings"]

    @requires_model
    def test_prediction_returns_calibrated_output(self, app, valid_record):
        with app.app_context():
            out = ps.run_prediction(ps.validate(valid_record)["record"])
        assert out["prediction"] in ("Typhoid", "No Typhoid")
        assert 0.0 <= out["confidence"] <= 1.0
        assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6
        assert out["reference"].startswith("ASM-")
        assert out["latency_ms"] > 0

    @requires_model
    def test_explanation_is_produced(self, app, valid_record):
        with app.app_context():
            out = ps.run_prediction(ps.validate(valid_record)["record"])
        assert out["explanation"]["method"].startswith("occlusion")

    @requires_model
    def test_triage_recommendation_levels(self):
        high = ps.triage_recommendation({"probability_positive": 0.97, "flagged_for_testing": True})
        low = ps.triage_recommendation({"probability_positive": 0.02, "flagged_for_testing": False})
        assert high["level"] == "high" and low["level"] == "minimal"

    def test_unknown_target_mode_rejected(self, app, valid_record):
        with app.app_context():
            with pytest.raises(ValueError):
                ps.run_prediction(valid_record, target_mode="trinary")


# ===========================================================================
# 9. Results — the full assessment journey
# ===========================================================================
class TestAssessmentFlow:
    @requires_model
    def test_assessment_is_persisted_with_provenance(self, client, app, valid_record):
        login(client, "doc@example.com")
        with app.app_context():
            patient_id = Patient.query.first().id
        payload = dict(valid_record)
        payload["patient_id"] = str(patient_id)
        payload["target_mode"] = "binary"
        resp = client.post("/clinic/assess", data=payload, follow_redirects=True)
        assert resp.status_code == 200
        with app.app_context():
            row = Assessment.query.first()
            assert row is not None
            assert row.prediction in ("Typhoid", "No Typhoid")
            assert row.model_kernel  # provenance recorded
            assert row.inputs  # inputs stored for reproducibility
            assert row.latency_ms > 0

    @requires_model
    def test_invalid_submission_is_reported_not_crashed(self, client):
        login(client, "doc@example.com")
        resp = client.post("/clinic/assess", data={"Age": "not-a-number"})
        assert resp.status_code == 200
        assert b"need attention" in resp.data or b"could not be completed" in resp.data

    @requires_model
    def test_api_predict(self, client, valid_record):
        login(client, "doc@example.com")
        resp = client.post("/api/predict", json={"record": valid_record})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["prediction"] in ("Typhoid", "No Typhoid")
        assert "triage" in body and "disclaimer" in body

    def test_api_validation_error(self, client):
        login(client, "doc@example.com")
        resp = client.post("/api/predict", json={"record": {"Age": 30}})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "validation_failed"

    def test_api_requires_authentication(self, client, valid_record):
        assert client.post("/api/predict", json={"record": valid_record}).status_code in (302, 401)

    def test_api_auditor_cannot_predict(self, client, valid_record):
        login(client, "audit@example.com")
        assert client.post("/api/predict", json={"record": valid_record}).status_code == 403

    def test_api_schema_endpoint(self, client):
        login(client, "doc@example.com")
        body = client.get("/api/schema").get_json()
        assert "Widal Test" in body["levels"]
        assert body["optional"]


# ===========================================================================
# 10. Logging and audit trail
# ===========================================================================
class TestAuditTrail:
    def test_successful_login_recorded(self, client, app):
        login(client, "doc@example.com")
        with app.app_context():
            assert AuditLog.query.filter_by(action="login", success=True).count() == 1

    def test_failed_login_recorded(self, client, app):
        client.post("/auth/login", data={"email": "doc@example.com", "password": "wrong"})
        with app.app_context():
            entry = AuditLog.query.filter_by(action="login_failed").first()
            assert entry is not None and entry.success is False

    def test_access_denial_recorded(self, client, app):
        login(client, "doc@example.com")
        client.get("/admin/users")
        with app.app_context():
            assert AuditLog.query.filter_by(action="access_denied", category="security").count() >= 1

    def test_patient_creation_recorded(self, client, app):
        login(client, "doc@example.com")
        client.post("/clinic/patients/new", data={
            "patient_code": "PT-AUDIT", "full_name": "Audited Patient"
        }, follow_redirects=True)
        with app.app_context():
            assert AuditLog.query.filter_by(action="patient_created").count() == 1

    @requires_model
    def test_prediction_recorded(self, client, app, valid_record):
        login(client, "doc@example.com")
        client.post("/clinic/assess", data=dict(valid_record, target_mode="binary"),
                    follow_redirects=True)
        with app.app_context():
            assert AuditLog.query.filter_by(action="prediction_created").count() == 1

    def test_audit_entries_capture_context(self, client, app):
        login(client, "doc@example.com")
        with app.app_context():
            entry = AuditLog.query.filter_by(action="login").first()
            assert entry.user_email == "doc@example.com"
            assert entry.category == "auth"
            assert entry.ip_address is not None


# ===========================================================================
# 11. Error handling and validation
# ===========================================================================
class TestErrorHandling:
    def test_404_page(self, client):
        login(client, "doc@example.com")
        resp = client.get("/no-such-page")
        assert resp.status_code == 404
        assert b"Page not found" in resp.data

    def test_404_json_for_api_paths(self, client):
        login(client, "doc@example.com")
        resp = client.get("/api/nothing-here")
        assert resp.status_code == 404
        assert resp.get_json()["status"] == 404

    def test_403_page_rendered(self, client):
        login(client, "doc@example.com")
        resp = client.get("/admin/")
        assert resp.status_code == 403
        assert b"Not permitted" in resp.data

    def test_missing_assessment_returns_404(self, client):
        login(client, "doc@example.com")
        assert client.get("/clinic/assessments/ASM-NOPE").status_code == 404

    def test_api_rejects_non_json(self, client):
        login(client, "doc@example.com")
        resp = client.post("/api/predict", data="plain text", content_type="text/plain")
        assert resp.status_code == 400

    def test_security_headers_present(self, client):
        headers = client.get("/auth/login").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in headers

    def test_request_id_header(self, client):
        assert "X-Request-ID" in client.get("/auth/login").headers


# ===========================================================================
# 12. Production configuration and health
# ===========================================================================
class TestProductionReadiness:
    def test_health_endpoint(self, client):
        body = client.get("/health").get_json()
        assert body["status"] in ("ok", "degraded")
        assert "database" in body["checks"]

    def test_testing_config_isolated(self, app):
        assert app.config["ENV_NAME"] == "testing"
        assert app.config["SQLALCHEMY_DATABASE_URI"] == "sqlite:///:memory:"

    def test_production_config_flags_secure_cookies(self):
        from webapp.config import ProductionConfig

        assert ProductionConfig.SESSION_COOKIE_SECURE is True
        assert ProductionConfig.DEBUG is False

    def test_production_config_validation_detects_default_secret(self, monkeypatch):
        from webapp.config import validate_production_config

        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        application = create_app("production")
        problems = validate_production_config(application)
        assert any("SECRET_KEY" in p for p in problems)

    def test_postgres_url_is_normalised(self):
        from webapp.config import _normalise_db_url

        assert _normalise_db_url("postgres://u:p@h:5432/d").startswith("postgresql+psycopg2://")

    def test_external_provider_url_keeps_its_query_string(self):
        """A hosted provider's connection string must survive normalisation.

        Neon and similar append `?sslmode=require`; dropping it turns TLS off.
        """
        from webapp.config import _normalise_db_url

        url = _normalise_db_url(
            "postgresql://u:p@ep-x.eu-central-1.aws.neon.tech/typhoid"
            "?sslmode=require&channel_binding=require"
        )
        assert url.startswith("postgresql+psycopg2://")
        assert url.endswith("?sslmode=require&channel_binding=require")

    def test_already_normalised_url_is_left_alone(self):
        from webapp.config import _normalise_db_url

        url = "postgresql+psycopg2://u:p@h/d"
        assert _normalise_db_url(url) == url

    def test_connection_pool_survives_an_idle_suspending_database(self):
        """External free-tier Postgres suspends when idle and drops the socket.

        `pool_pre_ping` discards a dead connection instead of raising, and
        `pool_recycle` keeps connections younger than the provider's own idle
        timeout. Without both, the first request after a quiet period fails.
        """
        from webapp.config import BaseConfig

        options = BaseConfig.SQLALCHEMY_ENGINE_OPTIONS
        assert options["pool_pre_ping"] is True
        assert 0 < options["pool_recycle"] <= 300

    def test_csrf_enabled_outside_testing(self):
        from webapp.config import DevelopmentConfig, ProductionConfig

        assert DevelopmentConfig.WTF_CSRF_ENABLED and ProductionConfig.WTF_CSRF_ENABLED

    def test_health_reports_the_database_engine(self, client):
        """A silent SQLite fall-back must be visible, not mistaken for success."""
        body = client.get("/health").get_json()
        assert "database_engine" in body["checks"]
        assert body["checks"]["database_engine"] == "sqlite"

    def test_health_never_leaks_the_connection_string(self, app, client):
        """Only the dialect name is reported — no host, database or credentials."""
        import json

        raw = json.dumps(client.get("/health").get_json())
        uri = app.config["SQLALCHEMY_DATABASE_URI"]
        assert uri not in raw
        for fragment in ("://", "@", "password", "sslmode"):
            assert fragment not in raw

    def test_production_on_sqlite_reports_degraded(self, monkeypatch):
        """The ephemeral fall-back is a failed deployment, not a healthy one."""
        monkeypatch.setenv("SECRET_KEY", "test-key-for-this-check")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        application = create_app("production")
        with application.test_client() as c:
            response = c.get("/health")
        body = response.get_json()
        assert response.status_code == 503
        assert body["status"] == "degraded"
        assert "ephemeral" in body["checks"]["database_warning"]

    def test_ml_package_is_not_imported_by_blueprints(self):
        """The separation must hold: only the service touches the ML package."""
        blueprint_dir = ROOT / "src" / "webapp" / "blueprints"
        for path in blueprint_dir.glob("*.py"):
            assert "typhoid_ml" not in path.read_text(encoding="utf-8"), path.name
