"""End-to-end integration: dataset -> trained artefact -> form -> prediction.

The feature space is defined in one place (`typhoid_ml.config`) but is consumed
in four: the training pipeline writes it into the model artefact, the schema
builds the assessment form from it, the validator accepts it, and the
prediction service passes it to the model. These tests assert that all four
still agree, so that a change to the policy cannot leave the deployed form
asking for one thing while the artefact expects another.

They are deliberately strict about the attributes dropped at supervisory
review: an attribute that is not obtainable at the point of care must not
reappear in the form by accident.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typhoid_ml.config import (  # noqa: E402
    DATA_PATH,
    DEFAULT_POLICY,
    DROPPED_AT_REVIEW,
    FEATURE_POLICIES,
    HEADACHE_FEATURE,
    HEADACHE_SOURCE,
    LEAKAGE_EXCLUDED,
    TARGET,
)
from typhoid_ml.data import load_dataset  # noqa: E402
from typhoid_ml.predict import form_schema, load_model, model_path, validate_record  # noqa: E402

MODEL_PRESENT = model_path("binary", DEFAULT_POLICY).exists()
requires_model = pytest.mark.skipif(not MODEL_PRESENT, reason="trained model not present")

POLICY = FEATURE_POLICIES[DEFAULT_POLICY]
DEPLOYED_FEATURES = POLICY["numeric"] + POLICY["categorical"]


@pytest.fixture(scope="module")
def df():
    return load_dataset()


@pytest.fixture(scope="module")
def schema():
    return form_schema()


@pytest.fixture
def app_factory():
    """A testing application whose ENV_NAME may be changed for one test."""
    from webapp import create_app

    app = create_app("testing")
    original = app.config.get("ENV_NAME")
    yield app
    app.config["ENV_NAME"] = original


# --------------------------------------------------------------------------
# The feature space itself
# --------------------------------------------------------------------------
def test_deployed_policy_is_the_post_review_feature_space():
    assert len(POLICY["numeric"]) == 2
    assert len(POLICY["categorical"]) == 11
    assert HEADACHE_FEATURE in POLICY["categorical"]


@pytest.mark.parametrize(
    "attribute", sorted(set(DROPPED_AT_REVIEW) | set(LEAKAGE_EXCLUDED) | {HEADACHE_SOURCE, TARGET})
)
def test_excluded_attribute_is_not_a_model_input(attribute):
    assert attribute not in DEPLOYED_FEATURES


# --------------------------------------------------------------------------
# Dataset -> derived column
# --------------------------------------------------------------------------
def test_headache_is_derived_from_the_supplied_column(df):
    """The recoding is exactly: a recorded Headache is Yes, everything else No."""
    assert HEADACHE_FEATURE not in pd.read_csv(DATA_PATH).columns
    assert ((df[HEADACHE_SOURCE] == "Headache") == (df[HEADACHE_FEATURE] == "Yes")).all()


def test_no_deployed_attribute_has_a_missing_value(df):
    """Completeness is why the form can require every field."""
    assert df[DEPLOYED_FEATURES].isna().sum().sum() == 0


# --------------------------------------------------------------------------
# Trained artefact -> form
# --------------------------------------------------------------------------
@requires_model
@pytest.mark.parametrize("target_mode", ["binary", "multiclass"])
def test_artefact_expects_the_deployed_feature_space(target_mode):
    if not model_path(target_mode, DEFAULT_POLICY).exists():
        pytest.skip(f"{target_mode} model not trained for {DEFAULT_POLICY}")
    _, meta = load_model(target_mode, DEFAULT_POLICY)
    assert meta["feature_policy"] == DEFAULT_POLICY
    assert meta["numeric_features"] + meta["categorical_features"] == DEPLOYED_FEATURES


def test_form_collects_exactly_the_model_inputs(schema):
    assert sorted(schema["numeric"] + schema["categorical"]) == sorted(DEPLOYED_FEATURES)


def test_every_field_is_rendered_in_a_group():
    """A field present in the schema but in no group would never be asked for."""
    from webapp.blueprints.clinician import _grouped_fields
    from webapp.services import prediction_service as ps

    rendered = [i["name"] for g in _grouped_fields(ps.form_schema()) for i in g["fields"]]
    assert sorted(rendered) == sorted(DEPLOYED_FEATURES)


# --------------------------------------------------------------------------
# Form options -> training data
# --------------------------------------------------------------------------
@pytest.mark.parametrize("column", POLICY["categorical"])
def test_offered_options_all_occur_in_the_training_data(column, df, schema):
    """An option the model never saw during training must not be selectable."""
    assert set(schema["levels"][column]) == {str(v) for v in df[column].dropna().unique()}


@pytest.mark.parametrize("column", POLICY["numeric"])
def test_stated_numeric_range_matches_the_training_data(column, df, schema):
    assert schema["ranges"][column]["min"] == float(df[column].min())
    assert schema["ranges"][column]["max"] == float(df[column].max())


# --------------------------------------------------------------------------
# Form -> model
# --------------------------------------------------------------------------
@requires_model
def test_a_submitted_record_reaches_the_model_unchanged(df):
    """What the web layer computes must equal what the artefact computes directly."""
    from webapp.services import prediction_service as ps

    row = df.iloc[7]
    submitted = {c: str(row[c]) for c in DEPLOYED_FEATURES}

    validated = validate_record(submitted)
    assert set(validated["record"]) == set(DEPLOYED_FEATURES)
    assert validated["warnings"] == {}

    model, _ = load_model("binary", DEFAULT_POLICY)
    positive = list(model.classes_).index("Typhoid")
    direct = model.predict_proba(pd.DataFrame([{c: row[c] for c in DEPLOYED_FEATURES}]))[0][positive]

    through_app = ps.validate_and_predict(submitted, target_mode="binary")
    assert through_app["probabilities"]["Typhoid"] == pytest.approx(direct, abs=1e-12)


def test_deployed_threshold_matches_the_published_analysis():
    """`TRIAGE_THRESHOLD` must be the cut-off the threshold analysis recommends."""
    import json

    from webapp.config import BaseConfig

    report = ROOT / "reports" / "threshold_analysis_binary.json"
    if not report.exists():
        pytest.skip("threshold analysis not generated")
    recommended = json.loads(report.read_text())["recommended"]["threshold"]
    assert BaseConfig.TRIAGE_THRESHOLD == pytest.approx(recommended)


# --------------------------------------------------------------------------
# Deployment misconfiguration is stated, not silent
# --------------------------------------------------------------------------
def test_template_filters_are_registered():
    """`pct` and `dt` are used by the dashboard; a missing one is a 500."""
    from webapp import create_app

    app = create_app("testing")
    assert "pct" in app.jinja_env.filters
    assert "dt" in app.jinja_env.filters


def test_no_deployment_warning_outside_production():
    from webapp import _deployment_warning, create_app

    app = create_app("testing")
    with app.app_context():
        assert _deployment_warning(app) is None


def test_production_on_the_sqlite_fallback_is_announced(app_factory):
    """The silent fallback erases accounts on restart and signs users out.

    It must be visible in the interface, not only in `/health`, because the
    symptom it produces looks like a fault in the application rather than a
    missing environment variable. The check is driven by `ENV_NAME` and the
    engine in use, so a testing application relabelled as production exercises
    exactly the same path without building a second application.
    """
    import uuid

    from webapp import _deployment_warning
    from webapp.extensions import db as _db
    from webapp.models import Role, User

    app = app_factory
    app.config["ENV_NAME"] = "production"
    email = f"doc-{uuid.uuid4().hex[:8]}@example.com"
    with app.app_context():
        _db.create_all()
        u = User(full_name="Dr Test", email=email, role=Role.CLINICIAN)
        u.set_password("ClinicPass2026")
        _db.session.add(u)
        _db.session.commit()

        assert _db.engine.dialect.name == "sqlite"
        assert "DATABASE_URL" in (_deployment_warning(app) or "")

        c = app.test_client()
        c.post("/auth/login", data={"email": email, "password": "ClinicPass2026"})
        body = c.get("/clinic/").get_data(as_text=True)
        assert "This deployment is not storing data" in body

        _db.session.remove()
        _db.drop_all()
