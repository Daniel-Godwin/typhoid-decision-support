"""Application-layer tests: schema, validation, form rendering and JSON API."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typhoid_ml.app import app as flask_app  # noqa: E402
from typhoid_ml.predict import ValidationError, form_schema, validate_record  # noqa: E402

MODEL_PRESENT = (ROOT / "models" / "svm_binary.joblib").exists()
requires_model = pytest.mark.skipif(not MODEL_PRESENT, reason="trained model not present")


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def valid_record():
    """A reference record in the deployed (post-review) feature space."""
    return dict(form_schema()["reference"])


@pytest.fixture(scope="module")
def routine_record():
    """A reference record in the pre-review comparator feature space."""
    return dict(form_schema("routine")["reference"])


def test_schema_covers_every_model_feature():
    """The deployed policy collects the thirteen post-review attributes."""
    schema = form_schema()
    assert len(schema["numeric"]) == 2
    assert len(schema["categorical"]) == 11
    assert set(schema["levels"]) == set(schema["categorical"])
    assert set(schema["reference"]) == set(schema["numeric"]) | set(schema["categorical"])


def test_comparator_schema_is_unchanged():
    """The routine policy is kept for Chapter Four's comparison and must not drift."""
    schema = form_schema("routine")
    assert len(schema["numeric"]) == 4
    assert len(schema["categorical"]) == 16


def test_dropped_attributes_are_absent_from_the_deployed_schema():
    """Nothing dropped at supervisory review may reach the deployed model."""
    from typhoid_ml.config import DROPPED_AT_REVIEW, HEADACHE_SOURCE, LEAKAGE_EXCLUDED

    collected = set(form_schema()["reference"])
    for attribute in list(DROPPED_AT_REVIEW) + LEAKAGE_EXCLUDED + [HEADACHE_SOURCE]:
        assert attribute not in collected, f"{attribute} is still collected"


def test_validation_accepts_reference_record(valid_record):
    out = validate_record(valid_record)
    assert set(out["record"]) == set(valid_record)
    assert out["warnings"] == {}


def test_validation_rejects_missing_field(valid_record):
    bad = dict(valid_record)
    bad.pop("Age")
    with pytest.raises(ValidationError) as exc:
        validate_record(bad)
    assert "Age" in json.loads(str(exc.value))


def test_deployed_form_has_no_optional_field():
    """Every attribute that was absent for some patients was dropped at review.

    The point-of-care form therefore asks nothing that may be left blank, so no
    entry the clinician skips can move the result.
    """
    assert form_schema()["optional"] == []


def test_comparator_optional_fields_match_dataset_missingness():
    """Only attributes that are actually absent for some patients may be optional."""
    assert set(form_schema("routine")["optional"]) == {
        "Gastrointestinal Symptoms",
        "Neurological Symptoms",
        "Ongoing Infection in Society",
    }


@pytest.mark.parametrize("blank", ["", "Not recorded", None])
def test_optional_field_may_be_left_unanswered(routine_record, blank):
    field = "Neurological Symptoms"
    out = validate_record(dict(routine_record, **{field: blank}), "routine")
    value = out["record"][field]
    assert isinstance(value, float) and value != value  # NaN, so the imputer sees it


def test_unanswered_optional_field_is_not_treated_as_a_category(routine_record):
    """A blank must impute, not encode as an unseen category.

    Encoding it as unknown (-1) changes the prediction, which is the bug this
    guards against.
    """
    baseline = validate_record(routine_record, "routine")["record"]
    blanked = validate_record(
        dict(routine_record, **{"Neurological Symptoms": "Not recorded"}), "routine"
    )["record"]
    assert set(baseline) == set(blanked)


def test_required_field_still_rejects_not_recorded(valid_record):
    with pytest.raises(ValidationError) as exc:
        validate_record(dict(valid_record, Gender="Not recorded"))
    assert "Gender" in json.loads(str(exc.value))


def test_validation_rejects_unknown_category(valid_record):
    bad = dict(valid_record, Gender="Unspecified")
    with pytest.raises(ValidationError) as exc:
        validate_record(bad)
    assert "Gender" in json.loads(str(exc.value))


def test_validation_rejects_non_numeric(valid_record):
    bad = dict(valid_record, Age="forty")
    with pytest.raises(ValidationError) as exc:
        validate_record(bad)
    assert "Age" in json.loads(str(exc.value))


def test_out_of_range_numeric_warns_but_passes(valid_record):
    out = validate_record(dict(valid_record, Age=140))
    assert "Age" in out["warnings"]
    assert out["record"]["Age"] == 140


def test_index_renders_dropdowns(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "<select" in body
    assert "Open Defecation" in body  # a real dataset category level
    # Excluded features must never be collected as inputs.
    for banned in (
        "Blood Culture Result",
        "Complications",
        "Typhoid Status",
        "White Blood Cell Count",
        "Platelet Count",
        "Widal Test",
        "Typhidot Test",
        "Gastrointestinal Symptoms",
        "Ongoing Infection in Society",
        "Neurological Symptoms",
    ):
        assert f'name="{banned}"' not in body
    assert 'name="Headache"' in body


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_api_schema_endpoint(client):
    data = client.get("/api/schema").get_json()
    assert data["numeric"] and data["categorical"]
    assert "Headache" in data["levels"]
    assert "Widal Test" not in data["levels"]


def test_api_rejects_invalid_payload(client):
    resp = client.post("/api/predict", json={"record": {"Age": 30}})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "validation_failed"


def test_api_rejects_unknown_target_mode(client, valid_record):
    resp = client.post("/api/predict", json={"record": valid_record, "target_mode": "quaternary"})
    assert resp.status_code == 400


@requires_model
def test_api_predict_returns_calibrated_result(client, valid_record):
    resp = client.post("/api/predict", json={"record": valid_record})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["prediction"] in ("Typhoid", "No Typhoid")
    assert 0.0 <= data["confidence"] <= 1.0
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-6
    assert data["explanation"]["method"].startswith("occlusion")


@requires_model
def test_api_rejects_a_blank_in_the_deployed_schema(client, valid_record):
    """No deployed field is optional, so a blank is an error rather than an imputation."""
    record = dict(valid_record, Headache="Not recorded")
    resp = client.post("/api/predict", json={"record": record})
    assert resp.status_code == 400
    assert "Headache" in resp.get_json()["fields"]


@requires_model
def test_form_submission_renders_prediction(client, valid_record):
    resp = client.post("/", data=valid_record)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Model confidence" in body
