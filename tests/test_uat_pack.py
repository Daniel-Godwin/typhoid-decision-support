"""The acceptance-test pack must agree with what the application actually returns.

A pack whose expected column disagrees with the deployed system turns every
mismatch into a reported defect and wastes the reviewer's time. These tests run
each of the ten records through `prediction_service.validate_and_predict` — the
same call the assessment form makes — and compare against the committed
expectations.

Regenerate the pack with `python scripts/make_uat_samples.py` after any
retraining, then run these tests before sending it to anyone.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

UAT = ROOT / "reports" / "uat"
PAYLOADS = UAT / "uat_api_payloads.json"
EXPECTED = UAT / "uat_expected.json"

pytestmark = pytest.mark.skipif(
    not (PAYLOADS.exists() and EXPECTED.exists()),
    reason="acceptance pack not generated; run scripts/make_uat_samples.py",
)


def load():
    payloads = json.loads(PAYLOADS.read_text(encoding="utf-8"))
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    by_case = {c["case"]: c for c in expected["cases"]}
    return payloads, by_case, expected


def test_pack_has_ten_distinct_cases():
    payloads, by_case, _ = load()
    assert len(payloads) == 10
    assert len({p["patient_code"] for p in payloads}) == 10
    assert set(by_case) == {p["case"] for p in payloads}


def test_pack_is_balanced_between_the_classes():
    _, by_case, _ = load()
    recorded = [c["recorded_diagnosis"] for c in by_case.values()]
    assert recorded.count("Typhoid") == 5
    assert recorded.count("No Typhoid") == 5


def test_pack_includes_cases_the_model_gets_wrong():
    """A pack of only easy cases would demonstrate nothing."""
    _, by_case, _ = load()
    disagreements = [c for c in by_case.values() if not c["agrees_with_record"]]
    assert disagreements, "the pack must contain at least one known failure"
    assert all(c["kind"] == "known-difficult" for c in disagreements)


def test_threshold_matches_the_deployed_configuration():
    """The pack's expected labels are only valid at the configured cut-off."""
    from webapp.config import BaseConfig

    _, _, expected = load()
    assert expected["threshold"] == pytest.approx(BaseConfig.TRIAGE_THRESHOLD)


@pytest.mark.parametrize("case", range(1, 11))
def test_expected_output_matches_the_application(case):
    from webapp.services import prediction_service as ps

    payloads, by_case, meta = load()
    payload = next(p for p in payloads if p["case"] == case)
    want = by_case[case]

    outcome = ps.validate_and_predict(
        payload["record"], target_mode="binary", policy="routine"
    )
    probability = outcome["probabilities"]["Typhoid"]
    label = "Typhoid" if probability >= meta["threshold"] else "No Typhoid"

    assert label == want["expected_prediction"], (
        f"case {case}: pack says {want['expected_prediction']}, "
        f"application returns {label}"
    )
    assert probability == pytest.approx(want["expected_probability"], abs=5e-4)


@pytest.mark.parametrize("case", range(1, 11))
def test_every_record_passes_validation(case):
    """A reviewer must never meet a validation error on a supplied record."""
    from webapp.services import prediction_service as ps

    payloads, _, _ = load()
    payload = next(p for p in payloads if p["case"] == case)
    result = ps.validate(payload["record"])
    assert result is not None
