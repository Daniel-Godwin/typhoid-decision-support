# System Testing Report

_Generated 05 September 2026, 10:30 UTC by `python scripts/make_test_report.py`. Regenerate after any change._

## Summary

| Metric | Value |
|---|---:|
| Total test cases | 128 |
| Passed | 128 |
| Failed | 0 |
| Skipped | 0 |
| Pass rate | 100.0% |

## Results by verification area

| # | Verification area | Cases | Passed | Failed | Result |
|---|---|---:|---:|---:|---|
| 1 | User authentication | 12 | 12 | 0 | Pass |
| 2 | Authorisation and access control | 12 | 12 | 0 | Pass |
| 3 | Administrative functionality | 5 | 5 | 0 | Pass |
| 4 | Clinician functionality | 6 | 6 | 0 | Pass |
| 5 | Database operations | 3 | 3 | 0 | Pass |
| 6 | Patient data management | 4 | 4 | 0 | Pass |
| 7 | Model loading and prediction | 10 | 10 | 0 | Pass |
| 8 | Preprocessing and feature handling | 10 | 10 | 0 | Pass |
| 9 | Prediction results and triage | 7 | 7 | 0 | Pass |
| 10 | System logging and audit trail | 6 | 6 | 0 | Pass |
| 11 | Error handling and validation | 7 | 7 | 0 | Pass |
| 12 | Production readiness | 7 | 7 | 0 | Pass |
| — | Dataset integrity | 11 | 11 | 0 | Pass |
| — | Decision-threshold analysis | 12 | 12 | 0 | Pass |

## Component coverage

| Test module | Component under test | Cases | Passed |
|---|---|---:|---:|
| `tests/test_app.py` | ML reference interface and prediction service | 20 | 20 |
| `tests/test_data.py` | Dataset loading, target construction, feature policy | 11 | 11 |
| `tests/test_pipeline.py` | Preprocessing, SMOTENC, encoding, pipeline integrity | 10 | 10 |
| `tests/test_threshold.py` | Threshold sweep, operating points, calibration, prevalence adjustment | 12 | 12 |
| `tests/test_webapp.py` | Web application, security, database, deployment readiness | 75 | 75 |

## Failures

None. Every test case passed.

## Non-functional verification

| Property | Method | Result |
|---|---|---|
| Inference latency | Measured over the held-out test partition | 9.96 ms per record |
| Batch throughput | Batch prediction over 2000 records | 6329.7 records/second |
| Responsive layout | Rendered at 1440, 834 and 390 px | See `reports/screenshots/` |
| Security headers | Inspected on every response | CSP, HSTS, X-Frame-Options, nosniff present |
| Production start-up | gunicorn with 2 workers and 4 threads | Health check returns `ok` |
| ML/web separation | Automated check that no blueprint imports `typhoid_ml` | Enforced by test |
