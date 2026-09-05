# Typhoid Fever Diagnostic Decision-Support System

A production-ready clinical decision-support system built on a Support Vector
Machine model for typhoid fever assessment in resource-limited settings.

Two independent packages joined at exactly one point:

```
src/typhoid_ml/   machine-learning pipeline   — training, evaluation, inference
src/webapp/       web application             — auth, roles, dashboards, API
                  └── services/prediction_service.py is the only bridge
```

No blueprint imports `typhoid_ml`; a test enforces it. The model can be
retrained without touching a view, and the application can be restructured
without risk to the trained artefacts or their preprocessing requirements.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:FLASK_APP = "run_dev.py"
python -m flask init-db
python -m flask seed-demo
python run_dev.py
```

Open http://127.0.0.1:5000 and sign in:

| Role | Email | Password |
|---|---|---|
| Administrator | `admin@example.com` | `AdminPass2026` |
| Clinician | `clinician@example.com` | `ClinicPass2026` |
| Auditor | `auditor@example.com` | `AuditPass2026` |

## Features

**Security and access** — PBKDF2 password hashing, account lockout after five
failed attempts, CSRF protection, session hardening, CSP and HSTS headers,
open-redirect protection, rate limiting, three roles with enforced separation.

**Clinical workflow** — patient register, assessment form driven by the dataset
schema so an unrecognised value can never reach the model, calibrated
probability, triage recommendation keyed to a configurable referral threshold,
per-prediction explanation of which findings moved the result.

**Administration** — user management, account enable/disable/unlock, append-only
activity log with category and outcome filters, system status page reporting
database health and model provenance.

**Interface** — a self-contained design system (no CDN dependency), responsive
from 390 px to desktop, mobile navigation drawer, accessible focus states, skip
link, print styles, reduced-motion support.

## Commands

| Command | Purpose |
|---|---|
| `flask init-db` | Create tables |
| `flask create-admin` | Create an administrator interactively |
| `flask bootstrap` | Tables plus initial admin from environment variables (deployment) |
| `flask seed-demo` | Demonstration accounts and patients |
| `flask stats` | Row counts |
| `python run_dev.py` | Development server |
| `gunicorn wsgi:app` | Production server |

## Machine-learning pipeline

Unchanged and independently runnable:

```powershell
python scripts/audit_dataset.py
python scripts/train_model.py --mode all --target binary
python scripts/train_model.py --mode all --target multiclass
python scripts/audit_any_dataset.py --file data/<file> --target <column>
python scripts/baseline_rules.py
python scripts/threshold_analysis.py --policy all
python scripts/make_report.py
python scripts/make_chapter4.py
```

Pipeline: audit → feature policy → stratified 80:20 split → impute and ordinal
encode → SMOTENC inside the pipeline → one-hot and standardise → SVM
(linear/polynomial/RBF) → Grid Search with 3-fold stratified CV on macro F1 →
Platt calibration → held-out evaluation with subgroup analysis.

## Testing

```powershell
python -m pytest                          # 128 tests
python scripts/make_test_report.py        # testing report for the write-up
python scripts/ui_walkthrough.py          # browser walkthrough + screenshots
python scripts/demo_evaluation.py         # small-sample evaluation via the API
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness and readiness probe |
| `/api/schema` | GET | Field names, permitted values, ranges |
| `/api/models` | GET | Model availability and metadata |
| `/api/predict` | POST | Run a prediction |
| `/api/assessments` | GET | Recent assessments |
| `/api/assessments/<ref>` | GET | One assessment with inputs and explanation |

## Deployment

See `docs/DEPLOYMENT.md`. `render.yaml` provisions the web service and
PostgreSQL database; `Dockerfile` and `Procfile` are provided as alternatives.

## Documentation

| File | Contents |
|---|---|
| `docs/SYSTEM_ARCHITECTURE.md` | Layering, request lifecycle, data model, security controls |
| `docs/DEPLOYMENT.md` | Step-by-step Render deployment and verification checklist |
| `docs/ARCHITECTURE.md` | Machine-learning pipeline design |
| `docs/DATASET_AUDIT.md` | **Read before interpreting any result** — dataset integrity findings |
| `docs/THESIS_ALIGNMENT.md` | Corrections needed in Chapters 1–3 and recommended framing |
| `docs/CHAPTER_1_3_CORRECTIONS.md` | Paste-ready replacement text |
| `reports/SYSTEM_TEST_REPORT.md` | Test results mapped to the twelve verification areas |
| `reports/THRESHOLD_ANALYSIS.md` | Decision cut-off, cost weighting, calibration, prevalence transportability |
| `reports/CHAPTER_4_DRAFT.md` | Chapter 4 narrative generated from the artefacts |

## Important

The Kaggle dataset used to train the shipped models encodes the label
deterministically in `Fever Duration (Days)`; a one-line rule reproduces almost
the entire reported accuracy. Two further public typhoid datasets were audited
and found to have comparable problems. See `docs/DATASET_AUDIT.md` before citing
any performance figure.

Research prototype. Not a medical device. Decision support only — it does not
replace blood culture, confirmatory laboratory testing or clinical judgement.
