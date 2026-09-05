# System Architecture

The system is two independent packages joined at exactly one point.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                              │
│   templates/ (Jinja2)          static/css/app.css   static/js/app.js     │
│   base · auth · clinician · admin · errors                               │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│                          APPLICATION LAYER   src/webapp/                  │
│                                                                          │
│   __init__.py     application factory, error handlers, context           │
│   config.py       Development / Testing / Production configuration        │
│   extensions.py   SQLAlchemy · Login · CSRF · Migrate · Limiter          │
│   security.py     role decorators, audit recording, security headers      │
│   logging_setup.py  rotating file / stdout logging, request correlation   │
│   forms.py        WTForms validation for every submitted form            │
│   cli.py          init-db · create-admin · bootstrap · seed-demo · stats  │
│                                                                          │
│   blueprints/     main · auth · clinician · admin · api                  │
│                   ── HTTP routing only, no ML imports ──                  │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│              SERVICE BOUNDARY   src/webapp/services/                      │
│                     prediction_service.py                                 │
│   The ONLY module in the web application that imports typhoid_ml.         │
│   Exposes: form_schema · validate · run_prediction · triage_recommendation│
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│              MACHINE LEARNING PACKAGE   src/typhoid_ml/                   │
│              (unchanged by the web application)                           │
│                                                                          │
│   config.py       feature policies, target modes, search spaces          │
│   data.py         loading, auditing, target construction                 │
│   preprocessing.py  imputation · ordinal encode · SMOTENC · scale/one-hot │
│   model.py        SVM pipeline, calibration, GridSearchCV                │
│   train.py        audit → baseline → grid → final orchestration          │
│   evaluate.py     metrics, confusion matrices, curves, subgroup analysis  │
│   explain.py      occlusion attribution                                   │
│   predict.py      model loading, schema, validation, inference           │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────────────┐
│         PERSISTENCE                                                       │
│   models/*.joblib          trained artefacts + metadata (read-only)       │
│   data/*.csv               source datasets (read-only)                    │
│   PostgreSQL / SQLite      users · patients · assessments · audit_logs    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Why the boundary matters

`src/webapp/blueprints/` contains no reference to `typhoid_ml`. This is enforced
by a test (`test_ml_package_is_not_imported_by_blueprints`) which fails the build
if any blueprint imports the ML package directly.

The consequence is that the model can be retrained, re-tuned or replaced without
touching a view, and the web application can be restructured without any risk to
the trained artefacts or their preprocessing requirements. The ML package has no
knowledge that a web application exists — it can still be driven entirely from
the command line, and its own test suite (41 tests) runs without Flask installed
in the import path.

## Request lifecycle

```text
 1. before_request   assign request id, start timer
 2. before_request   force password change if flagged
 3. Flask-Login      resolve session → current_user
 4. @roles_required  authorise, or record access_denied and abort 403
 5. view             validate input via WTForms or the prediction service
 6. service          delegate to typhoid_ml (schema, validate, predict, explain)
 7. persistence      write Assessment row with full provenance
 8. audit            append AuditLog row
 9. after_request    apply security headers, log method/path/status/duration
```

## Data model

| Table | Purpose | Notable columns |
|---|---|---|
| `users` | Accounts and credentials | `role`, `password_hash`, `failed_logins`, `locked_until`, `must_change_password` |
| `patients` | Patient register | `patient_code` (unique), `created_by_id` |
| `assessments` | One row per prediction | `input_json`, `model_kernel`, `model_trained_at`, `prediction`, `confidence`, `explanation_json`, `latency_ms` |
| `audit_logs` | Append-only activity trail | `action`, `category`, `success`, `ip_address`, `user_agent` |

`assessments` stores the submitted inputs and the identity of the model that
produced the result, so any historical prediction can be reproduced and
explained months later — a requirement for clinical auditability.

## Roles

| Role | Clinical features | Administration | Activity log |
|---|---|---|---|
| Administrator | full | full | read |
| Clinician | own patients and assessments | none | none |
| Auditor | read-only | none | read |

Administrators pass every role check. Clinicians see only their own assessments
and patients. Auditors may read but never create, which is enforced in the views
and in the API.

## Security controls

| Control | Implementation |
|---|---|
| Password storage | PBKDF2-SHA256, 600,000 iterations |
| Password policy | ≥ 10 characters, letters and digits, common passwords rejected |
| Brute force | Account lock after 5 failures for 15 minutes; rate limit on the login route |
| Session | HttpOnly, SameSite=Lax, Secure in production, 60-minute lifetime, strong protection |
| CSRF | Flask-WTF token on every form; API exempt but session-authenticated |
| Headers | CSP, X-Frame-Options DENY, nosniff, Referrer-Policy, HSTS in production |
| Open redirect | `next` parameter restricted to same-site relative paths |
| Enumeration | Identical response for unknown account and wrong password |
| Input validation | WTForms server-side, plus the ML pipeline's own schema validation |
| Request size | 8 MB cap |

## Configuration

Every environment-specific or secret value is read from an environment
variable — see `.env.example`. `validate_production_config()` logs an error at
start-up if `SECRET_KEY` is still the development default or `DATABASE_URL` is
unset in production, so a misconfigured deploy reports the problem rather than
running insecurely in silence.
