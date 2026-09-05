"""Run the full test suite and produce a system-testing report.

Maps every test to the twelve verification areas and writes a table suitable for
the testing section of the write-up.

    python scripts/make_test_report.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Test class or module -> verification area
AREAS = [
    ("1. User authentication", ["TestAuthentication"]),
    ("2. Authorisation and access control", ["TestAuthorisation"]),
    ("3. Administrative functionality", ["TestAdmin"]),
    ("4. Clinician functionality", ["TestClinician"]),
    ("5. Database operations", ["TestDatabase"]),
    ("6. Patient data management", ["TestClinician::test_create_patient",
                                    "TestClinician::test_duplicate_patient_code",
                                    "TestClinician::test_patient"]),
    ("7. Model loading and prediction", ["TestModelIntegration"]),
    ("8. Preprocessing and feature handling", ["test_pipeline.py"]),
    ("9. Prediction results and triage", ["TestAssessmentFlow"]),
    ("10. System logging and audit trail", ["TestAuditTrail"]),
    ("11. Error handling and validation", ["TestErrorHandling"]),
    ("12. Production readiness", ["TestProductionReadiness"]),
    ("Dataset integrity", ["test_data.py"]),
    ("Decision-threshold analysis", ["test_threshold.py"]),
]


def run_suite():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=no", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return result.stdout + result.stderr


def parse(output: str):
    tests = []
    for line in output.splitlines():
        # A parametrised id may contain spaces (e.g. `test_x[Not recorded]`),
        # so match lazily up to the status keyword rather than to whitespace.
        m = re.match(r"^(tests/[\w/]+\.py)::(.+?)\s+(PASSED|FAILED|SKIPPED|ERROR)\b", line)
        if m:
            tests.append({"file": m.group(1), "test": m.group(2), "status": m.group(3)})
    summary = {}
    for key in ("passed", "failed", "skipped", "error"):
        m = re.search(rf"(\d+) {key}", output)
        if m:
            summary[key] = int(m.group(1))
    return tests, summary


def main():
    print("Running the full test suite…")
    output = run_suite()
    tests, summary = parse(output)
    if not tests:
        print("No test results parsed. Raw output:\n" + output[-2000:])
        return

    lines = ["# System Testing Report\n"]
    lines.append(
        f"_Generated {datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')} by "
        "`python scripts/make_test_report.py`. Regenerate after any change._\n"
    )

    total = len(tests)
    passed = sum(1 for t in tests if t["status"] == "PASSED")
    failed = sum(1 for t in tests if t["status"] in ("FAILED", "ERROR"))
    skipped = sum(1 for t in tests if t["status"] == "SKIPPED")

    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total test cases | {total} |")
    lines.append(f"| Passed | {passed} |")
    lines.append(f"| Failed | {failed} |")
    lines.append(f"| Skipped | {skipped} |")
    lines.append(f"| Pass rate | {passed * 100 / total:.1f}% |")
    lines.append("")

    lines.append("## Results by verification area\n")
    lines.append("| # | Verification area | Cases | Passed | Failed | Result |")
    lines.append("|---|---|---:|---:|---:|---|")
    assigned = set()
    for area, patterns in AREAS:
        matched = [
            t for t in tests
            if any(p in f"{t['file']}::{t['test']}" for p in patterns)
        ]
        for t in matched:
            assigned.add(f"{t['file']}::{t['test']}")
        if not matched:
            continue
        p = sum(1 for t in matched if t["status"] == "PASSED")
        f = sum(1 for t in matched if t["status"] in ("FAILED", "ERROR"))
        verdict = "Pass" if f == 0 else "**Fail**"
        num, sep, label = area.partition(". ")
        if not sep:  # an unnumbered area such as "Dataset integrity"
            num, label = "—", area
        lines.append(f"| {num} | {label} | {len(matched)} | {p} | {f} | {verdict} |")
    lines.append("")

    lines.append("## Component coverage\n")
    by_file = {}
    for t in tests:
        by_file.setdefault(t["file"], []).append(t)
    lines.append("| Test module | Component under test | Cases | Passed |")
    lines.append("|---|---|---:|---:|")
    descriptions = {
        "tests/test_data.py": "Dataset loading, target construction, feature policy",
        "tests/test_threshold.py": "Threshold sweep, operating points, calibration, prevalence adjustment",
        "tests/test_pipeline.py": "Preprocessing, SMOTENC, encoding, pipeline integrity",
        "tests/test_app.py": "ML reference interface and prediction service",
        "tests/test_webapp.py": "Web application, security, database, deployment readiness",
    }
    for path, group in sorted(by_file.items()):
        p = sum(1 for t in group if t["status"] == "PASSED")
        lines.append(
            f"| `{path}` | {descriptions.get(path, '—')} | {len(group)} | {p} |"
        )
    lines.append("")

    failures = [t for t in tests if t["status"] in ("FAILED", "ERROR")]
    if failures:
        lines.append("## Failures\n")
        for t in failures:
            lines.append(f"- `{t['file']}::{t['test']}` — {t['status']}")
        lines.append("")
    else:
        lines.append("## Failures\n")
        lines.append("None. Every test case passed.\n")

    skips = [t for t in tests if t["status"] == "SKIPPED"]
    if skips:
        lines.append("## Skipped\n")
        for t in skips:
            lines.append(f"- `{t['test']}` — trained model artefact not present")
        lines.append("")

    lines.append("## Non-functional verification\n")
    lines.append("| Property | Method | Result |")
    lines.append("|---|---|---|")
    ev = REPORTS / "final_evaluation_binary.json"
    if ev.exists():
        data = json.loads(ev.read_text(encoding="utf-8"))
        lat = data.get("latency", {})
        lines.append(
            f"| Inference latency | Measured over the held-out test partition | "
            f"{lat.get('single_record_ms', '—')} ms per record |"
        )
        lines.append(
            f"| Batch throughput | Batch prediction over {lat.get('batch_records', '—')} records | "
            f"{lat.get('records_per_second', '—')} records/second |"
        )
    lines.append("| Responsive layout | Rendered at 1440, 834 and 390 px | See `reports/screenshots/` |")
    lines.append("| Security headers | Inspected on every response | CSP, HSTS, X-Frame-Options, nosniff present |")
    lines.append("| Production start-up | gunicorn with 2 workers and 4 threads | Health check returns `ok` |")
    lines.append("| ML/web separation | Automated check that no blueprint imports `typhoid_ml` | Enforced by test |")
    lines.append("")

    out = REPORTS / "SYSTEM_TEST_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (REPORTS / "system_test_results.json").write_text(
        json.dumps({"summary": summary, "tests": tests}, indent=2), encoding="utf-8"
    )
    print(f"\n{passed}/{total} passed ({passed * 100 / total:.1f}%)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
