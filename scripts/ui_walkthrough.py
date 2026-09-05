"""Browser walkthrough: exercises the whole system and captures screenshots.

Signs in as each role, runs a real assessment through the interface, visits every
page, and captures desktop, tablet and mobile renderings. Produces the evidence
used for the system-testing section of the write-up.

    python scripts/run_dev.py           # in one terminal
    python scripts/ui_walkthrough.py    # in another
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "reports" / "screenshots"

VIEWPORTS = {
    "desktop": {"width": 1440, "height": 960},
    "tablet": {"width": 834, "height": 1112},
    "mobile": {"width": 390, "height": 844},
}

ACCOUNTS = {
    "admin": ("admin@example.com", "AdminPass2026"),
    "clinician": ("clinician@example.com", "ClinicPass2026"),
    "auditor": ("auditor@example.com", "AuditPass2026"),
}


def sign_in(page, base, email, password):
    page.goto(f"{base}/auth/login", wait_until="networkidle")
    page.fill('input[name="email"]', email)
    page.fill('input[name="password"]', password)
    page.locator('input[type="submit"], button[type="submit"]').first.click()
    page.wait_for_load_state("networkidle")


def shot(page, name, results, full=True):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path), full_page=full)
    results.append({"screenshot": name, "url": page.url, "title": page.title()})
    print(f"  captured {name:44s} {page.url}")


def run_assessment(page, base, results):
    from typhoid_ml.predict import form_schema

    schema = form_schema("routine")
    record = dict(schema["reference"])
    record.update({
        "Age": 26, "Fever Duration (Days)": 9,
        "Widal Test": "High O & H Antibody", "Typhidot Test": "IgM Positive",
        "Gastrointestinal Symptoms": "Diarrhea", "Neurological Symptoms": "Delirium",
        "Skin Manifestations": "Yes", "Water Source Type": "River",
        "Sanitation Facilities": "Open Defecation", "Hand Hygiene": "No",
        "White Blood Cell Count": 3800, "Platelet Count": 95000, "Location": "Endemic",
    })

    page.goto(f"{base}/clinic/assess", wait_until="networkidle")
    shot(page, "05-clinician-assessment-form", results)

    for key, value in record.items():
        locator = page.locator(f'[name="{key}"]')
        if locator.count() == 0:
            continue
        if locator.evaluate("el => el.tagName") == "SELECT":
            page.select_option(f'[name="{key}"]', str(value))
        else:
            page.fill(f'[name="{key}"]', str(value))

    options = page.locator('select[name="patient_id"] option')
    if options.count() > 1:
        page.select_option('select[name="patient_id"]', index=1)

    page.locator('input[type="submit"], button[type="submit"]').first.click()
    page.wait_for_load_state("networkidle")
    shot(page, "06-clinician-assessment-result", results)
    return page.url


def validation_check(page, base, results):
    page.goto(f"{base}/clinic/patients/new", wait_until="networkidle")
    page.fill('input[name="patient_code"]', "PT-1001")   # already exists
    page.fill('input[name="full_name"]', "Duplicate Code")
    page.locator('input[type="submit"], button[type="submit"]').first.click()
    page.wait_for_load_state("networkidle")
    shot(page, "12-validation-error", results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # ---------- desktop, full journey ----------
        ctx = browser.new_context(viewport=VIEWPORTS["desktop"])
        page = ctx.new_page()

        print("Anonymous:")
        page.goto(f"{base}/", wait_until="networkidle")
        shot(page, "01-landing", results)
        page.goto(f"{base}/auth/login", wait_until="networkidle")
        shot(page, "02-login", results)

        print("Failed sign-in:")
        page.fill('input[name="email"]', "clinician@example.com")
        page.fill('input[name="password"]', "wrong-password")
        page.locator('input[type="submit"], button[type="submit"]').first.click()
        page.wait_for_load_state("networkidle")
        shot(page, "03-login-failed", results)

        print("Clinician:")
        sign_in(page, base, *ACCOUNTS["clinician"])
        shot(page, "04-clinician-dashboard", results)
        detail_url = run_assessment(page, base, results)
        page.goto(f"{base}/clinic/assessments", wait_until="networkidle")
        shot(page, "07-clinician-assessments", results)
        page.goto(f"{base}/clinic/patients", wait_until="networkidle")
        shot(page, "08-clinician-patients", results)
        page.goto(f"{base}/clinic/patients/1", wait_until="networkidle")
        shot(page, "09-clinician-patient-detail", results)
        validation_check(page, base, results)

        print("Forbidden area:")
        page.goto(f"{base}/admin/users", wait_until="networkidle")
        shot(page, "13-forbidden-403", results)
        page.goto(f"{base}/no-such-page", wait_until="networkidle")
        shot(page, "14-not-found-404", results)
        ctx.close()

        print("Administrator:")
        ctx = browser.new_context(viewport=VIEWPORTS["desktop"])
        page = ctx.new_page()
        sign_in(page, base, *ACCOUNTS["admin"])
        shot(page, "15-admin-dashboard", results)
        page.goto(f"{base}/admin/users", wait_until="networkidle")
        shot(page, "16-admin-users", results)
        page.goto(f"{base}/admin/users/new", wait_until="networkidle")
        shot(page, "17-admin-user-form", results)
        page.goto(f"{base}/admin/logs", wait_until="networkidle")
        shot(page, "18-admin-activity-log", results)
        page.goto(f"{base}/admin/system", wait_until="networkidle")
        shot(page, "19-admin-system-status", results)
        page.goto(f"{base}/auth/profile", wait_until="networkidle")
        shot(page, "20-profile", results)
        ctx.close()

        print("Auditor:")
        ctx = browser.new_context(viewport=VIEWPORTS["desktop"])
        page = ctx.new_page()
        sign_in(page, base, *ACCOUNTS["auditor"])
        shot(page, "21-auditor-dashboard", results)
        page.goto(f"{base}/clinic/assess", wait_until="networkidle")
        shot(page, "22-auditor-blocked-from-assessing", results)
        ctx.close()

        # ---------- responsive ----------
        print("Responsive rendering:")
        for label, viewport in VIEWPORTS.items():
            if label == "desktop":
                continue
            ctx = browser.new_context(viewport=viewport)
            page = ctx.new_page()
            page.goto(f"{base}/auth/login", wait_until="networkidle")
            shot(page, f"23-{label}-login", results)
            sign_in(page, base, *ACCOUNTS["clinician"])
            shot(page, f"24-{label}-dashboard", results)
            page.goto(f"{base}/clinic/assess", wait_until="networkidle")
            shot(page, f"25-{label}-assessment-form", results)
            if detail_url:
                page.goto(detail_url, wait_until="networkidle")
                shot(page, f"26-{label}-assessment-result", results)
            ctx.close()

        # ---------- mobile navigation drawer ----------
        ctx = browser.new_context(viewport=VIEWPORTS["mobile"])
        page = ctx.new_page()
        sign_in(page, base, *ACCOUNTS["clinician"])
        page.click("[data-nav-toggle]")
        page.wait_for_timeout(400)
        shot(page, "27-mobile-navigation-open", results, full=False)
        ctx.close()

        browser.close()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "walkthrough.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{len(results)} screenshots written to {OUT}")


if __name__ == "__main__":
    main()
