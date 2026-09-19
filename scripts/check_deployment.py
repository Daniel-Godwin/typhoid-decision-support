"""Check a running deployment and say plainly what, if anything, is wrong.

Reads the public `/health` endpoint and turns it into a verdict. No credentials
are needed and nothing is written.

    python scripts/check_deployment.py
    python scripts/check_deployment.py --url http://127.0.0.1:5000

Exit status is 0 when the deployment is serviceable and 1 when it is not, so
the script can also be used as a smoke test after a deploy.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "https://typhoid-decision-support.onrender.com"
EXPECTED_FEATURES = 13


def fetch(url: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/health", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def schema(url: str, timeout: float) -> dict | None:
    """The public schema endpoint, if this build exposes it without a session."""
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/api/schema", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="A sleeping free-tier instance can take about a minute to wake.",
    )
    args = ap.parse_args()

    print(f"Checking {args.url}\n(a sleeping instance takes up to a minute to answer)\n")
    try:
        body = fetch(args.url, args.timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            body = json.loads(exc.read().decode("utf-8"))
        else:
            print(f"UNREACHABLE  HTTP {exc.code} from /health")
            return 1
    except Exception as exc:
        print(f"UNREACHABLE  {exc.__class__.__name__}: {exc}")
        return 1

    checks = body.get("checks", {})
    problems: list[str] = []

    print(f"  status      {body.get('status')}")
    print(f"  environment {body.get('environment')}")
    print(f"  version     {body.get('version')}")
    print(f"  database    {checks.get('database')} ({checks.get('database_engine')})")
    print(f"  models      {checks.get('models')}")

    if checks.get("database") != "ok":
        problems.append(
            "The database is not reachable. Nothing can be saved and every sign-in "
            "will fail."
        )
    if body.get("environment") == "production" and checks.get("database_engine") == "sqlite":
        problems.append(
            "DATABASE_URL is not set, so the service is using a temporary local "
            "database. It is emptied on every restart, which erases accounts and "
            "patients and signs users out mid-session. Set DATABASE_URL in the "
            "hosting platform's environment settings and redeploy."
        )
    models = checks.get("models") or {}
    for mode, state in models.items():
        if state != "available":
            problems.append(
                f"The {mode} model artefact is missing from the deployed build. "
                "Confirm models/*.joblib are committed and not excluded by .gitignore."
            )

    sch = schema(args.url, args.timeout)
    if sch is not None:
        n = len(sch.get("numeric", [])) + len(sch.get("categorical", []))
        print(f"  form fields {n}")
        if n != EXPECTED_FEATURES:
            problems.append(
                f"The running build collects {n} attributes, not {EXPECTED_FEATURES}. "
                "The deployed code predates the feature reduction; push and redeploy."
            )
        stale = sorted(
            set(sch.get("categorical", []))
            & {"Widal Test", "Typhidot Test", "Gastrointestinal Symptoms",
               "Ongoing Infection in Society", "Neurological Symptoms",
               "Typhoid Vaccination Status"}
        )
        if stale:
            problems.append(
                "The running build still collects attributes dropped at supervisory "
                f"review: {', '.join(stale)}. Push and redeploy."
            )
    else:
        print("  form fields (requires a session; skipped)")

    print()
    if problems:
        print(f"NOT SERVICEABLE — {len(problems)} problem(s):\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}\n")
        return 1

    print("SERVICEABLE — database reachable, models loaded, feature space current.")
    print("If users are still being signed out, the cause is not in this check;")
    print("capture the hosting platform's log line for the failing request.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
