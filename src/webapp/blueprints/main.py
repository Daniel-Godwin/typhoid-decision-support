"""Public routes: landing page and health probe."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, redirect, render_template, url_for
from flask_login import current_user

from ..extensions import db
from ..services import prediction_service as ps

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(
            url_for("admin.dashboard") if current_user.is_admin else url_for("clinician.dashboard")
        )
    return render_template("index.html")


@main_bp.get("/health")
def health():
    """Liveness and readiness probe. Used by Render and by monitoring."""
    checks = {}
    ok = True

    try:
        db.session.execute(db.text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc.__class__.__name__}"
        ok = False

    # Which engine answered, so a silent fall-back to the local SQLite file is
    # visible rather than being mistaken for a working deployment. In
    # production that file is on an ephemeral disk and is emptied on every
    # restart. Only the dialect name is reported — never the host, the
    # database name or any part of the connection string.
    try:
        dialect = db.engine.dialect.name
    except Exception:
        dialect = "unknown"
    checks["database_engine"] = dialect
    if current_app.config.get("ENV_NAME") == "production" and dialect == "sqlite":
        checks["database_warning"] = (
            "running on the ephemeral SQLite fallback; DATABASE_URL is not set, "
            "and all stored data is lost on restart"
        )
        ok = False

    model_state = {}
    for mode in ps.available_target_modes():
        available = ps.model_is_available(mode)
        model_state[mode] = "available" if available else "missing"
        if mode == current_app.config["DEFAULT_TARGET_MODE"] and not available:
            ok = False
    checks["models"] = model_state

    body = {
        "status": "ok" if ok else "degraded",
        "environment": current_app.config.get("ENV_NAME"),
        "version": current_app.config.get("APP_VERSION"),
        "checks": checks,
    }
    return jsonify(body), (200 if ok else 503)
