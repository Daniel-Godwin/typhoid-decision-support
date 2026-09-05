"""Access control, audit recording and password policy."""
from __future__ import annotations

import re
from functools import wraps

from flask import abort, current_app, flash, redirect, request, url_for
from flask_login import current_user

from .extensions import db
from .models import AuditLog, Role


def record_audit(
    action: str,
    *,
    category: str = "general",
    entity_type: str | None = None,
    entity_id=None,
    detail: str | None = None,
    success: bool = True,
    user=None,
    commit: bool = True,
) -> AuditLog:
    """Write one row to the activity trail and mirror it into the app log."""
    actor = user if user is not None else (current_user if current_user.is_authenticated else None)
    entry = AuditLog(
        user_id=getattr(actor, "id", None),
        user_email=getattr(actor, "email", None) or "anonymous",
        action=action,
        category=category,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        detail=detail,
        success=success,
        ip_address=_client_ip(),
        user_agent=(request.user_agent.string[:300] if request else None),
    )
    db.session.add(entry)
    if commit:
        db.session.commit()

    logger = current_app.logger
    message = (
        f"audit action={action} category={category} "
        f"user={entry.user_email} entity={entity_type}:{entity_id} success={success}"
    )
    (logger.info if success else logger.warning)(message)
    return entry


def _client_ip() -> str | None:
    if not request:
        return None
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return (request.remote_addr or "")[:64] or None


def roles_required(*roles: str):
    """Restrict a view to the given roles. Administrators always pass."""

    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for("auth.login", next=request.full_path))
            if current_user.role not in roles and not current_user.is_admin:
                record_audit(
                    "access_denied",
                    category="security",
                    detail=f"{request.method} {request.path} requires {roles}",
                    success=False,
                )
                abort(403)
            return view(*args, **kwargs)

        return wrapper

    return decorator


def admin_required(view):
    return roles_required(Role.ADMIN)(view)


def clinician_required(view):
    """Clinical features. Auditors are read-only and must not reach these."""
    return roles_required(Role.CLINICIAN)(view)


def password_problems(password: str, minimum: int = 10) -> list[str]:
    """Password policy, returned as a list so the form can show every failure."""
    issues = []
    if len(password or "") < minimum:
        issues.append(f"must be at least {minimum} characters")
    if not re.search(r"[A-Za-z]", password or ""):
        issues.append("must contain a letter")
    if not re.search(r"\d", password or ""):
        issues.append("must contain a number")
    if (password or "").lower() in {
        "password12", "password123", "changeme123", "typhoid123", "administrator",
    }:
        issues.append("is too common")
    return issues


def require_password_change_guard():
    """Force a password reset before anything else when the flag is set."""
    if not current_user.is_authenticated or not getattr(current_user, "must_change_password", False):
        return None
    allowed = {"auth.change_password", "auth.logout", "static", "main.health"}
    if request.endpoint in allowed:
        return None
    flash("Please set a new password before continuing.", "warning")
    return redirect(url_for("auth.change_password"))


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "font-src 'self' data:; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'"
    ),
}


def apply_security_headers(response):
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    if current_app.config.get("ENV_NAME") == "production":
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    return response
