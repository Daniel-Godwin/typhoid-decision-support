"""Administrative area: users, activity log, system status."""
from __future__ import annotations

import platform
import sys
from datetime import timedelta

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import func, or_

from ..extensions import db
from ..forms import UserForm
from ..models import Assessment, AuditLog, Patient, Role, User, utcnow
from ..security import admin_required, record_audit, roles_required
from ..services import prediction_service as ps

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/")
@login_required
@admin_required
def dashboard():
    since = utcnow() - timedelta(days=30)

    stats = {
        "users": User.query.count(),
        "active_users": User.query.filter_by(is_active_account=True).count(),
        "locked_users": User.query.filter(User.locked_until.isnot(None)).count(),
        "patients": Patient.query.count(),
        "assessments": Assessment.query.count(),
        "assessments_30d": Assessment.query.filter(Assessment.created_at >= since).count(),
        "flagged": Assessment.query.filter(Assessment.flagged_for_testing.is_(True)).count(),
        "audit_events": AuditLog.query.count(),
        "failed_logins_30d": AuditLog.query.filter(
            AuditLog.action == "login_failed", AuditLog.created_at >= since
        ).count(),
    }

    by_role = dict(
        db.session.query(User.role, func.count(User.id)).group_by(User.role).all()
    )
    recent_activity = AuditLog.query.order_by(AuditLog.created_at.desc()).limit(12).all()
    busiest = (
        db.session.query(User.full_name, func.count(Assessment.id).label("n"))
        .join(Assessment, Assessment.clinician_id == User.id)
        .group_by(User.id)
        .order_by(func.count(Assessment.id).desc())
        .limit(5)
        .all()
    )

    models = {}
    for mode in ps.available_target_modes():
        available = ps.model_is_available(mode)
        meta = ps.model_metadata(mode) if available else None
        models[mode] = {"available": available, "meta": meta}

    return render_template(
        "admin/dashboard.html",
        stats=stats,
        by_role=by_role,
        recent_activity=recent_activity,
        busiest=busiest,
        models=models,
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
@admin_bp.route("/users")
@login_required
@admin_required
def users():
    page = request.args.get("page", 1, type=int)
    search = (request.args.get("q") or "").strip()
    role = request.args.get("role", "")

    query = User.query
    if search:
        query = query.filter(
            or_(User.full_name.ilike(f"%{search}%"), User.email.ilike(f"%{search}%"))
        )
    if role in Role.ALL:
        query = query.filter(User.role == role)

    pagination = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    return render_template("admin/users.html", pagination=pagination, search=search, role=role)


@admin_bp.route("/users/new", methods=["GET", "POST"])
@login_required
@admin_required
def user_new():
    form = UserForm()
    if form.validate_on_submit():
        user = User(
            full_name=form.full_name.data.strip(),
            email=form.email.data.lower().strip(),
            role=form.role.data,
            facility=(form.facility.data or "").strip() or None,
            is_active_account=form.is_active_account.data,
            must_change_password=form.must_change_password.data,
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        record_audit("user_created", category="user_admin", entity_type="User",
                     entity_id=user.id, detail=f"{user.email} role={user.role}")
        flash(f"Account created for {user.full_name}.", "success")
        return redirect(url_for("admin.users"))
    return render_template("admin/user_form.html", form=form, user=None)


@admin_bp.route("/users/<int:user_id>/edit", methods=["GET", "POST"])
@login_required
@admin_required
def user_edit(user_id: int):
    user = db.get_or_404(User, user_id)
    form = UserForm(obj=user, editing_user=user)
    if form.validate_on_submit():
        if user.id == current_user.id and form.role.data != Role.ADMIN:
            flash("You cannot remove your own administrator role.", "error")
            return render_template("admin/user_form.html", form=form, user=user), 400
        if user.id == current_user.id and not form.is_active_account.data:
            flash("You cannot disable your own account.", "error")
            return render_template("admin/user_form.html", form=form, user=user), 400

        user.full_name = form.full_name.data.strip()
        user.email = form.email.data.lower().strip()
        user.role = form.role.data
        user.facility = (form.facility.data or "").strip() or None
        user.is_active_account = form.is_active_account.data
        user.must_change_password = form.must_change_password.data
        changed_password = False
        if form.password.data:
            user.set_password(form.password.data)
            changed_password = True
        db.session.commit()
        record_audit(
            "user_updated", category="user_admin", entity_type="User", entity_id=user.id,
            detail=f"{user.email} role={user.role}" + (" password reset" if changed_password else ""),
        )
        flash(f"{user.full_name}'s account has been updated.", "success")
        return redirect(url_for("admin.users"))
    return render_template("admin/user_form.html", form=form, user=user)


@admin_bp.route("/users/<int:user_id>/toggle", methods=["POST"])
@login_required
@admin_required
def user_toggle(user_id: int):
    user = db.get_or_404(User, user_id)
    if user.id == current_user.id:
        flash("You cannot disable your own account.", "error")
        return redirect(url_for("admin.users"))
    user.is_active_account = not user.is_active_account
    db.session.commit()
    record_audit(
        "user_enabled" if user.is_active_account else "user_disabled",
        category="user_admin", entity_type="User", entity_id=user.id, detail=user.email,
    )
    flash(
        f"{user.full_name}'s account has been "
        f"{'enabled' if user.is_active_account else 'disabled'}.",
        "success",
    )
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/unlock", methods=["POST"])
@login_required
@admin_required
def user_unlock(user_id: int):
    user = db.get_or_404(User, user_id)
    user.locked_until = None
    user.failed_logins = 0
    db.session.commit()
    record_audit("user_unlocked", category="security", entity_type="User",
                 entity_id=user.id, detail=user.email)
    flash(f"{user.full_name}'s account has been unlocked.", "success")
    return redirect(url_for("admin.users"))


# ---------------------------------------------------------------------------
# Activity log
# ---------------------------------------------------------------------------
@admin_bp.route("/logs")
@login_required
@roles_required(Role.ADMIN, Role.AUDITOR)
def logs():
    page = request.args.get("page", 1, type=int)
    category = request.args.get("category", "")
    outcome = request.args.get("outcome", "")
    search = (request.args.get("q") or "").strip()

    query = AuditLog.query
    if category in AuditLog.CATEGORIES:
        query = query.filter(AuditLog.category == category)
    if outcome == "failed":
        query = query.filter(AuditLog.success.is_(False))
    elif outcome == "success":
        query = query.filter(AuditLog.success.is_(True))
    if search:
        query = query.filter(
            or_(
                AuditLog.action.ilike(f"%{search}%"),
                AuditLog.user_email.ilike(f"%{search}%"),
                AuditLog.detail.ilike(f"%{search}%"),
            )
        )

    pagination = query.order_by(AuditLog.created_at.desc()).paginate(
        page=page, per_page=40, error_out=False
    )
    return render_template(
        "admin/logs.html",
        pagination=pagination,
        category=category,
        outcome=outcome,
        search=search,
        categories=AuditLog.CATEGORIES,
        top_actions=AuditLog.counts_by_action(8),
    )


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------
@admin_bp.route("/system")
@login_required
@admin_required
def system():
    models = {}
    for mode in ps.available_target_modes():
        available = ps.model_is_available(mode)
        models[mode] = {
            "available": available,
            "meta": ps.model_metadata(mode) if available else None,
        }

    try:
        db.session.execute(db.text("SELECT 1"))
        database_ok = True
        database_error = None
    except Exception as exc:  # pragma: no cover - defensive
        database_ok = False
        database_error = str(exc)

    uri = current_app.config.get("SQLALCHEMY_DATABASE_URI", "")
    engine = uri.split(":", 1)[0] if uri else "unknown"

    return render_template(
        "admin/system.html",
        models=models,
        artefacts=ps.available_models(),
        policies=ps.available_policies(),
        runtime={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "environment": current_app.config.get("ENV_NAME"),
            "debug": current_app.debug,
            "database_engine": engine,
            "database_ok": database_ok,
            "database_error": database_error,
            "triage_threshold": current_app.config["TRIAGE_THRESHOLD"],
            "default_target": current_app.config["DEFAULT_TARGET_MODE"],
            "default_policy": current_app.config["DEFAULT_FEATURE_POLICY"],
            "rate_limiting": current_app.config.get("RATELIMIT_ENABLED"),
            "session_lifetime": str(current_app.config["PERMANENT_SESSION_LIFETIME"]),
        },
    )
