"""Authentication: sign in, sign out, password management, profile."""
from __future__ import annotations

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from urllib.parse import urlparse

from ..extensions import db, limiter
from ..forms import ChangePasswordForm, LoginForm, ProfileForm
from ..models import User
from ..security import record_audit

auth_bp = Blueprint("auth", __name__)


def _safe_next(target: str | None) -> str | None:
    """Only allow same-site redirects, to prevent open-redirect abuse."""
    if not target:
        return None
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc:
        return None
    if not target.startswith("/"):
        return None
    return target


def home_for(user) -> str:
    return url_for("admin.dashboard") if user.is_admin else url_for("clinician.dashboard")


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("20 per 15 minutes", methods=["POST"])
def login():
    if current_user.is_authenticated:
        return redirect(home_for(current_user))

    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data.lower().strip()
        user = User.query.filter_by(email=email).first()

        if user is None:
            # Same message and roughly the same work either way, so the response
            # does not reveal whether an account exists.
            record_audit(
                "login_failed",
                category="auth",
                detail=f"unknown account: {email}",
                success=False,
            )
            flash("Incorrect email or password.", "error")
            return render_template("auth/login.html", form=form), 401

        if user.is_locked:
            record_audit(
                "login_blocked", category="security", user=user,
                detail="account temporarily locked", success=False,
            )
            flash(
                "This account is temporarily locked after repeated failed sign-ins. "
                "Try again shortly or contact an administrator.",
                "error",
            )
            return render_template("auth/login.html", form=form), 423

        if not user.is_active_account:
            record_audit(
                "login_blocked", category="security", user=user,
                detail="account disabled", success=False,
            )
            flash("This account has been disabled. Contact an administrator.", "error")
            return render_template("auth/login.html", form=form), 403

        if not user.check_password(form.password.data):
            locked = user.register_failed_login(
                current_app.config["LOGIN_MAX_ATTEMPTS"],
                current_app.config["LOGIN_LOCKOUT_MINUTES"],
            )
            db.session.commit()
            record_audit(
                "login_failed", category="auth", user=user,
                detail="incorrect password" + (" — account locked" if locked else ""),
                success=False,
            )
            if locked:
                flash(
                    "Too many failed attempts. This account is locked for "
                    f"{current_app.config['LOGIN_LOCKOUT_MINUTES']} minutes.",
                    "error",
                )
            else:
                flash("Incorrect email or password.", "error")
            return render_template("auth/login.html", form=form), 401

        user.register_successful_login(request.headers.get("X-Forwarded-For", request.remote_addr))
        db.session.commit()
        login_user(user, remember=form.remember.data)
        record_audit("login", category="auth", user=user, detail=f"role={user.role}")
        flash(f"Signed in as {user.full_name}.", "success")

        return redirect(_safe_next(request.args.get("next")) or home_for(user))

    if form.errors:
        return render_template("auth/login.html", form=form), 400
    return render_template("auth/login.html", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
    record_audit("logout", category="auth")
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/password", methods=["GET", "POST"])
@login_required
def change_password():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if not current_user.check_password(form.current_password.data):
            record_audit(
                "password_change_failed", category="security",
                detail="current password incorrect", success=False,
            )
            flash("Your current password is incorrect.", "error")
            return render_template("auth/change_password.html", form=form), 400

        current_user.set_password(form.new_password.data)
        current_user.must_change_password = False
        db.session.commit()
        record_audit("password_changed", category="security")
        flash("Your password has been updated.", "success")
        return redirect(home_for(current_user))

    return render_template("auth/change_password.html", form=form)


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    form = ProfileForm(obj=current_user)
    if form.validate_on_submit():
        current_user.full_name = form.full_name.data.strip()
        current_user.facility = (form.facility.data or "").strip() or None
        db.session.commit()
        record_audit("profile_updated", category="user_admin", entity_type="User",
                     entity_id=current_user.id)
        flash("Your profile has been updated.", "success")
        return redirect(url_for("auth.profile"))
    return render_template("auth/profile.html", form=form)
