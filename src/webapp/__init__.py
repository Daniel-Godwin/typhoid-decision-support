"""Application factory for the Typhoid Diagnostic Decision Support system.

Layering:

    webapp/                     web application (this package)
      blueprints/               HTTP routing only
      services/                 the sole boundary to the ML package
    typhoid_ml/                 machine-learning pipeline, independent of the web

No blueprint imports `typhoid_ml` directly. That separation keeps the trained
artefacts and their preprocessing requirements insulated from web-layer change.
"""
from __future__ import annotations

import os

from flask import Flask, render_template, request

from .config import get_config, validate_production_config
from .extensions import csrf, db, limiter, login_manager, migrate
from .logging_setup import configure_logging, register_request_logging
from .security import apply_security_headers, require_password_change_guard

__version__ = "2.0.0"


def create_app(config_name: str | None = None) -> Flask:
    app = Flask(
        __name__,
        instance_relative_config=False,
        static_folder="static",
        template_folder="templates",
    )
    config_class = get_config(config_name)
    app.config.from_object(config_class)
    # ProductionConfig declares the database URI as a property; resolve it.
    if isinstance(getattr(config_class, "SQLALCHEMY_DATABASE_URI", None), property):
        app.config["SQLALCHEMY_DATABASE_URI"] = config_class().SQLALCHEMY_DATABASE_URI

    app.config["INSTANCE_DIR"].mkdir(parents=True, exist_ok=True) if app.config.get(
        "INSTANCE_DIR"
    ) else None
    from .config import INSTANCE_DIR

    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

    configure_logging(app)
    for problem in validate_production_config(app):
        app.logger.error("configuration problem: %s", problem)

    _init_extensions(app)
    _register_blueprints(app)
    _register_error_handlers(app)
    _register_context(app)
    register_request_logging(app)

    app.after_request(apply_security_headers)
    app.before_request(require_password_change_guard)

    from . import cli  # noqa: F401  (registers CLI commands)

    cli.register(app)

    # Warm the model that will actually serve requests. Under the debug
    # reloader the factory runs twice (supervisor + worker); only the worker
    # needs the artefacts, so the parent skips this and start-up halves.
    reloader_parent = app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
    if app.config.get("PRELOAD_MODELS") and not reloader_parent:
        with app.app_context():
            from .services import prediction_service

            modes = [app.config["DEFAULT_TARGET_MODE"]]
            if app.config.get("PRELOAD_ALL_MODELS"):
                modes = list(prediction_service.available_target_modes())
            status = prediction_service.preload(tuple(modes))
            app.logger.info("model preload: %s", status)

    app.logger.info(
        "application ready name=%s version=%s env=%s",
        app.config["APP_NAME"],
        __version__,
        app.config.get("ENV_NAME"),
    )
    return app


def _init_extensions(app: Flask) -> None:
    db.init_app(app)
    migrate.init_app(app, db, render_as_batch=True)
    login_manager.init_app(app)
    csrf.init_app(app)
    if app.config.get("RATELIMIT_ENABLED"):
        limiter.init_app(app)


def _register_blueprints(app: Flask) -> None:
    from .blueprints.admin import admin_bp
    from .blueprints.api import api_bp
    from .blueprints.auth import auth_bp
    from .blueprints.clinician import clinician_bp
    from .blueprints.main import main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(clinician_bp, url_prefix="/clinic")
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(api_bp, url_prefix="/api")
    csrf.exempt(api_bp)  # token-free JSON API; protected by login and rate limits


def _register_error_handlers(app: Flask) -> None:
    def wants_json() -> bool:
        return request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json"

    def handler(code: str, title: str, message: str, status: int):
        def _handle(error):  # noqa: ANN001
            if wants_json():
                return {"error": title, "message": message, "status": status}, status
            return render_template(
                "errors/error.html", code=code, title=title, message=message
            ), status

        return _handle

    app.register_error_handler(
        400, handler("400", "Bad request", "The request could not be understood.", 400)
    )
    app.register_error_handler(
        401, handler("401", "Sign-in required", "Please sign in to continue.", 401)
    )
    app.register_error_handler(
        403,
        handler(
            "403",
            "Not permitted",
            "Your account does not have access to this area. If you believe this is "
            "wrong, contact an administrator.",
            403,
        ),
    )
    app.register_error_handler(
        404,
        handler("404", "Page not found", "The page you requested does not exist.", 404),
    )
    app.register_error_handler(
        413,
        handler("413", "Request too large", "The submitted data exceeded the size limit.", 413),
    )
    app.register_error_handler(
        429,
        handler(
            "429",
            "Too many requests",
            "You have made too many requests in a short period. Please wait and try again.",
            429,
        ),
    )

    @app.errorhandler(500)
    @app.errorhandler(Exception)
    def _internal(error):  # noqa: ANN001
        db.session.rollback()
        app.logger.exception("unhandled exception: %s", error)
        status = getattr(error, "code", 500)
        if not isinstance(status, int) or status < 400:
            status = 500
        if status != 500:
            # Let registered handlers deal with known HTTP errors.
            return error
        if wants_json():
            return {"error": "Internal server error", "status": 500}, 500
        return render_template(
            "errors/error.html",
            code="500",
            title="Something went wrong",
            message="An unexpected error occurred. It has been logged for review.",
        ), 500


def _register_context(app: Flask) -> None:
    from .models import Role

    @app.context_processor
    def inject_globals():
        from datetime import datetime, timezone

        return {
            "APP_NAME": app.config["APP_NAME"],
            "APP_VERSION": __version__,
            "ENV_NAME": app.config.get("ENV_NAME"),
            "Role": Role,
            "now": datetime.now(timezone.utc),
        }

    @app.template_filter("pct")
    def _pct(value, digits=1):
        try:
            return f"{float(value) * 100:.{digits}f}%"
        except (TypeError, ValueError):
            return "—"

    @app.template_filter("dt")
    def _dt(value, fmt="%d %b %Y, %H:%M"):
        if not value:
            return "—"
        try:
            return value.strftime(fmt)
        except AttributeError:
            return str(value)
