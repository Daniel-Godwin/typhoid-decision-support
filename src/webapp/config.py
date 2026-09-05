"""Configuration classes for each environment.

Every setting that differs between development and production, or that is a
secret, is read from an environment variable. Nothing sensitive is hard-coded.
"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

# Repository root: .../src/webapp/config.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
INSTANCE_DIR = ROOT / "instance"
LOG_DIR = ROOT / "logs"


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _normalise_db_url(url: str) -> str:
    """Render supplies `postgres://`; SQLAlchemy 2 requires `postgresql+psycopg2://`."""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


class BaseConfig:
    # --- core -------------------------------------------------------------
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-key-change-in-production")
    APP_NAME = "Typhoid Diagnostic Decision Support"
    APP_VERSION = "2.0.0"

    # --- database ---------------------------------------------------------
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True, "pool_recycle": 280}

    # --- session / cookies ------------------------------------------------
    PERMANENT_SESSION_LIFETIME = timedelta(
        minutes=int(os.environ.get("SESSION_LIFETIME_MINUTES", "60"))
    )
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_DURATION = timedelta(days=7)

    # --- security ---------------------------------------------------------
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB request cap
    LOGIN_MAX_ATTEMPTS = int(os.environ.get("LOGIN_MAX_ATTEMPTS", "5"))
    LOGIN_LOCKOUT_MINUTES = int(os.environ.get("LOGIN_LOCKOUT_MINUTES", "15"))
    PASSWORD_MIN_LENGTH = 10

    # --- machine learning -------------------------------------------------
    # The web layer never reaches past this: it asks the prediction service,
    # which is the only module that imports the ML package.
    DEFAULT_TARGET_MODE = os.environ.get("DEFAULT_TARGET_MODE", "binary")
    DEFAULT_FEATURE_POLICY = os.environ.get("DEFAULT_FEATURE_POLICY", "routine")
    # Probability at or above which a case is flagged for confirmatory testing.
    # 0.08 is the cost-optimal cut-off derived in reports/THRESHOLD_ANALYSIS.md
    # for the routine policy, weighting a missed case as ten times as damaging
    # as an unnecessary confirmatory test. It costs nothing in specificity
    # relative to the library default of 0.5. Regenerate with
    # `python scripts/threshold_analysis.py` after any retraining.
    TRIAGE_THRESHOLD = float(os.environ.get("TRIAGE_THRESHOLD", "0.08"))
    PRELOAD_MODELS = _bool("PRELOAD_MODELS", True)
    # Load every model at start-up rather than only the default one.
    # Off by default: the four-class artefact is 11 MB and is loaded
    # on first use instead.
    PRELOAD_ALL_MODELS = _bool("PRELOAD_ALL_MODELS", False)

    # --- logging ----------------------------------------------------------
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_DIR = LOG_DIR
    LOG_TO_STDOUT = _bool("LOG_TO_STDOUT", False)
    LOG_MAX_BYTES = 2 * 1024 * 1024
    LOG_BACKUP_COUNT = 5

    # --- rate limiting ----------------------------------------------------
    RATELIMIT_ENABLED = _bool("RATELIMIT_ENABLED", True)
    RATELIMIT_STORAGE_URI = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")
    RATELIMIT_DEFAULT = os.environ.get("RATELIMIT_DEFAULT", "300 per hour")

    # --- bootstrap admin --------------------------------------------------
    BOOTSTRAP_ADMIN_EMAIL = os.environ.get("BOOTSTRAP_ADMIN_EMAIL")
    BOOTSTRAP_ADMIN_PASSWORD = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD")

    PREFERRED_URL_SCHEME = "https"


class DevelopmentConfig(BaseConfig):
    ENV_NAME = "development"
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{INSTANCE_DIR / 'typhoid_dev.db'}"
    )
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False


class TestingConfig(BaseConfig):
    ENV_NAME = "testing"
    TESTING = True
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False
    PRELOAD_MODELS = False
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False
    LOGIN_MAX_ATTEMPTS = 3


class ProductionConfig(BaseConfig):
    ENV_NAME = "production"
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    LOG_TO_STDOUT = _bool("LOG_TO_STDOUT", True)

    @property
    def SQLALCHEMY_DATABASE_URI(self):  # noqa: N802
        url = os.environ.get("DATABASE_URL")
        if url:
            return _normalise_db_url(url)
        # Falls back to a file database so a misconfigured deploy still starts
        # and reports the problem, rather than crashing on import.
        return f"sqlite:///{INSTANCE_DIR / 'typhoid_prod.db'}"


CONFIGS = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(name: str | None = None):
    name = (name or os.environ.get("FLASK_ENV") or "development").lower()
    return CONFIGS.get(name, DevelopmentConfig)


def validate_production_config(app) -> list[str]:
    """Configuration problems that must not reach production silently."""
    problems = []
    if app.config.get("ENV_NAME") != "production":
        return problems
    if app.config["SECRET_KEY"] == "dev-only-key-change-in-production":
        problems.append("SECRET_KEY is still the development default")
    if not os.environ.get("DATABASE_URL"):
        problems.append("DATABASE_URL is not set; falling back to a local SQLite file")
    return problems
