"""Application logging: rotating files locally, stdout in production."""
from __future__ import annotations

import logging
import sys
import time
import uuid
from logging.handlers import RotatingFileHandler

from flask import g, has_request_context, request

FORMAT = "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s"


class RequestIdFilter(logging.Filter):
    """Attach a per-request id so a single request can be traced across lines."""

    def filter(self, record):
        record.request_id = getattr(g, "request_id", "-") if has_request_context() else "-"
        return True


def configure_logging(app) -> None:
    level = getattr(logging, str(app.config.get("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    formatter = logging.Formatter(FORMAT)
    request_filter = RequestIdFilter()

    for handler in list(app.logger.handlers):
        app.logger.removeHandler(handler)

    if app.config.get("LOG_TO_STDOUT"):
        handler = logging.StreamHandler(sys.stdout)
    else:
        log_dir = app.config["LOG_DIR"]
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / "application.log",
            maxBytes=app.config.get("LOG_MAX_BYTES", 2 * 1024 * 1024),
            backupCount=app.config.get("LOG_BACKUP_COUNT", 5),
            encoding="utf-8",
        )
    handler.setFormatter(formatter)
    handler.addFilter(request_filter)
    handler.setLevel(level)

    app.logger.addHandler(handler)
    app.logger.setLevel(level)
    app.logger.propagate = False

    # Route werkzeug's access log through the same handler.
    werkzeug = logging.getLogger("werkzeug")
    werkzeug.handlers = [handler]
    werkzeug.setLevel(logging.WARNING)

    app.logger.info(
        "logging configured env=%s level=%s destination=%s",
        app.config.get("ENV_NAME"),
        logging.getLevelName(level),
        "stdout" if app.config.get("LOG_TO_STDOUT") else "file",
    )


def register_request_logging(app) -> None:
    @app.before_request
    def _start_timer():
        g.request_id = uuid.uuid4().hex[:8]
        g.request_started = time.perf_counter()

    @app.after_request
    def _log_request(response):
        started = getattr(g, "request_started", None)
        if started is None or request.endpoint == "static":
            return response
        duration_ms = (time.perf_counter() - started) * 1000
        level = app.logger.warning if response.status_code >= 400 else app.logger.info
        level(
            "%s %s -> %s in %.1fms",
            request.method,
            request.full_path.rstrip("?"),
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = getattr(g, "request_id", "-")
        return response
