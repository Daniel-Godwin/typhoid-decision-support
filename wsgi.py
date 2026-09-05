"""WSGI entry point for gunicorn and for `flask` CLI commands.

    gunicorn wsgi:app
    FLASK_APP=wsgi.py flask bootstrap
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from webapp import create_app  # noqa: E402

app = create_app(os.environ.get("FLASK_ENV", "production"))

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
