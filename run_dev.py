"""Start the application locally in development mode.

Prints its own start-up banner. The application routes werkzeug's logger
through the file handler at WARNING level so that access lines are not
duplicated on the console, which also suppresses werkzeug's own
" * Running on http://…" line — without this banner the server looks hung
while it is in fact already listening.
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("FLASK_ENV", "development")

_started = time.perf_counter()

from webapp import create_app  # noqa: E402

app = create_app("development")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 5000))


def _banner() -> None:
    display = "localhost" if HOST in ("127.0.0.1", "0.0.0.0") else HOST
    elapsed = time.perf_counter() - _started
    print("", flush=True)
    print(f"  {app.config['APP_NAME']} v{app.config['APP_VERSION']}", flush=True)
    print(f"  Ready in {elapsed:.1f}s  ·  environment: development", flush=True)
    print(f"  Open  ->  http://{display}:{PORT}", flush=True)
    print(f"  Logs  ->  {ROOT / 'logs' / 'application.log'}", flush=True)
    print("  Press CTRL+C to stop.", flush=True)
    print("", flush=True)


if __name__ == "__main__":
    # Under the debug reloader this module is executed twice: once by the
    # supervisor and once by the worker that actually binds the socket.
    # Only the worker should announce the address.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _banner()
    app.run(host=HOST, port=PORT, debug=True)
