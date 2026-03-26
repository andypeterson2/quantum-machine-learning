"""Entry point: ``python -m classifiers``

Port selection
--------------
Flask's debug reloader spawns a child process that re-executes this module.
The chosen port is written to the ``CLASSIFIERS_PORT`` environment variable
*before* :func:`~classifiers.server.create_app` is called.  Child processes inherit
the parent's environment, so the reloader child reads the same port value
rather than probing for a new one.
"""

import logging
import os

from .server import create_app

# Configure structured logging for the entire package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("classifiers")


def _get_ssl_context():
    """Return (cert, key) paths if dev certs exist, else None."""
    from pathlib import Path
    for d in [
        Path(os.environ.get("DEV_CERT_DIR", "")),
        Path(__file__).resolve().parents[2] / ".certs",
    ]:
        cert, key = d / "cert.pem", d / "key.pem"
        if cert.is_file() and key.is_file():
            return (str(cert), str(key))
    return None


# Persist the chosen port so Werkzeug's reloader child inherits it and binds
# to the same port instead of probing for a new one.
port = int(os.environ.get("CLASSIFIERS_PORT") or 5001)
os.environ["CLASSIFIERS_PORT"] = str(port)

app = create_app()

# Only print from the outer watcher process — WERKZEUG_RUN_MAIN is set in the
# reloader child, which would otherwise produce duplicate output.
ssl_ctx = _get_ssl_context()
scheme = "https" if ssl_ctx else "http"

if not os.environ.get("WERKZEUG_RUN_MAIN"):
    logger.info("Running on %s://localhost:%d", scheme, port)

host = os.environ.get("CLASSIFIERS_HOST", "127.0.0.1")
app.run(debug=True, host=host, port=port, ssl_context=ssl_ctx)
