"""Flask application factory.

All shared application state (model registry, persistence layer) is attached
to ``app.extensions`` inside :func:`create_app` rather than at module level.
This makes each factory call fully self-contained, which is required for:

* Clean unit testing (each test gets its own isolated app instance).
* Correct behaviour under Werkzeug's reloader (the child process re-imports
  this module and calls :func:`create_app` again; with module-level state it
  would create a second, orphaned registry).

Dependency Inversion
--------------------
Route handlers access shared state through ``current_app.extensions[...]``
rather than importing a concrete object from this module.  The routes depend
on the *interface* (dict key + expected type) rather than the *implementation*.

Plugin Discovery
----------------
:func:`~classifiers.plugin_registry.discover_plugins` is called during
application startup, walking ``classifiers/datasets/`` to auto-register
every dataset plugin.  Adding a new dataset is therefore zero-config.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from .connections import ConnectionTracker
from .model_registry import ModelRegistry
from .persistence import ModelPersistence
from .plugin_registry import discover_plugins

#: Default checkpoint directory: ``<project_root>/models/``
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def create_app(models_dir: Path | None = None) -> Flask:
    """Create and configure the Flask application.

    Shared services are stored in ``app.extensions`` under the keys
    ``"registry"`` and ``"persistence"``.  Dataset plugins are discovered
    automatically from ``classifiers/datasets/``.

    Args:
        models_dir: Override the directory used for ``.pt`` checkpoints.
            Defaults to ``<project_root>/models``.

    Returns:
        A fully configured :class:`~flask.Flask` application instance ready
        to serve requests.
    """
    # API-only service: the frontend is owned by the portal, so no static/template
    # serving (static_folder=None disables the default /static/<path> route too).
    app = Flask(__name__, static_folder=None)
    app.config["SECRET_KEY"] = os.environ.get("CLASSIFIERS_SECRET_KEY") or os.urandom(32).hex()
    # NOTE the anchored regex: flask-cors treats any entry containing "*" as an
    # UNANCHORED-at-the-end regex, so the old "http://localhost:*" allowed the
    # registrable origin http://localhostevil.com. The ^...$ form does not.
    CORS(
        app,
        origins=os.environ.get(
            "CLASSIFIERS_CORS_ORIGINS",
            r"^https?://localhost(:\d+)?$,https://andypeterson.dev",
        ).split(","),
    )

    # Cap request-body size. Predict images and JSON bodies are a few KB; anything
    # larger is rejected with 413 before parsing (a DoS guard). Overridable via env.
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.environ.get("CLASSIFIERS_MAX_CONTENT_LENGTH") or 2 * 1024 * 1024  # 2 MB
    )

    # Auto-discover dataset plugins (mnist, iris, etc.)
    discover_plugins()

    # Attach services — accessible inside any request context via current_app
    app.extensions["registry"] = ModelRegistry()
    app.extensions["persistence"] = ModelPersistence(
        models_dir or _DEFAULT_MODELS_DIR
    )
    app.extensions["start_time"] = time.monotonic()

    tracker = ConnectionTracker()
    app.extensions["connections"] = tracker

    # Background thread sweeps stale clients every 30 s.
    def _sweep_loop() -> None:
        while True:
            time.sleep(30)
            tracker.sweep(timeout=90)

    sweep_thread = threading.Thread(target=_sweep_loop, daemon=True)
    sweep_thread.start()

    from .routes import register_routes
    register_routes(app)

    # Uniform JSON error envelope for framework-raised errors (404/405/500, ...).
    from .routes.errors import register_error_handlers

    register_error_handlers(app)

    # Origin guard (the gate's teeth): reject anything that didn't arrive through the
    # gateway, which injects X-Origin-Secret. /health stays public so the host's health
    # check + scale-to-zero wake work. Inert until ORIGIN_SECRET is set, so it's safe to
    # land before the gateway is wired. See andypeterson-gateway/PHASE3-DEPLOY.md §3.
    @app.before_request
    def _origin_guard():
        want = os.environ.get("ORIGIN_SECRET")
        if want and request.path != "/health":
            got = request.headers.get("X-Origin-Secret") or ""
            # compare_digest: a plain != on a secret leaks timing.
            if not hmac.compare_digest(got, want):
                return jsonify({"error": {"code": "forbidden", "message": "origin"}}), 403
        return None

    return app
