"""Flask blueprint registration for the classifiers package.

Three blueprints are registered (there is no ``/`` route — API-only service):

* **main** — ``/health``, ``/api``, and the ``/api/datasets`` listing.
* **connection** — the SSE heartbeat (``/connect``, ``/pong``, ``/disconnect``).
* **dataset** — all dataset-scoped routes under ``/d/<dataset>/``.
"""

from __future__ import annotations

from flask import Flask


def register_routes(app: Flask) -> None:
    """Discover and register all route blueprints with *app*.

    Imports are deferred to this function to avoid circular-import issues
    at module load time.

    Args:
        app: The Flask application instance to register blueprints on.
    """
    from .connection_routes import bp as connection_bp
    from .dataset_routes import bp as dataset_bp
    from .main import bp as main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(connection_bp)
    app.register_blueprint(dataset_bp)
