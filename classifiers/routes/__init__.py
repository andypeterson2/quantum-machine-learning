"""Flask blueprint registration for the classifiers package.

Two blueprints are registered:

* **main** — root redirect (``/``) and ``/api/datasets`` listing.
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
    from .main import bp as main_bp
    from .dataset_routes import bp as dataset_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(dataset_bp)
