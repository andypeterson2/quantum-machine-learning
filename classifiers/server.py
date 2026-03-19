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

from pathlib import Path

from flask import Flask
from flask_cors import CORS

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
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = "classifiers-dev-secret"
    CORS(app)

    # Auto-discover dataset plugins (mnist, iris, etc.)
    discover_plugins()

    # Attach services — accessible inside any request context via current_app
    app.extensions["registry"] = ModelRegistry()
    app.extensions["persistence"] = ModelPersistence(
        models_dir or _DEFAULT_MODELS_DIR
    )

    from .routes import register_routes
    register_routes(app)

    return app
