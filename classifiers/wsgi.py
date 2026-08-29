"""Production WSGI entry point.

The dev entry (``python -m classifiers``) runs Flask's Werkzeug server; the
container runs gunicorn against this module instead (``classifiers.wsgi:app``).

ONE gunicorn worker only — the model registry is per-process, in-memory state
created inside :func:`~classifiers.server.create_app` (see the Dockerfile CMD and
that module's docstring). Concurrency comes from threads within the single worker.
"""

from .server import create_app

app = create_app()
