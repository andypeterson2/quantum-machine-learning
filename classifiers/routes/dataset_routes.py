"""Dataset-scoped routes — blueprint shell with shared request hooks.

All endpoints live under ``/d/<dataset>/`` and share a
:meth:`~flask.Blueprint.url_value_preprocessor` hook that resolves the
dataset slug to a :class:`~classifiers.dataset_plugin.DatasetPlugin` stored
on :data:`flask.g`.

Endpoint groups are registered by sub-modules to honour the Single
Responsibility Principle:

* :mod:`~classifiers.routes.train_routes` — training
* :mod:`~classifiers.routes.eval_routes`  — evaluation, ensemble, ablation
* :mod:`~classifiers.routes.model_routes` — CRUD, predict, import/export
"""

from __future__ import annotations

from typing import Any

from flask import (
    Blueprint,
    Response,
    g,
)

from ..plugin_registry import get_plugin
from . import train_routes, eval_routes, model_routes
from .errors import error_response

bp = Blueprint("dataset", __name__, url_prefix="/d/<dataset>")


# ── Hook: resolve plugin once per request ────────────────────────────────────


@bp.url_value_preprocessor
def pull_dataset(endpoint: str | None, values: dict[str, Any] | None) -> None:
    """Extract ``dataset`` from the URL and store the plugin on ``g``.

    Using a :meth:`~flask.Blueprint.url_value_preprocessor` removes
    ``dataset`` from *values* so individual view functions don't need to
    accept it as an explicit keyword argument.
    """
    if values is None:
        return
    dataset_name = values.pop("dataset")
    plugin = get_plugin(dataset_name)
    if plugin is None:
        g.plugin = None
        g.dataset_error = f"Unknown dataset: {dataset_name!r}"
    else:
        g.plugin = plugin
        g.dataset_error = None


@bp.before_request
def reject_unknown_dataset() -> tuple[Response, int] | None:
    """Return 404 early if the dataset slug doesn't match any plugin."""
    if g.plugin is None:
        return error_response(g.dataset_error, 404)
    return None


# ── Register endpoint sub-modules ───────────────────────────────────────────

train_routes.register(bp)
eval_routes.register(bp)
model_routes.register(bp)
