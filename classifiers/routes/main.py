"""Top-level routes — root redirect and dataset listing API.

The root URL redirects to the first available dataset, and ``/api/datasets``
returns the list of registered plugins for the frontend dataset selector.
"""

from __future__ import annotations

from flask import Blueprint, Response, jsonify

from ..plugin_registry import get_plugin, list_plugins

bp = Blueprint("main", __name__)


@bp.get("/api/datasets")
def list_datasets() -> Response:
    """Return the list of registered dataset plugins.

    **Response body** (JSON)::

        [
            {"name": "mnist", "display_name": "MNIST Handwritten Digits", "input_type": "image"},
            {"name": "iris",  "display_name": "Iris Flower Classification", "input_type": "tabular"}
        ]
    """
    plugins = list_plugins()
    result = [
        {
            "name": p.name,
            "display_name": p.display_name,
            "input_type": p.input_type,
        }
        for p in plugins.values()
    ]
    return jsonify(result)


@bp.get("/api/datasets/<name>/config")
def dataset_config(name: str) -> Response | tuple[Response, int]:
    """Return the full UI configuration for a specific dataset.

    **Response body** (JSON)::

        {
            "ui_config": { ... },
            "model_types": ["Linear", "Conv", "QKernel"]
        }
    """
    plugin = get_plugin(name)
    if plugin is None:
        return jsonify({"error": f"Unknown dataset: {name!r}"}), 404
    return jsonify({
        "ui_config": plugin.get_ui_config(),
        "model_types": list(plugin.get_model_types().keys()),
    })
