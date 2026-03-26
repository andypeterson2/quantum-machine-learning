"""Top-level routes — root redirect and dataset listing API.

The root URL redirects to the first available dataset, and ``/api/datasets``
returns the list of registered plugins for the frontend dataset selector.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify, redirect, render_template, send_from_directory, url_for

from ..plugin_registry import get_plugin, list_plugins

bp = Blueprint("main", __name__)

_UI_KIT_DIR = Path(__file__).resolve().parents[2] / "ui-kit"


@bp.get("/")
def index() -> Response:
    """Redirect root URL to the first available dataset."""
    plugins = list_plugins()
    if plugins:
        first = next(iter(plugins))
        return redirect(url_for("main.dataset_index", dataset=first))
    return "No datasets registered", 404


@bp.get("/d/<dataset>/")
def dataset_index(dataset: str) -> Response | tuple[str, int]:
    """Serve the SPA entry point for a specific dataset."""
    plugin = get_plugin(dataset)
    if plugin is None:
        return jsonify({"error": f"Unknown dataset: {dataset!r}"}), 404
    ui_config = plugin.get_ui_config()
    model_types = list(plugin.get_model_types().keys())
    return render_template(
        "index.html",
        ui_config=ui_config,
        model_types=model_types,
        ui_kit=_UI_KIT_DIR.is_dir(),
    )


@bp.get("/ui-kit/<path:filename>")
def ui_kit_static(filename: str) -> Response:
    """Serve files from the ui-kit directory."""
    return send_from_directory(str(_UI_KIT_DIR), filename)


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
