"""Top-level API routes — health, discovery, and dataset listing.

This service is API-only; the frontend is owned by the portal. ``/api/datasets``
returns the list of registered plugins for the dataset selector, and
``/api/datasets/<name>/config`` returns a dataset's UI configuration.
"""

from __future__ import annotations

import time

from flask import Blueprint, Response, current_app, jsonify

from classifiers.plugin_registry import get_plugin, list_plugins

from .errors import error_response

bp = Blueprint("main", __name__)

SERVICE = "classifiers"


def _service_version() -> str:
    """Resolve the installed package version, falling back to the pyproject value."""
    try:
        from importlib.metadata import version

        return version("quantum-machine-learning")
    except Exception:
        return "0.2.0"


# SSE channels are not in the HTTP url-map; each has a synchronous REST
# equivalent (.../train/sync, .../evaluate/sync) — streaming is additive.
_STREAMING = [
    {
        "protocol": "sse",
        "channel": "train",
        "description": "Live training metrics stream (SSE twin of the sync train route).",
    },
    {
        "protocol": "sse",
        "channel": "evaluate",
        "description": "Live evaluation stream; live equivalent of the synchronous evaluate route.",
    },
]


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


@bp.get("/health")
def health() -> Response:
    """Return server health status, uptime, and connected-client count."""
    start = current_app.extensions.get("start_time", 0.0)
    tracker = current_app.extensions.get("connections")
    uptime_s = round(time.monotonic() - start, 1)
    return jsonify({
        "status": "ok",
        "service": SERVICE,
        "version": _service_version(),
        "uptime_s": uptime_s,
        "uptime": uptime_s,  # legacy alias (pre-contract clients)
        "clients": tracker.count if tracker else 0,
        "timestamp": time.time(),
    })


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
        return error_response(f"Unknown dataset: {name!r}", 404, code="unknown_dataset")
    return jsonify({
        "ui_config": plugin.get_ui_config(),
        "model_types": list(plugin.get_model_types().keys()),
    })


@bp.get("/api")
def api_index() -> Response:
    """Discovery index: every HTTP endpoint plus streaming channels."""
    seen: set[tuple[str, str]] = set()
    endpoints = []
    for rule in current_app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        path = str(rule)
        view = current_app.view_functions.get(rule.endpoint)
        summary = ((getattr(view, "__doc__", "") or "").strip().splitlines() or [""])[0].strip()
        for method in (rule.methods or set()) - {"HEAD", "OPTIONS"}:
            if (method, path) in seen:
                continue
            seen.add((method, path))
            endpoints.append({"method": method, "path": path, "summary": summary})
    endpoints.sort(key=lambda e: (e["path"], e["method"]))
    return jsonify(
        {
            "service": SERVICE,
            "version": _service_version(),
            "endpoints": endpoints,
            "streaming": _STREAMING,
        }
    )
