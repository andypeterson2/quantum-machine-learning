"""Model management endpoints — list, delete, import/export, load from disk.

Separated from the main dataset blueprint to honour the Single Responsibility
Principle: this module only manages model lifecycle (CRUD on registry entries
and disk checkpoints).
"""

from __future__ import annotations

import base64
import inspect
import io
import re
from pathlib import Path
from typing import Any

import mistune

from flask import Response, current_app, g, jsonify, request

from ..predictor import Predictor
from .errors import error_response


def _read_model_section(plugin, model_type: str) -> str | None:
    """Extract and render the MODELS.md section for *model_type*.

    Locates the ``MODELS.md`` file next to the plugin's source file,
    extracts the ``## <model_type> (...)`` section, and returns it as
    rendered HTML.  Returns ``None`` when no matching section is found.
    """
    plugin_dir = Path(inspect.getfile(type(plugin))).parent
    md_path = plugin_dir / "MODELS.md"
    if not md_path.exists():
        return None
    text = md_path.read_text()
    # Match from "## ModelType (" to the next "---" divider or EOF.
    pattern = rf"(^## {re.escape(model_type)} \(.+?)(?=\n---|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    if not match:
        return None
    return mistune.html(match.group(1).strip())


def register(bp) -> None:
    """Attach model management routes to *bp*."""

    # ── Predict ──────────────────────────────────────────────────────────────

    @bp.post("/predict")
    def predict() -> tuple[Response, int] | Response:
        """Run every registered model on a user-supplied input and return predictions."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]

        model_items = registry.items(plugin.name)
        if not model_items:
            return error_response("No models trained yet")

        data = request.get_json(force=True)

        if plugin.input_type == "image":
            from PIL import Image

            b64: str = data.get("image", "")
            img_bytes = base64.b64decode(b64)
            raw_input: Any = Image.open(io.BytesIO(img_bytes)).convert("L")
        else:
            raw_input = data.get("features", {})

        results: dict[str, dict] = {}
        for model_name, entry in model_items:
            predictor = Predictor(entry.model, plugin)
            probs = predictor.predict(raw_input)
            pred_idx = int(probs.argmax())
            results[model_name] = {
                "prediction": plugin.class_labels[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probs": probs.tolist(),
            }

        return jsonify({"results": results})

    # ── List models ──────────────────────────────────────────────────────────

    @bp.get("/models")
    def list_models() -> Response:
        """Return metadata for all registered models in this dataset."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]

        result: dict[str, dict] = {}
        for model_name, entry in registry.items(plugin.name):
            result[model_name] = {
                "model_type": entry.model_type,
                "epochs": entry.epochs,
                "batch_size": entry.batch_size,
                "lr": entry.lr,
                "num_params": entry.num_params,
                "training_history": entry.training_history or [],
                "eval_result": {
                    "accuracy": entry.eval_result.accuracy,
                    "avg_loss": entry.eval_result.avg_loss,
                    "per_class_accuracy": entry.eval_result.per_class_accuracy,
                    "num_params": entry.eval_result.num_params,
                }
                if entry.eval_result
                else None,
            }
        return jsonify(result)

    # ── Model info (rendered MODELS.md section) ────────────────────────────

    @bp.get("/model-info/<model_type>")
    def model_info(model_type: str) -> Response | tuple[Response, int]:
        """Return the rendered HTML description for *model_type*."""
        html = _read_model_section(g.plugin, model_type)
        if html is None:
            return error_response(f"No info for model type '{model_type}'", 404)
        return jsonify({"html": html})

    # ── Delete model ─────────────────────────────────────────────────────────

    @bp.delete("/models/<name>")
    def delete_model(name: str) -> Response:
        """Remove a model from the in-memory registry (session-only)."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]
        registry.remove(plugin.name, name)
        return jsonify({"ok": True})

    # ── Export model ─────────────────────────────────────────────────────────

    @bp.post("/models/<name>/export")
    def export_model(name: str) -> Response | tuple[Response, int]:
        """Save a registered model to a ``.pt`` checkpoint file on disk."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]
        persistence = current_app.extensions["persistence"]

        entry = registry.get(plugin.name, name)
        if entry is None:
            return error_response(f"Model '{name}' not found", 404)

        filename = persistence.save(name, entry)
        return jsonify({"ok": True, "filename": filename})

    # ── List disk models ─────────────────────────────────────────────────────

    @bp.get("/models/disk")
    def list_disk_models() -> Response:
        """List ``.pt`` checkpoint files available for this dataset."""
        plugin = g.plugin
        persistence = current_app.extensions["persistence"]
        all_files = persistence.list_files()
        dataset_files = [
            f for f in all_files if f.get("dataset", "mnist") == plugin.name
        ]
        return jsonify(dataset_files)

    # ── Load disk model ──────────────────────────────────────────────────────

    @bp.post("/models/disk/<filename>/load")
    def load_disk_model(filename: str) -> Response | tuple[Response, int]:
        """Load a ``.pt`` checkpoint from disk into the in-memory registry."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]
        persistence = current_app.extensions["persistence"]

        try:
            data = persistence.load(filename)
        except ValueError as exc:
            return error_response(str(exc))
        except FileNotFoundError as exc:
            return error_response(str(exc), 404)

        if data["dataset"] != plugin.name:
            return error_response(
                f"Checkpoint is for dataset '{data['dataset']}', "
                f"not '{plugin.name}'"
            )

        # Resolve name collision
        base: str = data["name"]
        name: str = base
        i = 2
        while registry.get(plugin.name, name) is not None:
            name = f"{base} ({i})"
            i += 1

        registry.add(
            plugin.name,
            name,
            data["model"],
            model_type=data["model_type"],
            epochs=data["epochs"],
            batch_size=data["batch_size"],
            lr=data["lr"],
        )
        new_entry = registry.get(plugin.name, name)
        if new_entry is not None:
            new_entry.training_history = data.get("training_history", [])
            new_entry.num_params = data.get("num_params")

        return jsonify(
            {
                "ok": True,
                "name": name,
                "model_type": data["model_type"],
                "epochs": data["epochs"],
                "batch_size": data["batch_size"],
                "lr": data["lr"],
                "num_params": data.get("num_params"),
            }
        )
