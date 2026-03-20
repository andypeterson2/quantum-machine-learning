"""Evaluation endpoints — single model, ensemble, and ablation study.

Separated from the main dataset blueprint to honour the Single Responsibility
Principle: this module only orchestrates model evaluation.
"""

from __future__ import annotations

import queue
import threading

from flask import Response, current_app, g, jsonify, request

from ..evaluator import Evaluator
from .errors import error_response
from .sse import sse_response


def register(bp) -> None:
    """Attach evaluation routes to *bp*."""

    # ── Evaluate ─────────────────────────────────────────────────────────────

    @bp.post("/evaluate")
    def evaluate() -> Response | tuple[dict, int]:
        """Evaluate every registered model for this dataset via SSE."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]

        model_items = registry.items(plugin.name)
        if not model_items:
            return error_response("No models to evaluate")

        q: queue.Queue[dict | None] = queue.Queue()

        def run() -> None:
            try:
                evaluator = Evaluator()
                test_loader = plugin.get_test_loader(1000)
                all_results: dict[str, dict] = {}

                for model_name, entry in model_items:
                    q.put({"type": "status", "msg": f"Evaluating '{model_name}'..."})
                    ev = evaluator.evaluate(
                        entry.model,
                        test_loader,
                        plugin.num_classes,
                        plugin.class_labels,
                        on_status=lambda msg: q.put({"type": "status", "msg": msg}),
                    )
                    registry.update_eval_result(plugin.name, model_name, ev)
                    all_results[model_name] = {
                        "accuracy": ev.accuracy,
                        "avg_loss": ev.avg_loss,
                        "per_class_accuracy": ev.per_class_accuracy,
                        "num_params": ev.num_params,
                    }

                q.put({"type": "done", "results": all_results})
            except Exception as exc:
                q.put({"type": "error", "msg": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return sse_response(q)

    # ── Ensemble ─────────────────────────────────────────────────────────────

    @bp.post("/ensemble")
    def ensemble() -> Response | tuple[dict, int]:
        """Majority-vote ensemble evaluation across selected models."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]

        body = request.get_json(force=True)
        model_names: list[str] = body.get("model_names", [])
        if len(model_names) < 2:
            return error_response("Need at least 2 models for ensemble")

        models = []
        for mn in model_names:
            entry = registry.get(plugin.name, mn)
            if entry is None:
                return error_response(f"Model '{mn}' not found", 404)
            models.append(entry.model)

        evaluator = Evaluator()
        test_loader = plugin.get_test_loader(1000)
        result = evaluator.ensemble_evaluate(
            models, test_loader, plugin.num_classes, plugin.class_labels
        )
        return jsonify({
            "accuracy": result.accuracy,
            "avg_loss": result.avg_loss,
            "per_class_accuracy": result.per_class_accuracy,
        })

    # ── Ablation ─────────────────────────────────────────────────────────────

    @bp.post("/ablation")
    def ablation() -> Response | tuple[dict, int]:
        """Ablation study: zero out each layer and measure accuracy drop."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]

        body = request.get_json(force=True)
        model_name: str = body.get("model_name", "")
        entry = registry.get(plugin.name, model_name)
        if entry is None:
            return error_response(f"Model '{model_name}' not found", 404)

        q: queue.Queue[dict | None] = queue.Queue()

        def run() -> None:
            try:
                evaluator = Evaluator()
                test_loader = plugin.get_test_loader(1000)

                def on_status(msg: str | dict) -> None:
                    if isinstance(msg, dict):
                        q.put(msg)
                    else:
                        q.put({"type": "status", "msg": msg})

                results = evaluator.ablation_evaluate(
                    entry.model,
                    test_loader,
                    plugin.num_classes,
                    plugin.class_labels,
                    on_status=on_status,
                )
                summary = {
                    layer: {
                        "accuracy": r.accuracy,
                        "avg_loss": r.avg_loss,
                    }
                    for layer, r in results.items()
                }
                q.put({"type": "done", "results": summary})
            except Exception as exc:
                q.put({"type": "error", "msg": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return sse_response(q)
