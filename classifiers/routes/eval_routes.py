"""Evaluation endpoints — single model, ensemble, and ablation study.

``POST /evaluate`` streams per-model progress via SSE; ``POST /evaluate/sync``
runs the same evaluation inline and returns ``{"results": {...}}`` directly (the
cross-repo "curl-able" rule). Both share :func:`_evaluate_all`.

Separated from the main dataset blueprint to honour the Single Responsibility
Principle: this module only orchestrates model evaluation.
"""

from __future__ import annotations

import queue
import threading

from flask import Response, current_app, g, jsonify, request

from classifiers.evaluator import Evaluator

from .errors import error_response
from .sse import sse_response


def _evaluate_all(plugin, registry, on_status=None) -> dict:
    """Evaluate every registered model for *plugin*; return ``{name: metrics}``.

    Pure compute (plus registry updates) — no SSE — so it is shared by the async
    ``/evaluate`` stream and the synchronous ``/evaluate/sync`` route. *on_status*
    is an optional ``str -> None`` progress callback.
    """
    evaluator = Evaluator()
    test_loader = plugin.get_test_loader(1000)
    results: dict[str, dict] = {}
    for model_name, entry in registry.items(plugin.name):
        if on_status:
            on_status(f"Evaluating '{model_name}'...")
        ev = evaluator.evaluate(
            entry.model,
            test_loader,
            plugin.num_classes,
            plugin.class_labels,
            on_status=on_status or (lambda *_: None),
        )
        registry.update_eval_result(plugin.name, model_name, ev)
        results[model_name] = {
            "accuracy": ev.accuracy,
            "avg_loss": ev.avg_loss,
            "per_class_accuracy": ev.per_class_accuracy,
            "num_params": ev.num_params,
        }
    return results


# One registrar per blueprint: it enumerates every evaluation endpoint in one
# place, so the length is the route table, not tangled logic.
def register(bp) -> None:  # noqa: C901, PLR0915
    """Attach evaluation routes to *bp*."""

    # ── Evaluate ─────────────────────────────────────────────────────────────

    @bp.post("/evaluate")
    def evaluate() -> Response | tuple[Response, int]:
        """Evaluate every registered model for this dataset via SSE."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]
        slots = current_app.extensions["job_slots"]
        if not slots.acquire(blocking=False):
            return error_response("Server busy — try again shortly", 409, code="busy")

        if not registry.items(plugin.name):
            slots.release()
            return error_response("No models to evaluate")

        q: queue.Queue[dict | None] = queue.Queue()

        def run() -> None:
            try:
                results = _evaluate_all(
                    plugin, registry, on_status=lambda msg: q.put({"type": "status", "msg": msg})
                )
                q.put({"type": "done", "results": results})
            except Exception as exc:
                q.put({"type": "error", "msg": str(exc)})
            finally:
                slots.release()
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return sse_response(q)

    @bp.post("/evaluate/sync")
    def evaluate_sync() -> Response | tuple[Response, int]:
        """Evaluate every registered model synchronously; returns {"results": {...}}."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]
        slots = current_app.extensions["job_slots"]
        if not slots.acquire(blocking=False):
            return error_response("Server busy — try again shortly", 409, code="busy")
        try:
            if not registry.items(plugin.name):
                return error_response("No models to evaluate")
            try:
                results = _evaluate_all(plugin, registry)
            except Exception as exc:
                return error_response(str(exc), 500)
            return jsonify({"results": results})
        finally:
            slots.release()

    # ── Ensemble ─────────────────────────────────────────────────────────────

    @bp.post("/ensemble")
    def ensemble() -> Response | tuple[Response, int]:
        """Majority-vote ensemble evaluation across selected models."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]

        slots = current_app.extensions["job_slots"]
        if not slots.acquire(blocking=False):
            return error_response("Server busy — try again shortly", 409, code="busy")
        try:
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
        finally:
            slots.release()

    # ── Ablation ─────────────────────────────────────────────────────────────

    @bp.post("/ablation")
    def ablation() -> Response | tuple[Response, int]:
        """Ablation study: zero out each layer and measure accuracy drop."""
        plugin = g.plugin
        registry = current_app.extensions["registry"]

        body = request.get_json(force=True)
        model_name: str = body.get("model_name", "")
        entry = registry.get(plugin.name, model_name)
        if entry is None:
            return error_response(f"Model '{model_name}' not found", 404)

        slots = current_app.extensions["job_slots"]
        if not slots.acquire(blocking=False):
            return error_response("Server busy — try again shortly", 409, code="busy")

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
                slots.release()
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return sse_response(q)
