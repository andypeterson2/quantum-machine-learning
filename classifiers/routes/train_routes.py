"""Training endpoints — async (SSE) and synchronous.

``POST /train`` streams progress via Server-Sent Events; ``POST /train/sync``
runs the same training to completion and returns the final result directly in
the HTTP response (the cross-repo "curl-able" rule — every operation reachable
with no SSE client). Both forms share the setup, registration, and
result-shaping helpers below.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from flask import Response, current_app, g, jsonify, request

from ..trainer import Trainer
from ..training_config import TrainingConfig
from .errors import error_response
from .sse import sse_response


class _TrainingInputError(Exception):
    """Bad training-request input; carries an HTTP status for the error envelope."""

    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.msg = msg
        self.status = status


def _setup_trainer(plugin, registry, body) -> tuple[Trainer, str]:
    """Validate the request body and build ``(Trainer, model_name)``.

    Raises :class:`_TrainingInputError` on an unknown model type or a missing
    teacher model — callers map that to the JSON error envelope.
    """
    model_type_name: str = body.get("model_type", "")
    epochs: int = int(body.get("epochs", 3))
    batch_size: int = int(body.get("batch_size", 64))
    lr: float = float(body.get("lr", 1e-3))
    name: str = body.get("name") or registry.next_name(plugin.name)

    model_types = plugin.get_model_types()
    if model_type_name not in model_types:
        raise _TrainingInputError(f"Unknown model type: {model_type_name}")

    model_cls = model_types[model_type_name]
    train_loader = plugin.get_train_loader(batch_size)

    # Advanced training options
    patience = body.get("patience")
    val_gap = int(body.get("val_gap", 50))
    teacher_name: str | None = body.get("teacher")
    distill_weight = float(body.get("distill_weight", 0.5))

    config: TrainingConfig | None = None
    val_loader = None
    if patience is not None or teacher_name:
        teacher_model = None
        if teacher_name:
            teacher_entry = registry.get(plugin.name, teacher_name)
            if teacher_entry is None:
                raise _TrainingInputError(f"Teacher model '{teacher_name}' not found", 404)
            teacher_model = teacher_entry.model
        config = TrainingConfig(
            patience=int(patience) if patience is not None else None,
            val_gap=val_gap,
            teacher_model=teacher_model,
            distill_weight=distill_weight,
            teacher_process=None,
        )
        val_loader = plugin.get_val_loader(batch_size)

    trainer = Trainer(
        model_cls=model_cls,
        train_loader=train_loader,
        dataset=plugin.name,
        epochs=epochs,
        lr=lr,
        config=config,
        val_loader=val_loader,
    )
    return trainer, name


def _register_trained(registry, plugin, name: str, result) -> None:
    """Persist a finished training result into the model registry."""
    registry.add(
        plugin.name,
        name,
        result.model,
        model_type=result.model_type,
        epochs=result.epochs,
        batch_size=result.batch_size,
        lr=result.lr,
    )
    registry.update_training_meta(
        plugin.name,
        name,
        training_history=result.history,
        num_params=result.num_params,
    )


def _result_payload(name: str, result) -> dict[str, Any]:
    """Shape a finished training result into the response / ``done``-event body."""
    payload: dict[str, Any] = {
        "name": name,
        "model_type": result.model_type,
        "epochs": result.epochs,
        "epochs_completed": result.epochs_completed,
        "batch_size": result.batch_size,
        "lr": result.lr,
        "num_params": result.num_params,
        "stopped_early": result.stopped_early,
    }
    if result.best_val_accuracy is not None:
        payload["best_val_accuracy"] = result.best_val_accuracy
    if result.history:
        payload["history"] = result.history
    return payload


def register(bp) -> None:
    """Attach training routes to *bp*."""

    @bp.post("/train")
    def train() -> Response | tuple[Response, int]:
        """Train a new model and stream progress as Server-Sent Events.

        Reads model architecture, hyper-parameters, and display name from the
        JSON request body. SSE event shapes:

        * ``{"type": "status", "msg": "..."}`` — progress update.
        * ``{"type": "done", "name": "...", ...}`` — success with metadata.
        * ``{"type": "error", "msg": "..."}`` — unrecoverable error.
        """
        plugin = g.plugin
        registry = current_app.extensions["registry"]
        body = request.get_json(force=True)
        try:
            trainer, name = _setup_trainer(plugin, registry, body)
        except _TrainingInputError as exc:
            return error_response(exc.msg, exc.status)

        q: queue.Queue[dict | None] = queue.Queue()

        def run() -> None:
            """Worker executed in a daemon thread."""
            try:

                def on_status(msg: str | dict) -> None:
                    q.put(msg if isinstance(msg, dict) else {"type": "status", "msg": msg})

                result = trainer.train(on_status=on_status)
                _register_trained(registry, plugin, name, result)
                q.put({"type": "done", **_result_payload(name, result)})
            except Exception as exc:
                q.put({"type": "error", "msg": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return sse_response(q)

    @bp.post("/train/sync")
    def train_sync() -> Response | tuple[Response, int]:
        """Train to completion synchronously; returns the final result in the body.

        Same inputs as ``/train``; the response is the training result dict (the
        ``done`` event without its ``type`` field). Intended for scripts/CI.
        """
        plugin = g.plugin
        registry = current_app.extensions["registry"]
        body = request.get_json(force=True)
        try:
            trainer, name = _setup_trainer(plugin, registry, body)
        except _TrainingInputError as exc:
            return error_response(exc.msg, exc.status)
        try:
            result = trainer.train(on_status=lambda *_: None)
            _register_trained(registry, plugin, name, result)
        except Exception as exc:
            return error_response(str(exc), 500)
        return jsonify(_result_payload(name, result))
