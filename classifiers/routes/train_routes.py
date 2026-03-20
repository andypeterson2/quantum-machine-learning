"""Training endpoint — starts a training run and streams progress via SSE.

Separated from the main dataset blueprint to honour the Single Responsibility
Principle: this module only orchestrates model training.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from flask import Response, current_app, g, request

from ..trainer import Trainer
from .errors import error_response
from ..training_config import TrainingConfig
from .sse import sse_response


def register(bp) -> None:
    """Attach training routes to *bp*."""

    @bp.post("/train")
    def train() -> Response | tuple[dict, int]:
        """Train a new model and stream progress as Server-Sent Events.

        Reads model architecture, hyper-parameters, and display name from the
        JSON request body.  The training :class:`~torch.utils.data.DataLoader`
        is obtained from the active plugin (Dependency Inversion).

        **Request body** (JSON)::

            {
                "model_type": "CNN",
                "epochs": 3,
                "batch_size": 64,
                "lr": 0.001,
                "name": "My Model"      // optional
            }

        **SSE event shapes**:

        * ``{"type": "status", "msg": "..."}`` — progress update.
        * ``{"type": "done", "name": "...", ...}`` — training success with metadata.
        * ``{"type": "error", "msg": "..."}`` — unrecoverable error.
        """
        plugin = g.plugin
        registry = current_app.extensions["registry"]

        body = request.get_json(force=True)
        model_type_name: str = body.get("model_type", "")
        epochs: int = int(body.get("epochs", 3))
        batch_size: int = int(body.get("batch_size", 64))
        lr: float = float(body.get("lr", 1e-3))
        name: str = body.get("name") or registry.next_name(plugin.name)

        model_types = plugin.get_model_types()
        if model_type_name not in model_types:
            return error_response(f"Unknown model type: {model_type_name}")

        model_cls = model_types[model_type_name]
        train_loader = plugin.get_train_loader(batch_size)
        q: queue.Queue[dict | None] = queue.Queue()

        # Advanced training options
        patience = body.get("patience")
        val_gap = int(body.get("val_gap", 50))
        teacher_name: str | None = body.get("teacher")
        distill_weight = float(body.get("distill_weight", 0.5))

        config: TrainingConfig | None = None
        val_loader = None

        if patience is not None or teacher_name:
            teacher_model = None
            teacher_process = None
            if teacher_name:
                teacher_entry = registry.get(plugin.name, teacher_name)
                if teacher_entry is not None:
                    teacher_model = teacher_entry.model
            config = TrainingConfig(
                patience=int(patience) if patience is not None else None,
                val_gap=val_gap,
                teacher_model=teacher_model,
                distill_weight=distill_weight,
                teacher_process=teacher_process,
            )
            val_loader = plugin.get_val_loader(batch_size)

        def run() -> None:
            """Worker executed in a daemon thread."""
            try:

                def on_status(msg: str | dict) -> None:
                    if isinstance(msg, dict):
                        q.put(msg)
                    else:
                        q.put({"type": "status", "msg": msg})

                trainer = Trainer(
                    model_cls=model_cls,
                    train_loader=train_loader,
                    dataset=plugin.name,
                    epochs=epochs,
                    lr=lr,
                    config=config,
                    val_loader=val_loader,
                )
                result = trainer.train(on_status=on_status)
                registry.add(
                    plugin.name,
                    name,
                    result.model,
                    model_type=result.model_type,
                    epochs=result.epochs,
                    batch_size=result.batch_size,
                    lr=result.lr,
                )
                # Store extended info on the entry
                new_entry = registry.get(plugin.name, name)
                if new_entry is not None:
                    new_entry.training_history = result.history
                    new_entry.num_params = result.num_params

                done_event: dict[str, Any] = {
                    "type": "done",
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
                    done_event["best_val_accuracy"] = result.best_val_accuracy
                if result.history:
                    done_event["history"] = result.history
                q.put(done_event)
            except Exception as exc:
                q.put({"type": "error", "msg": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return sse_response(q)
