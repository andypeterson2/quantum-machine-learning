"""Generic training logic, decoupled from any specific dataset.

The :class:`Trainer` accepts a :class:`~torch.utils.data.DataLoader` provided
by the dataset plugin, so it has zero knowledge of data sources, transforms,
or normalisation constants.  Progress is reported through an optional callback
so callers can forward updates to any sink (SSE queue, stdout, etc.).

Advanced features (early stopping, validation, distillation, regularisation,
training-curve history) are enabled through an optional
:class:`~classifiers.training_config.TrainingConfig`.  When the config is
``None`` the trainer behaves identically to the original fixed-epoch Adam loop.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_model import BaseModel
from .training_config import HistoryEntry, TrainingConfig
from .types import StatusCallback

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Value object returned by :meth:`Trainer.train`.

    Attributes:
        model:             The trained model instance (best by val accuracy
                           when validation is active, otherwise final).
        model_type:        Architecture display name (e.g. ``"CNN"``).
        dataset:           Dataset slug (e.g. ``"mnist"``).
        epochs:            Number of epochs requested.
        epochs_completed:  Actual epochs completed (may be fewer with early stopping).
        batch_size:        Mini-batch size used during training.
        lr:                Learning rate used by the Adam optimiser.
        history:           Training curve data points (empty list if no validation).
        best_val_accuracy: Best validation accuracy achieved (``None`` if N/A).
        stopped_early:     Whether early stopping triggered.
        num_params:        Trainable parameter count.
    """

    model: BaseModel
    model_type: str
    dataset: str
    epochs: int
    batch_size: int
    lr: float
    epochs_completed: int = 0
    history: list[dict] = field(default_factory=list)
    best_val_accuracy: float | None = None
    stopped_early: bool = False
    num_params: int = 0


class Trainer:
    """Trains a classifier using Adam with optional advanced features.

    Args:
        model_cls:    :class:`BaseModel` subclass to instantiate and train.
        train_loader: Pre-built data loader (from the plugin).
        dataset:      Dataset slug stored in the result for downstream use.
        epochs:       Number of full passes over the training set.
        lr:           Initial learning rate for the Adam optimiser.
        config:       Optional :class:`TrainingConfig` enabling early stopping,
                      validation, distillation, and regularisation.
        val_loader:   Optional validation data loader for intermediate eval.
    """

    def __init__(
        self,
        model_cls: type[BaseModel],
        train_loader: DataLoader,
        dataset: str,
        epochs: int = 3,
        lr: float = 1e-3,
        config: TrainingConfig | None = None,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.model_cls = model_cls
        self.train_loader = train_loader
        self.dataset = dataset
        self.epochs = epochs
        self.lr = lr
        self.config = config
        self.val_loader = val_loader

    def train(self, on_status: StatusCallback | None = None) -> TrainResult:
        """Run the training loop and return a :class:`TrainResult`.

        Args:
            on_status: Optional callback invoked with a progress string
                       or a dict (for structured SSE events).

        Returns:
            A :class:`TrainResult` with the trained model and metadata.
        """

        def status(msg: str | dict) -> None:
            if on_status:
                on_status(msg)

        status("Preparing training data…")

        model = self.model_cls()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        cfg = self.config
        history: list[dict] = []
        best_acc = 0.0
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = -1
        stopped_early = False
        epochs_completed = 0

        # Distillation setup
        teacher = None
        if cfg and cfg.teacher_model is not None:
            teacher = cfg.teacher_model
            teacher.eval()

        global_batch = 0
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            running_loss = 0.0
            running_count = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()

                output = model(data)
                loss = type(model).loss_fn(output, target)

                # Distillation blending
                if teacher is not None and cfg is not None:
                    with torch.no_grad():
                        teacher_out = teacher(data)
                        if cfg.teacher_process is not None:
                            teacher_out = cfg.teacher_process(teacher_out)
                    distill_loss = F.mse_loss(output, teacher_out)
                    w = cfg.distill_weight
                    loss = (1 - w) * loss + w * distill_loss

                # Regularisation
                if cfg and cfg.regularization_fn is not None:
                    loss = loss + cfg.regularization_fn(model)

                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                running_loss += batch_loss
                running_count += 1
                global_batch += 1

                # Status update every 100 batches
                if batch_idx % 100 == 0:
                    status(
                        f"Epoch {epoch + 1}/{self.epochs} — "
                        f"batch {batch_idx}/{len(self.train_loader)} — "
                        f"loss: {batch_loss:.4f}"
                    )

                # Validation checkpoint
                if (
                    self.val_loader is not None
                    and cfg is not None
                    and global_batch % cfg.val_gap == 0
                ):
                    val_acc = self._validate(model)
                    avg_running = running_loss / running_count if running_count else 0
                    entry = HistoryEntry(
                        epoch=epoch,
                        batch=batch_idx,
                        train_loss=avg_running,
                        val_accuracy=val_acc,
                    )
                    history.append(entry.to_dict())
                    status(entry.to_dict())
                    running_loss = 0.0
                    running_count = 0

                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_epoch = epoch
                        best_model_state = copy.deepcopy(model.state_dict())

            epochs_completed = epoch + 1
            avg = total_loss / max(len(self.train_loader), 1)
            status(f"Epoch {epoch + 1}/{self.epochs} done — avg loss: {avg:.4f}")

            # Early stopping check
            if (
                cfg is not None
                and cfg.patience is not None
                and best_acc > 0.6
                and epoch > best_epoch + cfg.patience
            ):
                status(
                    f"Early stopping: no improvement for {cfg.patience} epochs "
                    f"(best val accuracy: {best_acc:.2%})"
                )
                stopped_early = True
                break

        # Use best model if we did validation
        if self.val_loader is not None and best_acc > 0:
            model.load_state_dict(best_model_state)

        status("Training complete!")
        logger.info(
            "Training finished: %s %d epochs, %d params, val_acc=%.4f",
            self.model_cls.name, epochs_completed, num_params,
            best_acc if best_acc > 0 else float("nan"),
        )
        return TrainResult(
            model=model,
            model_type=self.model_cls.name,
            dataset=self.dataset,
            epochs=self.epochs,
            batch_size=self.train_loader.batch_size or 0,
            lr=self.lr,
            epochs_completed=epochs_completed,
            history=history,
            best_val_accuracy=best_acc if best_acc > 0 else None,
            stopped_early=stopped_early,
            num_params=num_params,
        )

    def _validate(self, model: BaseModel) -> float:
        """Run a quick validation pass and return accuracy."""
        assert self.val_loader is not None
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
        model.train()
        return correct / total if total > 0 else 0.0
