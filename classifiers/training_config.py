"""Configuration and history types for advanced training features.

:class:`TrainingConfig` bundles optional advanced training parameters
(early stopping, validation, distillation, regularisation) so the
:class:`~classifiers.trainer.Trainer` constructor stays stable while
gaining new capabilities through composition.

:class:`HistoryEntry` records a single data point in the training curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from classifiers.base_model import BaseModel


@dataclass
class TrainingConfig:
    """Advanced training options passed to :class:`~classifiers.trainer.Trainer`.

    All fields are optional.  When the entire config is ``None`` the trainer
    behaves identically to the original fixed-epoch Adam loop.

    Attributes:
        patience:           Early-stopping patience in epochs.  ``None`` disables.
        val_gap:            Batches between validation checks.
        regularization_fn:  Callable ``(model) → scalar tensor`` added to the loss.
        teacher_model:      Frozen teacher model for knowledge distillation.
        distill_weight:     Blend ratio: ``(1-w)*true + w*distill``.
        teacher_process:    Post-processing applied to teacher output (e.g. softmax).
    """

    patience: int | None = None
    val_gap: int = 50
    regularization_fn: Callable[[torch.nn.Module], torch.Tensor] | None = None
    teacher_model: "BaseModel | None" = field(default=None, repr=False)
    distill_weight: float = 0.5
    teacher_process: Callable[[torch.Tensor], torch.Tensor] | None = None


@dataclass
class HistoryEntry:
    """A single data point in the training curve.

    Attributes:
        epoch:        Current epoch (0-indexed).
        batch:        Current batch within the epoch.
        train_loss:   Running average training loss at this point.
        val_accuracy: Validation accuracy (``None`` if not a validation step).
    """

    epoch: int
    batch: int
    train_loss: float
    val_accuracy: float | None = None

    def to_dict(self) -> dict:
        d: dict = {"epoch": self.epoch, "batch": self.batch, "train_loss": self.train_loss}
        if self.val_accuracy is not None:
            d["val_accuracy"] = self.val_accuracy
        return d
