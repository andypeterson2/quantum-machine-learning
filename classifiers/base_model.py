"""Abstract base class for all classifier model architectures.

The :class:`BaseModel` ABC defines the shared contract that every model across
every dataset must implement.  It is intentionally free of any dataset-specific
knowledge — input shape, class count, and normalisation are the concern of the
:class:`~classifiers.dataset_plugin.DatasetPlugin` that owns the model.

Model discovery is handled by the plugin system, not by a global registry.
Each :class:`~classifiers.dataset_plugin.DatasetPlugin` returns its compatible
model classes from :meth:`~DatasetPlugin.get_model_types`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(ABC, nn.Module):
    """Abstract base for all classifiers across all datasets.

    Concrete subclasses must:

    * Set :attr:`name` — architecture display name shown in the UI.
    * Set :attr:`description` — one-line human-readable summary.
    * Implement :meth:`forward`.
    * Optionally override :meth:`loss_fn` to use a non-standard training
      objective (e.g. multi-class hinge loss for SVM-style training).

    Attributes:
        name:        Architecture display name (class attribute).
        description: One-line description shown in the UI (class attribute).
    """

    name: str = ""
    description: str = ""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class scores (logits) for the input batch.

        The input shape depends on the dataset:

        * Image datasets: ``(N, C, H, W)`` (e.g. ``(N, 1, 28, 28)``).
        * Tabular datasets: ``(N, F)`` (e.g. ``(N, 4)``).

        Returns:
            Logit tensor of shape ``(N, num_classes)``.  Do **not** apply
            softmax — that is handled by the loss function during training
            and by :class:`~classifiers.predictor.Predictor` during inference.
        """
        ...

    @staticmethod
    def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the training loss for a batch of predictions.

        Override in a subclass to use a different objective (e.g.
        :func:`~classifiers.losses.multi_class_hinge_loss`).  The default is
        cross-entropy, appropriate for probabilistic classifiers.

        Args:
            output: Raw logit tensor of shape ``(N, num_classes)``.
            target: Ground-truth class indices of shape ``(N,)``.

        Returns:
            Scalar loss tensor.
        """
        return F.cross_entropy(output, target)
