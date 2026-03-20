"""Abstract base class for dataset plugins.

The :class:`DatasetPlugin` ABC is the single OCP extension point for adding
new datasets.  Each plugin bundles everything dataset-specific:

* Classification metadata (class count, labels)
* Data loading (train / test :class:`~torch.utils.data.DataLoader` instances)
* Input preprocessing (raw user input → model-ready tensor)
* Compatible model architectures

To add a new dataset, create a subpackage under ``classifiers/datasets/``,
subclass :class:`DatasetPlugin`, and call
:func:`~classifiers.plugin_registry.register_plugin` in the package's
``__init__.py``.  No changes to existing code are required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from .base_model import BaseModel


class DatasetPlugin(ABC):
    """Extension point for registering a new dataset with the platform.

    Attributes:
        name:           URL-safe slug used in routes (e.g. ``"mnist"``).
        display_name:   Human-readable name shown in the UI.
        input_type:     ``"image"`` for drawing-canvas datasets,
                        ``"tabular"`` for numeric-form datasets.
        num_classes:    Total number of target classes.
        class_labels:   Ordered list of human-readable class names.
        image_size:     ``(H, W)`` for image datasets, ``None`` otherwise.
        image_channels: Number of input channels (1 = grayscale, 3 = RGB),
                        ``None`` for tabular.
        feature_names:  Ordered list of feature column names for tabular
                        datasets, ``None`` for image.
    """

    name: str
    display_name: str
    input_type: Literal["image", "tabular"]
    num_classes: int
    class_labels: list[str]
    image_size: tuple[int, int] | None = None
    image_channels: int | None = None
    feature_names: list[str] | None = None

    # ── Data loading ──────────────────────────────────────────────────────────

    @abstractmethod
    def get_train_loader(self, batch_size: int) -> DataLoader:
        """Return a :class:`~torch.utils.data.DataLoader` over the training set.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        ...

    @abstractmethod
    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Return a :class:`~torch.utils.data.DataLoader` over the test set.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        ...

    # ── Inference preprocessing ───────────────────────────────────────────────

    @abstractmethod
    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """Convert raw user input to a model-ready tensor.

        For image datasets, *raw_input* is a :class:`PIL.Image.Image`.
        For tabular datasets, *raw_input* is a ``dict[str, float]`` mapping
        feature names to values.

        Args:
            raw_input: The unprocessed input from the frontend.

        Returns:
            A float tensor with a leading batch dimension, ready to pass
            directly to ``model.forward()``.
        """
        ...

    # ── Model types scoped to this dataset ────────────────────────────────────

    @abstractmethod
    def get_model_types(self) -> dict[str, type[BaseModel]]:
        """Return compatible model architectures for this dataset.

        Returns:
            A dict mapping display names (e.g. ``"CNN"``) to
            :class:`~classifiers.base_model.BaseModel` subclasses.
        """
        ...

    # ── Validation data ──────────────────────────────────────────────────────

    def get_val_loader(self, batch_size: int) -> DataLoader | None:
        """Return a :class:`~torch.utils.data.DataLoader` over a validation set.

        Override in plugins that support a separate validation split.  The
        default implementation returns ``None`` (no validation set).

        Args:
            batch_size: Number of samples per mini-batch.
        """
        return None

    # ── Defaults ──────────────────────────────────────────────────────────────

    def get_default_hyperparams(self) -> dict:
        """Return sensible default training hyper-parameters.

        Override to provide dataset-specific defaults (e.g. more epochs for
        small datasets like Iris).

        Returns:
            A dict with keys ``epochs``, ``batch_size``, ``lr``.
        """
        return {"epochs": 3, "batch_size": 64, "lr": 1e-3}

    # ── UI configuration ──────────────────────────────────────────────────────

    def get_ui_config(self) -> dict:
        """Return a JSON-serialisable dict passed to the frontend as ``UI_CONFIG``.

        The shared backend never inspects this dict; it is passed through to
        the Jinja template and then to JavaScript, maintaining OCP.

        Returns:
            A dict with all metadata the frontend needs to render the correct
            input widget and tables.
        """
        config: dict = {
            "name": self.name,
            "display_name": self.display_name,
            "input_type": self.input_type,
            "num_classes": self.num_classes,
            "class_labels": self.class_labels,
            "default_hyperparams": self.get_default_hyperparams(),
        }
        if self.input_type == "image":
            config["image_size"] = self.image_size
            config["image_channels"] = self.image_channels
        elif self.input_type == "tabular":
            config["feature_names"] = self.feature_names
        return config
