"""Iris flower classification dataset plugin.

Uses ``sklearn.datasets.load_iris()`` to provide a tiny (150-sample) tabular
dataset with 4 numeric features and 3 target classes.  Feature standardisation
(z-score) is computed from the training split and cached on the plugin instance
for reuse during test-set evaluation and single-sample inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from classifiers.base_model import BaseModel
from classifiers.dataset_plugin import DatasetPlugin


class IrisPlugin(DatasetPlugin):
    """Plugin for the Iris flower classification dataset.

    * 3 classes: setosa, versicolor, virginica
    * 4 continuous features: sepal length/width, petal length/width
    * Train / test split: 80 / 20 (stratified, fixed seed for reproducibility)
    """

    name = "iris"
    display_name = "Iris Flower Classification"
    input_type = "tabular"
    num_classes = 3
    class_labels = ["setosa", "versicolor", "virginica"]
    image_size = None
    image_channels = None
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    def __init__(self) -> None:
        super().__init__()
        self._train_X: torch.Tensor | None = None
        self._train_y: torch.Tensor | None = None
        self._test_X: torch.Tensor | None = None
        self._test_y: torch.Tensor | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    # ── Data loading ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load and split the dataset on first access, caching the result."""
        if self._train_X is not None:
            return

        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X = data.data.astype(np.float32)     # (150, 4)
        y = data.target.astype(np.int64)     # (150,)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_t = torch.from_numpy(X_train)
        self._mean = X_train_t.mean(dim=0)
        self._std = X_train_t.std(dim=0).clamp(min=1e-8)

        self._train_X = (X_train_t - self._mean) / self._std
        self._train_y = torch.from_numpy(y_train)
        self._test_X = (torch.from_numpy(X_test) - self._mean) / self._std
        self._test_y = torch.from_numpy(y_test)

    def get_train_loader(self, batch_size: int) -> DataLoader:
        """Return a :class:`DataLoader` over the standardised Iris training set.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        self._ensure_loaded()
        assert self._train_X is not None and self._train_y is not None
        ds = TensorDataset(self._train_X, self._train_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Return a :class:`DataLoader` over the standardised Iris test set.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        self._ensure_loaded()
        assert self._test_X is not None and self._test_y is not None
        ds = TensorDataset(self._test_X, self._test_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def get_val_loader(self, batch_size: int) -> DataLoader:
        """Split training data 80/20 to create a validation set.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        self._ensure_loaded()
        assert self._train_X is not None and self._train_y is not None
        n = len(self._train_X)
        split = int(n * 0.8)
        val_X = self._train_X[split:]
        val_y = self._train_y[split:]
        ds = TensorDataset(val_X, val_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """Convert a dict of feature values to a standardised tensor.

        The same mean/std computed from the training split is applied so that
        inference is consistent with training.

        Args:
            raw_input: A ``dict[str, float]`` mapping feature names to values.

        Returns:
            Float tensor of shape ``(1, 4)``.
        """
        self._ensure_loaded()
        assert self._mean is not None and self._std is not None
        assert self.feature_names is not None
        values = [float(raw_input[f]) for f in self.feature_names]
        tensor = torch.tensor([values], dtype=torch.float32)
        return (tensor - self._mean) / self._std

    # ── Model types ───────────────────────────────────────────────────────────

    def get_model_types(self) -> dict[str, type[BaseModel]]:
        """Return compatible architectures for Iris.

        Returns:
            ``{"Linear": IrisLinear, "SVM": IrisSVM, "QVC": IrisQVC}``
        """
        from .models import IrisLinear, IrisSVM, IrisQVC

        return {"Linear": IrisLinear, "SVM": IrisSVM, "QVC": IrisQVC}

    def get_default_hyperparams(self) -> dict:
        """Return Iris-tuned defaults: more epochs, smaller batch, lower lr.

        Returns:
            ``{"epochs": 50, "batch_size": 16, "lr": 0.01}``
        """
        return {"epochs": 50, "batch_size": 16, "lr": 0.01}
