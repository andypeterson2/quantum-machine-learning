"""Shared base for tabular dataset plugins.

Iris and BB84 differ in exactly one thing — where their raw arrays come from,
one from scikit-learn and one from a seeded simulation — and agreed, line for
line, on everything after that: z-score from the training split's statistics,
cache the tensors, serve train / validation / test loaders from an 80/20 split
of the training rows, and standardise a single sample the same way at inference.

That agreement was two copies. A third tabular dataset would have been a third.
Subclasses now supply :meth:`load_raw` and their metadata; the pipeline lives
here once.

This is an optional convenience, not a new requirement:
:class:`~classifiers.dataset_plugin.DatasetPlugin` remains the extension point,
and a plugin with different needs still implements it directly.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from classifiers.dataset_plugin import DatasetPlugin

#: Fraction of the training rows served as the training split; the rest is
#: validation. The test split is separate and never touched here.
TRAIN_FRACTION = 0.8

#: Floor on the per-feature standard deviation, so a constant feature cannot
#: divide by zero.
MIN_STD = 1e-8


class TabularPlugin(DatasetPlugin):
    """A dataset of numeric feature rows, standardised on its training split."""

    input_type = "tabular"
    image_size = None
    image_channels = None

    def __init__(self) -> None:
        super().__init__()
        self._train_X: torch.Tensor | None = None
        self._train_y: torch.Tensor | None = None
        self._test_X: torch.Tensor | None = None
        self._test_y: torch.Tensor | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    # ── What a subclass supplies ──────────────────────────────────────────────

    @abstractmethod
    def load_raw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return raw ``(train_X, train_y, test_X, test_y)``, unstandardised.

        Called once per plugin instance. ``train_X`` and ``test_X`` are float32
        ``(N, F)`` matrices; the label arrays are int64 ``(N,)``.
        """
        ...

    # ── The pipeline every tabular dataset shares ─────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load, standardise and cache the dataset on first access.

        The mean and standard deviation come from the training rows only, so
        the test split never informs the transformation applied to it.
        """
        if self._train_X is not None:
            return

        train_X, train_y, test_X, test_y = self.load_raw()

        train_t = torch.from_numpy(train_X)
        self._mean = train_t.mean(dim=0)
        self._std = train_t.std(dim=0).clamp(min=MIN_STD)

        self._train_X = (train_t - self._mean) / self._std
        self._train_y = torch.from_numpy(train_y)
        self._test_X = (torch.from_numpy(test_X) - self._mean) / self._std
        self._test_y = torch.from_numpy(test_y)

    def _split_at(self) -> int:
        self._ensure_loaded()
        assert self._train_X is not None
        return int(len(self._train_X) * TRAIN_FRACTION)

    def get_train_loader(self, batch_size: int) -> DataLoader:
        """Return a loader over the standardised training split.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        split = self._split_at()
        assert self._train_X is not None and self._train_y is not None
        ds = TensorDataset(self._train_X[:split], self._train_y[:split])
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def get_val_loader(self, batch_size: int) -> DataLoader:
        """Return a loader over the held-back tail of the training rows.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        split = self._split_at()
        assert self._train_X is not None and self._train_y is not None
        ds = TensorDataset(self._train_X[split:], self._train_y[split:])
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Return a loader over the standardised test split.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        self._ensure_loaded()
        assert self._test_X is not None and self._test_y is not None
        ds = TensorDataset(self._test_X, self._test_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def normalization(self) -> tuple[list[float], list[float]]:
        """Return the (mean, std) standardisation constants as plain lists.

        Computed from the training split; the exact constants any external
        consumer — e.g. the browser demo fed by :mod:`classifiers.web_export` —
        must reproduce.
        """
        self._ensure_loaded()
        assert self._mean is not None and self._std is not None
        return self._mean.tolist(), self._std.tolist()

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """Convert a dict of feature values to a standardised tensor.

        Args:
            raw_input: A ``dict[str, float]`` keyed by :attr:`feature_names`.

        Returns:
            Float tensor of shape ``(1, F)``.
        """
        self._ensure_loaded()
        assert self._mean is not None and self._std is not None
        assert self.feature_names is not None
        values = [float(raw_input[name]) for name in self.feature_names]
        tensor = torch.tensor([values], dtype=torch.float32)
        return (tensor - self._mean) / self._std
