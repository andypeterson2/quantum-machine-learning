"""BB84 eavesdropper-detection dataset plugin.

Classifies simulated quantum-key-distribution sessions as *clean* or
*eavesdropped* (intercept-resend attack) from the two observables an operator
actually sees: the quantum bit error rate and the sifted-key rate. The
sessions come from :mod:`classifiers.datasets.bb84.simulate` — a port of the
quantum-video-chat project's channel physics — generated on the fly with
fixed seeds, so the plugin needs no dataset download or cache anywhere,
including CI.

Quantum machine learning guarding quantum cryptography: the same platform
that trains the QVC/QSVM classifiers watches the protocol that distributes
quantum keys.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from classifiers.base_model import BaseModel
from classifiers.dataset_plugin import DatasetPlugin

from .simulate import generate_dataset

#: Session counts and seeds — fixed so every consumer (trainer, exporter,
#: drift tests) sees byte-identical data.
N_TRAIN = 2000
N_TEST = 500
TRAIN_SEED = 42
TEST_SEED = 43


class BB84Plugin(DatasetPlugin):
    """Plugin for BB84 eavesdropper detection.

    * 2 classes: clean, eavesdropped
    * 2 continuous features: qber, sifted_key_rate
    * Self-generated data (seeded simulation) — nothing to download.
    """

    name = "bb84"
    display_name = "BB84 Eavesdropper Detection"
    input_type = "tabular"
    num_classes = 2
    class_labels = ["clean", "eavesdropped"]
    image_size = None
    image_channels = None
    feature_names = ["qber", "sifted_key_rate"]

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
        """Simulate and standardise the dataset on first access."""
        if self._train_X is not None:
            return

        train_X, train_y = generate_dataset(N_TRAIN, TRAIN_SEED)
        test_X, test_y = generate_dataset(N_TEST, TEST_SEED)

        train_t = torch.from_numpy(train_X)
        self._mean = train_t.mean(dim=0)
        self._std = train_t.std(dim=0).clamp(min=1e-8)

        self._train_X = (train_t - self._mean) / self._std
        self._train_y = torch.from_numpy(train_y)
        self._test_X = (torch.from_numpy(test_X) - self._mean) / self._std
        self._test_y = torch.from_numpy(test_y)

    def get_train_loader(self, batch_size: int) -> DataLoader:
        """Return a :class:`DataLoader` over the standardised training set.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        self._ensure_loaded()
        assert self._train_X is not None and self._train_y is not None
        split = int(len(self._train_X) * 0.8)
        ds = TensorDataset(self._train_X[:split], self._train_y[:split])
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Return a :class:`DataLoader` over the standardised test set.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        self._ensure_loaded()
        assert self._test_X is not None and self._test_y is not None
        ds = TensorDataset(self._test_X, self._test_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def get_val_loader(self, batch_size: int) -> DataLoader:
        """Hold out the last 20% of the training set for validation.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        self._ensure_loaded()
        assert self._train_X is not None and self._train_y is not None
        split = int(len(self._train_X) * 0.8)
        ds = TensorDataset(self._train_X[split:], self._train_y[split:])
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def normalization(self) -> tuple[list[float], list[float]]:
        """Return the (mean, std) standardisation constants as plain lists.

        Computed from the training split; the exact constants any external
        consumer (e.g. the browser demo fed by :mod:`classifiers.web_export`)
        must reproduce.
        """
        self._ensure_loaded()
        assert self._mean is not None and self._std is not None
        return self._mean.tolist(), self._std.tolist()

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """Convert a dict of feature values to a standardised tensor.

        Args:
            raw_input: A ``dict[str, float]`` with ``qber`` and
                ``sifted_key_rate``.

        Returns:
            Float tensor of shape ``(1, 2)``.
        """
        self._ensure_loaded()
        assert self._mean is not None and self._std is not None
        assert self.feature_names is not None
        values = [float(raw_input[f]) for f in self.feature_names]
        tensor = torch.tensor([values], dtype=torch.float32)
        return (tensor - self._mean) / self._std

    # ── Model types ───────────────────────────────────────────────────────────

    def get_model_types(self) -> dict[str, type[BaseModel]]:
        """Return compatible architectures for BB84.

        Returns:
            ``{"Linear": BB84Linear, "SVM": BB84SVM}``, plus ``"QVC": BB84QVC``
            when PennyLane (the optional ``quantum`` extra) is installed.
        """
        from .models import BB84SVM, BB84Linear

        types: dict[str, type[BaseModel]] = {"Linear": BB84Linear, "SVM": BB84SVM}
        # Same lazy-optional pattern as the Iris plugin: only advertise QVC
        # when PennyLane is importable, so a lean deploy never offers a model
        # that raises ImportError at train time.
        try:
            import pennylane  # noqa: F401
        except ImportError:
            pass
        else:
            from .models import BB84QVC

            types["QVC"] = BB84QVC
        return types

    def get_default_hyperparams(self) -> dict:
        """Return BB84-tuned defaults for the 1600-sample training subset.

        Returns:
            ``{"epochs": 30, "batch_size": 32, "lr": 0.01}``
        """
        return {"epochs": 30, "batch_size": 32, "lr": 0.01}
