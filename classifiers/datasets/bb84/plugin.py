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

Standardisation, splitting and inference are the tabular pipeline in
:class:`~classifiers.datasets.tabular.TabularPlugin`.
"""

from __future__ import annotations

import numpy as np

from classifiers.base_model import BaseModel
from classifiers.datasets.tabular import TabularPlugin

from .simulate import generate_dataset

#: Session counts and seeds — fixed so every consumer (trainer, exporter,
#: drift tests) sees byte-identical data.
N_TRAIN = 2000
N_TEST = 500
TRAIN_SEED = 42
TEST_SEED = 43


class BB84Plugin(TabularPlugin):
    """Plugin for BB84 eavesdropper detection.

    * 2 classes: clean, eavesdropped
    * 2 continuous features: qber, sifted_key_rate
    * Self-generated data (seeded simulation) — nothing to download.
    """

    name = "bb84"
    display_name = "BB84 Eavesdropper Detection"
    num_classes = 2
    class_labels = ["clean", "eavesdropped"]
    feature_names = ["qber", "sifted_key_rate"]

    def load_raw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate both splits from their fixed seeds."""
        train_X, train_y = generate_dataset(N_TRAIN, TRAIN_SEED)
        test_X, test_y = generate_dataset(N_TEST, TEST_SEED)
        return train_X, train_y, test_X, test_y

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
