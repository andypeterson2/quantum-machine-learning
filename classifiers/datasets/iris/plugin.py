"""Iris flower classification dataset plugin.

Uses ``sklearn.datasets.load_iris()`` to provide a tiny (150-sample) tabular
dataset with 4 numeric features and 3 target classes.  The standardisation,
splitting and inference pipeline it shares with every tabular dataset lives in
:class:`~classifiers.datasets.tabular.TabularPlugin`.
"""

from __future__ import annotations

import numpy as np

from classifiers.base_model import BaseModel
from classifiers.datasets.tabular import TabularPlugin


class IrisPlugin(TabularPlugin):
    """Plugin for the Iris flower classification dataset.

    * 3 classes: setosa, versicolor, virginica
    * 4 continuous features: sepal length/width, petal length/width
    * Train / test split: 80 / 20 (stratified, fixed seed for reproducibility)
    """

    name = "iris"
    display_name = "Iris Flower Classification"
    num_classes = 3
    class_labels = ["setosa", "versicolor", "virginica"]
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    def load_raw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load the 150 samples and split them 80/20, stratified by class."""
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X = data.data.astype(np.float32)        # (150, 4)
        y = data.target.astype(np.int64)     # (150,)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, y_train, X_test, y_test

    # ── Model types ───────────────────────────────────────────────────────────

    def get_model_types(self) -> dict[str, type[BaseModel]]:
        """Return compatible architectures for Iris.

        Returns:
            ``{"Linear": IrisLinear, "SVM": IrisSVM}``, plus ``"QVC": IrisQVC`` when
            PennyLane (the optional ``quantum`` extra) is installed.
        """
        from .models import IrisLinear, IrisSVM

        types: dict[str, type[BaseModel]] = {"Linear": IrisLinear, "SVM": IrisSVM}
        # QVC needs PennyLane (the optional `quantum` extra, imported lazily deep in
        # IrisQVC). Only advertise it when PennyLane is importable, so a lean deploy
        # doesn't offer a model that raises ImportError at train time. Install the
        # `quantum` extra to light up the quantum classifier.
        try:
            import pennylane  # noqa: F401
        except ImportError:
            pass
        else:
            from .models import IrisQVC

            types["QVC"] = IrisQVC
        return types

    def get_default_hyperparams(self) -> dict:
        """Return Iris-tuned defaults: more epochs, smaller batch, lower lr.

        Returns:
            ``{"epochs": 50, "batch_size": 16, "lr": 0.01}``
        """
        return {"epochs": 50, "batch_size": 16, "lr": 0.01}
