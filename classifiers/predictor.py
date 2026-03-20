"""Dataset-agnostic inference: preprocess raw input and predict class probabilities.

:class:`Predictor` follows the Single Responsibility Principle — its only job
is to wrap a trained model with the preprocessing pipeline needed to turn raw
user input into a calibrated probability distribution.  It delegates all
dataset-specific preprocessing to the active
:class:`~classifiers.dataset_plugin.DatasetPlugin`, so it has no knowledge of
image sizes, normalisation constants, or feature schemas.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .base_model import BaseModel

if TYPE_CHECKING:
    from .dataset_plugin import DatasetPlugin


class Predictor:
    """Wraps a trained model and a dataset plugin to perform single-input prediction.

    The plugin's :meth:`~classifiers.dataset_plugin.DatasetPlugin.preprocess`
    method converts the raw user input (PIL image for image datasets, feature
    dict for tabular datasets) into a model-ready tensor.

    Args:
        model:  A trained :class:`~classifiers.base_model.BaseModel` instance.
        plugin: The :class:`~classifiers.dataset_plugin.DatasetPlugin` whose
                preprocessing matches the model's expected input format.

    Example::

        predictor = Predictor(trained_model, mnist_plugin)
        probs = predictor.predict(Image.open("digit.png"))
        print(f"Predicted: {probs.argmax()} ({probs.max():.1%} confidence)")
    """

    def __init__(self, model: BaseModel, plugin: "DatasetPlugin") -> None:
        self.model = model
        self.plugin = plugin

    def predict(self, raw_input: Any) -> np.ndarray:
        """Return a probability array for *raw_input*.

        Sets the model to eval mode, delegates preprocessing to the plugin,
        runs a forward pass with gradients disabled, and applies softmax to
        convert raw logits to probabilities.

        Args:
            raw_input: Unprocessed input from the frontend.  For image datasets
                       this is a :class:`PIL.Image.Image`; for tabular datasets
                       it is a ``dict[str, float]`` mapping feature names to values.

        Returns:
            A NumPy array of shape ``(num_classes,)`` with class probabilities
            that sum to 1.0.  Index ``i`` corresponds to ``plugin.class_labels[i]``.
        """
        self.model.eval()
        tensor = self.plugin.preprocess(raw_input)
        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1).squeeze().numpy()
        return probs
