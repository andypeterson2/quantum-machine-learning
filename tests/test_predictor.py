"""Unit tests for classifiers.predictor.Predictor."""

import numpy as np
import torch
from PIL import Image

from classifiers.predictor import Predictor


class TestPreprocessing:
    """Plugin preprocessing produces correct tensor shapes."""

    def test_output_shape(self, blank_image, mnist_plugin):
        tensor = mnist_plugin.preprocess(blank_image)
        assert tensor.shape == (1, 1, 28, 28)

    def test_output_dtype(self, blank_image, mnist_plugin):
        tensor = mnist_plugin.preprocess(blank_image)
        assert tensor.dtype == torch.float32

    def test_handles_rgb_input(self, mnist_plugin):
        """Plugin should convert RGB to grayscale gracefully."""
        rgb = Image.new("RGB", (280, 280), (128, 128, 128))
        tensor = mnist_plugin.preprocess(rgb)
        assert tensor.shape == (1, 1, 28, 28)

    def test_handles_different_sizes(self, mnist_plugin):
        """Should resize any input to 28x28."""
        small = Image.new("L", (14, 14), 128)
        tensor = mnist_plugin.preprocess(small)
        assert tensor.shape == (1, 1, 28, 28)

        large = Image.new("L", (560, 560), 128)
        tensor = mnist_plugin.preprocess(large)
        assert tensor.shape == (1, 1, 28, 28)


class TestPredict:
    def test_returns_probability_array(self, untrained_model, blank_image, mnist_plugin):
        predictor = Predictor(untrained_model, mnist_plugin)
        probs = predictor.predict(blank_image)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (10,)

    def test_probabilities_sum_to_one(self, untrained_model, drawn_image, mnist_plugin):
        predictor = Predictor(untrained_model, mnist_plugin)
        probs = predictor.predict(drawn_image)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_probabilities_non_negative(self, untrained_model, drawn_image, mnist_plugin):
        predictor = Predictor(untrained_model, mnist_plugin)
        probs = predictor.predict(drawn_image)
        assert (probs >= 0).all()

    def test_deterministic(self, untrained_model, drawn_image, mnist_plugin):
        predictor = Predictor(untrained_model, mnist_plugin)
        p1 = predictor.predict(drawn_image)
        p2 = predictor.predict(drawn_image)
        np.testing.assert_array_almost_equal(p1, p2)


class TestPredictWithLinear:
    """Predictor should work with any BaseModel, not just CNN."""

    def test_predict_with_linear_model(self, untrained_linear, drawn_image, mnist_plugin):
        predictor = Predictor(untrained_linear, mnist_plugin)
        probs = predictor.predict(drawn_image)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (10,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_predict_with_linear_blank(self, untrained_linear, blank_image, mnist_plugin):
        predictor = Predictor(untrained_linear, mnist_plugin)
        probs = predictor.predict(blank_image)
        assert (probs >= 0).all()
