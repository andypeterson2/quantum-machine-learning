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


class TestCenterDigit:
    """Drawn digits get MNIST's crop / fit-20x20 / centre-by-mass normalisation."""

    def test_corner_digit_is_scaled_and_centred(self):
        from classifiers.datasets.mnist.plugin import center_digit

        img = Image.new("L", (280, 280), 0)
        arr = np.array(img)
        arr[10:90, 10:60] = 255  # an 80x50 block in the top-left corner
        out = center_digit(Image.fromarray(arr))
        ys, _ = np.nonzero(out > 64)
        assert ys.max() - ys.min() + 1 in (19, 20, 21)  # longer side fills the 20px box
        cy = (out.sum(axis=1) * np.arange(28)).sum() / out.sum()
        cx = (out.sum(axis=0) * np.arange(28)).sum() / out.sum()
        assert abs(cy - 13.5) <= 1 and abs(cx - 13.5) <= 1

    def test_faint_grid_lines_do_not_count_as_ink(self):
        from classifiers.datasets.mnist.plugin import center_digit

        arr = np.zeros((280, 280), dtype=np.uint8)
        arr[::10, :] = 20  # the portal canvas's faint grid
        arr[:, ::10] = 20
        arr[100:180, 130:150] = 255  # a vertical stroke
        out = center_digit(Image.fromarray(arr))
        ys, xs = np.nonzero(out > 64)
        assert ys.max() - ys.min() + 1 in (19, 20, 21)
        assert xs.max() - xs.min() + 1 <= 7  # stays a thin stroke

    def test_blank_image_is_just_resized(self):
        from classifiers.datasets.mnist.plugin import center_digit

        out = center_digit(Image.new("L", (280, 280), 0))
        assert out.shape == (28, 28)
        assert out.max() == 0
