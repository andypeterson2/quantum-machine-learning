"""Shared fixtures for classifiers tests."""

import pytest
import torch
import numpy as np
from PIL import Image

from classifiers.datasets.mnist.models import MNISTNet, LinearNet, SVMNet
from classifiers.datasets.mnist.plugin import MNISTPlugin


@pytest.fixture
def mnist_plugin():
    """A fresh MNIST plugin instance."""
    return MNISTPlugin()


@pytest.fixture
def untrained_model():
    """A freshly initialized (random weights) MNISTNet."""
    return MNISTNet()


@pytest.fixture
def untrained_linear():
    """A freshly initialized LinearNet."""
    return LinearNet()


@pytest.fixture
def untrained_svm():
    """A freshly initialized SVMNet."""
    return SVMNet()


@pytest.fixture
def blank_image():
    """A blank 280x280 grayscale image (black canvas)."""
    return Image.new("L", (280, 280), 0)


@pytest.fixture
def drawn_image():
    """A 280x280 grayscale image with a white circle drawn in the center."""
    from PIL import ImageDraw

    img = Image.new("L", (280, 280), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 180, 180], fill=255)
    return img


@pytest.fixture
def sample_probs():
    """A realistic 10-element probability array (sums to 1)."""
    raw = np.array(
        [0.01, 0.01, 0.02, 0.85, 0.03, 0.02, 0.01, 0.02, 0.02, 0.01],
        dtype=np.float32,
    )
    return raw / raw.sum()


@pytest.fixture
def sample_batch():
    """A small batch of random tensors shaped like MNIST input."""
    return torch.randn(4, 1, 28, 28)


class _FakeLoader(list):
    """A list that also exposes a batch_size attribute like a real DataLoader."""

    def __init__(self, batches, batch_size):
        super().__init__(batches)
        self.batch_size = batch_size


def make_fake_train_loader(batch_size=16, n_batches=5):
    """Create a fake data loader yielding random MNIST-like batches."""
    batches = []
    for _ in range(n_batches):
        data = torch.randn(batch_size, 1, 28, 28)
        targets = torch.randint(0, 10, (batch_size,))
        batches.append((data, targets))
    return _FakeLoader(batches, batch_size)


def make_fake_test_loader(batch_size=100, n_batches=2):
    """Create a fake test loader with random MNIST-like batches."""
    batches = []
    for _ in range(n_batches):
        data = torch.randn(batch_size, 1, 28, 28)
        targets = torch.randint(0, 10, (batch_size,))
        batches.append((data, targets))
    return _FakeLoader(batches, batch_size)
