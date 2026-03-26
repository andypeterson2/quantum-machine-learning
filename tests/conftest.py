"""Shared fixtures and helpers for classifiers tests."""

import base64
import io
import json

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


def blank_png_b64(width: int = 280, height: int = 280) -> str:
    """Return a base64-encoded blank black PNG."""
    img = Image.new("L", (width, height), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse_sse(raw: bytes) -> list[dict]:
    """Parse raw SSE bytes into a list of event dicts."""
    events = []
    for chunk in raw.decode().split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data:"):
            continue
        events.append(json.loads(chunk[len("data:"):].strip()))
    return events


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
