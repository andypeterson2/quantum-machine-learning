"""MNIST handwritten-digit dataset plugin.

Encapsulates all MNIST-specific knowledge: normalisation constants, image
dimensions, data loading via ``torchvision.datasets.MNIST``, canvas-image
preprocessing, and the set of compatible model architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from classifiers.base_model import BaseModel
from classifiers.dataset_plugin import DatasetPlugin

#: Per-channel mean computed over the MNIST training set.
MNIST_MEAN: float = 0.1307
#: Per-channel standard deviation computed over the MNIST training set.
MNIST_STD: float = 0.3081
#: Side length (pixels) of MNIST images.
IMG_SIZE: int = 28
#: MNIST digits are fitted into a 20x20 box inside the 28x28 frame.
DIGIT_BOX: int = 20
#: Pixels at or above this (0-255) count as ink when finding the digit. Keeps the
#: faint grid lines the portal canvas draws from widening the box to the whole image.
INK_THRESHOLD: int = 32


def center_digit(image: Image.Image) -> np.ndarray:
    """MNIST's own normalisation for a hand-drawn digit.

    The training digits were cropped to their ink, fitted into a 20x20 box
    (aspect ratio kept) and centred by centre of mass in a 28x28 frame. A
    drawing that skips this step is off-centre and oddly scaled, and a plain 7
    reads as a 2 or a 3. The portal's in-browser tier does the same thing
    (website ``src/apps/classifiers/infer.ts`` ``preprocessDigit``).

    Args:
        image: A single-channel (``"L"``) PIL image of any size.

    Returns:
        A ``(28, 28)`` float32 array in ``[0, 255]``. A blank image is just
        resized, as before.
    """
    full = np.asarray(image, dtype=np.float32)
    ink = np.where(full >= INK_THRESHOLD, full, 0.0)
    ys, xs = np.nonzero(ink)
    if ys.size == 0:
        return np.asarray(image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS), dtype=np.float32)
    crop = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    scale = DIGIT_BOX / max(h, w)
    th, tw = max(1, round(h * scale)), max(1, round(w * scale))
    digit = Image.fromarray(crop, mode="F").resize((tw, th), Image.LANCZOS)
    scaled = np.clip(np.asarray(digit, dtype=np.float32), 0.0, 255.0)
    mass = float(scaled.sum())
    cy = float((scaled.sum(axis=1) * np.arange(th)).sum()) / mass
    cx = float((scaled.sum(axis=0) * np.arange(tw)).sum()) / mass
    oy = round((IMG_SIZE - 1) / 2 - cy)
    ox = round((IMG_SIZE - 1) / 2 - cx)
    out = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(IMG_SIZE, oy + th), min(IMG_SIZE, ox + tw)
    out[y0:y1, x0:x1] = scaled[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    return out


class MNISTPlugin(DatasetPlugin):
    """Plugin for the MNIST handwritten-digit dataset.

    * 10 classes (digits 0–9)
    * 28 × 28 single-channel (grayscale) images
    * Three model architectures: CNN, Linear, SVM
    """

    name = "mnist"
    display_name = "MNIST Handwritten Digits"
    input_type = "image"
    num_classes = 10
    class_labels = [str(i) for i in range(10)]
    image_size = (28, 28)
    image_channels = 1
    feature_names = None

    def get_train_loader(self, batch_size: int) -> DataLoader:
        """Load the MNIST training set with standard normalisation.

        Downloads the dataset on first call if not already cached.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ]
        )
        train_data = datasets.MNIST(
            str(Path(__file__).resolve().parent.parent.parent / "data"),
            train=True,
            download=True,
            transform=transform,
        )
        train_subset = Subset(train_data, range(55_000))
        return DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    def get_val_loader(self, batch_size: int) -> DataLoader:
        """Hold out the last 5,000 training samples as a validation set.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ]
        )
        full_train = datasets.MNIST(
            str(Path(__file__).resolve().parent.parent.parent / "data"),
            train=True,
            download=True,
            transform=transform,
        )
        val_subset = Subset(full_train, range(55_000, 60_000))
        return DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Load the MNIST test set with standard normalisation.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ]
        )
        test_data = datasets.MNIST(
            str(Path(__file__).resolve().parent.parent.parent / "data"),
            train=False,
            download=True,
            transform=transform,
        )
        return DataLoader(test_data, batch_size=batch_size, shuffle=False)

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """Convert a PIL image to a normalised MNIST-compatible tensor.

        Steps:

        1. Convert to single-channel grayscale.
        2. Crop to the ink, fit it into 20 × 20 and centre it by mass in a
           28 × 28 frame (:func:`center_digit`) — MNIST's own normalisation.
        3. Scale from [0, 255] to [0.0, 1.0].
        4. Add batch and channel dimensions → ``(1, 1, 28, 28)``.
        5. Z-score normalise with :data:`MNIST_MEAN` and :data:`MNIST_STD`.

        Args:
            raw_input: A :class:`PIL.Image.Image`.

        Returns:
            Float32 tensor of shape ``(1, 1, 28, 28)``.
        """
        image: Image.Image = raw_input
        arr = center_digit(image.convert("L")) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return (tensor - MNIST_MEAN) / MNIST_STD

    def get_model_types(self) -> dict[str, type[BaseModel]]:
        """Return MNIST-compatible architectures.

        Always includes CNN, Linear, SVM, Quadratic, Polynomial.
        Conditionally includes Qiskit models if ``qiskit`` is installed.
        """
        from .models import (
            LinearNet,
            MNISTNet,
            MNISTPolynomialNet,
            MNISTQuadraticNet,
            SVMNet,
        )

        types: dict[str, type[BaseModel]] = {
            "CNN": MNISTNet,
            "Linear": LinearNet,
            "SVM": SVMNet,
            "Quadratic": MNISTQuadraticNet,
            "Polynomial": MNISTPolynomialNet,
        }
        try:
            from .models import QiskitCNN, QiskitLinear

            types["Qiskit-CNN"] = QiskitCNN
            types["Qiskit-Linear"] = QiskitLinear
        except ImportError:
            pass
        return types
