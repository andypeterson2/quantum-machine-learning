"""MNIST handwritten-digit dataset plugin.

Encapsulates all MNIST-specific knowledge: normalisation constants, image
dimensions, data loading via ``torchvision.datasets.MNIST``, canvas-image
preprocessing, and the set of compatible model architectures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.utils.data import Subset

from classifiers.base_model import BaseModel
from classifiers.dataset_plugin import DatasetPlugin

#: Per-channel mean computed over the MNIST training set.
MNIST_MEAN: float = 0.1307
#: Per-channel standard deviation computed over the MNIST training set.
MNIST_STD: float = 0.3081
#: Side length (pixels) of MNIST images.
IMG_SIZE: int = 28


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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])
        train_data = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def get_val_loader(self, batch_size: int) -> DataLoader:
        """Hold out the last 5,000 training samples as a validation set.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])
        full_train = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        val_subset = Subset(full_train, range(55_000, 60_000))
        return DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Load the MNIST test set with standard normalisation.

        Args:
            batch_size: Number of samples per mini-batch.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])
        test_data = datasets.MNIST(
            "./data", train=False, download=True, transform=transform
        )
        return DataLoader(test_data, batch_size=batch_size, shuffle=False)

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """Convert a PIL image to a normalised MNIST-compatible tensor.

        Steps:

        1. Convert to single-channel grayscale.
        2. Resize to 28 × 28 with Lanczos resampling.
        3. Scale from [0, 255] to [0.0, 1.0].
        4. Add batch and channel dimensions → ``(1, 1, 28, 28)``.
        5. Z-score normalise with :data:`MNIST_MEAN` and :data:`MNIST_STD`.

        Args:
            raw_input: A :class:`PIL.Image.Image`.

        Returns:
            Float32 tensor of shape ``(1, 1, 28, 28)``.
        """
        image: Image.Image = raw_input
        img = image.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return (tensor - MNIST_MEAN) / MNIST_STD

    def get_model_types(self) -> dict[str, type[BaseModel]]:
        """Return MNIST-compatible architectures.

        Always includes CNN, Linear, SVM, Quadratic, Polynomial.
        Conditionally includes Qiskit models if ``qiskit`` is installed.
        """
        from .models import (
            MNISTNet, LinearNet, SVMNet, MNISTQuadraticNet, MNISTPolynomialNet,
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
