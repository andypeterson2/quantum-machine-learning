"""MNIST model architectures: a CNN, a linear model and a hinge-loss SVM."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from classifiers.base_model import BaseModel
from classifiers.losses import multi_class_hinge_loss

# CNN


class MNISTNet(BaseModel):
    """Two-layer convolutional network for MNIST digit classification.

    Architecture::

        Conv2d(1→32, k=3) → ReLU
        Conv2d(32→64, k=3) → ReLU → MaxPool2d(2)
        Flatten
        Linear(9216→128) → ReLU
        Linear(128→10)          ← raw logits
    """

    name = "CNN"
    description = "2-layer CNN with FC head"

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for input ``(N, 1, 28, 28)`` → ``(N, 10)``."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Linear (logistic regression)


class LinearNet(BaseModel):
    """Single linear layer — multinomial logistic regression for MNIST.

    Architecture::

        Flatten → Linear(784→10)
    """

    name = "Linear"
    description = "Logistic regression (single linear layer)"

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for input ``(N, 1, 28, 28)`` → ``(N, 10)``."""
        return self.fc(torch.flatten(x, 1))


# SVM (hinge loss)


class SVMNet(BaseModel):
    """Linear SVM for MNIST trained with multi-class hinge loss.

    Architecture::

        Flatten → Linear(784→10)
    """

    name = "SVM"
    description = "Linear SVM (multi-class hinge loss)"

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw scores for input ``(N, 1, 28, 28)`` → ``(N, 10)``."""
        return self.fc(x.view(x.size(0), -1))

    @staticmethod
    def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Delegate to Weston-Watkins multi-class hinge loss."""
        return multi_class_hinge_loss(output, target)

