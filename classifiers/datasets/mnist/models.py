"""MNIST model architectures.

Includes classical models (CNN, Linear, SVM), feature-expansion models
(Quadratic, Polynomial), and optional Qiskit quantum-hybrid models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from classifiers.base_model import BaseModel
from classifiers.layers import Quadratic, Polynomial
from classifiers.losses import multi_class_hinge_loss


# ── CNN ───────────────────────────────────────────────────────────────────────


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


# ── Linear (logistic regression) ─────────────────────────────────────────────


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


# ── SVM (hinge loss) ─────────────────────────────────────────────────────────


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
        """Delegate to Crammer-Singer multi-class hinge loss."""
        return multi_class_hinge_loss(output, target)


# ── Quadratic ────────────────────────────────────────────────────────────────


class MNISTQuadraticNet(BaseModel):
    """CNN with a quadratic feature-expansion layer.

    Architecture::

        Conv2d(1→6, k=5) → ReLU → MaxPool2d(2)
        Conv2d(6→16, k=5) → ReLU → MaxPool2d(2)
        Flatten → Linear(256→120) → ReLU
        Linear(120→32) → ReLU
        Quadratic(32→16) → ReLU
        Linear(16→10)
    """

    name = "Quadratic"
    description = "CNN + quadratic expansion layer"

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 32)
        self.quad = Quadratic(32, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.quad(x))
        return self.fc5(x)


# ── Polynomial ───────────────────────────────────────────────────────────────


class MNISTPolynomialNet(BaseModel):
    """CNN with polynomial (log-linear-exp) layers.

    Architecture::

        Conv2d(1→6, k=5) → ReLU → MaxPool2d(2)
        Conv2d(6→16, k=5) → ReLU → MaxPool2d(2)
        Flatten → Linear(256→120) → ReLU
        Polynomial(120→84) → ReLU
        Linear(84→32) → ReLU
        Polynomial(32→16) → ReLU
        Linear(16→10)
    """

    name = "Polynomial"
    description = "CNN + polynomial (log-linear-exp) layers"

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.poly1 = Polynomial(120, 84)
        self.fc3 = nn.Linear(84, 32)
        self.poly2 = Polynomial(32, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.poly1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.poly2(x))
        return self.fc5(x)


# ── Qiskit quantum models ───────────────────────────────────────────────────


class QiskitCNN(BaseModel):
    """CNN with a Qiskit quantum circuit layer.

    Architecture::

        Conv2d(1→6, k=5) → ReLU → MaxPool2d(2)
        Conv2d(6→16, k=5) → ReLU → MaxPool2d(2)
        Flatten → Linear(256→120) → ReLU
        Linear(120→84) → ReLU
        Linear(84→10) → Sigmoid(×0.8)
        Linear(10→3) → QiskitQLayer(3)
        Linear(3→10)

    Requires ``qiskit`` and ``qiskit-aer``.
    """

    name = "Qiskit-CNN"
    description = "CNN + Qiskit quantum circuit layer"

    def __init__(self) -> None:
        super().__init__()
        from classifiers.qiskit_layers import QiskitQLayer

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 3)
        self.q1 = QiskitQLayer(3)
        self.fc5 = nn.Linear(3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) * 0.8
        x = self.fc4(x)
        x = self.q1(x)
        return self.fc5(x)


class QiskitLinear(BaseModel):
    """Linear model with a Qiskit quantum circuit layer.

    Architecture::

        Flatten → Linear(784→84) → ReLU
        Linear(84→10) → Sigmoid(×0.8)
        Linear(10→3) → QiskitQLayer(3)
        Linear(3→10)

    Requires ``qiskit`` and ``qiskit-aer``.
    """

    name = "Qiskit-Linear"
    description = "Linear + Qiskit quantum circuit layer"

    def __init__(self) -> None:
        super().__init__()
        from classifiers.qiskit_layers import QiskitQLayer

        self.fc1 = nn.Linear(784, 84)
        self.fc2 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(10, 3)
        self.q1 = QiskitQLayer(3)
        self.fc4 = nn.Linear(3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) * 0.8
        x = self.fc3(x)
        x = self.q1(x)
        return self.fc4(x)
