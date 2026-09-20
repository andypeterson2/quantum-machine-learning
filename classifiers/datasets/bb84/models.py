"""BB84 model architectures: Linear (logistic regression), SVM, and QVC.

All three models accept standardised BB84 session features of shape ``(N, 2)``
— (qber, sifted_key_rate), z-scored with training-set statistics — and return
raw class scores (logits) of shape ``(N, 2)`` for (clean, eavesdropped).

The Quantum Variational Classifier (QVC) simulates a 2-qubit parameterised
quantum circuit using PennyLane's ``default.qubit`` statevector backend, with
gradients flowing through the circuit via PyTorch backpropagation — the same
pattern as the Iris QVC, sized for two features.
"""

from __future__ import annotations

import torch
from torch import nn

from classifiers.base_model import BaseModel
from classifiers.losses import multi_class_hinge_loss
from classifiers.qvc import build_qvc_layer

# Quantum circuit constants

#: Number of qubits — one per session feature (qber, sifted_key_rate).
_N_QUBITS: int = 2

#: Number of strongly-entangling variational layers.
#: 2 layers × 2 wires × 3 rotation params = 12 trainable parameters.
_N_LAYERS: int = 2


# Classical models

class BB84Linear(BaseModel):
    """Logistic regression for BB84 — 2 input features, 2 output classes.

    Architecture::

        Linear(2→2)   ← raw logits
    """

    name = "Linear"
    description = "Logistic regression (qber + sifted_key_rate → clean/eavesdropped)"

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for input ``(N, 2)`` → ``(N, 2)``."""
        return self.fc(x)


class BB84SVM(BaseModel):
    """Linear SVM for BB84 trained with multi-class hinge loss.

    Architecture::

        Linear(2→2)   ← raw scores
    """

    name = "SVM"
    description = "Linear SVM (hinge loss, qber + sifted_key_rate → clean/eavesdropped)"

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw scores for input ``(N, 2)`` → ``(N, 2)``."""
        return self.fc(x)

    @staticmethod
    def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Delegate to Weston-Watkins multi-class hinge loss."""
        return multi_class_hinge_loss(output, target)


# Quantum model

class BB84QVC(BaseModel):
    """Quantum Variational Classifier for BB84 using PennyLane.

    Simulates a 2-qubit parameterised quantum circuit on PennyLane's
    ``default.qubit`` statevector backend, trained end-to-end with PyTorch
    backpropagation. Quantum machine learning classifying the health of a
    quantum-cryptography session.

    **Parameter count:** ``_N_LAYERS × _N_QUBITS × 3 = 2 × 2 × 3 = 12``
    trainable rotation angles.
    """

    name = "QVC"
    description = "Quantum Variational Classifier (2 qubits, 2 layers, PennyLane)"

    def __init__(self) -> None:
        super().__init__()
        self.qlayer = build_qvc_layer(_N_QUBITS, _N_LAYERS, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class scores for input ``(N, 2)`` → ``(N, 2)``.

        Args:
            x: Standardised session features, shape ``(N, 2)``.

        Returns:
            Class score tensor of shape ``(N, 2)`` with values in ``[−1, 1]``,
            used directly as logits — see :class:`~classifiers.datasets.iris.
            models.IrisQVC` on what that bound costs in confidence.
        """
        return self.qlayer(x)
