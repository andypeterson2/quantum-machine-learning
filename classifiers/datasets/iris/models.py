"""Iris model architectures: Linear (logistic regression), SVM, and QVC.

All three models accept standardised Iris features of shape ``(N, 4)`` and
return raw class scores (logits) of shape ``(N, 3)``.

The Quantum Variational Classifier (QVC) simulates a 4-qubit parameterised
quantum circuit using PennyLane's ``default.qubit`` statevector backend.
Gradients flow through the circuit via PyTorch backpropagation, so QVC
training uses the same Adam loop as the classical models with no changes to
the shared :class:`~classifiers.trainer.Trainer`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from classifiers.base_model import BaseModel
from classifiers.losses import multi_class_hinge_loss

# ── Quantum circuit constants ─────────────────────────────────────────────────

#: Number of qubits — one per Iris feature (sepal/petal length/width).
_N_QUBITS: int = 4

#: Number of strongly-entangling variational layers.
#: 2 layers × 4 wires × 3 rotation params = 24 trainable parameters.
_N_LAYERS: int = 2


def _build_qvc_layer():
    """Construct and return a PennyLane :class:`~pennylane.qnn.TorchLayer`.

    Deferred to a function so that PennyLane is only imported when
    :class:`IrisQVC` is instantiated — keeping PennyLane an optional
    dependency for users who only need the classical models.

    The variational circuit:

    1. **AngleEmbedding** — encodes the 4 standardised Iris features as
       Y-rotation angles on qubits 0–3.  Standardised features lie roughly
       in ``[−2, 2]``, which maps naturally to qubit rotation angles.
    2. **StronglyEntanglingLayers** — ``_N_LAYERS`` layers of single-qubit
       rotations (RX, RY, RZ) interleaved with CNOT entanglers covering all
       pairs of qubits.
    3. **Measurement** — Pauli-Z expectation values on qubits 0, 1, 2 yield
       three real numbers in ``[−1, 1]``, used directly as class logits.

    Gradients are computed via ``diff_method="backprop"``, which propagates
    through the full PennyLane statevector simulation using PyTorch autograd.

    Returns:
        A :class:`~pennylane.qnn.TorchLayer` with trainable weight tensor of
        shape ``(_N_LAYERS, _N_QUBITS, 3)``.
    """
    import pennylane as qml

    dev = qml.device("default.qubit", wires=_N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        """Parameterised quantum circuit for IrisQVC.

        Args:
            inputs:  Standardised Iris feature vector, shape ``(4,)``.
            weights: Variational rotation parameters, shape
                     ``(_N_LAYERS, _N_QUBITS, 3)``.

        Returns:
            List of three Pauli-Z expectation values: ``[⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩]``.
        """
        # Encode features as rotation angles on qubits 0-3
        qml.AngleEmbedding(inputs, wires=range(_N_QUBITS), rotation="Y")
        # Variational ansatz with full entanglement between all qubit pairs
        qml.StronglyEntanglingLayers(weights, wires=range(_N_QUBITS))
        # Project onto 3 class scores via Pauli-Z measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]

    weight_shapes = {"weights": (_N_LAYERS, _N_QUBITS, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


# ── Classical models ──────────────────────────────────────────────────────────

class IrisLinear(BaseModel):
    """Logistic regression for Iris — 4 input features, 3 output classes.

    Architecture::

        Linear(4→3)   ← raw logits
    """

    name = "Linear"
    description = "Logistic regression (4 features → 3 classes)"

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for input ``(N, 4)`` → ``(N, 3)``."""
        return self.fc(x)


class IrisSVM(BaseModel):
    """Linear SVM for Iris trained with multi-class hinge loss.

    Architecture::

        Linear(4→3)   ← raw scores
    """

    name = "SVM"
    description = "Linear SVM (multi-class hinge loss, 4 → 3)"

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw scores for input ``(N, 4)`` → ``(N, 3)``."""
        return self.fc(x)

    @staticmethod
    def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Delegate to Crammer-Singer multi-class hinge loss."""
        return multi_class_hinge_loss(output, target)


# ── Quantum model ─────────────────────────────────────────────────────────────

class IrisQVC(BaseModel):
    """Quantum Variational Classifier for Iris using PennyLane.

    Simulates a 4-qubit parameterised quantum circuit on PennyLane's
    ``default.qubit`` statevector backend.  Trained end-to-end with PyTorch
    backpropagation — no quantum-specific training loop required.

    Architecture::

        AngleEmbedding(4 features → 4 qubits, rotation=Y)
        StronglyEntanglingLayers(n_layers=2, n_wires=4)
        Measure ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩ → 3 class scores

    **Parameter count:** ``_N_LAYERS × _N_QUBITS × 3 = 2 × 4 × 3 = 24``
    trainable rotation angles.

    **Training tip:** QVC converges well with the Iris plugin's defaults
    (50 epochs, lr=0.01, batch_size=16).  The circuit evaluation is
    performed sample-by-sample so training is slower than the classical
    models, but remains practical for Iris's 120-sample training set.
    """

    name = "QVC"
    description = "Quantum Variational Classifier (4 qubits, 2 layers, PennyLane)"

    def __init__(self) -> None:
        super().__init__()
        self.qlayer = _build_qvc_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class scores for input ``(N, 4)`` → ``(N, 3)``.

        Each sample in the batch is evaluated through the quantum circuit
        independently.  The output stack of Pauli-Z expectation values is
        used directly as logits for cross-entropy loss.

        Args:
            x: Standardised Iris features, shape ``(N, 4)``.

        Returns:
            Class score tensor of shape ``(N, 3)`` with values in
            ``[−1, 1]`` (usable as logits).
        """
        return self.qlayer(x)
