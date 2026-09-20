"""The variational circuit both tabular plugins classify with.

Iris and BB84 run the same quantum classifier at different widths: encode the
standardised features as Y rotations, entangle them through a few strongly
entangling layers, and read Pauli-Z expectation values off the leading qubits as
class scores. Only the qubit count and the number of scores differ.

That agreement was two copies, the way the HHL circuit was three (see
:mod:`classifiers.hhl`). A third tabular dataset would have been a third.

PennyLane is imported lazily, so the module can be imported wherever the rest of
the package can, with or without the optional ``quantum`` extra installed.
"""

from __future__ import annotations

import torch


def build_qvc_layer(n_qubits: int, n_layers: int, n_outputs: int):
    """Return a PennyLane :class:`~pennylane.qnn.TorchLayer` for *n_qubits*.

    The circuit:

    1. **AngleEmbedding** — encodes *n_qubits* standardised features as Y-rotation
       angles. Standardised features lie roughly in ``[-2, 2]``, which maps
       naturally onto rotation angles.
    2. **StronglyEntanglingLayers** — *n_layers* layers of single-qubit rotations
       (RX, RY, RZ) interleaved with CNOT entanglers across all qubit pairs.
    3. **Measurement** — Pauli-Z expectation values on the first *n_outputs*
       qubits, each in ``[-1, 1]`` and used directly as a class score.

    Gradients come from ``diff_method="backprop"``, which propagates through the
    full statevector simulation using PyTorch autograd — so a QVC trains in the
    same Adam loop as the classical models, with no quantum-specific branch in
    :class:`~classifiers.trainer.Trainer`.

    Args:
        n_qubits:  Qubits, one per input feature.
        n_layers:  Strongly-entangling layers.
        n_outputs: Leading qubits to measure, one per class. Must not exceed
                   *n_qubits*.

    Returns:
        A :class:`~pennylane.qnn.TorchLayer` whose trainable weight tensor has
        shape ``(n_layers, n_qubits, 3)``.

    Raises:
        ValueError: If *n_outputs* exceeds *n_qubits*.
    """
    if n_outputs > n_qubits:
        raise ValueError(f"cannot measure {n_outputs} of {n_qubits} qubits")

    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        """Encode *inputs*, entangle, and measure the leading qubits.

        Args:
            inputs:  Standardised feature vector of shape ``(n_qubits,)``.
            weights: Rotation parameters of shape ``(n_layers, n_qubits, 3)``.

        Returns:
            A list of *n_outputs* Pauli-Z expectation values.
        """
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_outputs)]

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})
