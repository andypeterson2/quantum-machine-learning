"""Gradient correctness for the Qiskit layer, against real Qiskit.

The rest of the Qiskit tests run on stubs and check shapes, which is why two
wrong gradients survived: the input gradients came back reversed, and the
measured quantity was each qubit's *share of the ones* rather than its own
probability, which the parameter-shift rule does not apply to.

These tests use an exact statevector executor — no shot noise — so the
parameter-shift gradients can be compared with central finite differences and
the comparison is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")

from classifiers.qiskit_layers import (  # noqa: E402
    _ExampleCircuit,
    _IndependentInterpret,
    _RunCircuit,
)

N_QUBITS = 3
EPS = 1e-4


class _ExactExecutor:
    """Exact outcome probabilities, shaped like the sampler's counts."""

    def run(self, qc) -> np.ndarray:
        from qiskit.quantum_info import Statevector

        probs = Statevector.from_instruction(
            qc.remove_final_measurements(inplace=False)
        ).probabilities_dict()
        scale = 1 << 20
        counts = {key + " " + "0" * len(key): p * scale for key, p in probs.items()}
        return _IndependentInterpret()(counts)


@pytest.fixture
def circuit():
    return _ExampleCircuit(N_QUBITS, executor=_ExactExecutor())


def _run(circuit, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(circuit.run(w.tolist(), x.tolist()), dtype=torch.float64)


def _finite_difference(circuit, w, x, g, *, index: int, wrt: str) -> float:
    """Central difference of ``g · f`` with respect to one w or x component."""
    target = w if wrt == "w" else x
    bump = torch.zeros_like(target, dtype=torch.float64)
    bump[index] = EPS
    w64, x64 = w.double(), x.double()
    if wrt == "w":
        plus, minus = _run(circuit, w64 + bump, x64), _run(circuit, w64 - bump, x64)
    else:
        plus, minus = _run(circuit, w64, x64 + bump), _run(circuit, w64, x64 - bump)
    return float((plus - minus) @ g.double() / (2 * EPS))


@pytest.fixture
def autograd(circuit):
    """Backward once; return the circuit, the gradients, and the inputs used."""
    torch.manual_seed(0)
    w = torch.randn(len(circuit.params), requires_grad=True)
    x = torch.randn(1, N_QUBITS, requires_grad=True)
    g = torch.tensor([1.0, -2.0, 0.5])
    (_RunCircuit.apply(circuit, w, x)[0] @ g).backward()
    return {"circuit": circuit, "w": w, "x": x, "g": g}


def test_input_gradients_match_finite_differences(autograd) -> None:
    """Each input gradient belongs to the input it was computed for.

    The reversed columns this pins down were invisible to a shape check, and
    wrong-order gradients still train — just not toward the loss.
    """
    expected = [
        _finite_difference(
            autograd["circuit"], autograd["w"], autograd["x"][0], autograd["g"],
            index=k, wrt="x",
        )
        for k in range(N_QUBITS)
    ]
    assert autograd["x"].grad[0].tolist() == pytest.approx(expected, abs=1e-3)


def test_weight_gradients_match_finite_differences(autograd) -> None:
    expected = [
        _finite_difference(
            autograd["circuit"], autograd["w"], autograd["x"][0], autograd["g"],
            index=k, wrt="w",
        )
        for k in range(autograd["w"].numel())
    ]
    assert autograd["w"].grad.tolist() == pytest.approx(expected, abs=1e-3)


def test_input_gradients_are_not_symmetric(circuit) -> None:
    """Guard for the test above: these gradients differ per input, so a
    reversed vector cannot pass by accident."""
    torch.manual_seed(0)
    w = torch.randn(len(circuit.params), requires_grad=True)
    x = torch.randn(1, N_QUBITS, requires_grad=True)
    (_RunCircuit.apply(circuit, w, x)[0] @ torch.tensor([1.0, -2.0, 0.5])).backward()
    grad = x.grad[0]
    assert not torch.allclose(grad, grad.flip(0), atol=1e-3)


class TestMeasuredQuantity:
    """The layer measures per-qubit probabilities, in qubit order."""

    def test_probabilities_are_per_qubit_not_shares(self) -> None:
        # Qubit 0 is 1 in both outcomes; qubit 2 in neither.
        out = _IndependentInterpret()({"001": 250, "011": 750})
        assert out == pytest.approx([1.0, 0.75, 0.0])

    def test_bitstrings_are_read_in_qubit_order(self) -> None:
        """Qiskit prints qubit 0 on the right."""
        out = _IndependentInterpret()({"100": 1000})
        assert out == pytest.approx([0.0, 0.0, 1.0])

    def test_no_shots_gives_zeros(self) -> None:
        assert _IndependentInterpret()({"000": 1024}) == pytest.approx([0.0, 0.0, 0.0])

    def test_exact_executor_matches_statevector(self, circuit) -> None:
        """Sanity check on the test's own executor: with all-zero weights the
        circuit is RX(x) per qubit, so P(qubit i == 1) = sin²(x_i / 2)."""
        x = torch.tensor([0.3, -1.1, 2.0])
        got = _run(circuit, torch.zeros(len(circuit.params)), x)
        assert got.tolist() == pytest.approx(torch.sin(x / 2).pow(2).tolist(), abs=1e-6)
