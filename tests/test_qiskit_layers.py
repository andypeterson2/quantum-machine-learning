"""Unit tests for classifiers.qiskit_layers (Qiskit quantum circuit layer).

All Qiskit dependencies are mocked so the tests run without Qiskit installed.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Mock Qiskit modules before importing the module under test
# ---------------------------------------------------------------------------

def _install_qiskit_mocks():
    """Insert fake qiskit / qiskit_aer modules into sys.modules."""
    qiskit_mod = ModuleType("qiskit")
    qiskit_circuit = ModuleType("qiskit.circuit")
    qiskit_aer = ModuleType("qiskit_aer")

    # ParameterVector: behaves like a list of symbolic placeholders.
    # Must be hashable since it's used as a dict key in assign_parameters.
    class FakeParameterVector(list):
        def __init__(self, name, length):
            super().__init__(range(length))
            self.name = name
            self._id = id(self)

        def __hash__(self):
            return self._id

        def __eq__(self, other):
            return self is other

    # QuantumCircuit stub — records gate calls; assign_parameters returns self.
    class FakeQuantumCircuit:
        def __init__(self, *args):
            self._num_qubits = args[0] if args else 0

        def rx(self, *a, **kw): ...
        def rxx(self, *a, **kw): ...
        def rzz(self, *a, **kw): ...
        def barrier(self, *a, **kw): ...
        def measure_all(self, *a, **kw): ...

        def assign_parameters(self, mapping, *, inplace=False, flat_input=False):
            return FakeQuantumCircuit(self._num_qubits)

    qiskit_mod.QuantumCircuit = FakeQuantumCircuit
    qiskit_mod.transpile = lambda qc, backend: qc
    qiskit_circuit.ParameterVector = FakeParameterVector

    # Aer stub
    class FakeBackend:
        def run(self, qc, shots=1024):
            return self

        def result(self):
            return self

        def get_counts(self):
            return {"000": 500, "111": 500}

    class FakeAer:
        @staticmethod
        def get_backend(name):
            return FakeBackend()

    qiskit_aer.Aer = FakeAer

    for name, mod in [
        ("qiskit", qiskit_mod),
        ("qiskit.circuit", qiskit_circuit),
        ("qiskit_aer", qiskit_aer),
    ]:
        sys.modules[name] = mod

    return FakeParameterVector, FakeQuantumCircuit, FakeBackend


_FakeParameterVector, _FakeQuantumCircuit, _FakeBackend = _install_qiskit_mocks()

from classifiers.qiskit_layers import (  # noqa: E402
    _ExampleCircuit,
    _Head,
    _IndependentInterpret,
    _ParametricCircuit,
    _QCExecutor,
    _QCSampler,
    _RunCircuit,
    QiskitQLayer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubExecutor(_QCExecutor):
    """Deterministic executor that returns a fixed array instead of running
    a real quantum circuit.  The output dimension matches *input_dim*."""

    def __init__(self, output_dim: int):
        self.output_dim = output_dim

    def run(self, qc) -> np.ndarray:
        return np.full(self.output_dim, 0.5, dtype=np.float32)


class _CountingExecutor(_QCExecutor):
    """Executor that counts how many times ``run`` is called."""

    def __init__(self, output_dim: int):
        self.output_dim = output_dim
        self.call_count = 0

    def run(self, qc) -> np.ndarray:
        self.call_count += 1
        rng = np.random.default_rng(self.call_count)
        return rng.random(self.output_dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: _IndependentInterpret
# ---------------------------------------------------------------------------

class TestIndependentInterpret:
    def test_uniform_counts(self):
        interp = _IndependentInterpret()
        counts = {"000": 500, "111": 500}
        result = interp(counts)
        assert result.shape == (3,)
        # bits "111" contribute to all 3 positions
        expected_total = 500 * 3  # only '1' bits counted
        assert np.isclose(result.sum(), 1.0)

    def test_single_outcome(self):
        interp = _IndependentInterpret()
        counts = {"101": 1000}
        result = interp(counts)
        assert result.shape == (3,)
        # bits 0 and 2 are '1', bit 1 is '0'
        assert result[0] > 0
        assert result[1] == 0.0
        assert result[2] > 0
        assert np.isclose(result.sum(), 1.0)

    def test_all_zeros_outcome(self):
        interp = _IndependentInterpret()
        counts = {"000": 1024}
        result = interp(counts)
        assert result.shape == (3,)
        # No '1' bits, total is 0 -> output should be all zeros.
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))

    def test_mixed_outcomes(self):
        interp = _IndependentInterpret()
        counts = {"10": 300, "01": 700}
        result = interp(counts)
        assert result.shape == (2,)
        assert np.isclose(result.sum(), 1.0)
        # bit-0 is '1' in "10" (300 times), bit-1 is '1' in "01" (700 times)
        assert result[0] == pytest.approx(300.0 / 1000.0)
        assert result[1] == pytest.approx(700.0 / 1000.0)


# ---------------------------------------------------------------------------
# Tests: _ParametricCircuit.run (mock-based)
# ---------------------------------------------------------------------------

class TestParametricCircuit:
    def _make_pc(self, input_dim=3, num_params=6):
        executor = _StubExecutor(output_dim=input_dim)

        def builder(params, inputs):
            return _FakeQuantumCircuit(len(inputs))

        return _ParametricCircuit(builder, num_params, input_dim, executor)

    def test_run_returns_correct_shape(self):
        pc = self._make_pc(input_dim=3, num_params=6)
        result = pc.run([0.0] * 6, [0.1, 0.2, 0.3])
        assert result.shape == (3,)

    def test_run_calls_executor(self):
        executor = _CountingExecutor(output_dim=3)

        def builder(params, inputs):
            return _FakeQuantumCircuit(len(inputs))

        pc = _ParametricCircuit(builder, 6, 3, executor)
        pc.run([0.0] * 6, [0.1, 0.2, 0.3])
        assert executor.call_count == 1

    def test_assign_parameters_called(self):
        """Verify that assign_parameters is invoked with the correct keys."""
        executor = _StubExecutor(output_dim=2)

        def builder(params, inputs):
            qc = _FakeQuantumCircuit(len(inputs))
            return qc

        pc = _ParametricCircuit(builder, 4, 2, executor)
        pc.qc.assign_parameters = MagicMock(return_value=_FakeQuantumCircuit(2))
        pc.run([1.0, 2.0, 3.0, 4.0], [0.5, 0.6])
        pc.qc.assign_parameters.assert_called_once()
        call_kwargs = pc.qc.assign_parameters.call_args
        mapping = call_kwargs[0][0]
        assert pc.params in mapping
        assert pc.inputs in mapping


# ---------------------------------------------------------------------------
# Tests: _ExampleCircuit
# ---------------------------------------------------------------------------

class TestExampleCircuit:
    def test_construction(self):
        executor = _StubExecutor(output_dim=3)
        ec = _ExampleCircuit(input_dim=3, executor=executor)
        assert len(ec.params) == 4  # 2 * (input_dim - 1), linear topology
        assert len(ec.inputs) == 3

    def test_run(self):
        executor = _StubExecutor(output_dim=3)
        ec = _ExampleCircuit(input_dim=3, executor=executor)
        result = ec.run([0.0] * 6, [0.1, 0.2, 0.3])
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# Tests: _RunCircuit forward
# ---------------------------------------------------------------------------

class TestRunCircuitForward:
    def _make_pc(self, input_dim=3):
        executor = _StubExecutor(output_dim=input_dim)

        def builder(params, inputs):
            return _FakeQuantumCircuit(len(inputs))

        return _ParametricCircuit(builder, 2 * input_dim, input_dim, executor)

    def test_forward_shape(self):
        pc = self._make_pc(input_dim=3)
        w = torch.zeros(6)
        x = torch.randn(4, 3)
        result = _RunCircuit.apply(pc, w, x)
        assert result.shape == (4, 3)

    def test_forward_values(self):
        """StubExecutor returns 0.5 for every element."""
        pc = self._make_pc(input_dim=2)
        w = torch.zeros(4)
        x = torch.randn(2, 2)
        result = _RunCircuit.apply(pc, w, x)
        expected = torch.full((2, 2), 0.5)
        assert torch.allclose(result, expected)

    def test_forward_single_sample(self):
        pc = self._make_pc(input_dim=3)
        w = torch.zeros(6)
        x = torch.randn(1, 3)
        result = _RunCircuit.apply(pc, w, x)
        assert result.shape == (1, 3)


# ---------------------------------------------------------------------------
# Tests: _RunCircuit backward (finite-difference gradient)
# ---------------------------------------------------------------------------

class TestRunCircuitBackward:
    def test_gradient_shapes(self):
        """Backward pass should produce gradients with correct shapes for w
        and x_batch."""
        input_dim = 3
        num_params = 2 * input_dim
        executor = _CountingExecutor(output_dim=input_dim)

        def builder(params, inputs):
            return _FakeQuantumCircuit(len(inputs))

        pc = _ParametricCircuit(builder, num_params, input_dim, executor)

        w = torch.zeros(num_params, requires_grad=True)
        x = torch.randn(1, input_dim, requires_grad=True)

        result = _RunCircuit.apply(pc, w, x)
        loss = result.sum()
        loss.backward()

        assert w.grad is not None
        assert w.grad.shape == (num_params,)
        assert x.grad is not None
        assert x.grad.shape == (1, input_dim)

    def test_gradient_nonzero(self):
        """With a non-constant executor the finite-difference gradients should
        generally be non-zero."""
        input_dim = 2
        num_params = 4
        executor = _CountingExecutor(output_dim=input_dim)

        def builder(params, inputs):
            return _FakeQuantumCircuit(len(inputs))

        pc = _ParametricCircuit(builder, num_params, input_dim, executor)

        w = torch.zeros(num_params, requires_grad=True)
        x = torch.randn(1, input_dim, requires_grad=True)

        result = _RunCircuit.apply(pc, w, x)
        loss = result.sum()
        loss.backward()

        # At least one gradient element should be non-zero.
        assert w.grad.abs().sum() > 0 or x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Tests: _Head
# ---------------------------------------------------------------------------

class TestHead:
    def _make_head(self, input_dim=3):
        executor = _StubExecutor(output_dim=input_dim)

        def builder(params, inputs):
            return _FakeQuantumCircuit(len(inputs))

        pc = _ParametricCircuit(builder, 2 * input_dim, input_dim, executor)
        return _Head(pc)

    def test_forward_shape(self):
        head = self._make_head(input_dim=3)
        x = torch.randn(4, 3)
        out = head(x)
        assert out.shape == (4, 3)

    def test_weights_initialized_zero(self):
        head = self._make_head(input_dim=3)
        assert torch.allclose(head.w, torch.zeros(6))

    def test_is_nn_module(self):
        head = self._make_head()
        assert isinstance(head, nn.Module)

    def test_parameters_registered(self):
        head = self._make_head(input_dim=2)
        param_names = [n for n, _ in head.named_parameters()]
        assert "w" in param_names


# ---------------------------------------------------------------------------
# Tests: QiskitQLayer
# ---------------------------------------------------------------------------

class TestQiskitQLayer:
    def _make_layer(self, input_dim=3, num_heads=1):
        """Build a QiskitQLayer with all Qiskit internals replaced by stubs."""
        executor = _StubExecutor(output_dim=input_dim)

        def builder(params, inputs):
            return _FakeQuantumCircuit(len(inputs))

        pc = _ParametricCircuit(builder, 2 * input_dim, input_dim, executor)

        # Manually construct the layer to bypass _check_qiskit / _ExampleCircuit.
        layer = QiskitQLayer.__new__(QiskitQLayer)
        nn.Module.__init__(layer)
        layer.heads = nn.ModuleList([_Head(pc) for _ in range(num_heads)])
        num_outputs = input_dim
        layer.reduce = (
            nn.Linear(num_heads * num_outputs, num_outputs)
            if num_heads > 1
            else nn.Identity()
        )
        return layer

    def test_single_head_forward_shape(self):
        layer = self._make_layer(input_dim=3, num_heads=1)
        x = torch.randn(4, 3)
        out = layer(x)
        assert out.shape == (4, 3)

    def test_multi_head_forward_shape(self):
        layer = self._make_layer(input_dim=3, num_heads=2)
        x = torch.randn(4, 3)
        out = layer(x)
        assert out.shape == (4, 3)

    def test_single_head_uses_identity_reduce(self):
        layer = self._make_layer(input_dim=3, num_heads=1)
        assert isinstance(layer.reduce, nn.Identity)

    def test_multi_head_uses_linear_reduce(self):
        layer = self._make_layer(input_dim=3, num_heads=2)
        assert isinstance(layer.reduce, nn.Linear)

    def test_is_nn_module(self):
        layer = self._make_layer()
        assert isinstance(layer, nn.Module)

    def test_deterministic_eval(self):
        layer = self._make_layer(input_dim=2, num_heads=1)
        layer.eval()
        x = torch.randn(2, 2)
        with torch.no_grad():
            a = layer(x)
            b = layer(x)
        assert torch.allclose(a, b)
