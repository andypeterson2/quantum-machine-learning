"""Qiskit-based trainable quantum circuit layer for PyTorch.

This module consolidates the quantum integration code from the
Digit-Classifier research codebase into a single file.  All Qiskit
imports are contained here — the rest of the classifiers package never
imports Qiskit directly, so the dependency is optional.

Public API:

* :class:`QiskitQLayer` — multi-headed trainable parametric quantum
  circuit layer that plugs into any ``nn.Module`` / ``BaseModel``.

Usage::

    from classifiers.qiskit_layers import QiskitQLayer

    layer = QiskitQLayer(input_dim=3, num_heads=1)
    out = layer(x_batch)  # (N, input_dim)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

# Dependency check

def _check_qiskit() -> None:
    """Raise a clear ``ImportError`` if Qiskit is not installed."""
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "Qiskit models require the 'qiskit' and 'qiskit-aer' packages. "
            "Install with:  pip install qiskit qiskit-aer"
        ) from err


# Executor (circuit runner)

class _QCExecutor(ABC):
    """Abstract base for quantum circuit execution strategies."""

    @abstractmethod
    def run(self, qc: QuantumCircuit) -> np.ndarray:  # noqa: F821
        ...


class _IndependentInterpret:
    """Per-qubit probability of measuring 1: ``output[i] = P(qubit i == 1)``.

    Two things this has to get right, because the layer's gradients depend on
    them. The counts are divided by the shot total, not by the number of ones,
    so each entry is that qubit's own probability rather than its share of the
    ones (a share is not linear in the state, which the parameter-shift rule in
    :class:`_RunCircuit` requires). And Qiskit's bitstrings are printed with the
    highest qubit on the left, so they are read right-to-left to keep
    ``output[i]`` on qubit ``i``.
    """

    def __call__(self, counts: dict[str, int]) -> np.ndarray:
        output_dim = len(next(iter(counts)).split(" ")[0])
        output = np.zeros(output_dim, dtype=np.float32)
        shots = 0
        for outcome, freq in counts.items():
            bits = outcome.split(" ")[0][::-1]  # qubit 0 first
            shots += freq
            for qubit, bit in enumerate(bits):
                if bit == "1":
                    output[qubit] += freq
        return output / shots if shots > 0 else output


class _QCSampler(_QCExecutor):
    """Run a quantum circuit by sampling with the Aer QASM simulator.

    The simulator seed is drawn from torch's generator when the sampler is
    built, so a run seeded through
    :func:`~classifiers.seeding.seed_everything` (as
    :class:`~classifiers.trainer.Trainer` does) repeats exactly, while an
    unseeded run still samples freshly. Pass *seed* to fix it directly.
    """

    def __init__(self, shots: int = 2**13, seed: int | None = None) -> None:
        _check_qiskit()
        from qiskit_aer import Aer

        self.backend = Aer.get_backend("qasm_simulator")
        self.interpret = _IndependentInterpret()
        self.shots = shots
        self.seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if seed is None else seed

    def run(self, qc: QuantumCircuit, shots: int | None = None) -> np.ndarray:  # noqa: F821
        from qiskit import transpile

        shots = shots or self.shots
        compiled = transpile(qc, self.backend)
        result = self.backend.run(compiled, shots=shots, seed_simulator=self.seed).result()
        counts = result.get_counts()
        return self.interpret(counts)


# Parametric circuit

class _ParametricCircuit:
    """A quantum circuit with reassignable parameter values."""

    def __init__(
        self,
        qc_builder: Callable,
        num_params: int,
        input_dim: int,
        executor: _QCExecutor,
    ) -> None:
        from qiskit.circuit import ParameterVector

        self.params = ParameterVector("params", num_params)
        self.inputs = ParameterVector("inputs", input_dim)
        self.qc = qc_builder(self.params, self.inputs)
        self.executor = executor

    def run(self, w: list[float], x: list[float]) -> np.ndarray:
        bound = self.qc.assign_parameters(
            {self.params: w, self.inputs: x}, inplace=False, flat_input=False
        )
        return self.executor.run(bound)


class _ExampleCircuit(_ParametricCircuit):
    """3-qubit parametric circuit with RX encoding + RXX/RZZ entanglement.

    Uses a hardware-efficient ansatz with linear entanglement topology
    to minimize circuit depth while maintaining expressibility.

    Gate count: ``n`` RX (encoding) + ``2*(n-1)`` entangling gates
    (reduced from ``2*n`` by removing the redundant wrap-around connection).
    """

    def __init__(self, input_dim: int, executor: _QCExecutor | None = None) -> None:
        if executor is None:
            executor = _QCSampler()
        # Linear topology: 2*(n-1) entangling params instead of 2*n
        num_params = 2 * (input_dim - 1) if input_dim > 1 else 0
        super().__init__(_ExampleCircuit._builder, num_params, input_dim, executor)

    @staticmethod
    def _builder(params, inputs):
        from qiskit import QuantumCircuit

        n = len(inputs)
        qc = QuantumCircuit(n, n)
        # Feature encoding via RX rotations
        for i in range(n):
            qc.rx(inputs[i], i)
        qc.barrier()
        # Linear entanglement (no cyclic wrap) — reduces depth by 1 layer
        for i in range(n - 1):
            qc.rxx(params[2 * i], i, i + 1)
            qc.rzz(params[2 * i + 1], i, i + 1)
        qc.barrier()
        qc.measure_all()
        return qc


# Autograd bridge

class _RunCircuit(Function):
    """Custom autograd Function: forward runs the circuit, backward computes
    parameter-shift gradients for both the weights and the inputs."""

    @staticmethod
    def forward(ctx, pc: _ParametricCircuit, w: torch.Tensor, x_batch: torch.Tensor):
        ctx.pc = pc
        w_list = w.tolist()
        values = [pc.run(w_list, x_batch[s].tolist()) for s in range(len(x_batch))]
        result = torch.tensor(np.array(values), dtype=torch.float32)
        ctx.save_for_backward(result, w, x_batch)
        return result

    @staticmethod
    def _estimate_partial(
        f: Callable, v: torch.Tensor, pos: int, delta: float = np.pi / 2
    ) -> torch.Tensor:
        """Partial derivative by the parameter-shift rule.

        For a Pauli rotation gate the derivative of an expectation value is
        exactly ``df/dθ = [f(θ + π/2) − f(θ − π/2)] / 2``. It holds for every
        parameter here — the inputs enter through RX, the weights through
        RXX/RZZ — and for the measured quantity, because
        :class:`_IndependentInterpret` returns per-qubit probabilities, which
        are expectation values of projectors. With a sampling executor each
        evaluation carries shot noise, so the estimate is unbiased rather than
        exact.
        """
        e = F.one_hot(torch.tensor([pos]), num_classes=v.shape[-1]).flatten().float()
        fv_plus = f(v + delta * e)
        fv_minus = f(v - delta * e)
        return torch.tensor(fv_plus - fv_minus, dtype=torch.float32) / 2

    @staticmethod
    def backward(ctx, grad_output):
        _, w, x_batch = ctx.saved_tensors
        w_list = w.tolist()

        batch_df_dw, batch_df_dx = [], []
        for j in range(len(x_batch)):
            x = x_batch[j]
            x_list = x.tolist()
            g = grad_output[j]

            df_dw = []
            for k in range(w.shape[-1]):
                df_dw_k = _RunCircuit._estimate_partial(
                    # Bind x_list per iteration — the lambda is consumed
                    # immediately, but the explicit default makes that safe.
                    f=lambda ww, x_list=x_list: ctx.pc.run(ww.tolist(), x_list),
                    v=w,
                    pos=k,
                )
                df_dw.append(torch.dot(df_dw_k, g))
            batch_df_dw.append(df_dw)

            df_dx = []
            for k in range(x.shape[-1]):
                df_dx_k = _RunCircuit._estimate_partial(
                    f=lambda xx: ctx.pc.run(w_list, xx.tolist()), v=x, pos=k
                )
                df_dx.append(torch.dot(df_dx_k, g))
            batch_df_dx.append(df_dx)

        # df_dx[k] is the derivative with respect to x[k] — the shift was applied
        # at position k — so the columns are already in input order.
        batch_df_dw = torch.tensor(batch_df_dw).sum(dim=0)
        batch_df_dx = torch.tensor(batch_df_dx)
        return None, batch_df_dw, batch_df_dx


# Public layer

class _Head(nn.Module):
    """Single-headed trainable parametric circuit."""

    def __init__(self, pc: _ParametricCircuit) -> None:
        super().__init__()
        self.pc = pc
        self.w = nn.Parameter(torch.zeros(len(pc.params)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RunCircuit.apply(self.pc, self.w, x)


class QiskitQLayer(nn.Module):
    """Multi-headed trainable quantum circuit layer (Qiskit backend).

    Each head independently trains the same parametric circuit architecture
    with its own set of weights.  Head outputs are concatenated and optionally
    reduced via a learned linear projection.

    Args:
        input_dim: Number of qubits / input features.
        num_heads: Number of independent circuit heads (default 1).
    """

    def __init__(self, input_dim: int, num_heads: int = 1) -> None:
        _check_qiskit()
        super().__init__()
        self.heads = nn.ModuleList([_Head(_ExampleCircuit(input_dim)) for _ in range(num_heads)])
        # Probe the output dimension.
        num_outputs = self.heads[0](torch.zeros((1, input_dim))).shape[1]
        self.reduce: nn.Module | Callable = (
            nn.Linear(num_heads * num_outputs, num_outputs)
            if num_heads > 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [head(x) for head in self.heads]
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.reduce(x)
