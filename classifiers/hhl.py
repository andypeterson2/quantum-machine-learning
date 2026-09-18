"""The paper's optimized HHL circuit, and the analysis that reads it.

Yang, Awan & Vall-Llosera, "SVM on NISQ Computers" (arXiv:1909.11988), Fig. 10:
a 4-qubit circuit that solves ``F α = y`` for the fixed
``F = [[1, 0.5], [0.5, 1]]`` the preprocessing unit pins every dataset onto, and
the readout that turns its measured distribution back into ``α``.

Three places need this: the notebook that recreates the paper,
``tools/hardware_run.py`` which runs the same circuit on IBM hardware, and the
exporter that ships the resulting rule. It used to exist three times — the tool
carried a comment reading "any change there must be mirrored here" — and the
copies had already drifted.

Qiskit is imported lazily, so this module can be imported anywhere the rest of
the package can, including the notebook's own venv, which has no torch.
"""

from __future__ import annotations

import numpy as np

#: The paper's published IBMQX2 numbers (its own yardstick).
PAPER_REFERENCE = {
    "ibmqx2_optimized_depth7_djs": 0.130,
    "ibmqx2_baseline_depth20_djs": 0.603,
    "note": (
        "the 0.603 baseline is a depth-20 unoptimized circuit this repo does "
        "not build; only the optimized circuit is compared like-for-like"
    ),
}

#: Shots per run, as in the paper and the notebook.
DEFAULT_SHOTS = 8192

#: Counts give only |amplitude|; alpha's sign pattern comes from the ideal solution.
ALPHA_SIGN_NOTE = (
    "alpha magnitudes are measured (sqrt of P(0001), P(0011)); the (+, -) sign "
    "pattern is taken from the ideal solution F^-1 y, not measured"
)


def h_theta(theta: float):
    """The paper's H(θ) gate: ``[[cos2θ, sin2θ], [sin2θ, −cos2θ]]``."""
    from qiskit.circuit.library import UnitaryGate

    matrix = np.array(
        [
            [np.cos(2 * theta), np.sin(2 * theta)],
            [np.sin(2 * theta), -np.cos(2 * theta)],
        ]
    )
    return UnitaryGate(matrix, label=f"H(pi/{round(np.pi / theta)})")


def build_hhl(*, measure: bool = False, barriers: bool = False):
    """The optimized HHL circuit for ``F=[[1,0.5],[0.5,1]]``, ``y=(1,−1)/√2``.

    Qubits 0..3 are the paper's q1..q4: q1q2 the eigenvalue register, q3 the
    solution, q4 the ancilla. Depth 8 as built; the paper counts 7 after
    cancelling two X gates.

    Part A does phase estimation in the eigenbasis frame (where |1> = |−>): q3
    takes |y>, q2 the half-bit both eigenvalues share, and q1 the ones-bit
    computed from q3, leaving |01> for lambda 0.5 and |11> for 1.5. Part B
    inverts the eigenvalue onto the ancilla with the paper's H(θ) gate, and
    part C undoes the estimation.

    Args:
        measure:  Append a measurement of all four qubits.
        barriers: Mark the paper's three parts (A, B, C), for drawing.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4, 4 if measure else 0)
    # Part A: phase estimation.
    qc.x(2)
    qc.x(1)
    qc.cx(2, 0)
    qc.x(0)
    if barriers:
        qc.barrier(label="A")
    # Part B: eigenvalue inversion onto the ancilla.
    qc.append(h_theta(np.pi / 8).control(1, ctrl_state=0), [0, 3])
    qc.append(h_theta(np.pi / 10).control(1, ctrl_state=1), [0, 3])
    if barriers:
        qc.barrier(label="B")
    # Part C: inverse phase estimation.
    qc.x(0)
    qc.cx(2, 0)
    qc.x(1)
    qc.h(2)
    if measure:
        qc.measure(range(4), range(4))
    return qc


def paper_key(qiskit_key: str) -> str:
    """Qiskit's little-endian ``c3c2c1c0`` bitstring to the paper's |q1q2q3q4⟩."""
    return qiskit_key[::-1]


def ideal_probs() -> dict[str, float]:
    """Exact ``|amplitude|²`` of the unmeasured circuit, paper-keyed."""
    from qiskit.quantum_info import Statevector

    state = Statevector.from_instruction(build_hhl())
    return {format(i, "04b")[::-1]: float(p) for i, p in enumerate(np.abs(state.data) ** 2)}


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2, so it lands in ``[0, 1]`` (Eqs. 32-33)."""

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        return float(
            np.sum(np.where(x > 0, x * np.log2(np.maximum(x, 1e-12) / np.maximum(y, 1e-12)), 0))
        )

    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def alpha_from_counts(probs: dict[str, float]) -> list[float]:
    """The paper's shot readout: ``α = (√P(0001), −√P(0011))``, paper-keyed.

    The magnitudes are measured; the signs are the ideal solution's (see
    :data:`ALPHA_SIGN_NOTE`).
    """
    return [float(np.sqrt(probs.get("0001", 0.0))), float(-np.sqrt(probs.get("0011", 0.0)))]


def analyse(counts: dict[str, int], shots: int) -> dict:
    """D_JS against the ideal distribution, the α readout, and P(q4=1)."""
    probs = {paper_key(k): v / shots for k, v in counts.items()}
    ideal = ideal_probs()
    states = sorted(ideal)
    p_ideal = np.array([ideal.get(s, 0.0) for s in states])
    p_measured = np.array([probs.get(s, 0.0) for s in states])
    return {
        "js_divergence_vs_ideal": round(js_divergence(p_ideal, p_measured), 4),
        "alpha": [round(a, 8) for a in alpha_from_counts(probs)],
        "p_q4_success": round(sum(p for s, p in probs.items() if s.endswith("1")), 4),
    }
