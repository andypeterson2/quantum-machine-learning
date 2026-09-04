"""Run the paper-recreation HHL circuit on real IBM Quantum hardware.

The notebook (``notebooks/qsvm-iris/``) rebuilds Yang, Awan & Vall-Llosera's
optimized 4-qubit HHL circuit (arXiv:1909.11988) and measures its output
distribution on Aer — including under a noise model standing in for the
retired IBMQX2 device the paper ran on in 2019. This tool closes that loop:
it executes the same circuit on a current IBM backend and computes the
paper's own yardstick — the Jensen–Shannon divergence between the ideal and
measured distributions — against the paper's 2019 number (D_JS = 0.130 for
the optimized depth-7 circuit on IBMQX2).

Two jobs are submitted: **raw** (no error mitigation — closest to 2019
conditions) and **mitigated** (dynamical decoupling + Pauli twirling — what
the 2025 stack adds in software). Results are cached as a committed artifact
under ``exports/hardware/`` (spend once, show forever); the notebook's final
section renders the comparison from the cache and stays green without it.

Credentials: the saved qiskit account (``~/.qiskit/qiskit-ibm.json``) or the
``IBM_QUANTUM_TOKEN`` env var. This tool never prints or stores the token.

Usage::

    python tools/hardware_run.py submit [--backend NAME] [--shots 8192]
    python tools/hardware_run.py fetch   # poll the pending jobs, write artifact

Honest-comparison caveat, recorded in the artifact: the paper's 0.603
baseline was a *depth-20 unoptimized* HHL this repo does not build; only the
optimized circuit's 0.130 is compared like-for-like.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from classifiers.web_export import provenance_base  # noqa: E402

logger = logging.getLogger("hardware_run")

OUT_DIR = REPO_ROOT / "exports" / "hardware"
PENDING = OUT_DIR / "pending.json"

#: The paper's published IBMQX2 numbers (its own yardstick).
PAPER_REFERENCE = {
    "ibmqx2_optimized_depth7_djs": 0.130,
    "ibmqx2_baseline_depth20_djs": 0.603,
    "note": (
        "the 0.603 baseline is a depth-20 unoptimized circuit this repo does "
        "not build; only the optimized circuit is compared like-for-like"
    ),
}

DEFAULT_SHOTS = 8192  # as in the paper and the notebook


# ── The circuit (duplicated from notebook cell 10 — the notebook is not an
#    importable module; any change there must be mirrored here) ───────────────


def _h_theta(theta: float):
    """Paper's H(θ) gate: [[cos2θ, sin2θ], [sin2θ, −cos2θ]]."""
    from qiskit.circuit.library import UnitaryGate

    m = np.array(
        [
            [np.cos(2 * theta), np.sin(2 * theta)],
            [np.sin(2 * theta), -np.cos(2 * theta)],
        ]
    )
    return UnitaryGate(m, label=f"H(pi/{round(np.pi / theta)})")


def build_hhl(*, measure: bool = False):
    """Optimized HHL for F=[[1,0.5],[0.5,1]], y=(1,−1)/√2 (paper Fig. 10)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4, 4 if measure else 0)
    qc.x(2)
    qc.x(1)
    qc.cx(2, 0)
    qc.x(0)
    qc.append(_h_theta(np.pi / 8).control(1, ctrl_state=0), [0, 3])
    qc.append(_h_theta(np.pi / 10).control(1, ctrl_state=1), [0, 3])
    qc.x(0)
    qc.cx(2, 0)
    qc.x(1)
    qc.h(2)
    if measure:
        qc.measure(range(4), range(4))
    return qc


# ── Analysis helpers (mirroring notebook cell 19) ─────────────────────────────


def paper_key(qiskit_key: str) -> str:
    """Qiskit's little-endian ``c3c2c1c0`` bitstring → the paper's |q1q2q3q4⟩."""
    return qiskit_key[::-1]


def ideal_probs() -> dict[str, float]:
    """Exact |amplitude|² of the unmeasured circuit, paper-keyed."""
    from qiskit.quantum_info import Statevector

    sv = Statevector.from_instruction(build_hhl())
    return {format(i, "04b")[::-1]: float(p) for i, p in enumerate(np.abs(sv.data) ** 2)}


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence, base 2 (paper Eqs. 32–33)."""

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        return float(
            np.sum(np.where(x > 0, x * np.log2(np.maximum(x, 1e-12) / np.maximum(y, 1e-12)), 0))
        )

    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def alpha_from_counts(probs: dict[str, float]) -> list[float]:
    """The paper's shot readout: α = (√P(0001), −√P(0011)), paper-keyed."""
    return [float(np.sqrt(probs.get("0001", 0.0))), float(-np.sqrt(probs.get("0011", 0.0)))]


def analyse(counts: dict[str, int], shots: int) -> dict:
    """D_JS against the ideal distribution + the α readout, for one job."""
    probs = {paper_key(k): v / shots for k, v in counts.items()}
    ideal = ideal_probs()
    states = sorted(ideal)
    p_ideal = np.array([ideal.get(s, 0.0) for s in states])
    p_meas = np.array([probs.get(s, 0.0) for s in states])
    return {
        "js_divergence_vs_ideal": round(js_divergence(p_ideal, p_meas), 4),
        "alpha": [round(a, 8) for a in alpha_from_counts(probs)],
        "p_q4_success": round(sum(p for s, p in probs.items() if s.endswith("1")), 4),
    }


def qsvm_accuracies(alpha: list[float]) -> dict[str, float]:
    """Re-measure the deployed QSVM rules with a hardware-derived α.

    Uses the exporter's own derivation (weight vector from α, per-dataset
    Eq. 24 maps re-solved) over every spec-table dataset whose features are
    available locally (MNIST needs the openml cache and is skipped without it).
    """
    from classifiers import qsvm_export

    w = qsvm_export.weight_vector(np.array(alpha))
    out: dict[str, float] = {}
    for name, spec in qsvm_export.QSVM_DATASETS.items():
        try:
            feats, labels = spec["features_fn"]()
        except Exception as exc:
            logger.info("skipping %s accuracy (%s)", name, exc)
            continue
        t1 = feats[labels == 1].mean(axis=0)
        t2 = feats[labels == -1].mean(axis=0)
        a, b = qsvm_export.solve_map(t1, t2, *spec["cd"])
        mapping = {"a": a, "b": b, "c": spec["cd"][0], "d": spec["cd"][1]}
        out[name] = round(float((qsvm_export.decide(w, mapping, feats) == labels).mean()), 4)
    return out


# ── Submission ────────────────────────────────────────────────────────────────


def _sampler(backend, *, shots: int, mitigated: bool):
    """A SamplerV2 configured raw or with DD + twirling (nonogram's recipe)."""
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(backend)
    sampler.options.default_shots = shots
    if mitigated:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
    else:
        # Explicitly raw: 2019 conditions, no runtime-era error suppression.
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
    return sampler


def submit(backend_name: str | None, shots: int) -> None:
    """Transpile once, submit the raw + mitigated jobs, record ids."""
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()  # saved account or IBM_QUANTUM_TOKEN
    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(operational=True, simulator=False)
    )
    logger.info("backend: %s (%d qubits)", backend.name, backend.num_qubits)

    transpiled = transpile(build_hhl(measure=True), backend=backend, optimization_level=3)
    two_qubit = sum(1 for inst in transpiled.data if inst.operation.num_qubits == 2)
    depth = transpiled.depth()
    logger.info("transpiled: depth=%d, two-qubit gates=%d", depth, two_qubit)
    creg_names = [cr.name for cr in transpiled.cregs]

    jobs = {}
    for label in ("raw", "mitigated"):
        job = _sampler(backend, shots=shots, mitigated=label == "mitigated").run([transpiled])
        jobs[label] = job.job_id()
        logger.info("%s job submitted: %s", label, job.job_id())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(
        json.dumps(
            {
                "backend": backend.name,
                "shots": shots,
                "transpiled": {
                    "depth": depth,
                    "two_qubit_gates": two_qubit,
                    "optimization_level": 3,
                },
                "creg_names": creg_names,
                "jobs": jobs,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("pending run recorded in %s — run `fetch` once the queue clears", PENDING)


# ── Retrieval ─────────────────────────────────────────────────────────────────


def _counts(result, creg_names: list[str]) -> dict[str, int]:
    """Counts from a PubResult DataBin — register name first, then _fields."""
    data = result.data
    for name in creg_names:
        bit_array = getattr(data, name, None)
        if bit_array is not None and hasattr(bit_array, "get_counts"):
            return dict(bit_array.get_counts())
    for name in getattr(data, "_fields", []):
        bit_array = getattr(data, name, None)
        if bit_array is not None and hasattr(bit_array, "get_counts"):
            return dict(bit_array.get_counts())
    raise RuntimeError("could not locate a BitArray with get_counts() on the result DataBin")


def fetch() -> None:
    """Retrieve the pending jobs and write the committed artifact."""
    import qiskit
    from qiskit_ibm_runtime import QiskitRuntimeService

    pending = json.loads(PENDING.read_text())
    service = QiskitRuntimeService()
    shots = pending["shots"]

    payload: dict = {
        "kind": "hardware-run",
        "circuit": "optimized 4-qubit HHL (paper Fig. 10, depth 8 as built)",
        "paper": "arXiv:1909.11988",
        "backend": pending["backend"],
        "shots": shots,
        "transpiled": pending["transpiled"],
        "paper_reference": PAPER_REFERENCE,
        "jobs": {},
    }

    for label, job_id in pending["jobs"].items():
        job = service.job(job_id)
        status = str(job.status())
        logger.info("%s job %s: %s", label, job_id, status)
        result = job.result()  # blocks if still running
        counts = _counts(result[0], pending["creg_names"])
        entry = {"job_id": job_id, "counts": {paper_key(k): v for k, v in counts.items()}}
        entry.update(analyse(counts, shots))
        entry["qsvm_accuracy"] = qsvm_accuracies(entry["alpha"])
        payload["jobs"][label] = entry

    payload["ideal_probs"] = {k: round(v, 6) for k, v in ideal_probs().items()}
    runtime_version = importlib.metadata.version("qiskit-ibm-runtime")
    payload["provenance"] = provenance_base(
        {
            "model": "HHL",
            "paper": "arXiv:1909.11988",
            "derivation": (
                "notebook cell 10 circuit executed on real hardware; "
                "see tools/hardware_run.py"
            ),
        },
        {"qiskit": qiskit.__version__, "qiskit-ibm-runtime": runtime_version},
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"hhl-{pending['backend']}-{payload['provenance']['exported_at']}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    PENDING.unlink()
    for label, entry in payload["jobs"].items():
        logger.info(
            "%s: D_JS=%.4f  alpha=(%.4f, %.4f)  qsvm=%s",
            label,
            entry["js_divergence_vs_ideal"],
            entry["alpha"][0],
            entry["alpha"][1],
            entry["qsvm_accuracy"],
        )
    logger.info("artifact written: %s", out)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_submit = sub.add_parser("submit", help="transpile + submit the raw/mitigated job pair")
    p_submit.add_argument("--backend", default=None, help="backend name (default: least busy)")
    p_submit.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    sub.add_parser("fetch", help="retrieve the pending jobs and write the artifact")
    args = parser.parse_args()
    if args.command == "submit":
        submit(args.backend, args.shots)
    else:
        fetch()


if __name__ == "__main__":
    main()
