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

The circuit and the readout live in :mod:`classifiers.hhl`, which the notebook
imports too, so there is one definition rather than three.

Usage::

    python tools/hardware_run.py submit [--backend NAME] [--shots 8192]
    python tools/hardware_run.py fetch   # poll the pending jobs, write artifact
    python tools/hardware_run.py rescore [ARTIFACT]  # offline: re-score qsvm_accuracy

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

from classifiers.hhl import (  # noqa: E402
    ALPHA_SIGN_NOTE,
    DEFAULT_SHOTS,
    PAPER_REFERENCE,
    analyse,
    build_hhl,
    ideal_probs,
    paper_key,
)
from classifiers.web_export import provenance_base  # noqa: E402

logger = logging.getLogger("hardware_run")

OUT_DIR = REPO_ROOT / "exports" / "hardware"
PENDING = OUT_DIR / "pending.json"

#: Fixed so a repeat run gets the same circuit and the same physical qubits;
#: unseeded, a series could not tell a drifting chip from a different corner.
TRANSPILER_SEED = 1909

#: How ``qsvm_accuracy`` is scored, recorded beside it.
QSVM_ACCURACY_PROTOCOL = (
    "held-out: Eq. 24 map fit on each dataset's fit split, rule scored on its "
    "held-out split (classifiers.qsvm_export.fit_and_score), as in exports/web/qsvm-*.json"
)


def qsvm_accuracies(alpha: list[float]) -> dict[str, float]:
    """Held-out accuracy of every deployed QSVM rule under a hardware-derived α.

    Scored by the exporter's own :func:`~classifiers.qsvm_export.fit_and_score`
    (map fit on the fit split, accuracy on the held-out split), so the numbers
    are comparable with the committed exports' ``test_accuracy``. Only a dataset
    that cannot be fetched (MNIST needs openml) is skipped; any other failure
    raises rather than leaving a silently empty result.
    """
    from classifiers import qsvm_export

    out: dict[str, float] = {}
    for name in qsvm_export.QSVM_DATASETS:
        try:
            fit = qsvm_export.fit_and_score(name, np.array(alpha))
        except OSError as exc:
            logger.warning("skipping %s accuracy — dataset unavailable (%s)", name, exc)
            continue
        out[name] = round(fit.accuracy, 4)
    return out


# Submission


def layout_qubits(transpiled) -> list[int]:
    """The physical qubits the circuit's four virtual ones landed on.

    Args:
        transpiled: A transpiled circuit carrying a ``TranspileLayout``.

    Returns:
        Physical indices in virtual-qubit order, or ``[]`` if the circuit was
        never laid out (a backendless transpile).
    """
    layout = getattr(transpiled, "layout", None)
    if layout is None:
        return []
    return [int(q) for q in layout.final_index_layout(filter_ancillas=True)]


def two_qubit_pairs(transpiled) -> list[list[int]]:
    """The physical qubit pairs the two-qubit gates act on, deduplicated."""
    pairs = {
        tuple(sorted(transpiled.find_bit(q).index for q in inst.qubits))
        for inst in transpiled.data
        if inst.operation.num_qubits == 2
    }
    return [list(pair) for pair in sorted(pairs)]


def calibration_snapshot(backend, qubits: list[int], pairs: list[list[int]]) -> dict:
    """Coherence and error figures for the qubits this run actually used.

    Recorded at submission because they move between calibrations: two jobs a
    day apart on one fixed layout differ by whatever this captures, and without
    it a drifting ratio has no candidate explanation.

    Args:
        backend: The backend being submitted to.
        qubits:  Physical qubits from :func:`layout_qubits`.
        pairs:   Physical pairs from :func:`two_qubit_pairs`.

    Returns:
        Per-qubit and per-edge figures; keys are absent where the backend's
        target does not carry them, which is not an error.
    """
    target = getattr(backend, "target", None)
    if target is None:
        return {}
    snapshot: dict = {"qubits": {}, "edges": {}}
    for q in qubits:
        entry: dict = {}
        try:
            props = target.qubit_properties[q]
            entry["t1_seconds"] = props.t1
            entry["t2_seconds"] = props.t2
        except (AttributeError, IndexError, TypeError):
            pass
        measure = target.get("measure") if hasattr(target, "get") else None
        if measure and (q,) in measure:
            entry["readout_error"] = measure[(q,)].error
        snapshot["qubits"][str(q)] = entry
    for name in ("ecr", "cz", "cx"):
        if name not in getattr(target, "operation_names", ()):
            continue
        for pair in pairs:
            # The target keys one direction of each edge; which one is the
            # backend's business, so take whichever is there and stop.
            for ordered in (tuple(pair), tuple(reversed(pair))):
                if ordered in target[name]:
                    key = f"{name}:{ordered[0]}_{ordered[1]}"
                    snapshot["edges"][key] = target[name][ordered].error
                    break
    return snapshot


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


def submit(
    backend_name: str | None,
    shots: int,
    initial_layout: list[int] | None = None,
    seed_transpiler: int = TRANSPILER_SEED,
) -> None:
    """Transpile once, submit the raw + mitigated jobs, record ids.

    Args:
        backend_name:    Backend to run on; the least busy one when omitted.
        shots:           Shots per job.
        initial_layout:  Physical qubits to pin the circuit to. Pass the
            ``physical_qubits`` of an earlier run to repeat it on the same
            corner of the chip, which is what makes two runs comparable.
        seed_transpiler: Fixed so the same inputs give the same circuit.
    """
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()  # saved account or IBM_QUANTUM_TOKEN
    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(operational=True, simulator=False)
    )
    logger.info("backend: %s (%d qubits)", backend.name, backend.num_qubits)

    transpiled = transpile(
        build_hhl(measure=True),
        backend=backend,
        optimization_level=3,
        initial_layout=initial_layout,
        seed_transpiler=seed_transpiler,
    )
    two_qubit = sum(1 for inst in transpiled.data if inst.operation.num_qubits == 2)
    depth = transpiled.depth()
    qubits = layout_qubits(transpiled)
    pairs = two_qubit_pairs(transpiled)
    logger.info(
        "transpiled: depth=%d, two-qubit gates=%d, qubits=%s", depth, two_qubit, qubits
    )
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
                    "seed_transpiler": seed_transpiler,
                    "initial_layout_requested": initial_layout,
                    "physical_qubits": qubits,
                    "two_qubit_pairs": pairs,
                },
                "calibration": calibration_snapshot(backend, qubits, pairs),
                "creg_names": creg_names,
                "jobs": jobs,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("pending run recorded in %s — run `fetch` once the queue clears", PENDING)


# Retrieval


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
        "calibration": pending.get("calibration", {}),
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

    payload["alpha_note"] = ALPHA_SIGN_NOTE
    payload["qsvm_accuracy_provenance"] = _qsvm_accuracy_provenance()
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


def _qsvm_accuracy_provenance() -> dict:
    """Where and how the artifact's ``qsvm_accuracy`` values were scored."""
    return provenance_base(
        {"model": "QSVM", "protocol": QSVM_ACCURACY_PROTOCOL},
        {"numpy": np.__version__, "scikit-learn": importlib.metadata.version("scikit-learn")},
    )


def latest_artifact() -> Path:
    """The newest committed hardware artifact."""
    runs = sorted(OUT_DIR.glob("hhl-*.json"))
    if not runs:
        raise FileNotFoundError(f"no hhl-*.json artifact under {OUT_DIR}")
    return runs[-1]


def rescore(path: Path) -> None:
    """Re-score an artifact's ``qsvm_accuracy`` from its recorded alphas.

    Offline: no account, no jobs. The measured counts, D_JS, alpha and the run's
    own provenance are left as they are; only the derived accuracies and their
    scoring provenance are rewritten.
    """
    payload = json.loads(path.read_text())
    for label, entry in payload["jobs"].items():
        entry["qsvm_accuracy"] = qsvm_accuracies(entry["alpha"])
        logger.info("%s: qsvm=%s", label, entry["qsvm_accuracy"])
    payload["alpha_note"] = ALPHA_SIGN_NOTE
    payload["qsvm_accuracy_provenance"] = _qsvm_accuracy_provenance()
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("artifact rescored: %s", path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_submit = sub.add_parser("submit", help="transpile + submit the raw/mitigated job pair")
    p_submit.add_argument("--backend", default=None, help="backend name (default: least busy)")
    p_submit.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    p_submit.add_argument(
        "--initial-layout",
        default=None,
        help="comma-separated physical qubits, e.g. 29,51,36,28; repeats an earlier run's layout",
    )
    p_submit.add_argument("--seed-transpiler", type=int, default=TRANSPILER_SEED)
    sub.add_parser("fetch", help="retrieve the pending jobs and write the artifact")
    p_rescore = sub.add_parser("rescore", help="re-score qsvm_accuracy offline (no jobs)")
    p_rescore.add_argument("artifact", nargs="?", type=Path, help="default: the newest one")
    args = parser.parse_args()
    if args.command == "submit":
        layout = (
            [int(q) for q in args.initial_layout.split(",")] if args.initial_layout else None
        )
        submit(args.backend, args.shots, layout, args.seed_transpiler)
    elif args.command == "fetch":
        fetch()
    else:
        rescore(args.artifact or latest_artifact())


if __name__ == "__main__":
    main()
