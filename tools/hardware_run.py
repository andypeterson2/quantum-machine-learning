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
    python tools/hardware_run.py exposure [ARTIFACT ...]  # record the solution qubit's timing

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


def pending_path(backend_name: str) -> Path:
    """Where one backend's submitted job ids wait for ``fetch``.

    One file per backend: a series submits to several before the first clears,
    and a shared file would overwrite the ids of jobs already charged against
    the month's quota — losing results that cannot be got back without paying
    for them twice.
    """
    return OUT_DIR / f"pending-{backend_name}.json"

#: Fixed so a repeat run gets the same circuit and the same physical qubits;
#: unseeded, a series could not tell a drifting chip from a different corner.
TRANSPILER_SEED = 1909

#: Transpiled circuits, kept so a later run resubmits the same object; a
#: recalibration otherwise changes the circuit underneath a comparison.
ISA_DIR = REPO_ROOT / "exports" / "hardware" / "isa"

#: The circuit's four qubits in virtual order (the paper's q1..q4); the readout
#: splits on the solution qubit and the ancilla, so a branch-only bias needs them.
QUBIT_ROLES = ("eigenvalue_q1", "eigenvalue_q2", "solution_q3", "ancilla_q4")

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


def build_probes(backend, layout: list[int], delay_seconds: float) -> list:
    """Two circuits that measure what could be producing the readout bias.

    The HHL readout takes alpha from two amplitudes on the solution qubit, and
    every raw run so far has read the ``|1>`` one low. Two mechanisms would do
    that: the measurement misreporting ``|1>`` as ``|0>``, and the state
    decaying while the circuit runs. These separate them.

    ``readout`` prepares ``|1>`` on the solution qubit and measures at once, so
    its flip rate is the readout asymmetry plus whatever decays inside the
    measurement window. ``decay`` waits *delay_seconds* first, so its flip rate
    adds the idle loss over that time. For the two to add up to the circuit's
    own exposure, *delay_seconds* has to be the circuit's pre-measurement time
    alone: its measurement window is already in both probes. Both go in the
    same job as the circuit, on the same physical qubits, so they share one
    calibration exactly, which is the part a separate job could not give.

    Args:
        backend:       Backend to lay the probes out on.
        layout:        Physical qubits, in the circuit's virtual order.
        delay_seconds: How long the circuit's operations run before its
            measurement starts, from :func:`exposure_split`.

    Returns:
        ``[readout, decay]``, transpiled onto *layout*.
    """
    from qiskit import QuantumCircuit, transpile

    solution = QUBIT_ROLES.index("solution_q3")
    probes = []
    for delay in (None, delay_seconds):
        qc = QuantumCircuit(len(layout), len(layout))
        qc.x(solution)
        if delay:
            qc.barrier()
            qc.delay(delay, solution, unit="s")
        qc.barrier()
        qc.measure(range(len(layout)), range(len(layout)))
        # optimization_level=0: the point is to run exactly this, and a higher
        # level would cancel the X against the measurement.
        probes.append(
            transpile(qc, backend=backend, optimization_level=0, initial_layout=layout)
        )
    return probes


#: Instructions that take no device time: ``rz`` is a frame change.
_ZERO_LENGTH = frozenset({"rz", "barrier"})

#: Time units a ``Delay`` may carry, in seconds.
_UNIT_SECONDS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}


def _instruction_seconds(target, name: str, qubits: list[int]) -> float:
    """One instruction's length on these physical qubits, 0.0 when untimed."""
    if name in _ZERO_LENGTH:
        return 0.0
    for ordered in (tuple(qubits), tuple(reversed(qubits))):
        try:
            seconds = target[name][ordered].duration
        except (KeyError, TypeError):
            continue
        if seconds is not None:
            return float(seconds)
    return 0.0


def exposure_split(circuit, target, solution_qubit: int) -> dict:
    """How long the solution qubit spends in the circuit and in its measurement.

    The two probes report flip rates per unit of exposure, so turning them into
    a prediction for the circuit needs the circuit's exposure split the same
    way: the time its operations run, and the measurement window after them.
    ``estimate_duration`` gives only their sum. This walks the circuit with the
    target's instruction lengths, one clock per qubit, and reads off when the
    operations end and when the solution qubit's last pulse ends.

    Args:
        circuit:        A transpiled circuit whose qubit indices are physical.
        target:         The backend's ``Target``, for instruction lengths.
        solution_qubit: Physical index of the solution qubit.

    Returns:
        ``circuit_seconds`` (operations, up to the first measurement),
        ``solution_busy_seconds`` (when the solution qubit's last pulse ends),
        ``measure_seconds`` (the solution qubit's measurement window) and
        ``total_seconds`` (their sum, which is what the decay probe's delay was
        set to on runs before this field existed).

    Raises:
        ValueError: If the target has no measurement length for the qubit, in
            which case no split exists.
    """
    clocks: dict[int, float] = {}
    busy: dict[int, float] = {}
    for inst in circuit.data:
        name = inst.operation.name
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        if name == "measure":
            continue
        start = max((clocks.get(q, 0.0) for q in qubits), default=0.0)
        end = start + _instruction_seconds(target, name, qubits)
        for q in qubits:
            clocks[q] = end
            if name not in _ZERO_LENGTH:
                busy[q] = end
    measure = _instruction_seconds(target, "measure", [solution_qubit])
    if measure <= 0.0:
        raise ValueError(f"target has no measurement length for qubit {solution_qubit}")
    circuit_seconds = max(clocks.values(), default=0.0)
    return {
        "circuit_seconds": circuit_seconds,
        "solution_busy_seconds": busy.get(solution_qubit, 0.0),
        "measure_seconds": measure,
        "total_seconds": circuit_seconds + measure,
    }


def probe_delay_seconds(probe, dt: float | None) -> float | None:
    """The wait the decay probe actually carries, read off its own ``Delay``.

    Args:
        probe: The decay probe circuit, as stored.
        dt:    The backend's sample time, for a delay expressed in ``dt``.

    Returns:
        The delay in seconds, or ``None`` if the circuit has no delay.
    """
    for inst in probe.data:
        if inst.operation.name != "delay":
            continue
        unit = inst.operation.unit
        if unit == "dt":
            return None if dt is None else float(inst.operation.duration) * dt
        return float(inst.operation.duration) * _UNIT_SECONDS[unit]
    return None


def delay_covers(delay_seconds: float | None, circuit_seconds: float | None) -> str | None:
    """What the decay probe's wait stood in for, so a reader can scale its rate.

    Args:
        delay_seconds:   The probe's wait.
        circuit_seconds: The circuit's pre-measurement time.

    Returns:
        ``"circuit"`` when the wait matches the operations alone,
        ``"circuit and measurement window"`` when it is longer, ``None`` when
        either input is unknown.
    """
    if delay_seconds is None or circuit_seconds is None:
        return None
    if delay_seconds > circuit_seconds * 1.05:
        return "circuit and measurement window"
    return "circuit"


def isa_path(backend_name: str, layout: list[int] | None) -> Path:
    """Where this backend-and-layout's transpiled circuits live."""
    suffix = "-".join(str(q) for q in layout) if layout else "default"
    return ISA_DIR / f"{backend_name}-{suffix}.qpy"


def load_or_build_isa(backend, layout: list[int] | None, seed: int) -> tuple:
    """The circuits to submit, reused from disk when this layout has run before.

    Returns:
        ``(circuits, digest, reused)`` — the HHL circuit followed by its two
        probes, the sha256 of the stored file, and whether it came from disk.
    """
    import hashlib
    import io

    from qiskit import qpy, transpile

    path = isa_path(backend.name, layout)
    if path.is_file():
        with path.open("rb") as handle:
            circuits = list(qpy.load(handle))
        return circuits, hashlib.sha256(path.read_bytes()).hexdigest(), True

    hhl = transpile(
        build_hhl(measure=True),
        backend=backend,
        optimization_level=3,
        initial_layout=layout,
        seed_transpiler=seed,
    )
    qubits = layout_qubits(hhl)
    delay = exposure_split(hhl, backend.target, qubits[QUBIT_ROLES.index("solution_q3")])
    circuits = [hhl, *build_probes(backend, qubits, delay["circuit_seconds"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    qpy.dump(circuits, buffer)
    path.write_bytes(buffer.getvalue())
    return circuits, hashlib.sha256(path.read_bytes()).hexdigest(), False


def _duration(transpiled, backend) -> float:
    """The circuit's wall-clock length in seconds, 0.0 if it cannot be timed."""
    try:
        return float(transpiled.estimate_duration(backend.target, unit="s"))
    except Exception:  # an untimed probe is better than no run at all
        logger.warning("could not estimate the circuit duration; the decay probe will not wait")
        return 0.0


def qubit_roles(qubits: list[int]) -> dict[str, int]:
    """Which physical qubit took each of the circuit's four roles.

    Args:
        qubits: Physical indices in virtual order, from :func:`layout_qubits`.

    Returns:
        Role name to physical qubit; empty when the circuit was never laid out.
    """
    return dict(zip(QUBIT_ROLES, qubits, strict=True)) if qubits else {}


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
    # Two runs on one calibration measure the chip twice; across one they
    # measure drift as well, and the stamp is what tells them apart.
    properties = backend.properties() if hasattr(backend, "properties") else None
    updated = getattr(properties, "last_update_date", None)
    if updated is not None:
        snapshot["last_update_date"] = str(updated)
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
    snapshot["edges"] = _edge_errors(target, pairs)
    return snapshot


def _edge_errors(target, pairs: list[list[int]]) -> dict[str, float]:
    """Two-qubit gate error for each edge the circuit used."""
    errors: dict[str, float] = {}
    for name in ("ecr", "cz", "cx"):
        if name not in getattr(target, "operation_names", ()):
            continue
        for pair in pairs:
            # The target keys one direction of each edge, whichever suits the
            # backend, so take the one that is there.
            for ordered in (tuple(pair), tuple(reversed(pair))):
                if ordered in target[name]:
                    errors[f"{name}:{ordered[0]}_{ordered[1]}"] = target[name][ordered].error
                    break
    return errors


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


def _exposure_or_empty(transpiled, backend, qubits: list[int]) -> dict:
    """The solution qubit's timing split, or ``{}`` when the target cannot give it."""
    if not qubits:
        return {}
    try:
        return exposure_split(
            transpiled, backend.target, qubits[QUBIT_ROLES.index("solution_q3")]
        )
    except (ValueError, AttributeError) as exc:
        logger.warning("could not split the circuit's exposure: %s", exc)
        return {}


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
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()  # saved account or IBM_QUANTUM_TOKEN
    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(operational=True, simulator=False)
    )
    logger.info("backend: %s (%d qubits)", backend.name, backend.num_qubits)

    circuits, isa_digest, reused = load_or_build_isa(backend, initial_layout, seed_transpiler)
    transpiled = circuits[0]
    logger.info("ISA circuits %s (sha256 %s)", "reused" if reused else "built", isa_digest[:12])
    two_qubit = sum(1 for inst in transpiled.data if inst.operation.num_qubits == 2)
    depth = transpiled.depth()
    qubits = layout_qubits(transpiled)
    pairs = two_qubit_pairs(transpiled)
    logger.info(
        "transpiled: depth=%d, two-qubit gates=%d, qubits=%s", depth, two_qubit, qubits
    )
    creg_names = [cr.name for cr in transpiled.cregs]
    exposure = _exposure_or_empty(transpiled, backend, qubits)
    probe_delay = probe_delay_seconds(circuits[2], getattr(backend.target, "dt", None))

    jobs = {}
    for label in ("raw", "mitigated"):
        job = _sampler(backend, shots=shots, mitigated=label == "mitigated").run(circuits)
        jobs[label] = job.job_id()
        logger.info("%s job submitted: %s", label, job.job_id())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pending = pending_path(backend.name)
    pending.write_text(
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
                    "qubit_roles": qubit_roles(qubits),
                    "two_qubit_pairs": pairs,
                    "isa_sha256": isa_digest,
                    "isa_reused": reused,
                    "duration_seconds": _duration(transpiled, backend),
                    "exposure": exposure,
                },
                "calibration": calibration_snapshot(backend, qubits, pairs),
                "creg_names": creg_names,
                "probe_delay_seconds": probe_delay,
                "jobs": jobs,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("pending run recorded in %s — run `fetch` once the queue clears", pending)


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


def fetch(backend_name: str | None = None) -> None:
    """Retrieve every pending backend's jobs, or one named backend's.

    Args:
        backend_name: Fetch only this backend's pending run; all of them when
            omitted. A backend still queued is left alone rather than failing
            the others.
    """
    paths = (
        [pending_path(backend_name)]
        if backend_name
        else sorted(OUT_DIR.glob("pending-*.json"))
    )
    if not paths:
        raise FileNotFoundError(f"no pending run in {OUT_DIR}; submit one first")
    for path in paths:
        _fetch_one(path)


def _probe_rates(result, pending: dict) -> dict:
    """How often the solution qubit, prepared in ``|1>``, read back ``|0>``.

    The two probes ran in the same job as the circuit, so these rates belong to
    the same calibration as the readout they are meant to explain. ``readout``
    is the measurement's own asymmetry; ``decay`` adds the idle loss over the
    circuit's duration. If ``decay`` accounts for the circuit's ratio excess,
    the bias is a property of the qubit rather than of the gates.
    """
    solution = QUBIT_ROLES.index("solution_q3")
    rates = {}
    for name, index in (("readout", 1), ("decay", 2)):
        counts = _counts(result[index], pending["creg_names"])
        total = sum(counts.values())
        flipped = sum(
            n for key, n in counts.items() if paper_key(key)[solution] == "0"
        )
        rates[name] = {
            "prepared": "1",
            "read_zero": round(flipped / total, 6) if total else None,
            "shots": total,
        }
    delay = pending.get("probe_delay_seconds")
    if delay is None:
        delay = pending["transpiled"].get("duration_seconds")
    rates["decay"]["delay_seconds"] = delay
    rates["decay"]["delay_covers"] = delay_covers(
        delay, pending["transpiled"].get("exposure", {}).get("circuit_seconds")
    )
    return rates


def _artifact_path(backend: str, date: str, shots: int) -> Path:
    """A free filename for one run's record.

    The name carried only the backend and the date, so a second run of the same
    backend on one day overwrote the first — which is a measurement that cost
    quota and cannot be taken again. The shot count usually separates them;
    a counter covers the rest.
    """
    stem = f"hhl-{backend}-{date}-{shots}shots"
    candidate = OUT_DIR / f"{stem}.json"
    serial = 2
    while candidate.exists():
        candidate = OUT_DIR / f"{stem}-{serial}.json"
        serial += 1
    return candidate


def _fetch_one(path: Path) -> None:
    """Retrieve one pending backend's jobs and write its committed artifact."""
    import qiskit
    from qiskit_ibm_runtime import QiskitRuntimeService

    pending = json.loads(path.read_text())
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
        if len(result) >= 3:
            entry["probes"] = _probe_rates(result, pending)
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
    out = _artifact_path(pending["backend"], payload["provenance"]["exported_at"], shots)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    path.unlink()
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


def record_exposure(paths: list[Path]) -> None:
    """Add the solution qubit's timing split to artifacts written without it.

    Reads each artifact's stored ISA circuit back from ``exports/hardware/isa``
    and the instruction lengths from the live target; no job is submitted. The
    runs that first carried probes set the decay probe's wait to the circuit's
    whole length, measurement included, and their artifacts said only
    ``delay_seconds``. With the split beside it, the probe's rate can be scaled
    to the circuit's exposure offline, and each decay probe is labelled with
    what its wait covered. Counts and rates are left as they are.

    Args:
        paths: Artifacts to update in place.

    Raises:
        FileNotFoundError: If an artifact's ISA circuit is no longer stored.
    """
    from datetime import datetime, timezone

    from qiskit import qpy
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    for path in paths:
        payload = json.loads(path.read_text())
        transpiled = payload["transpiled"]
        qubits = transpiled.get("physical_qubits") or []
        if not qubits:
            logger.warning("%s: no layout on record, cannot split its exposure", path.name)
            continue
        stored = isa_path(payload["backend"], qubits)
        if not stored.is_file():
            raise FileNotFoundError(f"{path.name}: its ISA circuit {stored.name} is gone")
        with stored.open("rb") as handle:
            circuits = list(qpy.load(handle))
        target = service.backend(payload["backend"]).target
        exposure = exposure_split(circuits[0], target, qubits[QUBIT_ROLES.index("solution_q3")])
        exposure["durations_read_on"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        transpiled["exposure"] = exposure
        delay = probe_delay_seconds(circuits[2], target.dt) if len(circuits) >= 3 else None
        for job in payload["jobs"].values():
            decay = job.get("probes", {}).get("decay")
            if decay is None:
                continue
            if delay is not None:
                decay["delay_seconds"] = delay
            decay["delay_covers"] = delay_covers(
                decay["delay_seconds"], exposure["circuit_seconds"]
            )
        path.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info(
            "%s: circuit %.3f us + measure %.3f us; decay probe waited %.3f us (%s)",
            path.name,
            exposure["circuit_seconds"] * 1e6,
            exposure["measure_seconds"] * 1e6,
            (delay or 0.0) * 1e6,
            delay_covers(delay, exposure["circuit_seconds"]),
        )


def probe_artifacts() -> list[Path]:
    """Every committed artifact whose jobs carried the two probes."""
    return [
        path
        for path in sorted(OUT_DIR.glob("hhl-*.json"))
        if any("probes" in job for job in json.loads(path.read_text())["jobs"].values())
    ]


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
    p_fetch = sub.add_parser("fetch", help="retrieve the pending jobs and write the artifact")
    p_fetch.add_argument("--backend", default=None, help="default: every pending backend")
    p_rescore = sub.add_parser("rescore", help="re-score qsvm_accuracy offline (no jobs)")
    p_rescore.add_argument("artifact", nargs="?", type=Path, help="default: the newest one")
    p_exposure = sub.add_parser(
        "exposure", help="record the solution qubit's timing split (reads the target, no jobs)"
    )
    p_exposure.add_argument(
        "artifacts", nargs="*", type=Path, help="default: every artifact that carried probes"
    )
    args = parser.parse_args()
    if args.command == "submit":
        layout = (
            [int(q) for q in args.initial_layout.split(",")] if args.initial_layout else None
        )
        submit(args.backend, args.shots, layout, args.seed_transpiler)
    elif args.command == "fetch":
        fetch(args.backend)
    elif args.command == "exposure":
        record_exposure(args.artifacts or probe_artifacts())
    else:
        rescore(args.artifact or latest_artifact())


if __name__ == "__main__":
    main()
