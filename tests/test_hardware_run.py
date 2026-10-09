"""Offline checks for ``tools/hardware_run.py`` — no IBM account, no jobs.

``qsvm_accuracies`` once returned ``{}`` for every dataset: a changed return
type raised inside a blanket ``except`` and was logged as a skip. These tests
pin it to the exporter's held-out scoring and to a narrow skip path.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from classifiers import qsvm_export
from classifiers.qsvm_rule import decide, weight_vector
from classifiers.web_export import OUT_DIR, REPO_ROOT

TOOL = REPO_ROOT / "tools" / "hardware_run.py"

HARDWARE_DIR = REPO_ROOT / "exports" / "hardware"


@pytest.fixture(scope="module")
def hardware_run():
    # The image ships only the package, so the docker CI job has no tools/ to test.
    if not TOOL.is_file():
        pytest.skip("tools/ not shipped here")
    spec = importlib.util.spec_from_file_location("hardware_run", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def no_mnist(monkeypatch):
    """Stand in for an absent openml cache with no network (what fetch_openml raises)."""

    def unavailable():
        raise OSError("openml unreachable")

    spec = dict(qsvm_export.QSVM_DATASETS["mnist"], features_fn=unavailable)
    monkeypatch.setitem(qsvm_export.QSVM_DATASETS, "mnist", spec)


def test_scores_match_committed_exports(hardware_run, no_mnist) -> None:
    """With the shipped alpha, the tool reproduces each export's held-out accuracy."""
    got = hardware_run.qsvm_accuracies(qsvm_export.ALPHA_SHOTS.tolist())
    for name in ("iris",):
        payload = json.loads((OUT_DIR / f"qsvm-{name}.json").read_text())
        assert got[name] == payload["test_accuracy"], name


def test_unavailable_dataset_is_skipped(hardware_run, no_mnist) -> None:
    got = hardware_run.qsvm_accuracies([0.5, -0.5])
    assert "mnist" not in got
    assert set(got) == {"iris"}


def test_other_failures_raise(hardware_run, monkeypatch) -> None:
    """A bug in the derivation must surface, not be logged as a skip."""

    def broken(dataset, alpha):
        raise ValueError("too many values to unpack")

    monkeypatch.setattr(qsvm_export, "fit_and_score", broken)
    with pytest.raises(ValueError, match="unpack"):
        hardware_run.qsvm_accuracies([0.5, -0.5])


def test_exactly_one_run_is_the_deployed_one() -> None:
    """ALPHA_SHOTS is one recorded run's readout, and the exports are built
    from it. Every other run is a second measurement of the same circuit, so
    the property to hold is that one artifact matches and the rest do not —
    not that they all do, which was only true while there was one of them."""
    matches = [
        path.name
        for path in sorted(HARDWARE_DIR.glob("hhl-*.json"))
        if json.loads(path.read_text())["jobs"]["raw"]["alpha"]
        == pytest.approx(qsvm_export.ALPHA_SHOTS.tolist())
    ]
    assert len(matches) == 1, matches


@pytest.mark.parametrize("path", sorted(HARDWARE_DIR.glob("hhl-*.json")), ids=lambda p: p.name)
def test_every_readout_is_a_plausible_solution(path) -> None:
    """A readout is two square roots of probabilities with the sign pattern
    taken from the ideal solution, so each component sits in [-1, 1] and the
    ratio is near 1. A job that fell outside this is a broken run, not a
    measurement, and must not reach the comparison."""
    run = json.loads(path.read_text())
    for label, job in run["jobs"].items():
        first, second = job["alpha"]
        assert 0.0 < first <= 1.0, (path.name, label)
        assert -1.0 <= second < 0.0, (path.name, label)
        assert 0.8 < first / -second < 1.25, (path.name, label)


@pytest.mark.parametrize("path", sorted(HARDWARE_DIR.glob("hhl-*.json")), ids=lambda p: p.name)
def test_artifact_accuracies_are_held_out(path) -> None:
    """Recomputed from the splits, not read back.

    Comparing the artifact's numbers to the exports compares two stored outputs
    of one function at one input, which catches a rewritten file and nothing
    else. Each job's alpha is scored again here against the committed map, which
    is what its protocol now claims.
    """
    run = json.loads(path.read_text())
    assert run["qsvm_accuracy_provenance"]["training"]["protocol"].startswith("held-out")
    assert "not measured" in run["alpha_note"]

    for job in run["jobs"].values():
        alpha = np.array(job["alpha"])
        for name, claimed in job["qsvm_accuracy"].items():
            if name == "mnist":
                continue  # needs the openml cache; covered in test_alpha_sensitivity
            if name not in qsvm_export.QSVM_DATASETS:
                continue  # a dataset the sweep covered when this run was recorded
            spec = qsvm_export.QSVM_DATASETS[name]
            payload = json.loads((OUT_DIR / f"qsvm-{name}.json").read_text())
            split = spec["features_fn"]()
            flipped = payload["classes"] != list(spec["classes"])
            labels = -split.test_y if flipped else split.test_y
            scored = decide(weight_vector(alpha), payload["map"], split.test_x)
            assert claimed == pytest.approx(float((scored == labels).mean()), abs=5e-5), name


def test_fit_and_score_is_held_out() -> None:
    """The map is fit on the fit split and scored on a disjoint held-out split."""
    fit = qsvm_export.fit_and_score("iris", np.array([0.5, -0.5]))
    assert len(fit.split.train_y) == 70
    assert len(fit.split.test_y) == 30
    assert 0.0 < fit.accuracy <= 1.0


class TestTheRunIsRepeatable:
    """A series comparing days or devices needs each run pinned to the same
    physical qubits, and needs to record what those qubits were doing."""

    @staticmethod
    def _transpiled(hardware_run, layout=None):
        from qiskit import transpile
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        from classifiers.hhl import build_hhl

        backend = FakeTorino()
        return backend, transpile(
            build_hhl(measure=True),
            backend=backend,
            optimization_level=3,
            initial_layout=layout,
            seed_transpiler=hardware_run.TRANSPILER_SEED,
        )

    def test_the_same_seed_gives_the_same_qubits(self, hardware_run) -> None:
        """Unseeded, the pass manager may land anywhere, and two runs a day
        apart would differ by their layout as much as by the chip."""
        _, first = self._transpiled(hardware_run)
        _, second = self._transpiled(hardware_run)
        assert hardware_run.layout_qubits(first) == hardware_run.layout_qubits(second)
        assert len(hardware_run.layout_qubits(first)) == 4

    def test_a_layout_can_be_pinned_to_an_earlier_run(self, hardware_run) -> None:
        _, transpiled = self._transpiled(hardware_run, layout=[29, 51, 36, 28])
        assert hardware_run.layout_qubits(transpiled) == [29, 51, 36, 28]

    def test_the_two_qubit_pairs_are_recorded_once_each(self, hardware_run) -> None:
        _, transpiled = self._transpiled(hardware_run, layout=[29, 51, 36, 28])
        pairs = hardware_run.two_qubit_pairs(transpiled)
        assert pairs
        assert all(a < b for a, b in pairs), "pairs are stored in one orientation"
        assert len(pairs) == len({tuple(p) for p in pairs})

    def test_the_calibration_covers_what_the_run_used(self, hardware_run) -> None:
        backend, transpiled = self._transpiled(hardware_run, layout=[29, 51, 36, 28])
        qubits = hardware_run.layout_qubits(transpiled)
        pairs = hardware_run.two_qubit_pairs(transpiled)
        snapshot = hardware_run.calibration_snapshot(backend, qubits, pairs)
        assert sorted(snapshot["qubits"]) == sorted(str(q) for q in qubits)
        for entry in snapshot["qubits"].values():
            assert entry["t1_seconds"] > 0
            assert 0.0 <= entry["readout_error"] < 1.0
        assert len(snapshot["edges"]) == len(pairs), "one error per edge, not one per direction"

    def test_each_qubit_role_is_recorded(self, hardware_run) -> None:
        """The readout splits on the solution qubit and the ancilla, so a bias
        on one branch is only chaseable if the record says which physical qubit
        held which role."""
        _, transpiled = self._transpiled(hardware_run, layout=[29, 51, 36, 28])
        roles = hardware_run.qubit_roles(hardware_run.layout_qubits(transpiled))
        assert roles == {
            "eigenvalue_q1": 29,
            "eigenvalue_q2": 51,
            "solution_q3": 36,
            "ancilla_q4": 28,
        }

    def test_a_circuit_with_no_layout_has_no_roles(self, hardware_run) -> None:
        assert hardware_run.qubit_roles([]) == {}

    def test_the_calibration_says_when_it_was_taken(self, hardware_run) -> None:
        """Two runs on one calibration measure the chip twice; across one they
        measure drift as well. Without the stamp there is no telling which."""
        backend, transpiled = self._transpiled(hardware_run, layout=[29, 51, 36, 28])
        snapshot = hardware_run.calibration_snapshot(
            backend,
            hardware_run.layout_qubits(transpiled),
            hardware_run.two_qubit_pairs(transpiled),
        )
        assert snapshot["last_update_date"]

    def test_a_backend_without_a_target_is_not_an_error(self, hardware_run) -> None:
        """Only the snapshot is lost, and losing it must not lose the run."""
        assert hardware_run.calibration_snapshot(object(), [1], [[1, 2]]) == {}

    def test_a_circuit_that_was_never_laid_out_reports_no_qubits(self, hardware_run) -> None:
        from classifiers.hhl import build_hhl

        assert hardware_run.layout_qubits(build_hhl(measure=True)) == []


class TestTheExposureIsSplit:
    """The probes report flip rates per unit of exposure, so a prediction for
    the circuit needs its operations and its measurement window timed apart."""

    @staticmethod
    def _laid_out(hardware_run):
        from qiskit import transpile
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        from classifiers.hhl import build_hhl

        backend = FakeTorino()
        layout = [29, 51, 36, 28]
        circuit = transpile(
            build_hhl(measure=True),
            backend=backend,
            optimization_level=3,
            initial_layout=layout,
            seed_transpiler=hardware_run.TRANSPILER_SEED,
        )
        return backend, circuit, layout

    def test_operations_and_measurement_add_up_to_the_estimate(self, hardware_run) -> None:
        backend, circuit, layout = self._laid_out(hardware_run)
        solution = layout[hardware_run.QUBIT_ROLES.index("solution_q3")]
        split = hardware_run.exposure_split(circuit, backend.target, solution)
        assert 0.0 < split["circuit_seconds"] < split["measure_seconds"]
        assert 0.0 < split["solution_busy_seconds"] <= split["circuit_seconds"]
        estimate = circuit.estimate_duration(backend.target, unit="s")
        assert split["total_seconds"] == pytest.approx(estimate, rel=0.02)

    def test_the_decay_probe_waits_the_pre_measurement_time(self, hardware_run) -> None:
        """The probe's own measurement window already covers the circuit's, so a
        wait of the whole length would count that window twice."""
        backend, circuit, layout = self._laid_out(hardware_run)
        solution = layout[hardware_run.QUBIT_ROLES.index("solution_q3")]
        split = hardware_run.exposure_split(circuit, backend.target, solution)
        _, decay = hardware_run.build_probes(backend, layout, split["circuit_seconds"])
        waited = hardware_run.probe_delay_seconds(decay, backend.target.dt)
        assert waited == pytest.approx(split["circuit_seconds"], rel=0.01)
        assert hardware_run.delay_covers(waited, split["circuit_seconds"]) == "circuit"

    def test_a_wait_of_the_whole_length_is_labelled(self, hardware_run) -> None:
        assert (
            hardware_run.delay_covers(2.08e-6, 0.416e-6) == "circuit and measurement window"
        )
        assert hardware_run.delay_covers(None, 0.4e-6) is None

    def test_a_probe_without_a_delay_reports_none(self, hardware_run) -> None:
        backend, _, layout = self._laid_out(hardware_run)
        readout, _ = hardware_run.build_probes(backend, layout, 1e-6)
        assert hardware_run.probe_delay_seconds(readout, backend.target.dt) is None

    def test_a_target_without_a_measurement_length_raises(self, hardware_run) -> None:
        from classifiers.hhl import build_hhl

        class Bare:
            def __getitem__(self, name):
                raise KeyError(name)

        with pytest.raises(ValueError, match="measurement length"):
            hardware_run.exposure_split(build_hhl(measure=True), Bare(), 2)
