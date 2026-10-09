"""The mechanism accounting, held to its artifact and recomputed from the runs.

``tools/hardware_mechanism.py`` carries the hardware section's numbers: how
much of the raw readout bias the probes account for, the ratio pooled across
every run, and the range the hardware alphas span downstream. Each is recomputed
here from the stored counts and rates, offline, so the artifact cannot say
something the runs do not.
"""

from __future__ import annotations

import importlib.util
import json
from math import log, sqrt

import pytest

from classifiers.web_export import REPO_ROOT

TOOL = REPO_ROOT / "tools" / "hardware_mechanism.py"
HARDWARE_DIR = REPO_ROOT / "exports" / "hardware"
ARTIFACT = HARDWARE_DIR / "mechanism.json"


@pytest.fixture(scope="module")
def mechanism():
    # The image ships only the package, so the docker CI job has no tools/ to test.
    if not TOOL.is_file():
        pytest.skip("tools/ not shipped here")
    spec = importlib.util.spec_from_file_location("hardware_mechanism", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.is_file(), f"missing {ARTIFACT} — run `make hardware-mechanism`"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def runs() -> dict[tuple[str, str], dict]:
    """Every hardware artifact, keyed by (backend, job id of its raw arm)."""
    return {
        (run["backend"], run["jobs"]["raw"]["job_id"]): run
        for run in (json.loads(p.read_text()) for p in sorted(HARDWARE_DIR.glob("hhl-*.json")))
    }


def _run_for(runs: dict, row: dict) -> dict:
    matches = [
        run
        for run in runs.values()
        if run["backend"] == row["backend"]
        and any(job["job_id"] == row["job_id"] for job in run["jobs"].values())
    ]
    assert len(matches) == 1, row["job_id"]
    return matches[0]


def test_every_probe_carrying_job_is_accounted(artifact, runs) -> None:
    expected = {
        job["job_id"]
        for run in runs.values()
        for job in run["jobs"].values()
        if "probes" in job
    }
    assert expected, "no run carried probes; the accounting has nothing to work on"
    assert {row["job_id"] for row in artifact["probe_jobs"]} == expected


def test_the_corrected_prediction_recomputes(artifact, runs, mechanism) -> None:
    """Readout rate plus the idle rate scaled to the circuit's own exposure."""
    for row in artifact["probe_jobs"]:
        run = _run_for(runs, row)
        job = run["jobs"][row["arm"]]
        readout = job["probes"]["readout"]["read_zero"]
        decay = job["probes"]["decay"]["read_zero"]
        fraction = run["transpiled"]["exposure"]["circuit_seconds"] / (
            job["probes"]["decay"]["delay_seconds"]
        )
        delta = readout + (decay - readout) * fraction
        assert row["corrected_ratio"] == pytest.approx(mechanism.depletion_ratio(delta), abs=1e-6)
        assert row["readout_only_ratio"] == pytest.approx(
            mechanism.depletion_ratio(readout), abs=1e-6
        )
        if row["arm"] == "raw":
            assert row["fraction_explained"] == pytest.approx(
                log(row["corrected_ratio"]) / log(row["observed_ratio"]), abs=1e-3
            )
        else:
            assert row["fraction_explained"] is None


def test_the_shot_sd_carries_the_covariance_term(artifact, runs, mechanism) -> None:
    """Two cells of one multinomial draw covary; dropping that term understates
    the sd by about a tenth and inflates every z by the same."""
    for row in artifact["probe_jobs"]:
        counts = _run_for(runs, row)["jobs"][row["arm"]]["counts"]
        n_a, n_b = counts[mechanism.KEY_ALPHA1], counts[mechanism.KEY_ALPHA2]
        total = sum(counts.values())
        p_a, p_b = n_a / total, n_b / total
        independent = 0.5 * sqrt((1 - p_a) / (total * p_a) + (1 - p_b) / (total * p_b))
        assert row["observed_log_sd"] == pytest.approx(
            mechanism.log_ratio_sd(n_a, n_b, total), abs=1e-6
        )
        assert row["observed_log_sd"] > independent * 1.05


def test_the_residual_is_resolved_on_marrakesh_and_not_on_kingston(artifact) -> None:
    """The claim the writeup makes: the probes account for the bias on
    ibm_kingston, and leave a resolved remainder on ibm_marrakesh."""
    raw = {row["backend"]: row for row in artifact["probe_jobs"] if row["arm"] == "raw"}
    assert raw["ibm_marrakesh"]["resolved"]
    assert raw["ibm_marrakesh"]["residual_sigma"] > artifact["resolved_sigma"]
    assert not raw["ibm_kingston"]["resolved"]
    assert abs(raw["ibm_kingston"]["residual_sigma"]) < 2.0
    for row in raw.values():
        assert 0.0 < row["fraction_explained"] < 1.0, row["backend"]


def test_the_decay_probe_waited_longer_than_the_circuit_ran(artifact) -> None:
    """The correction scales the idle rate down because the probe's wait was the
    circuit's whole length, measurement included. A run whose probe waits the
    pre-measurement time alone needs no scaling, and this is where that run
    will first be noticed."""
    for row in artifact["probe_jobs"]:
        assert row["delay_seconds"] > row["circuit_seconds"], row["job_id"]
        assert row["exposure_fraction"] < 1.0
        assert row["delay_covers"] == "circuit and measurement window"


def test_the_exposure_split_adds_up(runs) -> None:
    """Operations plus measurement window is the circuit's whole length, which
    is what estimate_duration gave and what the probe's wait was set to."""
    for run in runs.values():
        exposure = run["transpiled"].get("exposure")
        if not any("probes" in job for job in run["jobs"].values()):
            continue
        assert exposure, f"{run['backend']} carried probes but has no exposure split"
        total = exposure["circuit_seconds"] + exposure["measure_seconds"]
        assert total == pytest.approx(run["transpiled"]["duration_seconds"], rel=0.02)
        assert 0.0 < exposure["solution_busy_seconds"] <= exposure["circuit_seconds"]
        assert exposure["measure_seconds"] > exposure["circuit_seconds"]


def test_the_pooled_raw_ratio_is_resolved_above_one(artifact) -> None:
    raw = artifact["pooled"]["raw"]
    assert raw["n"] >= 9
    assert raw["z"] > 5.0
    assert raw["ratio"] > 1.0
    assert raw["above_one"] == raw["n"]
    for backend, figures in raw["per_backend"].items():
        assert figures["z"] > 3.0, backend


def test_the_pooled_figures_recompute(artifact, runs, mechanism) -> None:
    rows = mechanism.every_job(list(runs.values()))
    again = mechanism.pooled_by_arm(rows)
    for arm in ("raw", "mitigated"):
        for key in ("n", "ratio", "log_se", "z", "cochran_q", "above_one"):
            assert artifact["pooled"][arm][key] == again[arm][key], (arm, key)


def test_every_downstream_delta_is_inside_the_band(artifact, runs) -> None:
    """Each hardware alpha moves MNIST accuracy by less than the fit sample
    alone does; the exact alpha's accuracy and the band come from the other
    two artifacts, so this also pins the three files to one another."""
    down = artifact["downstream"]
    assert down["n_jobs"] == sum(len(run["jobs"]) for run in runs.values())
    assert down["n_without_mnist"] == 0
    assert down["all_inside_band"]
    assert abs(down["delta_min"]) < down["band_sd"]
    assert abs(down["delta_max"]) < down["band_sd"]
    sensitivity = json.loads((REPO_ROOT / "exports" / "alpha-sensitivity.json").read_text())
    exact = next(c for c in sensitivity["comparisons"] if c["alpha"] == "exact")
    exact_accuracy = next(d for d in exact["datasets"] if d["dataset"] == "mnist")[
        "other_accuracy"
    ]
    assert down["exact_alpha_accuracy"] == exact_accuracy
    for job in down["jobs"]:
        run = _run_for(runs, job)
        claimed = run["jobs"][job["arm"]]["qsvm_accuracy"]["mnist"]
        assert job["delta"] == pytest.approx(claimed - exact_accuracy, abs=1e-6)


def test_depletion_ratio_is_one_at_zero_and_grows(mechanism) -> None:
    assert mechanism.depletion_ratio(0.0) == 1.0
    assert mechanism.depletion_ratio(0.03) == pytest.approx(1.0305, abs=1e-4)
    with pytest.raises(ValueError, match="positive"):
        mechanism.log_ratio_sd(0, 10, 100)
