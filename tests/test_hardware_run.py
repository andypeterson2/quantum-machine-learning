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
    for name in ("iris", "bb84"):
        payload = json.loads((OUT_DIR / f"qsvm-{name}.json").read_text())
        assert got[name] == payload["test_accuracy"], name


def test_unavailable_dataset_is_skipped(hardware_run, no_mnist) -> None:
    got = hardware_run.qsvm_accuracies([0.5, -0.5])
    assert "mnist" not in got
    assert set(got) == {"iris", "bb84"}


def test_other_failures_raise(hardware_run, monkeypatch) -> None:
    """A bug in the derivation must surface, not be logged as a skip."""

    def broken(dataset, alpha):
        raise ValueError("too many values to unpack")

    monkeypatch.setattr(qsvm_export, "fit_and_score", broken)
    with pytest.raises(ValueError, match="unpack"):
        hardware_run.qsvm_accuracies([0.5, -0.5])


@pytest.mark.parametrize("path", sorted(HARDWARE_DIR.glob("hhl-*.json")), ids=lambda p: p.name)
def test_artifact_accuracies_are_held_out(path) -> None:
    """The committed artifact's accuracies use the exports' protocol: the raw job's
    alpha is the one the exports ship, so its scores must equal theirs."""
    run = json.loads(path.read_text())
    assert run["qsvm_accuracy_provenance"]["training"]["protocol"].startswith("held-out")
    assert "not measured" in run["alpha_note"]
    raw = run["jobs"]["raw"]
    assert raw["alpha"] == pytest.approx(qsvm_export.ALPHA_SHOTS.tolist())
    for name, acc in raw["qsvm_accuracy"].items():
        payload = json.loads((OUT_DIR / f"qsvm-{name}.json").read_text())
        assert acc == payload["test_accuracy"], name


def test_fit_and_score_is_held_out() -> None:
    """The map is fit on the fit split and scored on a disjoint held-out split."""
    fit = qsvm_export.fit_and_score("iris", np.array([0.5, -0.5]))
    assert len(fit.split.train_y) == 70
    assert len(fit.split.test_y) == 30
    assert 0.0 < fit.accuracy <= 1.0
