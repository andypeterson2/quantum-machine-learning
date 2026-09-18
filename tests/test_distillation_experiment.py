"""The distillation artifact, and the claims allowed to rest on it.

The switch from MSE to temperature-softened KL (commit 4311232) was argued from
a three-seed MNIST comparison whose numbers lived only in the commit message —
no script, no seeds, no artifact — and the new loss was never measured at all.
``tools/distillation_experiment.py`` produces ``exports/distillation.json``;
these tests check the artifact is internally consistent and that the
documentation reports its direction honestly.
"""

from __future__ import annotations

import json
import statistics

import pytest

from classifiers.web_export import REPO_ROOT

ARTIFACT = REPO_ROOT / "exports" / "distillation.json"
README = REPO_ROOT / "README.md"


@pytest.fixture(scope="module")
def experiment() -> dict:
    assert ARTIFACT.is_file(), f"missing {ARTIFACT} — run tools/distillation_experiment.py"
    return json.loads(ARTIFACT.read_text())


class TestArtifact:
    def test_every_run_names_its_seed(self, experiment) -> None:
        """Without the seed the run is a number nobody can reproduce — the
        failure this experiment exists to correct."""
        seeds = [run["seed"] for run in experiment["runs"]]
        assert len(seeds) >= 3
        assert len(set(seeds)) == len(seeds)
        assert all(isinstance(seed, int) for seed in seeds)

    def test_each_accuracy_carries_its_interval(self, experiment) -> None:
        for run in experiment["runs"]:
            for arm in ("teacher", "student_alone", "student_distilled"):
                low, high = run[arm]["accuracy_ci"]
                assert low <= run[arm]["accuracy"] <= high, (run["seed"], arm)
                assert run[arm]["n"] > 0

    def test_deltas_match_their_arms(self, experiment) -> None:
        for run in experiment["runs"]:
            expected = run["student_distilled"]["accuracy"] - run["student_alone"]["accuracy"]
            assert run["delta"] == pytest.approx(expected, abs=1e-9)

    def test_summary_matches_the_runs(self, experiment) -> None:
        runs, summary = experiment["runs"], experiment["summary"]
        deltas = [run["delta"] for run in runs]
        assert summary["seeds"] == len(runs)
        assert summary["delta_mean"] == pytest.approx(statistics.fmean(deltas), abs=5e-5)
        assert summary["delta_range"] == [
            pytest.approx(min(deltas)), pytest.approx(max(deltas))
        ]
        assert summary["helps"] is (min(deltas) > 0)

    def test_records_the_loss_it_measured(self, experiment) -> None:
        """An artifact measuring the old MSE term would say something else."""
        training = experiment["provenance"]["training"]
        assert "KL" in training["loss"]
        assert experiment["distill_temperature"] > 0


class TestDocumentedEffect:
    """The README may not describe distillation more warmly than it measures."""

    def test_readme_states_the_measured_direction(self, experiment) -> None:
        readme = README.read_text()
        assert "exports/distillation.json" in readme, (
            "the README describes distillation but does not point at the measurement"
        )
        if not experiment["summary"]["helps"]:
            claim = readme.lower()
            for boast in ("improves the student", "boosts the student", "student improves"):
                assert boast not in claim, f"README claims {boast!r}, the artifact disagrees"
