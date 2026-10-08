"""What the hardware's alpha readout is worth, held to the artifact.

``tools/alpha_sensitivity.py`` answers a question the divergence figure cannot:
not how close the measured distribution sat to the ideal one, but how many
deployed predictions the difference actually changes. These hold the committed
numbers, and recompute them where the data is available.

The recompute matters because the obvious version of this test is circular.
``tests/test_hardware_run.py`` compares two stored outputs of one function at
one input; that catches a rewritten file and nothing else. Here the flips are
counted again from the splits.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from classifiers import qsvm_export
from classifiers.qsvm_rule import decide, weight_vector
from classifiers.web_export import OUT_DIR, REPO_ROOT

from .conftest import MNIST_OPENML_CACHE

ARTIFACT = REPO_ROOT / "exports" / "alpha-sensitivity.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.is_file(), f"missing {ARTIFACT} — run `make alpha-sensitivity`"
    return json.loads(ARTIFACT.read_text())


def _comparison(artifact: dict, label: str) -> dict:
    return next(c for c in artifact["comparisons"] if c["alpha"] == label)


def _row(artifact: dict, label: str, dataset: str) -> dict:
    return next(d for d in _comparison(artifact, label)["datasets"] if d["dataset"] == dataset)


class TestTheCommittedArtifact:
    """What ships has to be complete. The skip path below is for a cold machine,
    never for the file in the repository."""

    def test_nothing_was_skipped(self, artifact) -> None:
        for comparison in artifact["comparisons"]:
            assert comparison["skipped"] == [], comparison["alpha"]

    def test_every_dataset_is_present(self, artifact) -> None:
        expected = set(qsvm_export.QSVM_DATASETS)
        for comparison in artifact["comparisons"]:
            assert {d["dataset"] for d in comparison["datasets"]} == expected

    def test_the_deployed_alpha_is_the_exported_one(self, artifact) -> None:
        assert artifact["deployed_alpha"] == pytest.approx(qsvm_export.ALPHA_SHOTS.tolist())

    def test_the_ratio_is_the_one_scalar_the_hardware_supplied(self, artifact) -> None:
        """Scale cancels in ``np.sign``, so the ratio is the whole of what the
        readout puts into the boundary."""
        alpha = qsvm_export.ALPHA_SHOTS
        assert artifact["deployed_ratio"] == pytest.approx(alpha[0] / alpha[1], abs=5e-6)
        assert artifact["deployed_ratio_error"] == pytest.approx(
            abs(alpha[0] / alpha[1] + 1.0), abs=5e-6
        )

    def test_the_pooled_counts_add_up(self, artifact) -> None:
        for comparison in artifact["comparisons"]:
            rows = comparison["datasets"]
            pooled = comparison["pooled"]
            assert pooled["flips"] == sum(r["flips"] for r in rows)
            assert pooled["n"] == sum(r["n"] for r in rows)
            assert pooled["flip_rate"] == pytest.approx(pooled["flips"] / pooled["n"], abs=5e-5)

    def test_the_sign_pattern_is_recorded_as_unmeasured(self, artifact) -> None:
        """Half of alpha was never measured, and the artifact has to say so."""
        assert "not measured" in artifact["sign_note"]


class TestTheFlipsRecompute:
    """Counted again from the splits, not read back from the file."""

    @staticmethod
    def _recount(dataset: str, alpha: np.ndarray) -> tuple[int, int]:
        spec = qsvm_export.QSVM_DATASETS[dataset]
        payload = json.loads((OUT_DIR / f"qsvm-{dataset}.json").read_text())
        split = spec["features_fn"]()
        here = decide(np.array(payload["w"]), payload["map"], split.test_x)
        there = decide(weight_vector(alpha), payload["map"], split.test_x)
        return int((here != there).sum()), len(split.test_y)

    @pytest.mark.parametrize("dataset", ["iris"])
    def test_against_the_exact_alpha(self, artifact, dataset) -> None:
        flips, n = self._recount(dataset, np.array(artifact["compared_against"]["exact"]))
        row = _row(artifact, "exact", dataset)
        assert (flips, n) == (row["flips"], row["n"])

    @pytest.mark.skipif(
        not MNIST_OPENML_CACHE, reason="openml mnist_784 not cached here; tests never download"
    )
    def test_mnist_against_the_exact_alpha(self, artifact) -> None:
        flips, n = self._recount("mnist", np.array(artifact["compared_against"]["exact"]))
        row = _row(artifact, "exact", "mnist")
        assert (flips, n) == (row["flips"], row["n"])

    @pytest.mark.parametrize("dataset", ["iris"])
    def test_the_committed_accuracy_is_the_shipped_one(self, artifact, dataset) -> None:
        """The deployed arm is the export's own rule, so this doubles as a drift
        check on the browser payload."""
        payload = json.loads((OUT_DIR / f"qsvm-{dataset}.json").read_text())
        assert _row(artifact, "exact", dataset)["committed_accuracy"] == payload["test_accuracy"]


class TestWhatTheNumberSays:
    """The claims the README is allowed to make from this artifact."""

    def test_the_boundary_barely_moves(self, artifact) -> None:
        tilt = _comparison(artifact, "exact")["boundary_tilt_degrees"]
        assert 0 < tilt < 5, "a small tilt is the whole claim; this is no longer small"

    def test_iris_cannot_see_the_difference(self, artifact) -> None:
        """Thirty samples and a 1.6 degree tilt: zero flips, and that is a
        statement about the split, not about the hardware."""
        assert _row(artifact, "exact", "iris")["flips"] == 0

    def test_no_difference_clears_its_own_interval(self, artifact) -> None:
        """Where the hardware alpha moves a figure at all it moves it up, so the
        only thing keeping that from reading as an improvement is the interval:
        every alternative accuracy sits inside the committed one's."""
        rows = _comparison(artifact, "exact")["datasets"]
        for row in rows:
            low, high = row["committed_accuracy_ci"]
            assert low <= row["other_accuracy"] <= high, row["dataset"]
        assert any(row["accuracy_delta"] != 0 for row in rows), "nothing moved; the test is vacuous"

    def test_every_difference_sits_inside_its_interval(self, artifact) -> None:
        """Nothing here is resolved by these splits."""
        for row in _comparison(artifact, "exact")["datasets"]:
            low, high = row["committed_accuracy_ci"]
            assert low <= row["other_accuracy"] <= high, row["dataset"]


def test_the_readme_quotes_the_artifact(artifact) -> None:
    """A published number has to name the file it came from."""
    readme = (REPO_ROOT / "README.md").read_text()
    pooled = _comparison(artifact, "exact")["pooled"]
    assert "exports/alpha-sensitivity.json" in readme
    assert f"{pooled['flips']} of {pooled['n']:,}" in readme


class TestTheClassicalBaseline:
    """The comparator the writeups quote, so it comes from the artifact too."""

    def test_every_dataset_has_one(self, artifact) -> None:
        assert {b["dataset"] for b in artifact["classical_baseline"]} == set(
            qsvm_export.QSVM_DATASETS
        )

    def test_it_recomputes(self, artifact) -> None:
        from sklearn.linear_model import LogisticRegression

        for baseline in artifact["classical_baseline"]:
            if baseline["dataset"] == "mnist" and not MNIST_OPENML_CACHE:
                continue
            split = qsvm_export.QSVM_DATASETS[baseline["dataset"]]["features_fn"]()
            model = LogisticRegression(max_iter=1000).fit(split.train_x, split.train_y)
            scored = float((model.predict(split.test_x) == split.test_y).mean())
            assert baseline["accuracy"] == pytest.approx(scored, abs=5e-5), baseline["dataset"]

    def test_the_quantum_rule_does_not_beat_it(self, artifact) -> None:
        """Two numbers solved from the class means land under a fitted linear
        model on the same features, which is what the paper expects."""
        baselines = {b["dataset"]: b["accuracy"] for b in artifact["classical_baseline"]}
        for row in _comparison(artifact, "exact")["datasets"]:
            assert row["committed_accuracy"] <= baselines[row["dataset"]], row["dataset"]
