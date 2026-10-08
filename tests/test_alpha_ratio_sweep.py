"""The alpha-ratio sweep, held to its artifact and recomputed where it can be.

``tools/alpha_ratio_sweep.py`` exists to answer one question the sensitivity
comparison cannot: the comparison says what the gap between the measured alpha
and the exact one costs, which is only interesting if the exact one is where the
accuracy is. These tests pin that it is not, because that is the claim the
writeup makes and it would otherwise rest on a file nothing checks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from classifiers import qsvm_export
from classifiers.qsvm_rule import decide, weight_vector
from classifiers.web_export import OUT_DIR, REPO_ROOT

from .conftest import MNIST_OPENML_CACHE

ARTIFACT = REPO_ROOT / "exports" / "alpha-ratio-sweep.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.is_file(), f"missing {ARTIFACT} — run `make alpha-ratio-sweep`"
    return json.loads(ARTIFACT.read_text())


def _row(artifact: dict, dataset: str) -> dict:
    rows = [d for d in artifact["datasets"] if d["dataset"] == dataset]
    assert rows, f"{dataset} missing from the sweep"
    return rows[0]


def test_it_swept_every_dataset(artifact) -> None:
    covered = {d["dataset"] for d in artifact["datasets"]} | set(artifact["skipped"])
    assert covered == set(qsvm_export.QSVM_DATASETS)
    assert artifact["skipped"] == [], "a dataset was skipped; re-run with its data available"


def test_the_deployed_ratio_is_the_hardware_one(artifact) -> None:
    """The sweep has to be about the alpha that ships, or it is about nothing."""
    alpha = qsvm_export.ALPHA_SHOTS
    assert artifact["deployed_alpha"] == pytest.approx(alpha.tolist())
    assert artifact["deployed_ratio"] == pytest.approx(float(alpha[0] / -alpha[1]), abs=1e-6)
    assert artifact["exact_ratio"] == 1.0


@pytest.mark.parametrize("dataset", ["iris"])
def test_the_curve_recomputes(artifact, dataset) -> None:
    """Score the committed rule again at a few ratios and land on the artifact."""
    row = _row(artifact, dataset)
    spec = qsvm_export.QSVM_DATASETS[dataset]
    payload = json.loads((OUT_DIR / f"qsvm-{dataset}.json").read_text())
    split = spec["features_fn"]()
    flipped = payload["classes"] != list(spec["classes"])
    labels = -split.test_y if flipped else split.test_y

    for i in (0, len(row["ratios"]) // 2, len(row["ratios"]) - 1):
        w = weight_vector(np.array([row["ratios"][i], -1.0]))
        acc = float((decide(w, payload["map"], split.test_x) == labels).mean())
        assert round(acc, 4) == row["accuracies"][i], row["ratios"][i]


@pytest.mark.skipif(
    not MNIST_OPENML_CACHE, reason="openml mnist_784 not cached here; tests never download"
)
def test_the_exact_solution_is_not_the_optimum(artifact) -> None:
    """The finding the writeup rests on: accuracy peaks away from alpha's exact
    value, so the readout's precision is not what the classifier is limited by."""
    row = _row(artifact, "mnist")
    assert not row["exact_is_optimal"]
    assert row["best"]["accuracy"] > row["exact"]["accuracy"]
    assert row["best"]["ratio"] != pytest.approx(artifact["exact_ratio"])


def test_the_peak_is_inside_the_grid(artifact) -> None:
    """A peak at an edge would mean the grid, not the data, placed it."""
    for row in artifact["datasets"]:
        assert row["ratios"][0] < row["best"]["ratio"] < row["ratios"][-1], row["dataset"]


def test_every_curve_is_scored_on_the_whole_split(artifact) -> None:
    for row in artifact["datasets"]:
        payload = json.loads((OUT_DIR / f"qsvm-{row['dataset']}.json").read_text())
        assert row["n"] == payload["test_n"], row["dataset"]
        assert len(row["accuracies"]) == len(row["ratios"])


def test_the_marked_points_sit_on_the_curve(artifact) -> None:
    """exact and hardware are read off the same function as the grid, so a
    rounded grid point must agree with them where the two coincide."""
    for row in artifact["datasets"]:
        for name in ("exact", "hardware"):
            mark = row[name]
            ratios = row["ratios"]
            near = min(range(len(ratios)), key=lambda i: abs(ratios[i] - mark["ratio"]))
            if abs(ratios[near] - mark["ratio"]) < 1e-9:
                assert row["accuracies"][near] == mark["accuracy"], (row["dataset"], name)
            low, high = mark["accuracy_ci"]
            assert low <= mark["accuracy"] <= high
