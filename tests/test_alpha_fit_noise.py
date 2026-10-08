"""The redraw study, held to its artifact.

``tools/alpha_fit_noise.py`` is what keeps the sensitivity comparison from
being read as a result. On the committed split the measured alpha beats the
exact one by 0.42 points at p = 7.9e-6; across redrawn fit samples the same
difference averages to roughly nothing and changes sign. These tests pin that,
because it is the claim the writeup now makes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from classifiers import qsvm_export
from classifiers.qsvm_rule import decide, solve_map, weight_vector
from classifiers.web_export import REPO_ROOT

from .conftest import MNIST_OPENML_CACHE

ARTIFACT = REPO_ROOT / "exports" / "alpha-fit-noise.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.is_file(), f"missing {ARTIFACT} — run `make alpha-fit-noise`"
    return json.loads(ARTIFACT.read_text())


def _row(artifact: dict, dataset: str) -> dict:
    rows = [d for d in artifact["datasets"] if d["dataset"] == dataset]
    assert rows, f"{dataset} missing from the study"
    return rows[0]


def test_it_redrew_every_dataset(artifact) -> None:
    covered = {d["dataset"] for d in artifact["datasets"]} | set(artifact["skipped"])
    assert covered == set(qsvm_export.QSVM_DATASETS)
    assert artifact["skipped"] == [], "a dataset was skipped; re-run with its data available"


def test_it_compared_the_two_alphas_that_matter(artifact) -> None:
    assert artifact["deployed_alpha"] == pytest.approx(qsvm_export.ALPHA_SHOTS.tolist())
    assert artifact["exact_alpha"] == [0.5, -0.5]


def test_the_shipped_draw_is_not_among_the_redraws(artifact) -> None:
    """A study of what the sample does cannot include the sample it is about."""
    assert qsvm_export.MNIST_FIT_SEED not in artifact["redraw_seeds"]
    assert qsvm_export.SEED not in artifact["redraw_seeds"]


def test_enough_draws_to_mean_anything(artifact) -> None:
    for row in artifact["datasets"]:
        assert row["redraws"] >= 30, row["dataset"]
        assert len(row["draws"]) == row["redraws"]


def test_the_effect_is_smaller_than_the_noise_it_has_to_clear(artifact) -> None:
    """The finding. If this fails the writeup is wrong, not the test."""
    for row in artifact["datasets"]:
        assert row["delta_is_under_the_fit_noise"], row["dataset"]
        assert abs(row["delta_mean"]) < row["fit_noise_sd"], row["dataset"]


@pytest.mark.skipif(
    not MNIST_OPENML_CACHE, reason="openml mnist_784 not cached here; tests never download"
)
def test_on_mnist_the_direction_is_a_coin_flip(artifact) -> None:
    """The committed split has the measured alpha ahead at p = 7.9e-6. Redrawn,
    it wins about half the time, which is what makes that p-value a fact about
    one sample rather than about the readout."""
    row = _row(artifact, "mnist")
    better = row["hardware_better_in"]
    assert 0.3 < better / row["redraws"] < 0.7, better
    assert row["delta_min"] < 0 < row["delta_max"]


@pytest.mark.skipif(
    not MNIST_OPENML_CACHE, reason="openml mnist_784 not cached here; tests never download"
)
def test_significance_alone_would_have_misled(artifact) -> None:
    """Individual draws clear p < 0.05 often, and in both directions. A single
    split's p-value is why this study exists."""
    row = _row(artifact, "mnist")
    assert row["resolved_in"] > row["redraws"] // 4
    assert row["hardware_better_in"] > 0 and row["exact_better_in"] > 0


def test_a_draw_recomputes(artifact) -> None:
    """Refit one redrawn sample from scratch and land on the recorded row."""
    row = _row(artifact, "iris")
    draw = row["draws"][0]
    spec = qsvm_export.QSVM_DATASETS["iris"]
    c, d = spec["cd_candidates"][0]
    split = spec["features_fn"](draw["seed"])
    x, y = split.train_x, split.train_y
    a, b = solve_map(x[y == 1].mean(axis=0), x[y == -1].mean(axis=0), c, d)
    mapping = {"a": a, "b": b, "c": c, "d": d}
    for name, alpha in (
        ("hardware_accuracy", qsvm_export.ALPHA_SHOTS),
        ("exact_accuracy", np.array([0.5, -0.5])),
    ):
        pred = decide(weight_vector(alpha), mapping, split.test_x)
        assert round(float((pred == split.test_y).mean()), 6) == draw[name], name
    assert draw["n"] == len(split.test_y)


def test_every_draw_is_scored_on_what_it_left_out(artifact) -> None:
    """A draw scored on its own fit sample would flatter itself."""
    for row in artifact["datasets"]:
        assert {d["n"] for d in row["draws"]} == {row["held_out_n"]}, row["dataset"]
        assert row["held_out_n"] > 0
