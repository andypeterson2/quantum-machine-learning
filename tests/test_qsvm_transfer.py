"""The transfer study, held to its artifact.

``tools/qsvm_transfer.py`` carries two claims the shipped datasets cannot: that
the paper's preprocessing does not transfer off the corpus it was fitted to, and
that the hardware alpha's advantage on MNIST reverses about as often as it
holds. Both are stated in the writeup, so both are pinned here.

The test that matters most is :func:`test_no_pair_was_selected`. Forty-five
pairs reported is the whole defence against reading the good ones as a result.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from classifiers import qsvm_export
from classifiers.qsvm_rule import decide, solve_map, weight_vector
from classifiers.web_export import REPO_ROOT

from .conftest import FASHION_OPENML_CACHE

ARTIFACT = REPO_ROOT / "exports" / "qsvm-transfer.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.is_file(), f"missing {ARTIFACT} — run `make qsvm-transfer`"
    return json.loads(ARTIFACT.read_text())


def test_no_pair_was_selected(artifact) -> None:
    """Every pair of the ten classes, each exactly once. A missing pair would
    make the summary a choice rather than a census."""
    from tools.qsvm_transfer import CLASS_NAMES

    expected = {
        frozenset((CLASS_NAMES[a], CLASS_NAMES[b]))
        for a, b in itertools.combinations(range(len(CLASS_NAMES)), 2)
    }
    got = [frozenset(p["pair"]) for p in artifact["pairs"]]
    assert len(got) == 45
    assert set(got) == expected
    assert len(set(got)) == len(got), "a pair is reported twice"


def test_the_declared_pair_is_recorded_and_poor(artifact) -> None:
    """The one pair a rule could have named in advance scores near chance. That
    is why the artifact reports all of them instead of naming one."""
    declared = artifact["declared_pair"]
    assert "fit sample" in declared["rule"]
    row = [p for p in artifact["pairs"] if set(p["pair"]) == set(declared["pair"])]
    assert row, "the declared pair is not among the scored pairs"
    assert row[0]["accuracy"] < 0.8


def test_the_rule_does_not_transfer(artifact) -> None:
    """Claim one: on a corpus it was not fitted to, the paper's rule is beaten
    by a logistic regression on the same two features."""
    s = artifact["summary"]
    assert s["accuracy_median"] < s["logistic_regression_median"]
    assert s["beats_logistic_regression_on"] < s["pairs_scored"] / 2
    assert s["accuracy_min"] < 0.55, "at least one pair should be near chance"


def test_the_hardware_advantage_reverses(artifact) -> None:
    """Claim two: resolved in both directions, so MNIST's sign is not a
    property of the readout."""
    s = artifact["summary"]
    assert s["resolved_on"] > 10
    assert s["hardware_ahead_on"] > 0
    assert s["hardware_behind_on"] > 0


def test_every_pair_is_scored_on_a_real_split(artifact) -> None:
    for row in artifact["pairs"]:
        assert row["n"] > 10_000, row["pair"]
        low, high = row["accuracy_ci"]
        assert low <= row["accuracy"] <= high, row["pair"]
        assert row["discordant_hardware_right"] >= 0
        assert row["discordant_exact_right"] >= 0


def test_selection_used_a_validation_slice(artifact) -> None:
    """The paper fixes nothing for this corpus, so (c, d) and the orientation
    are chosen here — and must be chosen off the held-out split."""
    for row in artifact["pairs"]:
        sel = row["selection"]
        assert sel["candidates"] > 1, row["pair"]
        assert sel["validation_n"] > 0, row["pair"]
        assert 0.0 < sel["validation_accuracy"] <= 1.0, row["pair"]


@pytest.mark.skipif(
    not FASHION_OPENML_CACHE, reason="openml Fashion-MNIST not cached here; tests never download"
)
def test_a_pair_recomputes(artifact) -> None:
    """Refit one pair from scratch and land on the recorded accuracy."""
    from tools.qsvm_transfer import CLASS_NAMES, FIT_PER_CLASS, corpus, fit_indices

    row = artifact["pairs"][0]
    pos, neg = (CLASS_NAMES.index(name) for name in row["pair"])
    feats, labels = corpus()
    fit = fit_indices(labels, artifact["fit_seed"])
    assert len(fit[pos]) == FIT_PER_CLASS

    fit_idx = np.concatenate([fit[pos], fit[neg]])
    held = np.setdiff1d(np.where(np.isin(labels, [pos, neg]))[0], fit_idx)
    fit_x, fit_y = feats[fit_idx], np.where(labels[fit_idx] == pos, 1, -1)
    sel = row["selection"]
    train_y = -fit_y if sel["flip"] else fit_y
    a, b = solve_map(
        fit_x[train_y == 1].mean(axis=0), fit_x[train_y == -1].mean(axis=0), sel["c"], sel["d"]
    )
    mapping = {"a": a, "b": b, "c": sel["c"], "d": sel["d"]}
    held_y = np.where(labels[held] == pos, 1, -1)
    test_y = -held_y if sel["flip"] else held_y
    pred = decide(weight_vector(qsvm_export.ALPHA_SHOTS), mapping, feats[held])
    assert round(float((pred == test_y).mean()), 4) == row["accuracy"]
    assert len(test_y) == row["n"]
