"""Export the three-class Iris QSVM: three pairwise rules over all four features.

The paper's Iris experiment is two classes and two of the four measurements
(Sec. V-A2), and ``qsvm_export`` ships exactly that. This module ships a second
model beside it that covers setosa, versicolor and virginica and reads all four
columns, by running the paper's rule once per pair of classes and voting.

What is the paper's, unchanged:

- the non-offset least-squares SVM, ``b = 0``, boundary through the origin
- M = 2 training points, the two class means
- the per-coordinate affine map pinning those means onto fixed targets
- the L2 normalisation of Eq. 22 before scoring
- the alpha read off ibm_marrakesh. :func:`targets_nd` keeps the targets' inner
  product, so the kernel matrix, its eigenvalues (0.509 and 1.491) and alpha are
  the same numbers the 2-D rule uses. One hardware run serves all three rules,
  and ``w`` below is one vector, not three.

What is not the paper's, and must be said wherever this model is described:

- four coordinates. Amplitude encoding needs two qubits per training point
  instead of one, so the Fig. 9 kernel oracle would have to change. That circuit
  is computed classically here, and was not re-run.
- the vote. The paper defines no multiclass rule.

Output: ``exports/web/qsvm-iris-ovo.json``, shipped to the website by
``make sync-web``.

``qsvm_export.QSVM_DATASETS`` is deliberately left alone. It drives the alpha
sensitivity sweep and the hardware accuracy table, both of which measure the
paper's own binary rule; a three-class voting model has no row in either.
"""

from __future__ import annotations

import importlib.metadata
import itertools
import json
import logging
from typing import NamedTuple

import numpy as np

from classifiers.qsvm_export import ALPHA_SHOTS
from classifiers.qsvm_rule import margin_nd, solve_map_nd, targets_nd, weight_vector
from classifiers.stats import wilson_interval
from classifiers.web_export import OUT_DIR, SEED, provenance_base

logger = logging.getLogger(__name__)

#: The three species, in the order the Iris dataset numbers them.
CLASSES = ["setosa", "versicolor", "virginica"]

#: All four measurements, in the dataset's own column order.
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

#: Splits the cross-validated figure averages over. One 45-sample split cannot
#: separate this model from the next; the mean over many says what it is worth.
CV_SEEDS = range(20)

#: Fraction of the data held out, matching the binary Iris export.
TEST_FRACTION = 0.3


class Rule(NamedTuple):
    """One pairwise rule: which two classes, and the map that tells them apart."""

    positive: str
    negative: str
    a: np.ndarray
    b: np.ndarray


class Contest(NamedTuple):
    """One rule's verdict on one sample."""

    positive: str
    negative: str
    winner: str
    score: float
    lean: float


def iris_split(seed: int) -> tuple:
    """All four features and all three classes, split 70/30 by class.

    Args:
        seed: The split's random state.

    Returns:
        ``(train_x, train_y, test_x, test_y)`` with integer class indices.
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    return train_test_split(
        iris.data, iris.target, test_size=TEST_FRACTION, stratify=iris.target, random_state=seed
    )


def pairwise_rules(x: np.ndarray, y: np.ndarray, targets: np.ndarray) -> list[Rule]:
    """One rule per unordered pair of classes, the lower index riding the +1 ray.

    Args:
        x:       Training features, shape (N, d).
        y:       Integer class indices into :data:`CLASSES`.
        targets: The (2, d) widened targets.

    Returns:
        Three rules, in ``itertools.combinations`` order.
    """
    rules = []
    for pos, neg in itertools.combinations(range(len(CLASSES)), 2):
        a, b = solve_map_nd(x[y == pos].mean(axis=0), x[y == neg].mean(axis=0), targets)
        rules.append(Rule(CLASSES[pos], CLASSES[neg], a, b))
    return rules


def contests(rules: list[Rule], w: np.ndarray, x: np.ndarray) -> list[list[Contest]]:
    """Every rule's verdict on every sample.

    ``lean`` is the score as a share of the evidence behind it — the same ratio
    the binary rule reports — so one rule's confidence is comparable with
    another's, and with the two-feature model's.

    Args:
        rules: The pairwise rules.
        w:     The shared weight vector.
        x:     Features, shape (N, d).

    Returns:
        One list of three contests per sample.
    """
    out: list[list[Contest]] = [[] for _ in range(len(x))]
    for rule in rules:
        scores = margin_nd(w, rule.a, rule.b, x)
        mapped = x * rule.a + rule.b
        unit = mapped / np.maximum(np.linalg.norm(mapped, axis=1, keepdims=True), 1e-12)
        totals = np.abs(unit * w).sum(axis=1)
        for n, score in enumerate(scores):
            lean = float(score / totals[n]) if totals[n] else 0.0
            winner = rule.positive if score > 0 else rule.negative
            out[n].append(Contest(rule.positive, rule.negative, winner, float(score), lean))
    return out


def vote(sample: list[Contest]) -> str:
    """The class the most rules picked; ties go to the widest winning lean.

    Args:
        sample: One sample's three contests.

    Returns:
        The winning class name.
    """
    tally: dict[str, float] = dict.fromkeys(CLASSES, 0.0)
    weight: dict[str, float] = dict.fromkeys(CLASSES, 0.0)
    for c in sample:
        tally[c.winner] += 1.0
        weight[c.winner] += abs(c.lean)
    return max(CLASSES, key=lambda c: (tally[c], weight[c]))


def predict(rules: list[Rule], w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vote every sample, returning integer class indices.

    Args:
        rules: The pairwise rules.
        w:     The shared weight vector.
        x:     Features, shape (N, d).

    Returns:
        An (N,) array of indices into :data:`CLASSES`.
    """
    return np.array([CLASSES.index(vote(s)) for s in contests(rules, w, x)])


def fit_and_score(seed: int, alpha: np.ndarray) -> tuple:
    """Fit the three rules on one split's training half and score the other.

    Args:
        seed:  The split's random state.
        alpha: The two-element solution the rules share.

    Returns:
        ``(rules, w, hits, n, train_n)``.
    """
    targets = targets_nd(len(FEATURES))
    w = weight_vector(alpha) @ targets
    train_x, test_x, train_y, test_y = iris_split(seed)
    rules = pairwise_rules(train_x, train_y, targets)
    hits = int((predict(rules, w, test_x) == test_y).sum())
    return rules, w, hits, len(test_y), len(train_y)


def cross_validated_accuracy(alpha: np.ndarray) -> float:
    """Mean held-out accuracy over :data:`CV_SEEDS` independent splits.

    Args:
        alpha: The two-element solution the rules share.

    Returns:
        The mean accuracy, rounded to four decimals.
    """
    scores = [hits / n for _, _, hits, n, _ in (fit_and_score(s, alpha) for s in CV_SEEDS)]
    return round(float(np.mean(scores)), 4)


def build_payload() -> dict:
    """Derive, measure and assemble the three-class Iris payload."""
    targets = targets_nd(len(FEATURES))
    rules, w, hits, test_n, train_n = fit_and_score(SEED, ALPHA_SHOTS)
    # Four weight components and three maps of eight coefficients each, counted
    # the way the binary export counts its six: len(w) plus the map's numbers.
    num_params = len(w) + sum(len(r.a) + len(r.b) for r in rules)
    payload: dict = {
        "kind": "qsvm-ovo",
        "dataset": "iris",
        "classes": list(CLASSES),
        "w": w.tolist(),
        "targets": targets.tolist(),
        "rules": [
            {"pair": [r.positive, r.negative], "a": r.a.tolist(), "b": r.b.tolist()} for r in rules
        ],
        "features": list(FEATURES),
        "raw_input": "features",
        "test_accuracy": round(hits / test_n, 4),
        "test_accuracy_ci": list(wilson_interval(hits, test_n)),
        "train_n": train_n,
        "test_n": test_n,
        "test_protocol": "stratified 70/30 split of all 150 samples, three species",
        "cv_accuracy": cross_validated_accuracy(ALPHA_SHOTS),
        "cv_splits": len(CV_SEEDS),
        "cv_protocol": (
            f"mean over {len(CV_SEEDS)} independent stratified 70/30 splits; "
            "45 held-out samples cannot separate this model from the next on their own"
        ),
        "selection": {
            "protocol": (
                "nothing is selected: the targets spread each paper coordinate evenly "
                "over its half, and the lower-numbered class of each pair rides the +1 ray"
            ),
            "candidates": 1,
            "validation_n": 0,
            "validation_accuracy": None,
            "positive_class": CLASSES[0],
        },
        "num_params": num_params,
        "display": {"label": "QSVM one-vs-one (Yang et al. 2019)", "subset": "3 species"},
    }
    payload["provenance"] = provenance_base(
        {
            "model": "QSVM one-vs-one",
            "paper": "arXiv:1909.11988",
            "alpha": "hardware shot readout (0.50097561, -0.48513046), ibm_marrakesh raw 8192",
            "alpha_note": (
                "the same alpha the binary exports ship: the widened targets keep the "
                "kernel matrix, so all three pairwise rules share one hardware run"
            ),
            "derivation": (
                "per-coordinate affine map from each pair's class means onto the widened "
                "targets, then a vote; the widening and the vote are not the paper's"
            ),
            "split_seed": SEED,
            "sampling_seeds": {},
            "selection": "nothing selected; no validation slice is used",
        },
        {"numpy": np.__version__, "scikit-learn": importlib.metadata.version("scikit-learn")},
    )
    return payload


def main() -> None:
    """Export the three-class Iris rule."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    out = OUT_DIR / "qsvm-iris-ovo.json"
    out.write_text(json.dumps(payload) + "\n")
    logger.info(
        "%s  test_acc=%.4f  cv_acc=%.4f", out.name, payload["test_accuracy"], payload["cv_accuracy"]
    )


if __name__ == "__main__":
    main()
