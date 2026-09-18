"""Export the QSVM paper-recreation classifiers as browser-runnable weights.

The notebook in ``notebooks/qsvm-iris/`` recreates Yang, Awan & Vall-Llosera,
"SVM on NISQ Computers" (arXiv:1909.11988) end to end. Its *deployable* result
is tiny: after the paper's solved preprocessing map, both datasets share one
2-D linear decision rule::

    s = w[0] * (a*f1 + b) + w[1] * (c*f2 + d)      s > 0 -> class +1

This module re-derives the map coefficients closed-form the same way the
notebook does (class means -> the Eq. 24 solve against the paper's fixed
training geometry), pairs them with the notebook's quantum shot-readout
``alpha`` (the measured artifact of the recreation; the exact analytic
``alpha = (0.5, -0.5)`` is sign-identical on Iris and is recorded in the
provenance), fits the map on a training split, scores the rule on a held-out
split, and writes
``exports/web/qsvm-{iris,mnist,bb84}.json`` for the portfolio site's in-browser
demo tier — same conventions as :mod:`classifiers.web_export`.

Run via ``make export-qsvm``; ship with ``make sync-web``.
``tests/test_web_export.py`` drift-checks the committed exports in CI
(the Iris derivation fully; the MNIST parts only when the openml
``mnist_784`` cache is present — the tests themselves never download).

MNIST reaches this repo twice over: ``torchvision`` serves the platform's
own 28x28 tensors, while the paper recreation needs the flat ``mnist_784``
vectors that sklearn fetches from openml. Both are cached separately in CI.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
from typing import NamedTuple

import numpy as np

from classifiers.stats import wilson_interval
from classifiers.web_export import OUT_DIR, SEED, provenance_base

logger = logging.getLogger(__name__)

#: The paper's fixed training geometry (its two mapped training points).
TARGETS = np.array([[0.987, 0.159], [0.345, 0.935]])

#: alpha = (sqrt(P(0001)), -sqrt(P(0011))) from the HHL readout on ibm_marrakesh
#: (2026-09-04, job dad49jdnj4cs73adbp90, 8192 raw shots), a real quantum computer.
ALPHA_SHOTS = np.array([0.50097561, -0.48513046])

#: Second-dimension map (c, d): the paper's own values (Eq. 15) for its two
#: datasets, and for BB84 a grid :func:`choose_parameters` picks from.
IRIS_CD = (0.95, -0.42)
MNIST_CD = (0.5, -0.3)
BB84_CD_GRID = [(2.0, 0.02), (1.0, 0.02), (4.0, 0.02), (2.0, 0.1), (8.0, 0.01)]

#: Fraction of the fit split held back to choose the free parameters on.
VALIDATION_FRACTION = 0.25

#: Ink threshold for the paper's pixel-ratio features (0-255 grayscale).
INK_THRESHOLD = 127

#: Held-out MNIST digits per class, drawn from outside the 100-per-class fit sample.
MNIST_TEST_PER_CLASS = 500


class Split(NamedTuple):
    """Raw (N, 2) features and +1/-1 labels, fit and held-out."""

    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    protocol: str


def weight_vector(alpha: np.ndarray) -> np.ndarray:
    """w = alpha1*x1 + alpha2*x2 over the row-normalized training targets."""
    x_train = TARGETS / np.linalg.norm(TARGETS, axis=1, keepdims=True)
    return alpha[0] * x_train[0] + alpha[1] * x_train[1]


def solve_map(t1: np.ndarray, t2: np.ndarray, c: float, d: float) -> tuple[float, float]:
    """Solve the Eq. 24 affine map so the class means land on TARGETS' rays.

    Args:
        t1: (f1, f2) mean of the +1 class.
        t2: (f1, f2) mean of the -1 class.
        c:  Hand-picked slope for the second feature.
        d:  Hand-picked offset for the second feature.

    Returns:
        (a, b) such that (a*f1 + b, c*f2 + d) maps each mean parallel to its
        paper target.
    """
    v12, v22 = c * t1[1] + d, c * t2[1] + d
    if v12 <= 0 or v22 <= 0:
        raise ValueError("mapped second components must stay positive (paper Sec. IV-A)")
    req = np.array([v12 * TARGETS[0, 0] / TARGETS[0, 1], v22 * TARGETS[1, 0] / TARGETS[1, 1]])
    a, b = np.linalg.solve(np.array([[t1[0], 1.0], [t2[0], 1.0]]), req)
    return float(a), float(b)


def decide(w: np.ndarray, mapping: dict, feats: np.ndarray) -> np.ndarray:
    """Apply the deployed rule to (N, 2) raw features; returns sign(+1/-1).

    Args:
        w:       The 2-D weight vector.
        mapping: ``{"a", "b", "c", "d"}`` affine map coefficients.
        feats:   Raw feature matrix of shape (N, 2).
    """
    v = np.stack(
        [mapping["a"] * feats[:, 0] + mapping["b"], mapping["c"] * feats[:, 1] + mapping["d"]],
        axis=1,
    )
    return np.sign(v @ w)


def iris_features() -> Split:
    """The notebook's Iris subset, (sepal_width, petal_length) with setosa=+1,
    split 70/30 by class."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    mask = iris.target < 2
    feats = iris.data[mask][:, [1, 2]]
    labels = np.where(iris.target[mask] == 0, 1, -1)
    tx, vx, ty, vy = train_test_split(
        feats, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    return Split(tx, ty, vx, vy, "stratified 70/30 split of the 100 setosa/versicolor samples")


def _ink_ratios(images: np.ndarray) -> np.ndarray:
    """(HR, VR): left/right and top/bottom ink counts, with an empty half counted as 1."""
    binary = images > INK_THRESHOLD
    hr = binary[:, :, :14].sum(axis=(1, 2)) / np.maximum(binary[:, :, 14:].sum(axis=(1, 2)), 1)
    vr = binary[:, :14, :].sum(axis=(1, 2)) / np.maximum(binary[:, 14:, :].sum(axis=(1, 2)), 1)
    return np.stack([hr, vr], axis=1)


def mnist_features() -> Split:
    """The notebook's 6-vs-9 fit sample (100 per class) as (HR, VR) ink ratios, "6"=+1,
    and a disjoint held-out sample of MNIST_TEST_PER_CLASS per class.

    Requires the openml ``mnist_784`` cache (the notebook's first run created
    it); callers in CI must skip when it is absent.
    """
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(  # noqa: N806 — sklearn's feature-matrix convention
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff"
    )
    rng = np.random.default_rng(42)
    pool6, pool9 = np.where(y == "6")[0], np.where(y == "9")[0]
    idx6 = rng.choice(pool6, 100, replace=False)
    idx9 = rng.choice(pool9, 100, replace=False)
    held = np.random.default_rng(43)
    test6 = held.choice(np.setdiff1d(pool6, idx6), MNIST_TEST_PER_CLASS, replace=False)
    test9 = held.choice(np.setdiff1d(pool9, idx9), MNIST_TEST_PER_CLASS, replace=False)

    def sample(i6: np.ndarray, i9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        images = X[np.concatenate([i6, i9])].reshape(-1, 28, 28)
        labels = np.concatenate([np.ones(len(i6), dtype=int), -np.ones(len(i9), dtype=int)])
        return _ink_ratios(images), labels

    tx, ty = sample(idx6, idx9)
    vx, vy = sample(test6, test9)
    n_test = 2 * MNIST_TEST_PER_CLASS
    protocol = f"fit on the notebook's 200 digits; scored on {n_test} other 6s and 9s"
    return Split(tx, ty, vx, vy, protocol)


def bb84_features() -> Split:
    """The bb84 plugin's train and test splits as raw (qber, sifted_key_rate), eve=+1.

    Re-generates the exact seeded simulations the plugin serves (self-generated
    data — no cache, so the CI drift check runs this unconditionally).

    Labels arrive with *eavesdropped* on the +1 ray; which class actually ends
    up there is decided by :func:`choose_parameters` on a validation slice,
    because it matters — the Eq. 24 geometry places the boundary about 87% of
    the way from the +1 class mean toward the −1 mean, so the orientation moves
    the boundary between the sparse gap near the clean regime and the middle of
    the eavesdropped distribution.
    """
    from classifiers.datasets.bb84.plugin import N_TEST, N_TRAIN, TEST_SEED, TRAIN_SEED
    from classifiers.datasets.bb84.simulate import generate_dataset

    def sessions(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        feats, labels01 = generate_dataset(n, seed)
        return feats.astype(np.float64), np.where(labels01 == 1, 1, -1)  # eve is +1

    tx, ty = sessions(N_TRAIN, TRAIN_SEED)
    vx, vy = sessions(N_TEST, TEST_SEED)
    protocol = f"fit on the {N_TRAIN} training sessions; scored on the {N_TEST} test sessions"
    return Split(tx, ty, vx, vy, protocol)


#: Per-dataset export specs — adding a dataset is adding one entry here (plus
#: its features function above); build_payload has no dataset branches.
QSVM_DATASETS: dict[str, dict] = {
    "iris": {
        "features_fn": iris_features,
        "cd_candidates": [IRIS_CD],
        # Paper Sec. V-A2 names setosa the +1 class and Eq. 15 gives (c, d);
        # nothing here is the modeller's to choose.
        "free_parameters": False,
        "classes": ["setosa", "versicolor"],
        "features": ["sepal_width", "petal_length"],
        "raw_input": "features",
        "subset": "setosa vs versicolor",
        "extra": {},
    },
    "mnist": {
        "features_fn": mnist_features,
        "cd_candidates": [MNIST_CD],
        # Paper Sec. V-A1: "6" is the +1 class, (c, d) from Eq. 15.
        "free_parameters": False,
        "classes": ["6", "9"],
        "features": ["horizontal_ink_ratio", "vertical_ink_ratio"],
        "raw_input": "pixels",
        "subset": "6 vs 9",
        "extra": {"ink_threshold": INK_THRESHOLD},
    },
    "bb84": {
        "features_fn": bb84_features,
        "cd_candidates": BB84_CD_GRID,
        # The paper has no BB84 experiment, so both the orientation and (c, d)
        # are this repo's to pick — and so must be picked on validation data.
        "free_parameters": True,
        "classes": ["eavesdropped", "clean"],
        "features": ["qber", "sifted_key_rate"],
        "raw_input": "features",
        "subset": "eavesdropped vs clean",
        "extra": {},
    },
}


class Fit(NamedTuple):
    """One dataset's derived rule and its held-out score.

    ``hits`` is kept beside ``accuracy`` so callers can state the uncertainty:
    29/30 and 290/300 are the same accuracy with very different intervals.
    """

    w: np.ndarray
    mapping: dict
    split: Split
    accuracy: float
    hits: int
    choice: Choice


class Choice(NamedTuple):
    """The free parameters, and the validation evidence for them."""

    flip: bool
    c: float
    d: float
    validation_accuracy: float
    validation_n: int
    candidates: int


def _oriented(labels: np.ndarray, *, flip: bool) -> np.ndarray:
    """Labels with the classes swapped between the paper's two rays."""
    return -labels if flip else labels


def _solve_on(x: np.ndarray, y: np.ndarray, c: float, d: float) -> dict:
    """The Eq. 24 map fitted to these points' class means."""
    a, b = solve_map(x[y == 1].mean(axis=0), x[y == -1].mean(axis=0), c, d)
    return {"a": a, "b": b, "c": c, "d": d}


def choose_parameters(split: Split, spec: dict, w: np.ndarray) -> Choice:
    """Pick the orientation and (c, d) on a validation slice of the fit split.

    Two parameters here are the modeller's, not the paper's: which class rides
    the +1 ray (the Eq. 24 geometry puts the boundary about 87% of the way from
    the +1 mean toward the -1 mean, so the choice matters), and the second
    dimension's (c, d) where the paper gives no value. They were previously
    fixed by hand, justified by accuracies that this module only ever computed
    on the held-out split — which makes the published number optimistic.

    They are now chosen on a slice held back from the fit split, so the held-out
    split takes no part in the choice. Ties keep the first candidate, so the
    result is deterministic.

    Where the paper fixes both (Iris and MNIST are its own experiments), there
    is nothing to choose and its values stand: selecting them here would only
    add noise — on Iris's 18-sample validation slice it flips the orientation
    and costs 13 points of held-out accuracy.
    """
    from sklearn.model_selection import train_test_split

    if not spec["free_parameters"]:
        c, d = spec["cd_candidates"][0]
        return Choice(flip=False, c=c, d=d, validation_accuracy=float("nan"),
                      validation_n=0, candidates=1)

    fit_x, val_x, fit_y, val_y = train_test_split(
        split.train_x,
        split.train_y,
        test_size=VALIDATION_FRACTION,
        stratify=split.train_y,
        random_state=SEED,
    )

    best: Choice | None = None
    candidates = 0
    for flip in (False, True):
        fit_labels, val_labels = _oriented(fit_y, flip=flip), _oriented(val_y, flip=flip)
        for c, d in spec["cd_candidates"]:
            try:
                mapping = _solve_on(fit_x, fit_labels, c, d)
            except ValueError:
                # The mapped second components must stay positive (paper Sec.
                # IV-A); a candidate that breaks that is simply not available.
                continue
            candidates += 1
            accuracy = float((decide(w, mapping, val_x) == val_labels).mean())
            if best is None or accuracy > best.validation_accuracy:
                best = Choice(flip, c, d, accuracy, len(val_labels), candidates)
    if best is None:
        raise ValueError("no (c, d) candidate maps this dataset into the first quadrant")
    return best._replace(candidates=candidates)


def fit_and_score(dataset: str, alpha: np.ndarray, choice: Choice | None = None) -> Fit:
    """Fit the Eq. 24 map on the fit split and score the rule on the held-out split.

    The one derivation both the exporter and ``tools/hardware_run.py`` use, so a
    hardware alpha is scored exactly the way the shipped exports are. The free
    parameters come from :func:`choose_parameters` unless a *choice* is supplied.
    """
    spec = QSVM_DATASETS[dataset]
    w = weight_vector(alpha)
    split = spec["features_fn"]()
    if choice is None:
        choice = choose_parameters(split, spec, w)
    train_y = _oriented(split.train_y, flip=choice.flip)
    test_y = _oriented(split.test_y, flip=choice.flip)
    mapping = _solve_on(split.train_x, train_y, choice.c, choice.d)
    hits = int((decide(w, mapping, split.test_x) == test_y).sum())
    return Fit(w, mapping, split, hits / len(test_y), hits, choice)


def build_payload(dataset: str) -> dict:
    """Derive, measure, and assemble one dataset's qsvm export payload."""
    spec = QSVM_DATASETS[dataset]
    w, mapping, split, acc, hits, choice = fit_and_score(dataset, ALPHA_SHOTS)
    classes = list(reversed(spec["classes"])) if choice.flip else list(spec["classes"])
    payload: dict = {
        "kind": "qsvm",
        "dataset": dataset,
        "classes": classes,
        "w": w.tolist(),
        "map": mapping,
        "features": spec["features"],
        "raw_input": spec["raw_input"],
        "test_accuracy": round(acc, 4),
        "test_accuracy_ci": list(wilson_interval(hits, len(split.test_y))),
        "train_n": len(split.train_y),
        "test_n": len(split.test_y),
        "test_protocol": split.protocol,
        "selection": {
            "protocol": (
                f"orientation and (c, d) chosen on a stratified {VALIDATION_FRACTION:.0%} "
                "validation slice of the fit split; the held-out split takes no part"
                if spec["free_parameters"]
                else "orientation and (c, d) fixed by the paper; nothing selected here"
            ),
            "candidates": choice.candidates,
            "validation_n": choice.validation_n,
            "validation_accuracy": (
                round(choice.validation_accuracy, 4) if spec["free_parameters"] else None
            ),
            "positive_class": classes[0],
            "c": choice.c,
            "d": choice.d,
        },
        "num_params": 6,
        "display": {"label": "QSVM (Yang et al. 2019)", "subset": spec["subset"]},
        **spec["extra"],
    }
    payload["provenance"] = provenance_base(
        {
            "model": "QSVM",
            "paper": "arXiv:1909.11988",
            "alpha": "hardware shot readout (0.50097561, -0.48513046), ibm_marrakesh raw 8192",
            "alpha_note": (
                "adopted from exports/hardware/hhl-ibm_marrakesh-2026-09-04.json; "
                "the Aer readout (0.51048996, -0.49487372, seed 42) and the exact "
                "analytic alpha (0.5, -0.5) are sign-identical"
            ),
            "derivation": "closed-form Eq. 24 map from the training split's class means",
            "selection": (
                "free parameters (orientation, and (c, d) where the paper gives none) "
                "chosen on a validation slice of the fit split, never on the held-out split"
            ),
        },
        {"numpy": np.__version__, "scikit-learn": importlib.metadata.version("scikit-learn")},
    )
    return payload


def main() -> None:
    """Export every qsvm classifier in the spec table."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in QSVM_DATASETS:
        payload = build_payload(dataset)
        out = OUT_DIR / f"qsvm-{dataset}.json"
        out.write_text(json.dumps(payload) + "\n")
        logger.info("%s  test_acc=%.4f", out.name, payload["test_accuracy"])


if __name__ == "__main__":
    main()
