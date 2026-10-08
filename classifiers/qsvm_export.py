"""Export the QSVM paper-recreation classifiers as browser-runnable weights.

The notebook in ``notebooks/qsvm-iris/`` recreates Yang, Awan & Vall-Llosera,
"SVM on NISQ Computers" (arXiv:1909.11988) end to end. Its *deployable* result
is tiny: after the paper's solved preprocessing map, both datasets share one
2-D linear decision rule::

    s = w[0] * (a*f1 + b) + w[1] * (c*f2 + d)      s > 0 -> class +1

The rule itself — the Eq. 24 map, the weight vector and the ink-ratio features —
lives in :mod:`classifiers.qsvm_rule`, which the notebook imports too, so there
is one definition rather than two. This module re-derives the map coefficients
closed-form the same way the notebook does (class means -> the Eq. 24 solve
against the paper's fixed training geometry), pairs them with the shot-readout
``alpha`` (the measured artifact of the recreation; the exact analytic
``alpha = (0.5, -0.5)`` is sign-identical on Iris and is recorded in the
provenance), fits the map on a training split, scores the rule on a held-out
split, and writes
``exports/web/qsvm-{iris,mnist}.json`` for the portfolio site's in-browser
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
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from classifiers.qsvm_rule import (
    INK_THRESHOLD,
    decide,
    ink_ratios,
    solve_map,
    weight_vector,
)
from classifiers.stats import wilson_interval
from classifiers.web_export import OUT_DIR, SEED, provenance_base

logger = logging.getLogger(__name__)

#: alpha = (sqrt(P(0001)), -sqrt(P(0011))) from the HHL readout on ibm_marrakesh
#: (2026-09-04, job dad49jdnj4cs73adbp90, 8192 raw shots), a real quantum computer.
ALPHA_SHOTS = np.array([0.50097561, -0.48513046])

#: Second-dimension map (c, d): the paper's own values (Eq. 15), for the two
#: datasets it ran. :func:`choose_parameters` has nothing to pick here.
IRIS_CD = (0.95, -0.42)
MNIST_CD = (0.5, -0.3)

#: Fraction of the fit split held back to choose the free parameters on.
VALIDATION_FRACTION = 0.25

#: The notebook's ``rng_seed``, so the exporter fits on the same 100 digits per
#: class; everything else is held out, needing no second seed.
MNIST_FIT_SEED = 42


class Split(NamedTuple):
    """Raw (N, 2) features and +1/-1 labels, fit and held-out."""

    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    protocol: str


def iris_features(seed: int = SEED) -> Split:
    """The notebook's Iris subset, (sepal_width, petal_length) with setosa=+1,
    split 70/30 by class.

    Args:
        seed: The split's random state. The default is the committed one;
            :mod:`tools.alpha_fit_noise` varies it to measure how much the
            split alone moves the rule.
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    mask = iris.target < 2
    feats = iris.data[mask][:, [1, 2]]
    labels = np.where(iris.target[mask] == 0, 1, -1)
    tx, vx, ty, vy = train_test_split(
        feats, labels, test_size=0.3, stratify=labels, random_state=seed
    )
    return Split(tx, ty, vx, vy, "stratified 70/30 split of the 100 setosa/versicolor samples")


@lru_cache(maxsize=1)
def _mnist_corpus() -> tuple[np.ndarray, np.ndarray]:
    """The openml digits, parsed once.

    Parsing 70,000 rows takes seconds, and :mod:`tools.alpha_fit_noise` redraws
    the fit sample sixty times from the same corpus.
    """
    from sklearn.datasets import fetch_openml

    return fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff"
    )


def mnist_features(seed: int = MNIST_FIT_SEED) -> Split:
    """The notebook's 6-vs-9 fit sample (100 per class) as (HR, VR) ink ratios, "6"=+1,
    and every other 6 and 9 in the corpus as the held-out split.

    The held-out size is not a parameter: a sub-sample would only widen the
    interval on a figure the full corpus already settles, and the digits cost
    nothing once the file is cached.

    Requires the openml ``mnist_784`` cache (the notebook's first run created
    it); callers in CI must skip when it is absent.

    Args:
        seed: Draws the fit sample. The default is the notebook's;
            :mod:`tools.alpha_fit_noise` varies it to measure how much the
            sample alone moves the rule.
    """
    X, y = _mnist_corpus()  # noqa: N806 — sklearn's feature-matrix convention
    rng = np.random.default_rng(seed)
    pool6, pool9 = np.where(y == "6")[0], np.where(y == "9")[0]
    idx6 = rng.choice(pool6, 100, replace=False)
    idx9 = rng.choice(pool9, 100, replace=False)
    test6, test9 = np.setdiff1d(pool6, idx6), np.setdiff1d(pool9, idx9)

    def sample(i6: np.ndarray, i9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        images = X[np.concatenate([i6, i9])].reshape(-1, 28, 28)
        labels = np.concatenate([np.ones(len(i6), dtype=int), -np.ones(len(i9), dtype=int)])
        return ink_ratios(images), labels

    tx, ty = sample(idx6, idx9)
    vx, vy = sample(test6, test9)
    n_test = len(test6) + len(test9)
    protocol = f"fit on the notebook's 200 digits; scored on every other 6 and 9 ({n_test})"
    return Split(tx, ty, vx, vy, protocol)


#: Per-dataset export specs — adding a dataset is adding one entry here (plus
#: its features function above); build_payload has no dataset branches. ``seeds``
#: names whatever draws that dataset's samples, so the provenance records it.
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
        "seeds": {},
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
        "seeds": {"fit_sample": MNIST_FIT_SEED},
        "extra": {"ink_threshold": INK_THRESHOLD},
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


def paper_orientation(x: np.ndarray, y: np.ndarray, c: float, d: float) -> bool:
    """Whether the labels must be swapped to match the paper's ray assignment.

    The paper's two targets are not interchangeable: the +1 ray has the smaller
    second component (0.159 against 0.938 once normalised), so the +1 class is
    the one whose mapped second component is smaller. Reading the orientation
    off that, rather than searching for it, reproduces the paper's own choice
    on both of its datasets — setosa against versicolor, and "6" against "9" —
    which is the evidence that this is its convention rather than a new one.

    Searching instead costs more than it buys. A validation slice of a
    200-sample fit split is 50 images; choosing between two orientations on 50
    images is close to a coin toss, and the toss lands in the published number.

    Args:
        x: Fit features, shape (N, 2).
        y: Fit labels, +1 and -1 as the dataset spec names them.
        c: Second-dimension slope, whose sign decides which way the map orders
            the two class means.
        d: Second-dimension offset.

    Returns:
        ``True`` when the spec's +1 class belongs on the -1 ray.
    """
    pos = c * x[y == 1].mean(axis=0)[1] + d
    neg = c * x[y == -1].mean(axis=0)[1] + d
    return bool(pos > neg)


def choose_parameters(split: Split, spec: dict, w: np.ndarray) -> Choice:
    """Pick (c, d) on a validation slice of the fit split.

    One parameter here is the modeller's, not the paper's: the second
    dimension's (c, d), where the paper gives no value for a dataset it did not
    run. It was once fixed by hand, justified by accuracies this module only
    ever computed on the held-out split — which made the published number
    optimistic. It is now chosen on a slice held back from the fit split, so
    the held-out split takes no part. Ties keep the first candidate, so the
    result is deterministic.

    The orientation is not searched at all; :func:`paper_orientation` reads it
    off the geometry for whichever (c, d) is under test.

    Where the paper fixes (c, d) too — Iris and MNIST are its own experiments —
    there is nothing to choose and its values stand.
    """
    from sklearn.model_selection import train_test_split

    if not spec["free_parameters"]:
        c, d = spec["cd_candidates"][0]
        flip = paper_orientation(split.train_x, split.train_y, c, d)
        return Choice(flip=flip, c=c, d=d, validation_accuracy=float("nan"),
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
    for c, d in spec["cd_candidates"]:
        flip = paper_orientation(fit_x, fit_y, c, d)
        fit_labels, val_labels = _oriented(fit_y, flip=flip), _oriented(val_y, flip=flip)
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


def fit_and_score(dataset: str, alpha: np.ndarray) -> Fit:
    """Fit the Eq. 24 map on the fit split and score the rule on the held-out split.

    The one derivation both the exporter and ``tools/hardware_run.py`` use, so a
    hardware alpha is scored exactly the way the shipped exports are. The free
    parameters come from :func:`choose_parameters`.
    """
    spec = QSVM_DATASETS[dataset]
    w = weight_vector(alpha)
    split = spec["features_fn"]()
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
                f"(c, d) chosen on a stratified {VALIDATION_FRACTION:.0%} validation slice of "
                "the fit split, the held-out split taking no part; the orientation follows the "
                "paper's ray geometry and is not searched"
                if spec["free_parameters"]
                else "(c, d) fixed by the paper; the orientation follows its ray geometry"
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
            "split_seed": SEED,
            "sampling_seeds": spec["seeds"],
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
