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
provenance), measures the accuracies by running the rule, and writes
``exports/web/qsvm-{iris,mnist}.json`` for the portfolio site's in-browser
demo tier — same conventions as :mod:`classifiers.web_export`.

Run via ``make export-qsvm``; ship with ``make sync-web``.
``tests/test_web_export.py`` drift-checks the committed exports in CI
(the Iris derivation fully; the MNIST parts only when the openml
``mnist_784`` cache is present — CI never downloads datasets).
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
from datetime import datetime, timezone

import numpy as np

from classifiers.web_export import OUT_DIR, SEED, _git

logger = logging.getLogger(__name__)

#: The paper's fixed training geometry (its two mapped training points).
TARGETS = np.array([[0.987, 0.159], [0.345, 0.935]])

#: alpha as measured from the notebook's 8192-shot HHL readout
#: (seed_simulator=42): sqrt(P(0001)), -sqrt(P(0011)).
ALPHA_SHOTS = np.array([0.51048996, -0.49487372])

#: Hand-picked second-dimension map coefficients (c, d) per dataset — the
#: notebook keeps the paper's original OCR values for MNIST and re-picks
#: for Iris so the mapped means stay in the first quadrant.
IRIS_CD = (0.95, -0.42)
MNIST_CD = (0.5, -0.3)

#: Ink threshold for the paper's pixel-ratio features (0-255 grayscale).
INK_THRESHOLD = 127


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


def iris_features() -> tuple[np.ndarray, np.ndarray]:
    """The notebook's Iris subset: (sepal_width, petal_length), setosa=+1."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    mask = iris.target < 2
    feats = iris.data[mask][:, [1, 2]]
    labels = np.where(iris.target[mask] == 0, 1, -1)
    return feats, labels


def mnist_features() -> tuple[np.ndarray, np.ndarray]:
    """The notebook's 6-vs-9 subset as (HR, VR) ink ratios, "6"=+1.

    Requires the openml ``mnist_784`` cache (the notebook's first run created
    it); callers in CI must skip when it is absent.
    """
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml(  # noqa: N806 — sklearn's feature-matrix convention
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff"
    )
    rng = np.random.default_rng(42)
    idx6 = rng.choice(np.where(y == "6")[0], 100, replace=False)
    idx9 = rng.choice(np.where(y == "9")[0], 100, replace=False)
    images = X[np.concatenate([idx6, idx9])].reshape(-1, 28, 28)
    labels = np.concatenate([np.ones(100, dtype=int), -np.ones(100, dtype=int)])
    binary = images > INK_THRESHOLD
    hr = binary[:, :, :14].sum(axis=(1, 2)) / binary[:, :, 14:].sum(axis=(1, 2))
    vr = binary[:, :14, :].sum(axis=(1, 2)) / binary[:, 14:, :].sum(axis=(1, 2))
    return np.stack([hr, vr], axis=1), labels


def build_payload(dataset: str) -> dict:
    """Derive, measure, and assemble one dataset's qsvm export payload."""
    w = weight_vector(ALPHA_SHOTS)
    if dataset == "iris":
        feats, labels = iris_features()
        c, d = IRIS_CD
        classes = ["setosa", "versicolor"]
        features = ["sepal_width", "petal_length"]
        raw_input = "features"
        subset = "setosa vs versicolor"
    else:
        feats, labels = mnist_features()
        c, d = MNIST_CD
        classes = ["6", "9"]
        features = ["horizontal_ink_ratio", "vertical_ink_ratio"]
        raw_input = "pixels"
        subset = "6 vs 9"
    t1 = feats[labels == 1].mean(axis=0)
    t2 = feats[labels == -1].mean(axis=0)
    a, b = solve_map(t1, t2, c, d)
    mapping = {"a": a, "b": b, "c": c, "d": d}
    acc = float((decide(w, mapping, feats) == labels).mean())
    payload: dict = {
        "kind": "qsvm",
        "dataset": dataset,
        "classes": classes,
        "w": w.tolist(),
        "map": mapping,
        "features": features,
        "raw_input": raw_input,
        "test_accuracy": round(acc, 4),
        "num_params": 6,
        "display": {"label": "QSVM (Yang et al. 2019)", "subset": subset},
    }
    if dataset == "mnist":
        payload["ink_threshold"] = INK_THRESHOLD
    payload["provenance"] = {
        "source_repo": "quantum-machine-learning",
        "source_sha": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain", "--", ".", ":(exclude)exports/web")),
        "exported_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "seed": SEED,
        "training": {
            "model": "QSVM",
            "paper": "arXiv:1909.11988",
            "alpha": "shot readout (0.51048996, -0.49487372), seed 42",
            "alpha_note": "the exact analytic alpha (0.5, -0.5) is sign-identical on Iris",
            "derivation": "closed-form Eq. 24 map from class means; see notebooks/qsvm-iris/",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": importlib.metadata.version("scikit-learn"),
        },
    }
    return payload


def main() -> None:
    """Export both qsvm classifiers."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in ("iris", "mnist"):
        payload = build_payload(dataset)
        out = OUT_DIR / f"qsvm-{dataset}.json"
        out.write_text(json.dumps(payload) + "\n")
        logger.info("%s  test_acc=%.4f", out.name, payload["test_accuracy"])


if __name__ == "__main__":
    main()
