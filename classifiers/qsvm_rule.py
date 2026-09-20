"""The paper's decision rule, and the features it reads.

Yang, Awan & Vall-Llosera, "SVM on NISQ Computers" (arXiv:1909.11988) ends at a
2-D linear rule: map the raw features onto the paper's fixed training geometry
(Eq. 24), then take the sign of their dot product with a weight vector built
from the quantum solution's ``alpha``.

Three places need that arithmetic: the notebook that recreates the paper, the
exporter that ships the rule to the browser, and ``tools/hardware_run.py``, which
re-scores it under a hardware-measured ``alpha``. It used to exist twice, and the
copies had already drifted — the exporter guarded an empty image half and the
notebook divided by zero.

Like :mod:`classifiers.hhl`, this module imports nothing beyond numpy, so the
notebook's own venv can import it. Anything needing torch belongs in
:mod:`classifiers.qsvm_export` instead.
"""

from __future__ import annotations

import numpy as np

#: The paper's fixed training geometry (its two mapped training points).
TARGETS = np.array([[0.987, 0.159], [0.345, 0.935]])

#: Ink threshold for the paper's pixel-ratio features (0-255 grayscale).
INK_THRESHOLD = 127


def weight_vector(alpha: np.ndarray) -> np.ndarray:
    """``w = alpha1*x1 + alpha2*x2`` over the row-normalized training targets.

    Args:
        alpha: The two-element solution of ``F alpha = y``.

    Returns:
        The 2-D weight vector the decision rule takes the sign against.
    """
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

    Raises:
        ValueError: If either mapped second component is non-positive, which
            puts the point outside the first quadrant the paper works in.
    """
    v12, v22 = c * t1[1] + d, c * t2[1] + d
    if v12 <= 0 or v22 <= 0:
        raise ValueError("mapped second components must stay positive (paper Sec. IV-A)")
    req = np.array([v12 * TARGETS[0, 0] / TARGETS[0, 1], v22 * TARGETS[1, 0] / TARGETS[1, 1]])
    a, b = np.linalg.solve(np.array([[t1[0], 1.0], [t2[0], 1.0]]), req)
    return float(a), float(b)


def decide(w: np.ndarray, mapping: dict, feats: np.ndarray) -> np.ndarray:
    """Apply the deployed rule to (N, 2) raw features; returns sign(+1/-1).

    The mapped vector is not normalised first: scaling a vector leaves the sign
    of its dot product alone, so the boundary is the same either way.

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


def ink_ratios(images: np.ndarray) -> np.ndarray:
    """The paper's (HR, VR) pixel ratios for (N, 28, 28) grayscale digits.

    HR is left ink over right ink, VR is top over bottom. An empty denominator
    half counts as 1, so a digit drawn entirely on one side gives a finite ratio
    rather than an infinity that no downstream step can map.

    Args:
        images: Grayscale images of shape (N, 28, 28), 0-255.

    Returns:
        An (N, 2) matrix of (HR, VR).
    """
    binary = images > INK_THRESHOLD
    hr = binary[:, :, :14].sum(axis=(1, 2)) / np.maximum(binary[:, :, 14:].sum(axis=(1, 2)), 1)
    vr = binary[:, :14, :].sum(axis=(1, 2)) / np.maximum(binary[:, 14:, :].sum(axis=(1, 2)), 1)
    return np.stack([hr, vr], axis=1)
