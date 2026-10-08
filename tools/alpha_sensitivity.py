"""How much the hardware's alpha readout moves the deployed QSVM rule.

The paper's quantum step solves a 2x2 linear system for alpha. Its exact
solution is known in closed form — ``F`` and ``y`` are fixed constants and ``y``
is an eigenvector of ``F``, so alpha is proportional to ``(1, -1)`` before any
circuit runs. The hardware measured ``(0.50097561, -0.48513046)`` instead.

This measures what that difference is worth downstream: how many held-out
predictions change when the shipped rule's weight vector is rebuilt from the
exact alpha rather than the measured one.

Only ``w`` moves. The affine map, the label orientation and the split all come
from the committed ``exports/web/qsvm-*.json``, so the comparison isolates
alpha. ``qsvm_export.fit_and_score`` cannot do that: on BB84 the free parameters
are re-chosen per alpha, and under the mitigated readout it picks
``(c, d) = (1.0, 0.02)`` instead of ``(2.0, 0.1)`` and moves the map's ``a``
from 11.78 to 4.56 — a different model, not the same model under a different
alpha.

Two properties bound what the result can say. ``decide`` takes
``np.sign(v @ w)``, so scale cancels and only alpha's *ratio* reaches the
boundary: one scalar, not two. And alpha's ``(+, -)`` sign pattern is not
measured at all, but taken from the ideal solution — see
``classifiers.hhl.ALPHA_SIGN_NOTE``.

Usage::

    python tools/alpha_sensitivity.py
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from classifiers import qsvm_export  # noqa: E402
from classifiers.hhl import ALPHA_SIGN_NOTE  # noqa: E402
from classifiers.qsvm_rule import decide, weight_vector  # noqa: E402
from classifiers.stats import mcnemar_exact, paired_accuracy_delta, wilson_interval  # noqa: E402
from classifiers.web_export import provenance_base  # noqa: E402

logger = logging.getLogger("alpha-sensitivity")

OUT_PATH = REPO_ROOT / "exports" / "alpha-sensitivity.json"
WEB_DIR = REPO_ROOT / "exports" / "web"

#: The closed-form solution of ``F alpha = y``, up to the scale ``decide`` ignores.
ALPHA_EXACT = np.array([0.5, -0.5])

#: What the deployed rule is compared against. The rule itself is built from the
#: raw readout, so that is the fixed arm and these are the alternatives.
ALPHAS: dict[str, np.ndarray] = {
    "exact": ALPHA_EXACT,
    "hardware_mitigated": np.array([0.49361153, -0.48361836]),
}


def boundary_tilt_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """The angle between the two decision boundaries, in degrees."""
    a, b = weight_vector(first), weight_vector(second)
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, cosine)))))


def compare(dataset: str, alpha: np.ndarray) -> dict:
    """The committed rule against the same rule at *alpha*.

    The map and the orientation are read from the committed export rather than
    refitted, so ``w`` is the only thing that differs between the two arms.
    """
    spec = qsvm_export.QSVM_DATASETS[dataset]
    payload = json.loads((WEB_DIR / f"qsvm-{dataset}.json").read_text())
    committed = np.array(payload["w"])
    if not np.allclose(committed, weight_vector(qsvm_export.ALPHA_SHOTS)):
        raise ValueError(f"{dataset}: the committed w is not the hardware alpha's")

    split = spec["features_fn"]()
    flipped = payload["classes"] != list(spec["classes"])
    labels = -split.test_y if flipped else split.test_y

    scores = _scores(committed, payload["map"], split.test_x)
    if np.any(scores == 0):
        raise ValueError(f"{dataset}: a sample sits on the boundary, so a tie reads as a flip")

    here = np.sign(scores)
    there = decide(weight_vector(alpha), payload["map"], split.test_x)
    hits_here, hits_there = int((here == labels).sum()), int((there == labels).sum())
    n = len(labels)
    # The two arms score the same points, so the pairs where they agree say
    # nothing and only the discordant ones carry the comparison.
    right_here = here == labels
    right_there = there == labels
    b = int((right_here & ~right_there).sum())
    c = int((right_there & ~right_here).sum())
    delta, delta_low, delta_high = paired_accuracy_delta(b, c, n)
    return {
        "dataset": dataset,
        "n": n,
        "flips": int((here != there).sum()),
        "flip_rate": round(float((here != there).mean()), 4),
        "committed_accuracy": round(hits_here / n, 4),
        "committed_accuracy_ci": list(wilson_interval(hits_here, n)),
        "other_accuracy": round(hits_there / n, 4),
        "other_accuracy_ci": list(wilson_interval(hits_there, n)),
        "accuracy_delta": round((hits_here - hits_there) / n, 4),
        "discordant_committed_right": b,
        "discordant_other_right": c,
        "mcnemar_p": round(mcnemar_exact(b, c), 8),
        "paired_delta": delta,
        "paired_delta_ci": [delta_low, delta_high],
    }


def _scores(w: np.ndarray, mapping: dict, feats: np.ndarray) -> np.ndarray:
    """The signed distances ``decide`` takes the sign of."""
    v = np.stack(
        [mapping["a"] * feats[:, 0] + mapping["b"], mapping["c"] * feats[:, 1] + mapping["d"]],
        axis=1,
    )
    return v @ w


def compare_all(alpha: np.ndarray, label: str) -> dict:
    """Every dataset, deployed rule against one alternative alpha."""
    datasets, skipped = [], []
    for dataset in qsvm_export.QSVM_DATASETS:
        try:
            datasets.append(compare(dataset, alpha))
        except OSError as exc:  # the openml cache is absent on this machine
            logger.warning("skipping %s: %s", dataset, exc)
            skipped.append(dataset)
    flips = sum(d["flips"] for d in datasets)
    n = sum(d["n"] for d in datasets)
    b = sum(d["discordant_committed_right"] for d in datasets)
    c = sum(d["discordant_other_right"] for d in datasets)
    delta, delta_low, delta_high = paired_accuracy_delta(b, c, n) if n else (0.0, 0.0, 0.0)
    return {
        "alpha": label,
        "boundary_tilt_degrees": round(
            boundary_tilt_degrees(qsvm_export.ALPHA_SHOTS, alpha), 4
        ),
        "datasets": datasets,
        "pooled": {
            "n": n,
            "flips": flips,
            "flip_rate": round(flips / n, 4) if n else 0.0,
            "discordant_committed_right": b,
            "discordant_other_right": c,
            "mcnemar_p": round(mcnemar_exact(b, c), 8),
            "paired_delta": delta,
            "paired_delta_ci": [delta_low, delta_high],
        },
        "skipped": skipped,
    }


def classical_baseline(dataset: str) -> dict:
    """Logistic regression on the same two features, as the comparator.

    The quantum rule is two numbers; this is what fitting the same points
    without them is worth, so the writeups can quote it from an artifact rather
    than from memory.
    """
    from sklearn.linear_model import LogisticRegression

    split = qsvm_export.QSVM_DATASETS[dataset]["features_fn"]()
    model = LogisticRegression(max_iter=1000).fit(split.train_x, split.train_y)
    hits = int((model.predict(split.test_x) == split.test_y).sum())
    n = len(split.test_y)
    return {
        "dataset": dataset,
        "n": n,
        "accuracy": round(hits / n, 4),
        "accuracy_ci": list(wilson_interval(hits, n)),
    }


def main() -> None:
    """Compare the deployed rule against each alternative and write the artifact."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    comparisons = [compare_all(alpha, label) for label, alpha in ALPHAS.items()]
    # The one scalar the hardware contributes: scale cancels in np.sign, so the
    # ratio is the whole of what the readout puts into the boundary.
    deployed_ratio = float(qsvm_export.ALPHA_SHOTS[0] / qsvm_export.ALPHA_SHOTS[1])

    payload = {
        "kind": "alpha-sensitivity",
        "deployed_alpha": qsvm_export.ALPHA_SHOTS.tolist(),
        "deployed_ratio": round(deployed_ratio, 6),
        "deployed_ratio_error": round(abs(deployed_ratio + 1.0), 6),
        "compared_against": {label: alpha.tolist() for label, alpha in ALPHAS.items()},
        "comparisons": comparisons,
        "scale_invariance_note": (
            "decide() takes np.sign(v @ w), so scale cancels and only alpha's ratio "
            "moves the boundary — one measured scalar, not two"
        ),
        "sign_note": ALPHA_SIGN_NOTE,
        "classical_baseline": [
            classical_baseline(dataset) for dataset in qsvm_export.QSVM_DATASETS
        ],
        "provenance": provenance_base(
            {
                "model": "QSVM",
                "protocol": (
                    "the committed map, orientation and split from exports/web/qsvm-*.json "
                    "held fixed, with only w rebuilt from each alpha, so the comparison is "
                    "the readout error alone and not a refit"
                ),
            },
            {
                "numpy": np.__version__,
                "scikit-learn": importlib.metadata.version("scikit-learn"),
            },
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    for comparison in comparisons:
        pooled = comparison["pooled"]
        logger.info(
            "vs %s: %.2f degrees apart, %d/%d predictions changed",
            comparison["alpha"], comparison["boundary_tilt_degrees"],
            pooled["flips"], pooled["n"],
        )
    logger.info("wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
