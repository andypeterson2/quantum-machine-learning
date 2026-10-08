"""Does the paper's rule work on a corpus it was not designed for?

Yang et al. fit their preprocessing to one 28x28 task, MNIST 6 vs 9, and the
exports ship exactly that. Fashion-MNIST is the same shape and the same two ink
ratios apply unchanged, so it answers two questions the shipped datasets cannot.

First, transfer. The Eq. 24 map pins the two class means onto fixed targets, and
those targets came from the paper's own digits. A corpus whose class geometry
sits elsewhere has no reason to land where the map expects.

Second, direction. :mod:`tools.alpha_fit_noise` shows the hardware alpha's
advantage on MNIST is a coin flip across redraws of one fit sample. Forty-five
class pairs are forty-five independent chances for that advantage to hold or
reverse, on held-out splits of about 13,800 images each.

Every pair is reported. Choosing one would be choosing a result, and the
choosing is the part that would not survive review: the pair with the largest
class-mean VR gap on the fit sample, which is the only pre-declarable analogue
of the paper's own rotation pair, scores barely above chance. Within a pair the
orientation and ``(c, d)`` come from :func:`classifiers.qsvm_export.choose_parameters`
on a validation slice of the fit split, as they do for any dataset the paper
did not run.

Output: ``exports/qsvm-transfer.json``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import logging
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers import qsvm_export  # noqa: E402
from classifiers.qsvm_rule import decide, ink_ratios, solve_map, weight_vector  # noqa: E402
from classifiers.stats import mcnemar_exact, paired_accuracy_delta, wilson_interval  # noqa: E402
from classifiers.web_export import provenance_base  # noqa: E402

logger = logging.getLogger(__name__)

ARTIFACT = REPO_ROOT / "exports" / "qsvm-transfer.json"

#: The corpus. Same shape and same ink ratios as the paper's digits, different
#: subject matter. MIT licensed (Xiao et al., arXiv:1708.07747).
CORPUS = "Fashion-MNIST"

#: Fashion-MNIST's labels, in its own order.
CLASS_NAMES = (
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

#: Images per class in the fit sample, matching the paper's 100 per class.
FIT_PER_CLASS = 100

#: ``(c, d)`` candidates. The paper gives values for its own two datasets only,
#: so these are a spread around them for :func:`choose_parameters` to pick from.
CD_GRID = [(0.5, -0.3), (0.95, -0.42), (2.0, 0.02), (1.0, 0.02), (0.25, -0.1)]

ALPHA_EXACT = np.array([0.5, -0.5])


def corpus() -> tuple[np.ndarray, np.ndarray]:
    """The Fashion-MNIST images as (HR, VR) ink ratios, with integer labels."""
    from sklearn.datasets import fetch_openml

    data, target = fetch_openml(
        CORPUS, version=1, return_X_y=True, as_frame=False, parser="liac-arff"
    )
    return ink_ratios(data.reshape(-1, 28, 28)), target.astype(int)


def fit_indices(labels: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """``FIT_PER_CLASS`` images of each class, drawn once and shared by every pair."""
    rng = np.random.default_rng(seed)
    return {
        c: rng.choice(np.where(labels == c)[0], FIT_PER_CLASS, replace=False)
        for c in range(len(CLASS_NAMES))
    }


def declared_pair(feats: np.ndarray, fit: dict[int, np.ndarray]) -> dict:
    """The pair a rule stated in advance would have picked, and why.

    The paper's pair is a rotation pair, which VR separates by construction, so
    the analogue is the largest class-mean VR gap. Computed on the fit sample
    alone, so no held-out image takes part. Recorded because the pair it names
    scores near chance, which is the evidence that the better-scoring pairs in
    this artifact are not a pre-declared choice and must not be read as one.
    """
    vr = {c: float(feats[fit[c], 1].mean()) for c in range(len(CLASS_NAMES))}
    pairs = itertools.combinations(range(len(CLASS_NAMES)), 2)
    pos, neg = max(pairs, key=lambda p: abs(vr[p[0]] - vr[p[1]]))
    return {
        "rule": "largest class-mean VR gap on the fit sample; no held-out image takes part",
        "pair": [CLASS_NAMES[pos], CLASS_NAMES[neg]],
        "vr_gap": round(abs(vr[pos] - vr[neg]), 4),
    }


def one_pair(
    feats: np.ndarray, labels: np.ndarray, fit: dict[int, np.ndarray], pos: int, neg: int
) -> dict:
    """Fit the paper's rule to one class pair and score both alphas on the rest."""
    from sklearn.linear_model import LogisticRegression

    fit_idx = np.concatenate([fit[pos], fit[neg]])
    held_idx = np.setdiff1d(np.where(np.isin(labels, [pos, neg]))[0], fit_idx)
    fit_x, fit_y = feats[fit_idx], np.where(labels[fit_idx] == pos, 1, -1)
    held_x, held_y = feats[held_idx], np.where(labels[held_idx] == pos, 1, -1)

    spec = {"free_parameters": True, "cd_candidates": CD_GRID}
    split = qsvm_export.Split(fit_x, fit_y, held_x, held_y, "")
    w = weight_vector(qsvm_export.ALPHA_SHOTS)
    choice = qsvm_export.choose_parameters(split, spec, w)

    train_y = -fit_y if choice.flip else fit_y
    test_y = -held_y if choice.flip else held_y
    a, b = solve_map(
        fit_x[train_y == 1].mean(axis=0), fit_x[train_y == -1].mean(axis=0), choice.c, choice.d
    )
    mapping = {"a": a, "b": b, "c": choice.c, "d": choice.d}

    right_hw = decide(w, mapping, held_x) == test_y
    right_ex = decide(weight_vector(ALPHA_EXACT), mapping, held_x) == test_y
    n = len(test_y)
    disc_b = int((right_hw & ~right_ex).sum())
    disc_c = int((right_ex & ~right_hw).sum())
    delta, low, high = paired_accuracy_delta(disc_b, disc_c, n)
    baseline = LogisticRegression(max_iter=5000).fit(fit_x, fit_y).score(held_x, held_y)
    return {
        "pair": [CLASS_NAMES[pos], CLASS_NAMES[neg]],
        "n": n,
        "accuracy": round(float(right_hw.mean()), 4),
        "accuracy_ci": list(wilson_interval(int(right_hw.sum()), n)),
        "exact_accuracy": round(float(right_ex.mean()), 4),
        "logistic_regression": round(float(baseline), 4),
        "beats_logistic_regression": bool(right_hw.mean() > baseline),
        "discordant_hardware_right": disc_b,
        "discordant_exact_right": disc_c,
        "mcnemar_p": round(mcnemar_exact(disc_b, disc_c), 10),
        "paired_delta": delta,
        "paired_delta_ci": [low, high],
        "selection": {
            "flip": choice.flip,
            "c": choice.c,
            "d": choice.d,
            "candidates": choice.candidates,
            "validation_accuracy": round(choice.validation_accuracy, 4),
            "validation_n": choice.validation_n,
        },
    }


def summarise(pairs: list[dict]) -> dict:
    """What the forty-five pairs say, as two claims rather than a table."""
    resolved = [p for p in pairs if p["mcnemar_p"] < 0.05]
    ahead = [p for p in resolved if p["paired_delta"] > 0]
    deltas = [p["paired_delta"] for p in pairs]
    return {
        "pairs_scored": len(pairs),
        "accuracy_median": round(statistics.median(p["accuracy"] for p in pairs), 4),
        "accuracy_min": min(p["accuracy"] for p in pairs),
        "accuracy_max": max(p["accuracy"] for p in pairs),
        "logistic_regression_median": round(
            statistics.median(p["logistic_regression"] for p in pairs), 4
        ),
        "beats_logistic_regression_on": sum(p["beats_logistic_regression"] for p in pairs),
        "resolved_on": len(resolved),
        "hardware_ahead_on": len(ahead),
        "hardware_behind_on": len(resolved) - len(ahead),
        "delta_mean": round(statistics.fmean(deltas), 6),
        "delta_sd": round(statistics.stdev(deltas), 6),
    }


def build(seed: int = qsvm_export.MNIST_FIT_SEED) -> dict:
    """Score every class pair of the corpus."""
    feats, labels = corpus()
    fit = fit_indices(labels, seed)
    pairs = [
        one_pair(feats, labels, fit, pos, neg)
        for pos, neg in itertools.combinations(range(len(CLASS_NAMES)), 2)
    ]
    return {
        "kind": "qsvm-transfer",
        "corpus": CORPUS,
        "corpus_citation": "Xiao, Rasul & Vollgraf, arXiv:1708.07747",
        "fit_seed": seed,
        "fit_per_class": FIT_PER_CLASS,
        "deployed_alpha": qsvm_export.ALPHA_SHOTS.tolist(),
        "exact_alpha": ALPHA_EXACT.tolist(),
        "declared_pair": declared_pair(feats, fit),
        "summary": summarise(pairs),
        "pairs": pairs,
        "provenance": provenance_base(
            {
                "model": "QSVM",
                "paper": "arXiv:1909.11988",
                "protocol": (
                    "every class pair, none selected; orientation and (c, d) chosen per pair "
                    "on a validation slice of the fit split, scored on every other image"
                ),
            },
            {"numpy": np.__version__, "scikit-learn": importlib.metadata.version("scikit-learn")},
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Write the transfer artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    artifact = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1) + "\n")
    s = artifact["summary"]
    logger.info(
        "%d pairs: median %.3f (logistic regression %.3f), ahead of it on %d",
        s["pairs_scored"],
        s["accuracy_median"],
        s["logistic_regression_median"],
        s["beats_logistic_regression_on"],
    )
    logger.info(
        "hardware vs exact resolved on %d; ahead %d, behind %d",
        s["resolved_on"],
        s["hardware_ahead_on"],
        s["hardware_behind_on"],
    )
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
