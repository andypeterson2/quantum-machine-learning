"""The hardware alpha's effect, set beside the noise it has to clear.

:mod:`tools.alpha_sensitivity` establishes that the measured alpha and the exact
one give different predictions, and on MNIST that the difference is resolved:
McNemar p = 7.9e-6. A resolved difference is not yet a meaningful one, and the
obvious objection is that the rule is fitted to 200 digits drawn at random, so
redrawing them moves accuracy too. If that movement is larger than the gap
between the two alphas, the gap is a detail of this fit rather than a property
of the readout.

This redraws the fit sample :data:`REDRAWS` times, refits the Eq. 24 map on each
draw, and scores both alphas on everything that draw left out. It reports the
spread of the rule's accuracy across draws, and the paired difference between
the alphas within each draw. Nothing here is selected: both datasets take the
paper's own (c, d) and orientation, so a redraw changes the sample and nothing
else.

Output: ``exports/alpha-fit-noise.json``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import statistics
import sys
from math import sqrt
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers import qsvm_export  # noqa: E402
from classifiers.qsvm_rule import decide, solve_map, weight_vector  # noqa: E402
from classifiers.stats import Z_95, mcnemar_exact  # noqa: E402
from classifiers.web_export import provenance_base  # noqa: E402

logger = logging.getLogger(__name__)

ARTIFACT = REPO_ROOT / "exports" / "alpha-fit-noise.json"

#: Fit samples to draw. Enough that the spread's own standard error is about an
#: eighth of the spread, which is the precision the comparison needs.
REDRAWS = 60

#: Seeds for the redraws. The committed export's own seed is deliberately not
#: among them, so no draw here is the shipped rule.
REDRAW_SEEDS = tuple(range(1000, 1000 + REDRAWS))

ALPHA_EXACT = np.array([0.5, -0.5])


def one_draw(dataset: str, seed: int) -> dict:
    """Refit on one redrawn sample and score both alphas on what it left out.

    Args:
        dataset: A key of :data:`classifiers.qsvm_export.QSVM_DATASETS`.
        seed:    Draws the fit sample.

    Returns:
        Both accuracies, the discordant counts between them, and the sample size.
    """
    spec = qsvm_export.QSVM_DATASETS[dataset]
    c, d = spec["cd_candidates"][0]
    split = spec["features_fn"](seed)
    x, y = split.train_x, split.train_y
    a, b = solve_map(x[y == 1].mean(axis=0), x[y == -1].mean(axis=0), c, d)
    mapping = {"a": a, "b": b, "c": c, "d": d}

    hardware = decide(weight_vector(qsvm_export.ALPHA_SHOTS), mapping, split.test_x)
    exact = decide(weight_vector(ALPHA_EXACT), mapping, split.test_x)
    right_hw, right_ex = hardware == split.test_y, exact == split.test_y
    n = len(split.test_y)
    return {
        "seed": seed,
        "n": n,
        "hardware_accuracy": round(float(right_hw.mean()), 6),
        "exact_accuracy": round(float(right_ex.mean()), 6),
        "discordant_hardware_right": int((right_hw & ~right_ex).sum()),
        "discordant_exact_right": int((right_ex & ~right_hw).sum()),
    }


def summarise(dataset: str, draws: list[dict]) -> dict:
    """Turn the draws into the two numbers the comparison needs.

    ``fit_noise_sd`` is how far the rule's own accuracy moves when only the fit
    sample changes. ``delta_mean`` is how far the alphas move it within a draw.
    The second being smaller than the first is the finding.
    """
    deltas = [d["hardware_accuracy"] - d["exact_accuracy"] for d in draws]
    exact_acc = [d["exact_accuracy"] for d in draws]
    significant = [
        mcnemar_exact(d["discordant_hardware_right"], d["discordant_exact_right"]) < 0.05
        for d in draws
    ]
    # Consecutive draws share all but 200 of their held-out images, so the
    # spread across draws is what the interval comes from.
    standard_error = statistics.stdev(deltas) / sqrt(len(deltas))
    mean_delta = statistics.fmean(deltas)
    return {
        "dataset": dataset,
        "redraws": len(draws),
        "held_out_n": draws[0]["n"],
        "delta_mean": round(statistics.fmean(deltas), 6),
        "delta_sd": round(statistics.stdev(deltas), 6),
        "delta_min": round(min(deltas), 6),
        "delta_max": round(max(deltas), 6),
        "hardware_better_in": sum(d > 0 for d in deltas),
        "exact_better_in": sum(d < 0 for d in deltas),
        "resolved_in": sum(significant),
        "fit_noise_sd": round(statistics.stdev(exact_acc), 6),
        "exact_accuracy_mean": round(statistics.fmean(exact_acc), 6),
        "delta_mean_se": round(standard_error, 6),
        "delta_mean_ci": [
            round(mean_delta - Z_95 * standard_error, 6),
            round(mean_delta + Z_95 * standard_error, 6),
        ],
        # The question the artifact exists to answer.
        "delta_is_under_the_fit_noise": bool(
            abs(statistics.fmean(deltas)) < statistics.stdev(exact_acc)
        ),
        "draws": draws,
    }


def build() -> dict:
    """Redraw every dataset the exporter ships, skipping any without its data."""
    datasets, skipped = [], []
    for dataset in qsvm_export.QSVM_DATASETS:
        try:
            draws = [one_draw(dataset, seed) for seed in REDRAW_SEEDS]
        except OSError as exc:  # the openml cache is absent on this machine
            logger.warning("skipping %s: %s", dataset, exc)
            skipped.append(dataset)
            continue
        datasets.append(summarise(dataset, draws))
    return {
        "kind": "alpha-fit-noise",
        "deployed_alpha": qsvm_export.ALPHA_SHOTS.tolist(),
        "exact_alpha": ALPHA_EXACT.tolist(),
        "redraw_seeds": list(REDRAW_SEEDS),
        "datasets": datasets,
        "skipped": skipped,
        "provenance": provenance_base(
            {
                "model": "QSVM",
                "paper": "arXiv:1909.11988",
                "protocol": (
                    "the fit sample is redrawn; the paper's (c, d) and orientation are held, "
                    "and each draw is scored on everything it left out"
                ),
            },
            {"numpy": np.__version__, "scikit-learn": importlib.metadata.version("scikit-learn")},
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Write the fit-noise artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    artifact = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1) + "\n")
    for row in artifact["datasets"]:
        logger.info(
            "%-6s delta %+.4f sd %.4f (hardware better in %d/%d); fit noise sd %.4f",
            row["dataset"],
            row["delta_mean"],
            row["delta_sd"],
            row["hardware_better_in"],
            row["redraws"],
            row["fit_noise_sd"],
        )
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
