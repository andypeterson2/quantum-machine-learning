"""Accuracy as a function of the one number the quantum step contributes.

The deployed rule decides by ``sign(v · w)``, and ``w = alpha1*x1 + alpha2*x2``
over two fixed unit targets. Scaling alpha leaves the sign alone, so the only
quantity that reaches a prediction is the ratio ``alpha1 / -alpha2``: one
scalar, read off the hardware as 1.0327 against the exact solution's 1.

:mod:`tools.alpha_sensitivity` measures what the gap between those two values
costs. This measures the shape of the curve they sit on, which answers a
question the comparison cannot: is the exact solution where the accuracy is
best? It is not, on any dataset here, so the readout error being small is not
what makes the classifier work and a more precise alpha would not improve it.

The map, the orientation and the split come from the committed export, so the
ratio is the only thing that moves. Output: ``exports/alpha-ratio-sweep.json``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers import qsvm_export  # noqa: E402
from classifiers.qsvm_rule import decide, weight_vector  # noqa: E402
from classifiers.stats import wilson_interval  # noqa: E402
from classifiers.web_export import OUT_DIR, provenance_base  # noqa: E402

logger = logging.getLogger(__name__)

WEB_DIR = OUT_DIR
ARTIFACT = REPO_ROOT / "exports" / "alpha-ratio-sweep.json"

#: Ratios to score. Wide enough that the peak is inside it rather than at an
#: edge, and fine enough that the peak's position is not a grid artifact.
RATIO_GRID = np.round(np.arange(0.50, 2.001, 0.01), 2)

#: The exact least-squares solution's ratio, and the paper's own sign pattern.
EXACT_RATIO = 1.0


def ratio_of(alpha: np.ndarray) -> float:
    """The one scalar an alpha contributes, ``alpha1 / -alpha2``."""
    return float(alpha[0] / -alpha[1])


def sweep(dataset: str) -> dict:
    """Score the committed rule across :data:`RATIO_GRID`.

    Args:
        dataset: A key of :data:`classifiers.qsvm_export.QSVM_DATASETS`.

    Returns:
        The grid, the accuracy at each point, and where the named ratios land.

    Raises:
        ValueError: If the committed ``w`` is not the hardware alpha's, which
            would mean the export and this sweep disagree about the rule.
    """
    spec = qsvm_export.QSVM_DATASETS[dataset]
    payload = json.loads((WEB_DIR / f"qsvm-{dataset}.json").read_text())
    committed = np.array(payload["w"])
    if not np.allclose(committed, weight_vector(qsvm_export.ALPHA_SHOTS)):
        raise ValueError(f"{dataset}: the committed w is not the hardware alpha's")

    split = spec["features_fn"]()
    flipped = payload["classes"] != list(spec["classes"])
    labels = -split.test_y if flipped else split.test_y
    n = len(labels)

    def accuracy_at(ratio: float) -> tuple[float, int]:
        w = weight_vector(np.array([ratio, -1.0]))
        hits = int((decide(w, payload["map"], split.test_x) == labels).sum())
        return hits / n, hits

    accuracies, hits_at = [], []
    for ratio in RATIO_GRID:
        acc, hits = accuracy_at(float(ratio))
        accuracies.append(round(acc, 4))
        hits_at.append(hits)

    best = int(np.argmax(accuracies))
    hardware_ratio = ratio_of(qsvm_export.ALPHA_SHOTS)
    marks = {}
    for name, ratio in (("exact", EXACT_RATIO), ("hardware", hardware_ratio)):
        acc, hits = accuracy_at(ratio)
        marks[name] = {
            "ratio": round(ratio, 6),
            "accuracy": round(acc, 4),
            "accuracy_ci": list(wilson_interval(hits, n)),
        }
    return {
        "dataset": dataset,
        "n": n,
        "ratios": RATIO_GRID.tolist(),
        "accuracies": accuracies,
        "best": {
            "ratio": float(RATIO_GRID[best]),
            "accuracy": accuracies[best],
            "accuracy_ci": list(wilson_interval(hits_at[best], n)),
        },
        **marks,
        "exact_is_optimal": bool(abs(float(RATIO_GRID[best]) - EXACT_RATIO) < 1e-9),
    }


def build() -> dict:
    """Sweep every dataset the exporter ships, skipping any without its data."""
    datasets, skipped = [], []
    for dataset in qsvm_export.QSVM_DATASETS:
        try:
            datasets.append(sweep(dataset))
        except OSError as exc:  # the openml cache is absent on this machine
            logger.warning("skipping %s: %s", dataset, exc)
            skipped.append(dataset)
    return {
        "kind": "alpha-ratio-sweep",
        "deployed_alpha": qsvm_export.ALPHA_SHOTS.tolist(),
        "deployed_ratio": round(ratio_of(qsvm_export.ALPHA_SHOTS), 6),
        "exact_ratio": EXACT_RATIO,
        "datasets": datasets,
        "skipped": skipped,
        "provenance": provenance_base(
            {
                "model": "QSVM",
                "paper": "arXiv:1909.11988",
                "protocol": (
                    "held-out split, committed map and orientation; the alpha ratio is "
                    "the only quantity varied"
                ),
            },
            {"numpy": np.__version__, "scikit-learn": importlib.metadata.version("scikit-learn")},
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Write the sweep artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    artifact = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1) + "\n")
    for row in artifact["datasets"]:
        logger.info(
            "%-6s best ratio %.2f (%.4f); exact %.4f, hardware %.4f",
            row["dataset"],
            row["best"]["ratio"],
            row["best"]["accuracy"],
            row["exact"]["accuracy"],
            row["hardware"]["accuracy"],
        )
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
