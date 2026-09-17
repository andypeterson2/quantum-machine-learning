"""Measure the accuracies the model documentation claims.

The per-model ``MODELS.md`` files are served by ``/model-info``, so their
numbers are part of what this project publishes — and they were written by hand.
Two of them contradicted the committed exports (Iris Linear claimed ~95-97%
against a measured 90%; BB84 Linear claimed ~92-94% against 95.4%), and the rest
had no source at all.

This runs each model through the real plugin, Trainer and Evaluator at the
plugin's own default hyper-parameters, with a fixed seed, and writes accuracy,
its 95% interval, the sample count and provenance to
``exports/benchmarks.json``. ``tests/test_model_docs.py`` then holds the
documents to that file, so a claim cannot drift from what was measured.

Not every model is here. The Qiskit MNIST models evaluate a shot-sampled
circuit per sample, which is hours on a test split, so they stay unmeasured and
their documents say so rather than quoting a number nobody produced.

Usage::

    python tools/benchmark.py            # the fast set (seconds to a few minutes)
    python tools/benchmark.py --slow     # adds the MNIST CNN-backbone models
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from classifiers.evaluator import Evaluator  # noqa: E402
from classifiers.plugin_registry import discover_plugins, get_plugin  # noqa: E402
from classifiers.trainer import Trainer  # noqa: E402
from classifiers.web_export import provenance_base  # noqa: E402

logger = logging.getLogger("benchmark")

OUT_PATH = REPO_ROOT / "exports" / "benchmarks.json"

#: One seed for every run, so the file is reproducible (see classifiers.seeding).
SEED = 0

#: (dataset, model type) pairs measured by default — all of them finish in
#: seconds to a couple of minutes on a laptop CPU.
FAST: list[tuple[str, str]] = [
    ("iris", "Linear"),
    ("iris", "SVM"),
    ("iris", "QVC"),
    ("bb84", "Linear"),
    ("bb84", "SVM"),
    ("bb84", "QVC"),
    ("mnist", "Linear"),
    ("mnist", "SVM"),
]

#: Added by ``--slow``: MNIST's CNN-backbone models, minutes each on CPU.
SLOW: list[tuple[str, str]] = [
    ("mnist", "CNN"),
    ("mnist", "Quadratic"),
    ("mnist", "Polynomial"),
]


def measure(dataset: str, model_type: str) -> dict:
    """Train and evaluate one model the way the platform does."""
    plugin = get_plugin(dataset)
    if plugin is None:
        raise ValueError(f"no dataset plugin named {dataset!r}")
    model_types = plugin.get_model_types()
    if model_type not in model_types:
        raise LookupError(f"{dataset} has no {model_type} model here (optional dependency?)")

    hp = plugin.get_default_hyperparams()
    result = Trainer(
        model_cls=model_types[model_type],
        train_loader=plugin.get_train_loader(hp["batch_size"]),
        dataset=dataset,
        epochs=hp["epochs"],
        lr=hp["lr"],
        seed=SEED,
    ).train()

    ev = Evaluator().evaluate(
        result.model,
        plugin.get_test_loader(hp["batch_size"]),
        plugin.num_classes,
        plugin.class_labels,
    )
    logger.info(
        "%s/%s: %.4f (95%% CI %.4f-%.4f, n=%d)",
        dataset, model_type, ev.accuracy, ev.accuracy_ci[0], ev.accuracy_ci[1], ev.num_samples,
    )
    return {
        "dataset": dataset,
        "model_type": model_type,
        "accuracy": round(ev.accuracy, 4),
        "accuracy_ci": list(ev.accuracy_ci),
        "n": ev.num_samples,
        "num_params": ev.num_params,
        "training": {"seed": SEED, **hp},
    }


def main() -> None:
    """Measure every selected model and write the artifact."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slow", action="store_true", help="also measure the MNIST CNN-backbone models"
    )
    args = parser.parse_args()

    discover_plugins()
    pairs = FAST + (SLOW if args.slow else [])
    results = [measure(dataset, model_type) for dataset, model_type in pairs]

    payload = {
        "kind": "benchmarks",
        "results": results,
        "provenance": provenance_base(
            {
                "model": "all",
                "trainer": "classifiers.trainer.Trainer",
                "protocol": (
                    "trained at each plugin's default hyper-parameters with seed "
                    f"{SEED}; accuracy and its 95% Wilson interval measured on the "
                    "plugin's test split"
                ),
            },
            {
                "torch": torch.__version__,
                "scikit-learn": importlib.metadata.version("scikit-learn"),
            },
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s (%d models)", OUT_PATH, len(results))


if __name__ == "__main__":
    main()
