"""Measure the accuracies the model documentation claims.

The per-model ``MODELS.md`` files are served by ``/model-info``, so their
numbers are part of what this project publishes — and they were written by hand.
Two of them contradicted the committed exports (Iris Linear claimed ~95-97%
against a measured 90%; BB84 Linear claimed ~92-94% against 95.4%), and the rest
had no source at all.

This runs each model through the real plugin, Trainer and Evaluator at the
plugin's own default hyper-parameters, once per seed, and writes accuracy, its
95% interval, the sample count and provenance to ``exports/benchmarks.json``.
``tests/test_model_docs.py`` then holds the documents to that file, so a claim
cannot drift from what was measured.

Each model is measured at several seeds because the interval covers only one of
the two ways these numbers move. On Iris the seed moves accuracy further than
the test split does — the QVC spans 70% to 86.7% on initialisation alone — and a
reader given one run and its interval cannot see that. The published accuracy
stays ``HEADLINE_SEED``'s run, which keeps it reproducible and keeps its Wilson
interval defined; the spread across seeds rides beside it in ``summary``.

Not every model is here. The Qiskit MNIST models evaluate a shot-sampled
circuit per sample, which is hours on a test split, so they stay unmeasured and
their documents say so rather than quoting a number nobody produced.

Usage::

    python tools/benchmark.py              # the fast set, ten seeds each
    python tools/benchmark.py --slow       # adds the MNIST CNN-backbone models, three seeds
    python tools/benchmark.py --seeds 0    # one seed, for a smoke run
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from classifiers.evaluator import Evaluator  # noqa: E402
from classifiers.plugin_registry import discover_plugins, get_plugin  # noqa: E402
from classifiers.stats import spread  # noqa: E402
from classifiers.trainer import Trainer  # noqa: E402
from classifiers.web_export import provenance_base  # noqa: E402

logger = logging.getLogger("benchmark")

OUT_PATH = REPO_ROOT / "exports" / "benchmarks.json"

#: The seed whose run is published. The documents and the README quote this one,
#: so it has to appear in every run list or the artifact stops backing them.
HEADLINE_SEED = 0

#: Seeds every fast pair is repeated over. Ten is about ten minutes for the set.
FAST_SEEDS: tuple[int, ...] = tuple(range(10))

#: Seeds for the slow pairs, which are minutes each. Ten would be over an hour,
#: and a tier nobody reruns stays single-seed while looking like the others.
SLOW_SEEDS: tuple[int, ...] = (0, 1, 2)

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

#: Added by ``--slow``: MNIST's CNN, minutes per seed on CPU.
SLOW: list[tuple[str, str]] = [
    ("mnist", "CNN"),
]


def _resolve(dataset: str, model_type: str):
    """The plugin, the model class and the hyper-parameters, or raise.

    Checked once per pair rather than once per seed.
    """
    plugin = get_plugin(dataset)
    if plugin is None:
        raise ValueError(f"no dataset plugin named {dataset!r}")
    model_types = plugin.get_model_types()
    if model_type not in model_types:
        raise LookupError(f"{dataset} has no {model_type} model here (optional dependency?)")
    return plugin, model_types[model_type], plugin.get_default_hyperparams()


def measure(dataset: str, model_type: str, seed: int) -> dict:
    """Train and evaluate one model at one seed, the way the platform does."""
    plugin, model_cls, hp = _resolve(dataset, model_type)
    result = Trainer(
        model_cls=model_cls,
        train_loader=plugin.get_train_loader(hp["batch_size"]),
        dataset=dataset,
        epochs=hp["epochs"],
        lr=hp["lr"],
        seed=seed,
    ).train()

    ev = Evaluator().evaluate(
        result.model,
        plugin.get_test_loader(hp["batch_size"]),
        plugin.num_classes,
        plugin.class_labels,
    )
    logger.info(
        "%s/%s seed %d: %.4f (95%% CI %.4f-%.4f, n=%d)",
        dataset, model_type, seed,
        ev.accuracy, ev.accuracy_ci[0], ev.accuracy_ci[1], ev.num_samples,
    )
    return {
        "seed": seed,
        "accuracy": round(ev.accuracy, 4),
        "accuracy_ci": list(ev.accuracy_ci),
        "n": ev.num_samples,
        "num_params": ev.num_params,
    }


def summarise(runs: list[dict]) -> dict:
    """Mean and spread of one pair's accuracy across its seeds."""
    accuracies = [run["accuracy"] for run in runs]
    mean, deviation, (low, high) = spread(accuracies)
    headline = next(run["accuracy"] for run in runs if run["seed"] == HEADLINE_SEED)
    return {
        "seeds": len(runs),
        "accuracy_mean": mean,
        "accuracy_stdev": deviation,
        "accuracy_range": [low, high],
        "headline_seed": HEADLINE_SEED,
        "headline_is_best": headline == high,
    }


def measure_pair(dataset: str, model_type: str, seeds: Sequence[int]) -> dict:
    """Every seed for one pair, with the headline seed's run as the published one.

    The published accuracy stays a single reproducible run: a Wilson interval
    needs an integer success count, and a mean across seeds has none.
    """
    if HEADLINE_SEED not in seeds:
        raise ValueError(f"seeds {list(seeds)} omit the published seed {HEADLINE_SEED}")
    _, _, hp = _resolve(dataset, model_type)
    runs = [measure(dataset, model_type, seed) for seed in seeds]
    headline = next(run for run in runs if run["seed"] == HEADLINE_SEED)
    return {
        "dataset": dataset,
        "model_type": model_type,
        "accuracy": headline["accuracy"],
        "accuracy_ci": headline["accuracy_ci"],
        "n": headline["n"],
        "num_params": headline["num_params"],
        "training": {"seed": HEADLINE_SEED, **hp},
        "runs": [{k: run[k] for k in ("seed", "accuracy", "accuracy_ci")} for run in runs],
        "summary": summarise(runs),
    }


def main() -> None:
    """Measure every selected model and write the artifact."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slow", action="store_true", help="also measure the MNIST CNN-backbone models"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=f"seeds to repeat every pair over (default {len(FAST_SEEDS)} fast, "
             f"{len(SLOW_SEEDS)} slow)",
    )
    args = parser.parse_args()

    discover_plugins()
    override = tuple(args.seeds) if args.seeds else None
    tiers = [(FAST, override or FAST_SEEDS)]
    if args.slow:
        tiers.append((SLOW, override or SLOW_SEEDS))
    results = [
        measure_pair(dataset, model_type, seeds)
        for pairs, seeds in tiers
        for dataset, model_type in pairs
    ]

    payload = {
        "kind": "benchmarks",
        "results": results,
        "provenance": provenance_base(
            {
                "model": "all",
                "trainer": "classifiers.trainer.Trainer",
                "headline_seed": HEADLINE_SEED,
                "protocol": (
                    "each (dataset, model) trained once per seed at the plugin's own "
                    f"default hyper-parameters; the published accuracy and its 95% "
                    f"Wilson interval are seed {HEADLINE_SEED}'s run on the plugin's "
                    "test split, which is sampling error only. summary.accuracy_stdev "
                    "is the spread of the same measurement across seeds. The test "
                    "split does not depend on the seed, so the two components do not "
                    "mix"
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
    logger.info(
        "wrote %s (%d models, %d runs)",
        OUT_PATH, len(results), sum(len(r["runs"]) for r in results),
    )


if __name__ == "__main__":
    main()
