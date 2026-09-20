"""Does distillation help the student? Measure it, don't remember it.

The distillation term was changed from MSE on raw logits to temperature-softened
KL (commit 4311232) on the strength of a three-seed MNIST comparison — a linear
student at 92.05% alone against 85.95% distilled. Those numbers live only in
that commit message: no script, no seeds, no artifact. Nothing measured the new
loss at all, so the feature's own evidence could not be regenerated.

This runs the comparison end to end: for each seed, train the teacher, then
train the same student twice — alone, and distilled from that teacher — and
score all three on the test split. Results go to ``exports/distillation.json``
with per-seed accuracies, their intervals, the spread across seeds, and
provenance.

Runtime is a few minutes per seed on a laptop CPU (the teacher is a CNN), so
this is a committed artifact rather than something CI runs.

Usage::

    python tools/distillation_experiment.py [--seeds 0 1 2] [--dataset mnist]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from classifiers.evaluator import Evaluator  # noqa: E402
from classifiers.plugin_registry import discover_plugins, get_plugin  # noqa: E402
from classifiers.trainer import Trainer  # noqa: E402
from classifiers.training_config import TrainingConfig  # noqa: E402
from classifiers.web_export import provenance_base  # noqa: E402

logger = logging.getLogger("distillation")

OUT_PATH = REPO_ROOT / "exports" / "distillation.json"

#: Seeds to repeat the comparison over — three, as the original claim used.
DEFAULT_SEEDS = (0, 1, 2)

#: The pairing the model docs ask about: can a linear model absorb a CNN's
#: knowledge?
TEACHER = "CNN"
STUDENT = "Linear"

#: The route's own defaults, so this measures what a caller would get.
DISTILL_WEIGHT = 0.5
DISTILL_TEMPERATURE = 4.0


def _train(plugin, model_type: str, seed: int, config: TrainingConfig | None = None):
    """Train one arm of the comparison.

    No validation loader, deliberately. The trainer returns the best-validation
    checkpoint when it has one and the final weights when it does not, so giving
    the distilled arm a loader and the solo arm none compared two model-selection
    rules as well as two losses. The config here sets no patience, so the loader
    bought nothing else.
    """
    hp = plugin.get_default_hyperparams()
    return Trainer(
        model_cls=plugin.get_model_types()[model_type],
        train_loader=plugin.get_train_loader(hp["batch_size"]),
        dataset=plugin.name,
        epochs=hp["epochs"],
        lr=hp["lr"],
        config=config,
        seed=seed,
    ).train()


def _score(plugin, model) -> dict:
    hp = plugin.get_default_hyperparams()
    ev = Evaluator().evaluate(
        model,
        plugin.get_test_loader(hp["batch_size"]),
        plugin.num_classes,
        plugin.class_labels,
    )
    return {
        "accuracy": round(ev.accuracy, 4),
        "accuracy_ci": list(ev.accuracy_ci),
        "n": ev.num_samples,
    }


def run_seed(dataset: str, seed: int) -> dict:
    """Teacher, student alone, and student distilled — all at one seed."""
    plugin = get_plugin(dataset)
    if plugin is None:
        raise ValueError(f"no dataset plugin named {dataset!r}")

    teacher = _train(plugin, TEACHER, seed)
    alone = _train(plugin, STUDENT, seed)
    distilled = _train(
        plugin,
        STUDENT,
        seed,
        TrainingConfig(
            teacher_model=teacher.model,
            distill_weight=DISTILL_WEIGHT,
            distill_temperature=DISTILL_TEMPERATURE,
        ),
    )

    scores = {
        "seed": seed,
        "teacher": _score(plugin, teacher.model),
        "student_alone": _score(plugin, alone.model),
        "student_distilled": _score(plugin, distilled.model),
    }
    scores["delta"] = round(
        scores["student_distilled"]["accuracy"] - scores["student_alone"]["accuracy"], 4
    )
    logger.info(
        "seed %d: teacher %.4f, student %.4f alone -> %.4f distilled (%+.4f)",
        seed,
        scores["teacher"]["accuracy"],
        scores["student_alone"]["accuracy"],
        scores["student_distilled"]["accuracy"],
        scores["delta"],
    )
    return scores


def summarise(runs: list[dict]) -> dict:
    """Mean and spread of the effect across seeds."""
    deltas = [r["delta"] for r in runs]
    alone = [r["student_alone"]["accuracy"] for r in runs]
    distilled = [r["student_distilled"]["accuracy"] for r in runs]
    spread = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    return {
        "seeds": len(runs),
        "student_alone_mean": round(statistics.fmean(alone), 4),
        "student_distilled_mean": round(statistics.fmean(distilled), 4),
        "delta_mean": round(statistics.fmean(deltas), 4),
        "delta_stdev": round(spread, 4),
        "delta_range": [round(min(deltas), 4), round(max(deltas), 4)],
        "helps": bool(min(deltas) > 0),
    }


def main() -> None:
    """Run every seed and write the artifact."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--dataset", default="mnist")
    args = parser.parse_args()

    discover_plugins()
    runs = [run_seed(args.dataset, seed) for seed in args.seeds]
    summary = summarise(runs)
    logger.info(
        "distillation changes the student by %+.4f on average (stdev %.4f over %d seeds)",
        summary["delta_mean"], summary["delta_stdev"], summary["seeds"],
    )

    payload = {
        "kind": "distillation-experiment",
        "dataset": args.dataset,
        "teacher": TEACHER,
        "student": STUDENT,
        "distill_weight": DISTILL_WEIGHT,
        "distill_temperature": DISTILL_TEMPERATURE,
        "runs": runs,
        "summary": summary,
        "provenance": provenance_base(
            {
                "model": f"{STUDENT} distilled from {TEACHER}",
                "trainer": "classifiers.trainer.Trainer",
                "protocol": (
                    "per seed: train the teacher, then the student alone and distilled "
                    "from it, all at the plugin's default hyper-parameters and all "
                    "returning their final weights; accuracy and its 95% Wilson interval "
                    "measured on the test split"
                ),
                "loss": "KL(teacher || student) on outputs softened at T, scaled by T^2",
            },
            {
                "torch": torch.__version__,
                "scikit-learn": importlib.metadata.version("scikit-learn"),
            },
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
