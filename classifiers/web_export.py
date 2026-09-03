"""Export browser-runnable classifier weights for the portfolio site.

The portfolio site's zero-backend inference demo (``infer.js``) runs a compact
linear model entirely in the visitor's browser. The site claims those weights
come from *the same models this platform trains* — this module is what makes
that claim true. It trains each dataset's ``Linear`` model exactly the way the
platform itself does — same dataset plugin, same :class:`~classifiers.trainer.
Trainer`, same default hyper-parameters — and writes the weights plus the
exact normalisation constants the browser must reproduce to
``exports/web/<dataset>.json``, stamped with provenance (source commit, date,
framework versions, seed).

Workflow: ``make export-web`` regenerates ``exports/web/``; ``make sync-web``
copies the files into the website checkout. ``tests/test_web_export.py``
re-verifies the committed exports against the live plugins on every CI run,
so a plugin change that would silently strand the browser weights fails CI
here instead.

Determinism note: a re-run on the same machine/versions reproduces the same
weights (seeded); across platforms the floats may differ, which is why the
drift check re-evaluates the *committed* weights rather than retraining.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from classifiers.plugin_registry import discover_plugins, get_plugin
from classifiers.trainer import Trainer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from classifiers.dataset_plugin import DatasetPlugin

logger = logging.getLogger(__name__)

#: Global RNG seed for every export run.
SEED = 0

#: Repo root (parent of the ``classifiers`` package).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Canonical output directory, committed so CI can drift-check the exports.
OUT_DIR = REPO_ROOT / "exports" / "web"

#: Browser canvas pixel scale for image datasets (0-255 → 0-1 before z-score).
CANVAS_SCALE = 255.0


def evaluate_payload(payload: dict, test_loader: DataLoader) -> float:
    """Measure a payload's accuracy on a plugin's real test split.

    Runs the exported ``weight``/``bias`` (not the in-memory model) over the
    already-normalised tensors the plugin serves, so the exporter and the CI
    drift check score the exact same artifact the browser downloads.

    Args:
        payload:     An export payload with ``weight`` and ``bias`` lists.
        test_loader: The plugin's test loader (normalised samples).

    Returns:
        Top-1 accuracy in ``[0, 1]``.
    """
    weight = torch.tensor(payload["weight"], dtype=torch.float32)
    bias = torch.tensor(payload["bias"], dtype=torch.float32)
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = xb.flatten(1) @ weight.T + bias
            correct += int((logits.argmax(1) == yb).sum().item())
            total += int(yb.numel())
    return correct / total


def _git(*args: str) -> str:
    """Run a git command in the repo root and return its stripped stdout."""
    result = subprocess.run(  # noqa: S603 — fixed argv, no user input
        ["git", *args],  # noqa: S607 — git from PATH is the intended toolchain
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def provenance_base(training: dict, versions: dict) -> dict:
    """The provenance block every export carries — one definition, two exporters.

    The exports themselves are excluded from the dirty check: a fresh export
    always rewrites them, and they must not count as dirt in their own
    provenance.
    """
    return {
        "source_repo": "quantum-machine-learning",
        "source_sha": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain", "--", ".", ":(exclude)exports/web")),
        "exported_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "seed": SEED,
        "training": training,
        "versions": {"python": platform.python_version(), **versions},
    }


def _provenance(dataset: str, hyperparams: dict) -> dict:
    """Provenance for a Linear platform-model export."""
    data_dep = "scikit-learn" if dataset == "iris" else "torchvision"
    return provenance_base(
        {"model": "Linear", "trainer": "classifiers.trainer.Trainer", **hyperparams},
        {"torch": torch.__version__, data_dep: importlib.metadata.version(data_dep)},
    )


def _train_linear(plugin: DatasetPlugin) -> tuple[torch.nn.Linear, dict]:
    """Train the plugin's ``Linear`` model the way the platform does.

    Uses the plugin's own train loader and default hyper-parameters with the
    real :class:`Trainer` — no re-implemented data pipeline, so the exported
    weights cannot drift from what the backend trains.

    Returns:
        The trained model's single linear layer, and the hyper-parameters used.
    """
    hp = plugin.get_default_hyperparams()
    trainer = Trainer(
        model_cls=plugin.get_model_types()["Linear"],
        train_loader=plugin.get_train_loader(hp["batch_size"]),
        dataset=plugin.name,
        epochs=hp["epochs"],
        lr=hp["lr"],
    )
    result = trainer.train()
    return result.model.fc, hp


def _iris_feature_ranges(plugin: DatasetPlugin, batch_size: int) -> list[list[float]]:
    """Recover per-feature raw min/max from the plugin's normalised tensors.

    De-normalises every sample the plugin serves (train + val + test covers
    the full dataset), so the demo form's slider bounds come from the same
    pipeline as everything else.
    """
    mean_list, std_list = plugin.normalization()
    mean = torch.tensor(mean_list)
    std = torch.tensor(std_list)
    batches = []
    for loader in (
        plugin.get_train_loader(batch_size),
        plugin.get_val_loader(batch_size),
        plugin.get_test_loader(batch_size),
    ):
        batches.extend(xb * std + mean for xb, _ in loader)
    raw = torch.cat(batches)
    return [[float(raw[:, i].min()), float(raw[:, i].max())] for i in range(raw.shape[1])]


def _payload(plugin: DatasetPlugin) -> dict:
    """Train and assemble the browser payload for one dataset plugin."""
    fc, hp = _train_linear(plugin)
    payload: dict = {
        "kind": "linear",
        "dataset": plugin.name,
        "input": fc.in_features,
        "classes": list(plugin.class_labels),
        "weight": fc.weight.detach().tolist(),
        "bias": fc.bias.detach().tolist(),
    }
    if plugin.name == "iris":
        mean, std = plugin.normalization()
        payload["features"] = list(plugin.feature_names or [])
        payload["normalize"] = {"scale": 1.0, "mean": mean, "std": std}
        payload["feature_ranges"] = _iris_feature_ranges(plugin, hp["batch_size"])
    else:
        from classifiers.datasets.mnist.plugin import MNIST_MEAN, MNIST_STD

        payload["normalize"] = {
            "scale": CANVAS_SCALE,
            "mean": [MNIST_MEAN],
            "std": [MNIST_STD],
        }
    acc = evaluate_payload(payload, plugin.get_test_loader(512))
    payload["test_accuracy"] = round(acc, 4)
    payload["provenance"] = _provenance(plugin.name, hp)
    return payload


def export_dataset(name: str) -> Path:
    """Export one dataset's linear model to ``exports/web/<name>.json``."""
    plugin = get_plugin(name)
    if plugin is None:
        raise ValueError(f"no dataset plugin named {name!r}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    payload = _payload(plugin)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.json"
    out.write_text(json.dumps(payload) + "\n")
    size_kb = out.stat().st_size // 1024
    logger.info("%s  test_acc=%.4f  (%d KB)", out.name, payload["test_accuracy"], size_kb)
    return out


def main() -> None:
    """Export every browser-served dataset."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    discover_plugins()
    for name in ("iris", "mnist"):
        export_dataset(name)


if __name__ == "__main__":
    main()
