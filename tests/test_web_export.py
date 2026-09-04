"""Drift check for the browser model exports (``exports/web/``).

The portfolio site serves these weights for in-browser inference and claims
they come from the same models this platform trains. These tests enforce that
claim against the *committed* export files: if a dataset plugin's split,
normalisation, labels, or feature set changes without re-running
``make export-web``, CI fails here instead of the browser demo silently
drifting from the backend.

The accuracy checks deliberately re-evaluate the committed weights (pure
inference — deterministic across platforms) rather than retraining, and the
MNIST accuracy check only runs when the dataset is already cached locally —
CI never downloads datasets.
"""

from __future__ import annotations

import json
import re

import pytest
import torch

from classifiers.datasets.mnist.plugin import MNIST_MEAN, MNIST_STD
from classifiers.plugin_registry import discover_plugins, get_plugin
from classifiers.web_export import OUT_DIR, REPO_ROOT, evaluate_payload

# The actual test-set file, not just the directory — an empty MNIST/ dir exists
# in fresh checkouts and must still skip.
MNIST_CACHE = REPO_ROOT / "classifiers" / "data" / "MNIST" / "raw" / "t10k-images-idx3-ubyte"

PROVENANCE_KEYS = {
    "source_repo",
    "source_sha",
    "source_dirty",
    "exported_at",
    "seed",
    "training",
    "versions",
}


def _load(name: str) -> dict:
    path = OUT_DIR / f"{name}.json"
    assert path.is_file(), f"missing committed export {path} — run `make export-web`"
    return json.loads(path.read_text())


@pytest.fixture(scope="module", autouse=True)
def _plugins() -> None:
    discover_plugins()


class TestProvenance:
    """Every export must say exactly where it came from."""

    @pytest.mark.parametrize("name", ["iris", "mnist", "bb84"])
    def test_provenance_block(self, name: str) -> None:
        prov = _load(name)["provenance"]
        assert set(prov) >= PROVENANCE_KEYS
        assert prov["source_repo"] == "quantum-machine-learning"
        assert re.fullmatch(r"[0-9a-f]{40}", prov["source_sha"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", prov["exported_at"])
        assert isinstance(prov["seed"], int)
        assert prov["training"]["model"] == "Linear"
        assert {"epochs", "batch_size", "lr"} <= set(prov["training"])
        assert "torch" in prov["versions"]


class TestIrisExport:
    """Full drift check — Iris ships with scikit-learn, no download needed."""

    def test_metadata_matches_plugin(self) -> None:
        payload = _load("iris")
        plugin = get_plugin("iris")
        assert plugin is not None
        assert payload["kind"] == "linear"
        assert payload["input"] == 4
        assert payload["classes"] == plugin.class_labels
        assert payload["features"] == plugin.feature_names

    def test_normalization_matches_plugin(self) -> None:
        """The exact constants the browser applies must match the live plugin."""
        payload = _load("iris")
        plugin = get_plugin("iris")
        assert plugin is not None
        mean, std = plugin.normalization()
        assert payload["normalize"]["scale"] == 1.0
        assert payload["normalize"]["mean"] == pytest.approx(mean, abs=1e-6)
        assert payload["normalize"]["std"] == pytest.approx(std, abs=1e-6)

    def test_weight_shapes(self) -> None:
        payload = _load("iris")
        weight = torch.tensor(payload["weight"])
        bias = torch.tensor(payload["bias"])
        assert weight.shape == (3, 4)
        assert bias.shape == (3,)
        assert len(payload["feature_ranges"]) == 4
        assert all(lo < hi for lo, hi in payload["feature_ranges"])

    def test_accuracy_claim_reproduces(self) -> None:
        """The committed weights must score their claimed accuracy on the
        plugin's real test split — the strongest drift signal we have."""
        payload = _load("iris")
        plugin = get_plugin("iris")
        assert plugin is not None
        acc = evaluate_payload(payload, plugin.get_test_loader(64))
        assert round(acc, 4) == payload["test_accuracy"]
        assert acc >= 0.9


class TestMnistExport:
    """Metadata always; accuracy only when the dataset is cached (no CI download)."""

    def test_metadata_matches_plugin(self) -> None:
        payload = _load("mnist")
        plugin = get_plugin("mnist")
        assert plugin is not None
        assert payload["kind"] == "linear"
        assert payload["input"] == 28 * 28
        assert payload["classes"] == plugin.class_labels

    def test_normalization_matches_plugin(self) -> None:
        payload = _load("mnist")
        norm = payload["normalize"]
        assert norm["scale"] == 255.0
        assert norm["mean"] == pytest.approx([MNIST_MEAN])
        assert norm["std"] == pytest.approx([MNIST_STD])

    def test_weight_shapes(self) -> None:
        payload = _load("mnist")
        weight = torch.tensor(payload["weight"])
        bias = torch.tensor(payload["bias"])
        assert weight.shape == (10, 28 * 28)
        assert bias.shape == (10,)

    @pytest.mark.skipif(
        not MNIST_CACHE.is_file(), reason="MNIST not cached locally; CI never downloads"
    )
    def test_accuracy_claim_reproduces(self) -> None:
        payload = _load("mnist")
        plugin = get_plugin("mnist")
        assert plugin is not None
        acc = evaluate_payload(payload, plugin.get_test_loader(512))
        assert round(acc, 4) == payload["test_accuracy"]
        assert acc >= 0.85


class TestBb84Export:
    """Full drift check — bb84 data is self-generated, so CI runs everything."""

    def test_metadata_matches_plugin(self) -> None:
        payload = _load("bb84")
        plugin = get_plugin("bb84")
        assert plugin is not None
        assert payload["kind"] == "linear"
        assert payload["input"] == 2
        assert payload["classes"] == plugin.class_labels
        assert payload["features"] == plugin.feature_names

    def test_normalization_matches_plugin(self) -> None:
        """The exact constants the browser applies must match the live plugin."""
        payload = _load("bb84")
        plugin = get_plugin("bb84")
        assert plugin is not None
        mean, std = plugin.normalization()
        assert payload["normalize"]["scale"] == 1.0
        assert payload["normalize"]["mean"] == pytest.approx(mean, abs=1e-6)
        assert payload["normalize"]["std"] == pytest.approx(std, abs=1e-6)

    def test_weight_shapes(self) -> None:
        payload = _load("bb84")
        weight = torch.tensor(payload["weight"])
        bias = torch.tensor(payload["bias"])
        assert weight.shape == (2, 2)
        assert bias.shape == (2,)
        assert len(payload["feature_ranges"]) == 2
        assert all(lo < hi for lo, hi in payload["feature_ranges"])

    def test_accuracy_claim_reproduces(self) -> None:
        """The committed weights must score their claimed accuracy on the
        plugin's real (seeded, regenerated) test split."""
        payload = _load("bb84")
        plugin = get_plugin("bb84")
        assert plugin is not None
        acc = evaluate_payload(payload, plugin.get_test_loader(256))
        assert round(acc, 4) == payload["test_accuracy"]
        assert acc >= 0.9


# ── QSVM paper-recreation exports (classifiers/qsvm_export.py) ────────────────

from pathlib import Path  # noqa: E402

from sklearn.datasets import get_data_home  # noqa: E402

from classifiers import qsvm_export  # noqa: E402

# fetch_openml stores the ARFF cache under <data_home>/openml; presence of any
# openml cache is our (coarse but CI-safe) signal that mnist_784 is available.
_OPENML_DIR = Path(get_data_home()) / "openml"
MNIST_OPENML_CACHE = _OPENML_DIR.is_dir() and any(_OPENML_DIR.rglob("*.gz"))


class TestQsvmProvenance:
    """The qsvm exports carry the same provenance discipline, QSVM-flavoured."""

    @pytest.mark.parametrize("name", ["qsvm-iris", "qsvm-mnist", "qsvm-bb84"])
    def test_provenance_block(self, name: str) -> None:
        prov = _load(name)["provenance"]
        assert set(prov) >= PROVENANCE_KEYS
        assert prov["source_repo"] == "quantum-machine-learning"
        assert re.fullmatch(r"[0-9a-f]{40}", prov["source_sha"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", prov["exported_at"])
        assert prov["training"]["model"] == "QSVM"
        assert prov["training"]["paper"] == "arXiv:1909.11988"
        assert "scikit-learn" in prov["versions"]


class TestQsvmSchema:
    """Both files honour the browser contract for kind=qsvm."""

    @pytest.mark.parametrize("name", ["qsvm-iris", "qsvm-mnist", "qsvm-bb84"])
    def test_contract(self, name: str) -> None:
        payload = _load(name)
        assert payload["kind"] == "qsvm"
        assert payload["dataset"] in {"iris", "mnist", "bb84"}
        assert len(payload["classes"]) == 2
        assert len(payload["w"]) == 2
        assert set(payload["map"]) == {"a", "b", "c", "d"}
        assert all(isinstance(v, float) for v in payload["map"].values())
        assert len(payload["features"]) == 2
        assert payload["raw_input"] in {"features", "pixels"}
        assert 0.0 < payload["test_accuracy"] <= 1.0
        assert payload["num_params"] == 6


class TestQsvmIrisDrift:
    """Full re-derivation in CI — Iris ships with scikit-learn, no download."""

    def test_map_and_weights_rederive(self) -> None:
        payload = _load("qsvm-iris")
        feats, labels = qsvm_export.iris_features()
        t1 = feats[labels == 1].mean(axis=0)
        t2 = feats[labels == -1].mean(axis=0)
        a, b = qsvm_export.solve_map(t1, t2, *qsvm_export.IRIS_CD)
        assert payload["map"]["a"] == pytest.approx(a, abs=1e-9)
        assert payload["map"]["b"] == pytest.approx(b, abs=1e-9)
        w = qsvm_export.weight_vector(qsvm_export.ALPHA_SHOTS)
        assert payload["w"] == pytest.approx(w.tolist(), abs=1e-9)

    def test_accuracy_claim_reproduces(self) -> None:
        payload = _load("qsvm-iris")
        feats, labels = qsvm_export.iris_features()
        import numpy as np

        acc = float(
            (qsvm_export.decide(np.array(payload["w"]), payload["map"], feats) == labels).mean()
        )
        assert round(acc, 4) == payload["test_accuracy"]
        assert acc >= 0.9


class TestQsvmMnistDrift:
    """Re-derivation only when the openml mnist_784 cache exists (CI never downloads)."""

    @pytest.mark.skipif(
        not MNIST_OPENML_CACHE, reason="openml mnist_784 not cached locally; CI never downloads"
    )
    def test_accuracy_claim_reproduces(self) -> None:
        payload = _load("qsvm-mnist")
        feats, labels = qsvm_export.mnist_features()
        import numpy as np

        acc = float(
            (qsvm_export.decide(np.array(payload["w"]), payload["map"], feats) == labels).mean()
        )
        assert round(acc, 4) == payload["test_accuracy"]
        assert acc >= 0.85


class TestQsvmBb84Drift:
    """Full re-derivation in CI — the bb84 sessions are seeded simulation."""

    def test_map_and_weights_rederive(self) -> None:
        payload = _load("qsvm-bb84")
        feats, labels = qsvm_export.bb84_features()
        t1 = feats[labels == 1].mean(axis=0)
        t2 = feats[labels == -1].mean(axis=0)
        a, b = qsvm_export.solve_map(t1, t2, *qsvm_export.BB84_CD)
        assert payload["map"]["a"] == pytest.approx(a, abs=1e-9)
        assert payload["map"]["b"] == pytest.approx(b, abs=1e-9)
        w = qsvm_export.weight_vector(qsvm_export.ALPHA_SHOTS)
        assert payload["w"] == pytest.approx(w.tolist(), abs=1e-9)

    def test_accuracy_claim_reproduces(self) -> None:
        payload = _load("qsvm-bb84")
        feats, labels = qsvm_export.bb84_features()
        import numpy as np

        acc = float(
            (qsvm_export.decide(np.array(payload["w"]), payload["map"], feats) == labels).mean()
        )
        assert round(acc, 4) == payload["test_accuracy"]
        assert acc >= 0.9

    def test_eavesdropped_rides_the_plus_one_ray(self) -> None:
        """classes[0] (the s>0 class) must be 'eavesdropped' — the boundary
        placement argument in bb84_features() depends on it."""
        payload = _load("qsvm-bb84")
        assert payload["classes"] == ["eavesdropped", "clean"]
