"""Drift check for the browser model exports (``exports/web/``).

The portfolio site serves these weights for in-browser inference and claims
they come from the same models this platform trains. These tests enforce that
claim against the *committed* export files: if a dataset plugin's split,
normalisation, labels, or feature set changes without re-running
``make export-web``, CI fails here instead of the browser demo silently
drifting from the backend.

The accuracy checks deliberately re-evaluate the committed weights (pure
inference — deterministic across platforms) rather than retraining, and the
MNIST accuracy check only runs when the dataset is already cached locally.
The test run itself never downloads: CI populates its caches in a separate
step, once per cache key, before pytest starts.
"""

from __future__ import annotations

import json
import re

import pytest
import torch

from classifiers.datasets.mnist.plugin import MNIST_MEAN, MNIST_STD
from classifiers.plugin_registry import discover_plugins, get_plugin
from classifiers.web_export import OUT_DIR, REPO_ROOT, evaluate_payload

# The test-set file itself: an empty MNIST/ dir exists
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
        correct, total = evaluate_payload(payload, plugin.get_test_loader(64))
        assert round(correct / total, 4) == payload["test_accuracy"]
        assert total == payload["test_n"]
        assert correct / total >= 0.9


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
        not MNIST_CACHE.is_file(), reason="MNIST not cached here; tests never download"
    )
    def test_accuracy_claim_reproduces(self) -> None:
        payload = _load("mnist")
        plugin = get_plugin("mnist")
        assert plugin is not None
        correct, total = evaluate_payload(payload, plugin.get_test_loader(512))
        assert round(correct / total, 4) == payload["test_accuracy"]
        assert total == payload["test_n"]
        assert correct / total >= 0.85


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
        correct, total = evaluate_payload(payload, plugin.get_test_loader(256))
        assert round(correct / total, 4) == payload["test_accuracy"]
        assert total == payload["test_n"]
        assert correct / total >= 0.9


class TestReportedUncertainty:
    """Every committed accuracy ships the interval it was measured with."""

    @pytest.mark.parametrize(
        "name", ["iris", "mnist", "bb84", "qsvm-iris", "qsvm-mnist"]
    )
    def test_interval_brackets_the_claim(self, name: str) -> None:
        payload = _load(name)
        low, high = payload["test_accuracy_ci"]
        assert low <= payload["test_accuracy"] <= high
        assert 0.0 <= low < high <= 1.0

    @pytest.mark.parametrize(
        "name", ["iris", "mnist", "bb84", "qsvm-iris", "qsvm-mnist"]
    )
    def test_interval_matches_the_recorded_sample_count(self, name: str) -> None:
        """The interval must come from this export's own n, not a stale one."""
        from classifiers.stats import wilson_interval

        payload = _load(name)
        n = payload["test_n"]
        hits = round(payload["test_accuracy"] * n)
        assert payload["test_accuracy_ci"] == pytest.approx(wilson_interval(hits, n), abs=1e-4)

    def test_the_iris_splits_are_too_small_to_separate(self) -> None:
        """Documents why this exists: on 30 samples the linear baseline and the
        QSVM rule are not distinguishable, however far apart the point estimates
        look."""
        linear, qsvm = _load("iris"), _load("qsvm-iris")
        assert linear["test_n"] == qsvm["test_n"] == 30
        assert linear["test_accuracy_ci"][1] > qsvm["test_accuracy_ci"][0]


class TestRegeneration:
    """`make export-web` must still produce what is committed.

    The other checks re-score the committed weights, which catches a plugin
    change but not a broken exporter: nothing re-ran the training path. These
    two datasets need no download and train in seconds, so they can.

    Weights are compared loosely on purpose — the run is seeded, so it repeats
    exactly on one machine, but float arithmetic differs between platforms,
    which is why the accuracy claim is what CI enforces.
    """

    @pytest.mark.skipif(
        not (REPO_ROOT / ".git").exists(),
        reason="exporting stamps provenance from git; the image ships no checkout",
    )
    @pytest.mark.parametrize("name", ["iris", "bb84"])
    def test_training_path_reproduces_the_committed_export(self, name: str) -> None:
        from classifiers.web_export import SEED, _payload, seed_everything

        plugin = get_plugin(name)
        assert plugin is not None
        seed_everything(SEED)
        fresh = _payload(plugin)
        committed = _load(name)

        assert fresh["test_accuracy"] == pytest.approx(committed["test_accuracy"], abs=0.05)
        assert torch.tensor(fresh["weight"]).shape == torch.tensor(committed["weight"]).shape
        assert fresh["normalize"] == committed["normalize"]
        assert fresh["classes"] == committed["classes"]


# QSVM paper-recreation exports

from classifiers import qsvm_export  # noqa: E402

from .conftest import MNIST_OPENML_CACHE  # noqa: E402


class TestQsvmProvenance:
    """The qsvm exports carry the same provenance discipline, QSVM-flavoured."""

    @pytest.mark.parametrize("name", ["qsvm-iris", "qsvm-mnist"])
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

    @pytest.mark.parametrize("name", ["qsvm-iris", "qsvm-mnist"])
    def test_contract(self, name: str) -> None:
        payload = _load(name)
        assert payload["kind"] == "qsvm"
        assert payload["dataset"] in {"iris", "mnist"}
        assert len(payload["classes"]) == 2
        assert len(payload["w"]) == 2
        assert set(payload["map"]) == {"a", "b", "c", "d"}
        assert all(isinstance(v, float) for v in payload["map"].values())
        assert len(payload["features"]) == 2
        assert payload["raw_input"] in {"features", "pixels"}
        assert 0.0 < payload["test_accuracy"] <= 1.0
        assert payload["test_n"] > 0
        assert payload["train_n"] > 0
        assert payload["test_protocol"]
        assert payload["num_params"] == 6


class TestQsvmSelection:
    """Free parameters are chosen on validation data, never on the held-out split."""

    @pytest.mark.parametrize("name", ["qsvm-iris", "qsvm-mnist"])
    def test_selection_is_recorded(self, name: str) -> None:
        selection = _load(name)["selection"]
        assert selection["positive_class"] in _load(name)["classes"]
        assert "held-out" in selection["protocol"] or "fixed by the paper" in selection["protocol"]

    def test_paper_datasets_select_nothing(self) -> None:
        """Iris and MNIST are the paper's own experiments: it fixes the
        orientation and (c, d), so there is nothing to choose."""
        for name in ("qsvm-iris", "qsvm-mnist"):
            selection = _load(name)["selection"]
            assert selection["candidates"] == 1
            assert selection["validation_accuracy"] is None
            assert selection["validation_n"] == 0

    def test_a_dataset_the_paper_never_ran_would_choose_on_validation(self) -> None:
        """Every shipped dataset is one the paper ran, so nothing is selected.
        The selection path still has to work for one that is not."""
        import numpy as np

        spec = dict(
            qsvm_export.QSVM_DATASETS["iris"],
            free_parameters=True,
            cd_candidates=[(0.95, -0.42), (0.5, -0.3), (2.0, 0.1)],
        )
        split = qsvm_export.iris_features()
        choice = qsvm_export.choose_parameters(
            split, spec, qsvm_export.weight_vector(qsvm_export.ALPHA_SHOTS)
        )
        assert choice.candidates > 1
        assert choice.validation_n > 0
        assert 0.0 < choice.validation_accuracy <= 1.0
        assert (choice.c, choice.d) in spec["cd_candidates"]
        assert not np.isnan(choice.validation_accuracy)


class TestQsvmIrisDrift:
    """Full re-derivation in CI — Iris ships with scikit-learn, no download."""

    def test_map_and_weights_rederive(self) -> None:
        """Re-run the whole derivation, selection included, and land on the
        committed rule."""
        payload = _load("qsvm-iris")
        fit = qsvm_export.fit_and_score("iris", qsvm_export.ALPHA_SHOTS)
        assert payload["map"]["a"] == pytest.approx(fit.mapping["a"], abs=1e-9)
        assert payload["map"]["b"] == pytest.approx(fit.mapping["b"], abs=1e-9)
        assert payload["map"]["c"] == pytest.approx(fit.choice.c)
        assert payload["map"]["d"] == pytest.approx(fit.choice.d)
        assert payload["w"] == pytest.approx(fit.w.tolist(), abs=1e-9)

    def test_accuracy_claim_reproduces(self) -> None:
        payload = _load("qsvm-iris")
        split = qsvm_export.iris_features()
        import numpy as np

        pred = qsvm_export.decide(np.array(payload["w"]), payload["map"], split.test_x)
        acc = float((pred == split.test_y).mean())
        assert round(acc, 4) == payload["test_accuracy"]
        assert len(split.test_y) == payload["test_n"]
        assert acc >= 0.9


class TestQsvmMnistDrift:
    """Re-derivation only when the openml mnist_784 cache exists (tests never download)."""

    @pytest.mark.skipif(
        not MNIST_OPENML_CACHE, reason="openml mnist_784 not cached here; tests never download"
    )
    def test_accuracy_claim_reproduces(self) -> None:
        payload = _load("qsvm-mnist")
        split = qsvm_export.mnist_features()
        import numpy as np

        pred = qsvm_export.decide(np.array(payload["w"]), payload["map"], split.test_x)
        acc = float((pred == split.test_y).mean())
        assert round(acc, 4) == payload["test_accuracy"]
        assert len(split.test_y) == payload["test_n"]
        assert acc >= 0.85


class TestQsvmOvo:
    """The three-class Iris rule: three pairwise maps, one shared alpha."""

    def test_contract(self) -> None:
        payload = _load("qsvm-iris-ovo")
        assert payload["kind"] == "qsvm-ovo"
        assert payload["dataset"] == "iris"
        assert payload["classes"] == ["setosa", "versicolor", "virginica"]
        assert len(payload["features"]) == 4
        assert payload["raw_input"] == "features"
        assert len(payload["w"]) == 4
        assert len(payload["rules"]) == 3
        for rule in payload["rules"]:
            assert len(rule["pair"]) == 2
            assert set(rule["pair"]) <= set(payload["classes"])
            assert len(rule["a"]) == len(rule["b"]) == 4
        assert {tuple(r["pair"]) for r in payload["rules"]} == {
            ("setosa", "versicolor"),
            ("setosa", "virginica"),
            ("versicolor", "virginica"),
        }
        assert payload["num_params"] == 28
        assert payload["test_n"] == 45
        assert payload["train_n"] == 105

    def test_alpha_is_untouched_by_the_widening(self) -> None:
        """The widened targets keep the kernel matrix, which is what lets all
        three rules share one hardware alpha — and lets that alpha be the one
        the binary exports already shipped."""
        import numpy as np

        from classifiers.qsvm_rule import TARGETS

        targets = np.array(_load("qsvm-iris-ovo")["targets"])
        unit2 = TARGETS / np.linalg.norm(TARGETS, axis=1, keepdims=True)
        assert targets.shape == (2, 4)
        assert np.allclose(np.linalg.norm(targets, axis=1), 1.0)
        assert np.allclose(targets @ targets.T, unit2 @ unit2.T)

    def test_one_weight_vector_serves_every_rule(self) -> None:
        import numpy as np

        from classifiers.qsvm_rule import weight_vector

        payload = _load("qsvm-iris-ovo")
        expected = weight_vector(qsvm_export.ALPHA_SHOTS) @ np.array(payload["targets"])
        assert payload["w"] == pytest.approx(expected.tolist(), abs=1e-12)

    def test_every_rule_pins_its_class_means_to_the_targets(self) -> None:
        import numpy as np

        from classifiers import qsvm_ovo_export

        payload = _load("qsvm-iris-ovo")
        targets = np.array(payload["targets"])
        train_x, _, train_y, _ = qsvm_ovo_export.iris_split(qsvm_ovo_export.SEED)
        for rule in payload["rules"]:
            pos, neg = (qsvm_ovo_export.CLASSES.index(c) for c in rule["pair"])
            a, b = np.array(rule["a"]), np.array(rule["b"])
            assert np.allclose(train_x[train_y == pos].mean(axis=0) * a + b, targets[0])
            assert np.allclose(train_x[train_y == neg].mean(axis=0) * a + b, targets[1])

    def test_rules_and_accuracy_rederive(self) -> None:
        """Re-run the whole derivation and land on the committed rule."""
        import numpy as np

        from classifiers import qsvm_ovo_export

        payload = _load("qsvm-iris-ovo")
        rules, w, hits, test_n, train_n = qsvm_ovo_export.fit_and_score(
            qsvm_ovo_export.SEED, qsvm_export.ALPHA_SHOTS
        )
        assert payload["w"] == pytest.approx(w.tolist(), abs=1e-12)
        assert payload["test_n"] == test_n
        assert payload["train_n"] == train_n
        assert payload["test_accuracy"] == round(hits / test_n, 4)
        for committed, derived in zip(payload["rules"], rules, strict=True):
            assert committed["pair"] == [derived.positive, derived.negative]
            assert committed["a"] == pytest.approx(derived.a.tolist(), abs=1e-12)
            assert committed["b"] == pytest.approx(derived.b.tolist(), abs=1e-12)
            assert np.isfinite(derived.a).all()

    def test_it_beats_a_coin_and_says_how_sure_it_is(self) -> None:
        """One 45-sample split cannot pin this down, so the export carries a
        cross-validated figure beside it."""
        payload = _load("qsvm-iris-ovo")
        low, high = payload["test_accuracy_ci"]
        assert low <= payload["test_accuracy"] <= high
        assert payload["test_accuracy"] >= 0.80
        assert payload["cv_accuracy"] >= 0.80
        assert payload["cv_splits"] >= 10

    def test_nothing_is_selected(self) -> None:
        """Both free choices are fixed by construction, so no validation slice
        is used and none may be claimed."""
        selection = _load("qsvm-iris-ovo")["selection"]
        assert selection["candidates"] == 1
        assert selection["validation_n"] == 0
        assert selection["validation_accuracy"] is None

    def test_provenance_block(self) -> None:
        prov = _load("qsvm-iris-ovo")["provenance"]
        assert set(prov) >= PROVENANCE_KEYS
        assert prov["source_repo"] == "quantum-machine-learning"
        assert re.fullmatch(r"[0-9a-f]{40}", prov["source_sha"])
        assert prov["training"]["model"] == "QSVM one-vs-one"
        assert prov["training"]["paper"] == "arXiv:1909.11988"

    def test_the_binary_iris_rule_is_untouched(self) -> None:
        """This model ships beside the paper's own Iris experiment, not over it."""
        binary = _load("qsvm-iris")
        assert binary["kind"] == "qsvm"
        assert binary["classes"] == ["setosa", "versicolor"]
        assert binary["features"] == ["sepal_width", "petal_length"]
