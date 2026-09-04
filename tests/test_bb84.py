"""Unit and integration tests for the BB84 dataset plugin, simulator, and models."""

import numpy as np
import pytest
import torch

from classifiers.base_model import BaseModel
from classifiers.datasets.bb84.models import BB84SVM, BB84Linear
from classifiers.datasets.bb84.plugin import N_TEST, N_TRAIN, TEST_SEED, TRAIN_SEED, BB84Plugin
from classifiers.datasets.bb84.simulate import SessionConfig, generate_dataset, simulate_session
from classifiers.evaluator import EvalResult, Evaluator
from classifiers.predictor import Predictor
from classifiers.trainer import Trainer, TrainResult

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def bb84_plugin():
    return BB84Plugin()


@pytest.fixture
def bb84_features():
    """A typical clean session: low QBER, healthy sifted-key rate."""
    return {"qber": 0.02, "sifted_key_rate": 0.10}


# ── Simulator tests ───────────────────────────────────────────────────────────


class TestBb84Simulator:
    def test_deterministic_under_seed(self):
        a = generate_dataset(200, 7)
        b = generate_dataset(200, 7)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])

    def test_balanced_labels(self):
        _, y = generate_dataset(400, 1)
        assert int(y.sum()) == 200

    def test_qber_monotone_in_intercept_fraction(self):
        """More interception ⇒ more errors, on identical channel conditions."""
        rng = np.random.default_rng(0)
        qbers = []
        for fraction in (0.0, 0.5, 1.0):
            qber, _ = simulate_session(
                rng, SessionConfig(intercept_fraction=fraction, num_pulses=20000)
            )
            qbers.append(qber)
        assert qbers[0] < qbers[1] < qbers[2]
        # Full intercept-resend sits near the textbook 25%.
        assert 0.20 < qbers[2] < 0.30

    def test_lossy_eve_depresses_sifted_rate(self):
        """Photons Eve intercepts but fails to register never reach Bob."""
        rng = np.random.default_rng(0)
        _, clean_rate = simulate_session(rng, SessionConfig(num_pulses=50000))
        _, tapped_rate = simulate_session(
            rng,
            SessionConfig(intercept_fraction=1.0, eve_detector_efficiency=0.75, num_pulses=50000),
        )
        assert tapped_rate < clean_rate

    def test_regimes_overlap(self):
        """The classification task must be honest: neither regime is a lookup."""
        features, labels = generate_dataset(1000, 42)
        clean_qber = features[labels == 0, 0]
        eve_qber = features[labels == 1, 0]
        # Distributions overlap near the clean regime's noisy tail…
        assert clean_qber.max() > eve_qber.min()
        # …but remain distinguishable in aggregate.
        assert eve_qber.mean() > clean_qber.mean() + 0.05


# ── Plugin tests ──────────────────────────────────────────────────────────────


class TestBb84Plugin:
    def test_plugin_attributes(self, bb84_plugin):
        assert bb84_plugin.name == "bb84"
        assert bb84_plugin.input_type == "tabular"
        assert bb84_plugin.num_classes == 2
        assert bb84_plugin.class_labels == ["clean", "eavesdropped"]
        assert bb84_plugin.feature_names == ["qber", "sifted_key_rate"]

    def test_get_train_loader(self, bb84_plugin):
        loader = bb84_plugin.get_train_loader(batch_size=32)
        batch_data, batch_targets = next(iter(loader))
        assert batch_data.shape[1] == 2
        assert batch_targets.dtype == torch.int64

    def test_get_test_loader(self, bb84_plugin):
        loader = bb84_plugin.get_test_loader(batch_size=N_TEST)
        batch_data, batch_targets = next(iter(loader))
        assert batch_data.shape == (N_TEST, 2)
        assert batch_targets.shape == (N_TEST,)

    def test_get_val_loader(self, bb84_plugin):
        loader = bb84_plugin.get_val_loader(batch_size=64)
        assert loader is not None
        batch_data, _ = next(iter(loader))
        assert batch_data.shape[1] == 2

    def test_train_val_split_covers_training_set(self, bb84_plugin):
        n_train = sum(len(yb) for _, yb in bb84_plugin.get_train_loader(64))
        n_val = sum(len(yb) for _, yb in bb84_plugin.get_val_loader(64))
        assert n_train + n_val == N_TRAIN

    def test_seeds_are_distinct_splits(self):
        """Train and test must come from different simulation streams."""
        assert TRAIN_SEED != TEST_SEED

    def test_preprocess(self, bb84_plugin, bb84_features):
        tensor = bb84_plugin.preprocess(bb84_features)
        assert tensor.shape == (1, 2)
        assert tensor.dtype == torch.float32

    def test_get_model_types(self, bb84_plugin):
        types = bb84_plugin.get_model_types()
        assert "Linear" in types
        assert "SVM" in types

    def test_default_hyperparams(self, bb84_plugin):
        defaults = bb84_plugin.get_default_hyperparams()
        assert defaults["epochs"] == 30
        assert defaults["batch_size"] == 32
        assert defaults["lr"] == 0.01

    def test_ui_config(self, bb84_plugin):
        config = bb84_plugin.get_ui_config()
        assert config["name"] == "bb84"
        assert config["input_type"] == "tabular"
        assert config["feature_names"] == bb84_plugin.feature_names


# ── Model tests ───────────────────────────────────────────────────────────────


class TestBb84Models:
    @pytest.mark.parametrize("model_cls", [BB84Linear, BB84SVM])
    def test_is_base_model(self, model_cls):
        assert isinstance(model_cls(), BaseModel)

    @pytest.mark.parametrize("model_cls", [BB84Linear, BB84SVM])
    def test_output_shape(self, model_cls):
        model = model_cls()
        x = torch.randn(8, 2)
        out = model(x)
        assert out.shape == (8, 2)

    @pytest.mark.parametrize("model_cls", [BB84Linear, BB84SVM])
    def test_has_name_and_description(self, model_cls):
        assert len(model_cls.name) > 0
        assert len(model_cls.description) > 0

    def test_svm_uses_hinge_loss(self):
        out = torch.randn(4, 2)
        tgt = torch.randint(0, 2, (4,))
        svm_loss = BB84SVM.loss_fn(out, tgt)
        linear_loss = BB84Linear.loss_fn(out, tgt)
        assert abs(svm_loss.item() - linear_loss.item()) > 1e-3


# ── Integration: train + predict ──────────────────────────────────────────────


class TestBb84TrainPredict:
    def test_train_and_predict_linear(self, bb84_plugin, bb84_features):
        loader = bb84_plugin.get_train_loader(batch_size=32)
        trainer = Trainer(
            model_cls=BB84Linear, train_loader=loader,
            dataset="bb84", epochs=2, lr=0.05,
        )
        result = trainer.train()
        assert isinstance(result, TrainResult)
        assert result.model_type == "Linear"

        predictor = Predictor(result.model, bb84_plugin)
        probs = predictor.predict(bb84_features)
        assert probs.shape == (2,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_train_and_evaluate_svm(self, bb84_plugin):
        train_loader = bb84_plugin.get_train_loader(batch_size=32)
        test_loader = bb84_plugin.get_test_loader(batch_size=N_TEST)

        trainer = Trainer(
            model_cls=BB84SVM, train_loader=train_loader,
            dataset="bb84", epochs=2, lr=0.05,
        )
        result = trainer.train()

        evaluator = Evaluator()
        eval_result = evaluator.evaluate(
            result.model, test_loader, 2, bb84_plugin.class_labels,
        )
        assert isinstance(eval_result, EvalResult)
        assert 0.0 <= eval_result.accuracy <= 1.0
        assert len(eval_result.per_class_accuracy) == 2

    def test_linear_beats_chance_clearly(self, bb84_plugin):
        """An eavesdropper detector that can't beat a coin flip is decoration."""
        trainer = Trainer(
            model_cls=BB84Linear,
            train_loader=bb84_plugin.get_train_loader(batch_size=32),
            dataset="bb84", epochs=8, lr=0.05,
        )
        model = trainer.train().model
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in bb84_plugin.get_test_loader(256):
                correct += int((model(xb).argmax(1) == yb).sum())
                total += int(yb.numel())
        assert correct / total >= 0.85
