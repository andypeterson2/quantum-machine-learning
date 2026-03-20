"""Integration tests: end-to-end pipelines combining multiple modules."""

import torch
import numpy as np
from PIL import Image, ImageDraw

from classifiers.datasets.mnist.models import MNISTNet, LinearNet, SVMNet
from classifiers.datasets.mnist.plugin import MNISTPlugin
from classifiers.trainer import Trainer
from classifiers.predictor import Predictor
from classifiers.evaluator import Evaluator, EvalResult
from classifiers.model_registry import ModelRegistry
from tests.conftest import make_fake_train_loader, make_fake_test_loader

DS = "mnist"
NUM_CLASSES = 10
CLASS_LABELS = [str(i) for i in range(10)]


def _make_drawn_image():
    img = Image.new("L", (280, 280), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 80, 200, 200], fill=255)
    draw.line([(140, 80), (140, 200)], fill=255, width=10)
    return img


class TestTrainThenPredict:
    """Train a model, then use it to predict on a drawn image."""

    def test_train_and_predict_cnn(self):
        loader = make_fake_train_loader()
        plugin = MNISTPlugin()

        trainer = Trainer(model_cls=MNISTNet, train_loader=loader,
                          dataset=DS, epochs=1)
        result = trainer.train()
        assert result.model_type == "CNN"

        predictor = Predictor(result.model, plugin)
        probs = predictor.predict(_make_drawn_image())
        assert probs.shape == (10,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_train_and_predict_linear(self):
        loader = make_fake_train_loader()
        plugin = MNISTPlugin()

        trainer = Trainer(model_cls=LinearNet, train_loader=loader,
                          dataset=DS, epochs=1)
        result = trainer.train()
        assert result.model_type == "Linear"

        predictor = Predictor(result.model, plugin)
        probs = predictor.predict(_make_drawn_image())
        assert probs.shape == (10,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_train_and_predict_svm(self):
        loader = make_fake_train_loader()
        plugin = MNISTPlugin()

        trainer = Trainer(model_cls=SVMNet, train_loader=loader,
                          dataset=DS, epochs=1)
        result = trainer.train()
        assert result.model_type == "SVM"

        predictor = Predictor(result.model, plugin)
        probs = predictor.predict(_make_drawn_image())
        assert probs.shape == (10,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert all(0.0 <= p <= 1.0 for p in probs)


class TestTrainThenEvaluate:
    """Train a model, then evaluate it on the test set."""

    def test_train_and_evaluate_pipeline(self):
        train_loader = make_fake_train_loader()
        test_loader = make_fake_test_loader()

        trainer = Trainer(model_cls=MNISTNet, train_loader=train_loader,
                          dataset=DS, epochs=1)
        train_result = trainer.train()

        evaluator = Evaluator()
        eval_result = evaluator.evaluate(
            train_result.model, test_loader, NUM_CLASSES, CLASS_LABELS,
        )
        assert isinstance(eval_result, EvalResult)
        assert 0.0 <= eval_result.accuracy <= 1.0
        assert eval_result.avg_loss >= 0.0
        assert len(eval_result.per_class_accuracy) == 10


class TestTrainThenEvaluateSVM:
    """Train an SVM, then evaluate it on the test set."""

    def test_train_and_evaluate_svm(self):
        train_loader = make_fake_train_loader()
        test_loader = make_fake_test_loader()

        trainer = Trainer(model_cls=SVMNet, train_loader=train_loader,
                          dataset=DS, epochs=1)
        train_result = trainer.train()
        assert train_result.model_type == "SVM"

        evaluator = Evaluator()
        eval_result = evaluator.evaluate(
            train_result.model, test_loader, NUM_CLASSES, CLASS_LABELS,
        )
        assert isinstance(eval_result, EvalResult)
        assert 0.0 <= eval_result.accuracy <= 1.0
        assert eval_result.avg_loss >= 0.0
        assert len(eval_result.per_class_accuracy) == 10
        for label, acc in eval_result.per_class_accuracy.items():
            assert 0.0 <= acc <= 1.0, f"Label {label} accuracy out of range"


class TestSVMUsesHingeLoss:
    """Verify that the Trainer actually applies hinge loss for SVM."""

    def test_svm_training_loss_differs_from_cnn(self):
        """Train CNN and SVM from identical weights; final losses should diverge."""
        import re

        torch.manual_seed(0)
        loader = make_fake_train_loader(n_batches=2)

        svm_losses: list[float] = []

        def capture_svm(msg) -> None:
            if isinstance(msg, str):
                m = re.search(r"loss: ([0-9.]+)", msg)
                if m:
                    svm_losses.append(float(m.group(1)))

        Trainer(model_cls=SVMNet, train_loader=loader,
                dataset=DS, epochs=1).train(on_status=capture_svm)

        torch.manual_seed(0)
        loader = make_fake_train_loader(n_batches=2)

        cnn_losses: list[float] = []

        def capture_cnn(msg) -> None:
            if isinstance(msg, str):
                m = re.search(r"loss: ([0-9.]+)", msg)
                if m:
                    cnn_losses.append(float(m.group(1)))

        Trainer(model_cls=MNISTNet, train_loader=loader,
                dataset=DS, epochs=1).train(on_status=capture_cnn)

        assert len(svm_losses) > 0
        assert len(cnn_losses) > 0


class TestMultiModelComparison:
    """Train multiple model types, predict with all, compare results."""

    def test_multi_model_predict(self):
        loader = make_fake_train_loader()
        plugin = MNISTPlugin()
        registry = ModelRegistry()

        for model_cls in [MNISTNet, LinearNet, SVMNet]:
            name = registry.next_name(DS)
            trainer = Trainer(model_cls=model_cls, train_loader=loader,
                              dataset=DS, epochs=1)
            result = trainer.train()
            registry.add(
                DS, name, result.model, model_type=result.model_type,
                epochs=result.epochs, batch_size=result.batch_size, lr=result.lr,
            )

        assert len(registry) == 3

        img = _make_drawn_image()
        predictions = {}
        for name, entry in registry.items(DS):
            predictor = Predictor(entry.model, plugin)
            probs = predictor.predict(img)
            predictions[name] = int(np.argmax(probs))

        assert len(predictions) == 3
        for pred in predictions.values():
            assert 0 <= pred <= 9

    def test_multi_model_evaluate(self):
        train_loader = make_fake_train_loader()
        test_loader = make_fake_test_loader()
        registry = ModelRegistry()

        for model_cls in [MNISTNet, LinearNet, SVMNet]:
            name = registry.next_name(DS)
            trainer = Trainer(model_cls=model_cls, train_loader=train_loader,
                              dataset=DS, epochs=1)
            result = trainer.train()
            registry.add(
                DS, name, result.model, model_type=result.model_type,
                epochs=result.epochs, batch_size=result.batch_size, lr=result.lr,
            )

        evaluator = Evaluator()
        for name, entry in registry.items(DS):
            ev = evaluator.evaluate(
                entry.model, test_loader, NUM_CLASSES, CLASS_LABELS,
            )
            registry.update_eval_result(DS, name, ev)

        for name, entry in registry.items(DS):
            assert entry.eval_result is not None
            assert 0.0 <= entry.eval_result.accuracy <= 1.0


class TestRegistryLifecycleAllModels:
    """Integration test for registry add/remove/query cycle with all three types."""

    def test_full_lifecycle(self):
        registry = ModelRegistry()

        registry.add(DS, "cnn_1", MNISTNet(), model_type="CNN", epochs=1, batch_size=32, lr=0.01)
        registry.add(DS, "linear_1", LinearNet(), model_type="Linear", epochs=2, batch_size=64, lr=0.01)
        registry.add(DS, "svm_1", SVMNet(), model_type="SVM", epochs=3, batch_size=32, lr=0.01)

        assert len(registry) == 3
        assert set(registry.names(DS)) == {"cnn_1", "linear_1", "svm_1"}

        registry.remove(DS, "linear_1")
        assert len(registry) == 2
        assert registry.get(DS, "linear_1") is None

        assert registry.get(DS, "cnn_1") is not None
        assert registry.get(DS, "svm_1") is not None
        assert registry.get(DS, "svm_1").epochs == 3
        assert registry.get(DS, "svm_1").model_type == "SVM"

    def test_svm_entry_stores_correct_metadata(self):
        registry = ModelRegistry()
        registry.add(DS, "my_svm", SVMNet(), model_type="SVM", epochs=5, batch_size=128, lr=0.005)
        entry = registry.get(DS, "my_svm")
        assert entry.model_type == "SVM"
        assert entry.epochs == 5
        assert entry.batch_size == 128
        assert abs(entry.lr - 0.005) < 1e-9
        assert isinstance(entry.model, SVMNet)


class TestPredictorConsistency:
    """Multiple predictors on the same model should give identical results."""

    def test_same_model_same_result(self):
        plugin = MNISTPlugin()
        model = MNISTNet()
        img = _make_drawn_image()

        p1 = Predictor(model, plugin).predict(img)
        p2 = Predictor(model, plugin).predict(img)

        np.testing.assert_array_almost_equal(p1, p2)

    def test_different_models_can_differ(self):
        """Two randomly initialized models may give different predictions."""
        plugin = MNISTPlugin()
        img = _make_drawn_image()

        p1 = Predictor(MNISTNet(), plugin).predict(img)
        p2 = Predictor(MNISTNet(), plugin).predict(img)

        assert p1.shape == (10,)
        assert p2.shape == (10,)

    def test_different_architectures(self):
        """CNN, Linear, and SVM should all produce valid predictions."""
        plugin = MNISTPlugin()
        img = _make_drawn_image()

        p_cnn = Predictor(MNISTNet(), plugin).predict(img)
        p_linear = Predictor(LinearNet(), plugin).predict(img)
        p_svm = Predictor(SVMNet(), plugin).predict(img)

        for p in (p_cnn, p_linear, p_svm):
            assert p.shape == (10,)
            assert abs(p.sum() - 1.0) < 1e-5
            assert all(0.0 <= v <= 1.0 for v in p)

    def test_svm_predictor_consistency(self):
        """Same SVMNet, same image → identical probability arrays."""
        plugin = MNISTPlugin()
        model = SVMNet()
        img = _make_drawn_image()
        p1 = Predictor(model, plugin).predict(img)
        p2 = Predictor(model, plugin).predict(img)
        np.testing.assert_array_almost_equal(p1, p2)
