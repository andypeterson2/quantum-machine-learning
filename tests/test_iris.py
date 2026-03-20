"""Unit and integration tests for the Iris dataset plugin and models."""

import pytest
import torch

from classifiers.base_model import BaseModel
from classifiers.datasets.iris.models import IrisLinear, IrisSVM
from classifiers.datasets.iris.plugin import IrisPlugin
from classifiers.evaluator import Evaluator, EvalResult
from classifiers.predictor import Predictor
from classifiers.trainer import Trainer, TrainResult


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def iris_plugin():
    return IrisPlugin()


@pytest.fixture
def iris_features():
    """A typical Iris setosa sample."""
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }


# ── Plugin tests ──────────────────────────────────────────────────────────────


class TestIrisPlugin:
    def test_plugin_attributes(self, iris_plugin):
        assert iris_plugin.name == "iris"
        assert iris_plugin.input_type == "tabular"
        assert iris_plugin.num_classes == 3
        assert len(iris_plugin.class_labels) == 3
        assert iris_plugin.feature_names is not None
        assert len(iris_plugin.feature_names) == 4

    def test_get_train_loader(self, iris_plugin):
        loader = iris_plugin.get_train_loader(batch_size=16)
        batch_data, batch_targets = next(iter(loader))
        assert batch_data.shape[1] == 4
        assert batch_targets.dtype == torch.int64

    def test_get_test_loader(self, iris_plugin):
        loader = iris_plugin.get_test_loader(batch_size=100)
        batch_data, batch_targets = next(iter(loader))
        assert batch_data.shape[1] == 4

    def test_get_val_loader(self, iris_plugin):
        loader = iris_plugin.get_val_loader(batch_size=16)
        assert loader is not None
        batch_data, _ = next(iter(loader))
        assert batch_data.shape[1] == 4

    def test_preprocess(self, iris_plugin, iris_features):
        tensor = iris_plugin.preprocess(iris_features)
        assert tensor.shape == (1, 4)
        assert tensor.dtype == torch.float32

    def test_get_model_types(self, iris_plugin):
        types = iris_plugin.get_model_types()
        assert "Linear" in types
        assert "SVM" in types

    def test_default_hyperparams(self, iris_plugin):
        defaults = iris_plugin.get_default_hyperparams()
        assert defaults["epochs"] == 50
        assert defaults["batch_size"] == 16
        assert defaults["lr"] == 0.01

    def test_ui_config(self, iris_plugin):
        config = iris_plugin.get_ui_config()
        assert config["name"] == "iris"
        assert config["input_type"] == "tabular"
        assert config["feature_names"] == iris_plugin.feature_names


# ── Model tests ───────────────────────────────────────────────────────────────


class TestIrisModels:
    @pytest.mark.parametrize("model_cls", [IrisLinear, IrisSVM])
    def test_is_base_model(self, model_cls):
        assert isinstance(model_cls(), BaseModel)

    @pytest.mark.parametrize("model_cls", [IrisLinear, IrisSVM])
    def test_output_shape(self, model_cls):
        model = model_cls()
        x = torch.randn(8, 4)
        out = model(x)
        assert out.shape == (8, 3)

    @pytest.mark.parametrize("model_cls", [IrisLinear, IrisSVM])
    def test_has_name_and_description(self, model_cls):
        assert len(model_cls.name) > 0
        assert len(model_cls.description) > 0

    def test_svm_uses_hinge_loss(self):
        out = torch.randn(4, 3)
        tgt = torch.randint(0, 3, (4,))
        svm_loss = IrisSVM.loss_fn(out, tgt)
        linear_loss = IrisLinear.loss_fn(out, tgt)
        assert abs(svm_loss.item() - linear_loss.item()) > 1e-3


# ── Integration: train + predict ──────────────────────────────────────────────


class TestIrisTrainPredict:
    def test_train_and_predict_linear(self, iris_plugin, iris_features):
        loader = iris_plugin.get_train_loader(batch_size=16)
        trainer = Trainer(
            model_cls=IrisLinear, train_loader=loader,
            dataset="iris", epochs=2, lr=0.01,
        )
        result = trainer.train()
        assert isinstance(result, TrainResult)
        assert result.model_type == "Linear"

        predictor = Predictor(result.model, iris_plugin)
        probs = predictor.predict(iris_features)
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_train_and_evaluate_svm(self, iris_plugin):
        train_loader = iris_plugin.get_train_loader(batch_size=16)
        test_loader = iris_plugin.get_test_loader(batch_size=100)

        trainer = Trainer(
            model_cls=IrisSVM, train_loader=train_loader,
            dataset="iris", epochs=2, lr=0.01,
        )
        result = trainer.train()

        evaluator = Evaluator()
        eval_result = evaluator.evaluate(
            result.model, test_loader, 3, iris_plugin.class_labels,
        )
        assert isinstance(eval_result, EvalResult)
        assert 0.0 <= eval_result.accuracy <= 1.0
        assert len(eval_result.per_class_accuracy) == 3
