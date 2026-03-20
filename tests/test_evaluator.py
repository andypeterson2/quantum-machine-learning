"""Unit tests for classifiers.evaluator.Evaluator and EvalResult."""

import pytest
import torch
from unittest.mock import patch

from classifiers.evaluator import Evaluator, EvalResult
from classifiers.datasets.mnist.models import MNISTNet, LinearNet
from tests.conftest import make_fake_test_loader

NUM_CLASSES = 10
CLASS_LABELS = [str(i) for i in range(10)]


class TestEvalResult:
    def test_defaults(self):
        result = EvalResult(accuracy=0.95, avg_loss=0.1)
        assert result.per_class_accuracy == {}

    def test_with_per_class(self):
        pca = {str(i): 0.9 + i * 0.01 for i in range(10)}
        result = EvalResult(accuracy=0.95, avg_loss=0.1, per_class_accuracy=pca)
        assert result.per_class_accuracy["5"] == pytest.approx(0.95)


class TestEvaluator:
    def test_evaluate_returns_eval_result(self):
        loader = make_fake_test_loader(batch_size=10, n_batches=2)
        evaluator = Evaluator()
        model = MNISTNet()
        result = evaluator.evaluate(model, loader, NUM_CLASSES, CLASS_LABELS)
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert result.avg_loss >= 0.0
        assert len(result.per_class_accuracy) == 10

    def test_evaluate_with_linear_model(self):
        loader = make_fake_test_loader(batch_size=10, n_batches=2)
        evaluator = Evaluator()
        model = LinearNet()
        result = evaluator.evaluate(model, loader, NUM_CLASSES, CLASS_LABELS)
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.accuracy <= 1.0

    def test_evaluate_calls_status_callback(self):
        loader = make_fake_test_loader(batch_size=10, n_batches=3)
        statuses = []
        evaluator = Evaluator()
        model = MNISTNet()
        evaluator.evaluate(
            model, loader, NUM_CLASSES, CLASS_LABELS,
            on_status=statuses.append,
        )
        assert len(statuses) > 0
        assert any("Loading" in s for s in statuses)
        assert any("done" in s.lower() or "Evaluation" in s for s in statuses)

    def test_perfect_model_gets_full_accuracy(self):
        """If model always predicts correctly, accuracy should be 1.0."""
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        data = torch.randn(10, 1, 28, 28)
        loader = [(data, targets)]

        model = MNISTNet()
        logits = torch.zeros(10, 10)
        for i, t in enumerate(targets):
            logits[i, t] = 100.0

        with patch.object(model, "forward", return_value=logits):
            evaluator = Evaluator()
            result = evaluator.evaluate(model, loader, NUM_CLASSES, CLASS_LABELS)

        assert result.accuracy == pytest.approx(1.0)
        for label in CLASS_LABELS:
            assert result.per_class_accuracy[label] == pytest.approx(1.0)

    def test_no_status_callback_ok(self):
        loader = make_fake_test_loader(batch_size=10, n_batches=1)
        evaluator = Evaluator()
        model = MNISTNet()
        result = evaluator.evaluate(
            model, loader, NUM_CLASSES, CLASS_LABELS, on_status=None,
        )
        assert isinstance(result, EvalResult)

    def test_evaluate_has_num_params(self):
        loader = make_fake_test_loader(batch_size=10, n_batches=1)
        evaluator = Evaluator()
        model = MNISTNet()
        result = evaluator.evaluate(model, loader, NUM_CLASSES, CLASS_LABELS)
        assert result.num_params is not None
        assert result.num_params > 0


class TestEnsembleEvaluator:
    def test_ensemble_evaluate(self):
        loader = make_fake_test_loader(batch_size=10, n_batches=2)
        evaluator = Evaluator()
        models = [MNISTNet(), LinearNet()]
        result = evaluator.ensemble_evaluate(
            models, loader, NUM_CLASSES, CLASS_LABELS,
        )
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.accuracy <= 1.0


class TestAblationEvaluator:
    def test_ablation_evaluate(self):
        loader = make_fake_test_loader(batch_size=10, n_batches=1)
        evaluator = Evaluator()
        model = LinearNet()  # simple model with one layer
        results = evaluator.ablation_evaluate(
            model, loader, NUM_CLASSES, CLASS_LABELS,
        )
        assert isinstance(results, dict)
        assert len(results) > 0
        for layer_name, result in results.items():
            assert isinstance(result, EvalResult)
