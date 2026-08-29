"""Integration tests: verify every model type trains successfully.

Covers both direct Trainer usage and HTTP route (SSE) training for every
model registered in each dataset plugin.  Uses fake/small data loaders
to keep tests fast and avoid real dataset downloads.

Run with:
    python -m pytest tests/test_all_models_train.py -v
"""

from __future__ import annotations

import pytest
import torch

from classifiers.datasets.iris.models import IrisLinear, IrisSVM
from classifiers.datasets.mnist.models import (
    LinearNet,
    MNISTNet,
    MNISTPolynomialNet,
    MNISTQuadraticNet,
    SVMNet,
)
from classifiers.evaluator import Evaluator
from classifiers.server import create_app
from classifiers.trainer import Trainer, TrainResult
from tests.conftest import (
    make_fake_test_loader,
    make_fake_train_loader,
)
from tests.conftest import (
    parse_sse as _parse_sse,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


class _FakeLoader(list):
    """List with a batch_size attribute to mimic DataLoader."""

    def __init__(self, batches, batch_size):
        super().__init__(batches)
        self.batch_size = batch_size


def _make_iris_loader(n_samples=40, batch_size=16):
    """Synthetic Iris-like DataLoader (4 features, 3 classes)."""
    batches = []
    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        data = torch.randn(bs, 4)
        targets = torch.randint(0, 3, (bs,))
        batches.append((data, targets))
        remaining -= bs
    return _FakeLoader(batches, batch_size)


def _make_iris_test_loader(batch_size=20, n_samples=20):
    return _make_iris_loader(n_samples=n_samples, batch_size=batch_size)


# ── Check optional dependencies ─────────────────────────────────────────────

_HAS_QISKIT = False
try:
    import qiskit  # noqa: F401
    import qiskit_aer  # noqa: F401

    from classifiers.datasets.mnist.models import QiskitCNN, QiskitLinear
    _HAS_QISKIT = True
except ImportError:
    pass

_HAS_PENNYLANE = False
try:
    import pennylane  # noqa: F401

    from classifiers.datasets.iris.models import IrisQVC
    _HAS_PENNYLANE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Direct Trainer Tests — MNIST models
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainMNISTModels:
    """Train each MNIST model type via the Trainer and verify the result."""

    @pytest.fixture(autouse=True)
    def _loaders(self):
        self.train_loader = make_fake_train_loader(batch_size=8, n_batches=3)
        self.test_loader = make_fake_test_loader(batch_size=8, n_batches=2)

    def _train(self, model_cls, epochs=1):
        trainer = Trainer(
            model_cls=model_cls,
            train_loader=self.train_loader,
            dataset="mnist",
            epochs=epochs,
            lr=1e-3,
        )
        return trainer.train()

    def _assert_result(self, result, model_cls, epochs=1):
        assert isinstance(result, TrainResult)
        assert isinstance(result.model, model_cls)
        assert result.model_type == model_cls.name
        assert result.dataset == "mnist"
        assert result.epochs == epochs
        assert result.epochs_completed == epochs
        assert result.num_params > 0
        assert result.stopped_early is False

    def test_cnn(self):
        result = self._train(MNISTNet)
        self._assert_result(result, MNISTNet)

    def test_linear(self):
        result = self._train(LinearNet)
        self._assert_result(result, LinearNet)

    def test_svm(self):
        result = self._train(SVMNet)
        self._assert_result(result, SVMNet)

    def test_quadratic(self):
        result = self._train(MNISTQuadraticNet)
        self._assert_result(result, MNISTQuadraticNet)

    def test_polynomial(self):
        result = self._train(MNISTPolynomialNet)
        self._assert_result(result, MNISTPolynomialNet)

    @pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
    def test_qiskit_cnn(self):
        result = self._train(QiskitCNN)
        self._assert_result(result, QiskitCNN)

    @pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
    def test_qiskit_linear(self):
        result = self._train(QiskitLinear)
        self._assert_result(result, QiskitLinear)


# ═══════════════════════════════════════════════════════════════════════════════
# Direct Trainer Tests — Iris models
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainIrisModels:
    """Train each Iris model type via the Trainer and verify the result."""

    @pytest.fixture(autouse=True)
    def _loaders(self):
        self.train_loader = _make_iris_loader(n_samples=32, batch_size=8)
        self.test_loader = _make_iris_test_loader(n_samples=16, batch_size=16)

    def _train(self, model_cls, epochs=1):
        trainer = Trainer(
            model_cls=model_cls,
            train_loader=self.train_loader,
            dataset="iris",
            epochs=epochs,
            lr=1e-2,
        )
        return trainer.train()

    def _assert_result(self, result, model_cls, epochs=1):
        assert isinstance(result, TrainResult)
        assert isinstance(result.model, model_cls)
        assert result.model_type == model_cls.name
        assert result.dataset == "iris"
        assert result.epochs == epochs
        assert result.epochs_completed == epochs
        assert result.num_params > 0

    def test_linear(self):
        result = self._train(IrisLinear)
        self._assert_result(result, IrisLinear)

    def test_svm(self):
        result = self._train(IrisSVM)
        self._assert_result(result, IrisSVM)

    @pytest.mark.skipif(not _HAS_PENNYLANE, reason="pennylane not installed")
    def test_qvc(self):
        result = self._train(IrisQVC)
        self._assert_result(result, IrisQVC)


# ═══════════════════════════════════════════════════════════════════════════════
# Status callback — all models emit proper progress
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainStatusCallbacks:
    """Verify every model emits the expected status messages."""

    @pytest.mark.parametrize("model_cls", [
        MNISTNet, LinearNet, SVMNet, MNISTQuadraticNet, MNISTPolynomialNet,
    ])
    def test_mnist_model_emits_status(self, model_cls):
        loader = make_fake_train_loader(batch_size=8, n_batches=3)
        statuses = []
        trainer = Trainer(
            model_cls=model_cls, train_loader=loader,
            dataset="mnist", epochs=1, lr=1e-3,
        )
        trainer.train(on_status=statuses.append)
        str_msgs = [s for s in statuses if isinstance(s, str)]
        assert any("Preparing" in s for s in str_msgs)
        assert any("complete" in s.lower() for s in str_msgs)

    @pytest.mark.parametrize("model_cls", [IrisLinear, IrisSVM])
    def test_iris_model_emits_status(self, model_cls):
        loader = _make_iris_loader(n_samples=16, batch_size=8)
        statuses = []
        trainer = Trainer(
            model_cls=model_cls, train_loader=loader,
            dataset="iris", epochs=1, lr=1e-2,
        )
        trainer.train(on_status=statuses.append)
        str_msgs = [s for s in statuses if isinstance(s, str)]
        assert any("Preparing" in s for s in str_msgs)
        assert any("complete" in s.lower() for s in str_msgs)


# ═══════════════════════════════════════════════════════════════════════════════
# Forward pass — all models produce valid output shapes
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelForwardPass:
    """Verify each model's forward pass produces correct output shape."""

    @pytest.mark.parametrize("model_cls,expected_classes", [
        (MNISTNet, 10),
        (LinearNet, 10),
        (SVMNet, 10),
        (MNISTQuadraticNet, 10),
        (MNISTPolynomialNet, 10),
    ])
    def test_mnist_forward(self, model_cls, expected_classes):
        model = model_cls()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, expected_classes)

    @pytest.mark.parametrize("model_cls,expected_classes", [
        (IrisLinear, 3),
        (IrisSVM, 3),
    ])
    def test_iris_forward(self, model_cls, expected_classes):
        model = model_cls()
        x = torch.randn(4, 4)
        out = model(x)
        assert out.shape == (4, expected_classes)

    @pytest.mark.skipif(not _HAS_PENNYLANE, reason="pennylane not installed")
    def test_iris_qvc_forward(self):
        model = IrisQVC()
        x = torch.randn(2, 4)
        out = model(x)
        assert out.shape == (2, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation after training — all models can be evaluated
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateAfterTraining:
    """Train each model then evaluate — verify metrics are sane."""

    def _train_and_eval(self, model_cls, train_loader, test_loader,
                        dataset, num_classes, class_labels, lr=1e-3):
        trainer = Trainer(
            model_cls=model_cls, train_loader=train_loader,
            dataset=dataset, epochs=1, lr=lr,
        )
        result = trainer.train()
        evaluator = Evaluator()
        return evaluator.evaluate(
            result.model, test_loader, num_classes, class_labels,
        )

    @pytest.mark.parametrize("model_cls", [
        MNISTNet, LinearNet, SVMNet, MNISTQuadraticNet, MNISTPolynomialNet,
    ])
    def test_mnist_evaluate(self, model_cls):
        train_loader = make_fake_train_loader(batch_size=8, n_batches=3)
        test_loader = make_fake_test_loader(batch_size=8, n_batches=2)
        labels = [str(i) for i in range(10)]
        ev = self._train_and_eval(
            model_cls, train_loader, test_loader, "mnist", 10, labels,
        )
        assert 0.0 <= ev.accuracy <= 1.0
        assert ev.avg_loss >= 0.0
        assert len(ev.per_class_accuracy) == 10

    @pytest.mark.parametrize("model_cls", [IrisLinear, IrisSVM])
    def test_iris_evaluate(self, model_cls):
        train_loader = _make_iris_loader(n_samples=32, batch_size=8)
        test_loader = _make_iris_test_loader(n_samples=16, batch_size=16)
        labels = ["setosa", "versicolor", "virginica"]
        ev = self._train_and_eval(
            model_cls, train_loader, test_loader, "iris", 3, labels, lr=1e-2,
        )
        assert 0.0 <= ev.accuracy <= 1.0
        assert ev.avg_loss >= 0.0
        assert len(ev.per_class_accuracy) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Route Tests — SSE training for every model type
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainRouteAllMNISTModels:
    """Train each MNIST model via the /d/mnist/train HTTP route."""

    @pytest.fixture(autouse=True)
    def _app(self):
        self.application = create_app()
        self.application.config["TESTING"] = True
        self.client = self.application.test_client()

    @pytest.mark.parametrize("model_type", [
        "CNN", "Linear", "SVM", "Quadratic", "Polynomial",
    ])
    def test_train_model_via_route(self, model_type):
        from unittest.mock import patch
        fake_loader = make_fake_train_loader(batch_size=8, n_batches=3)
        with patch(
            "classifiers.datasets.mnist.plugin.MNISTPlugin.get_train_loader",
            return_value=fake_loader,
        ):
            res = self.client.post(
                "/d/mnist/train",
                json={
                    "model_type": model_type,
                    "epochs": 1,
                    "batch_size": 8,
                    "lr": 0.001,
                    "name": f"test-{model_type}",
                },
                content_type="application/json",
            )
        assert res.status_code == 200
        assert res.content_type.startswith("text/event-stream")
        events = _parse_sse(res.data)
        types = [e["type"] for e in events]
        assert "done" in types, f"No 'done' event for {model_type}: {types}"

        done = next(e for e in events if e["type"] == "done")
        assert done["name"] == f"test-{model_type}"
        assert done["model_type"] == model_type
        assert done["epochs_completed"] == 1
        assert done["num_params"] > 0

    def test_train_unknown_type_returns_400(self):
        res = self.client.post(
            "/d/mnist/train",
            json={"model_type": "NOPE", "epochs": 1, "batch_size": 8, "lr": 0.001},
            content_type="application/json",
        )
        assert res.status_code == 400


class TestTrainRouteAllIrisModels:
    """Train each Iris model via the /d/iris/train HTTP route."""

    @pytest.fixture(autouse=True)
    def _app(self):
        self.application = create_app()
        self.application.config["TESTING"] = True
        self.client = self.application.test_client()

    @pytest.mark.parametrize("model_type", ["Linear", "SVM"])
    def test_train_model_via_route(self, model_type):
        from unittest.mock import patch
        fake_loader = _make_iris_loader(n_samples=32, batch_size=8)
        with patch(
            "classifiers.datasets.iris.plugin.IrisPlugin.get_train_loader",
            return_value=fake_loader,
        ):
            res = self.client.post(
                "/d/iris/train",
                json={
                    "model_type": model_type,
                    "epochs": 1,
                    "batch_size": 8,
                    "lr": 0.01,
                    "name": f"test-{model_type}",
                },
                content_type="application/json",
            )
        assert res.status_code == 200
        events = _parse_sse(res.data)
        types = [e["type"] for e in events]
        assert "done" in types, f"No 'done' event for {model_type}: {types}"

        done = next(e for e in events if e["type"] == "done")
        assert done["name"] == f"test-{model_type}"
        assert done["model_type"] == model_type
        assert done["num_params"] > 0

    @pytest.mark.skipif(not _HAS_PENNYLANE, reason="pennylane not installed")
    def test_train_qvc_via_route(self):
        from unittest.mock import patch
        fake_loader = _make_iris_loader(n_samples=16, batch_size=8)
        with patch(
            "classifiers.datasets.iris.plugin.IrisPlugin.get_train_loader",
            return_value=fake_loader,
        ):
            res = self.client.post(
                "/d/iris/train",
                json={
                    "model_type": "QVC",
                    "epochs": 1,
                    "batch_size": 8,
                    "lr": 0.01,
                    "name": "test-QVC",
                },
                content_type="application/json",
            )
        assert res.status_code == 200
        events = _parse_sse(res.data)
        done = next(e for e in events if e["type"] == "done")
        assert done["name"] == "test-QVC"


# ═══════════════════════════════════════════════════════════════════════════════
# SSE event structure — verify all expected fields
# ═══════════════════════════════════════════════════════════════════════════════


class TestSSEEventStructure:
    """Verify SSE events from training have all required fields."""

    @pytest.fixture(autouse=True)
    def _app(self):
        self.application = create_app()
        self.application.config["TESTING"] = True
        self.client = self.application.test_client()

    def test_done_event_fields_mnist(self):
        from unittest.mock import patch
        fake_loader = make_fake_train_loader(batch_size=8, n_batches=3)
        with patch(
            "classifiers.datasets.mnist.plugin.MNISTPlugin.get_train_loader",
            return_value=fake_loader,
        ):
            res = self.client.post(
                "/d/mnist/train",
                json={"model_type": "CNN", "epochs": 2, "batch_size": 8,
                      "lr": 0.001, "name": "fields-test"},
                content_type="application/json",
            )
        events = _parse_sse(res.data)
        done = next(e for e in events if e["type"] == "done")
        required = {"type", "name", "model_type", "epochs", "epochs_completed",
                     "batch_size", "lr", "num_params", "stopped_early"}
        assert required.issubset(set(done.keys())), (
            f"Missing fields: {required - set(done.keys())}"
        )

    def test_status_events_include_progress(self):
        from unittest.mock import patch
        fake_loader = make_fake_train_loader(batch_size=8, n_batches=3)
        with patch(
            "classifiers.datasets.mnist.plugin.MNISTPlugin.get_train_loader",
            return_value=fake_loader,
        ):
            res = self.client.post(
                "/d/mnist/train",
                json={"model_type": "Linear", "epochs": 1, "batch_size": 8,
                      "lr": 0.001, "name": "progress-test"},
                content_type="application/json",
            )
        events = _parse_sse(res.data)
        status_msgs = [e["msg"] for e in events if e["type"] == "status"]
        assert any("Preparing" in m for m in status_msgs)
        assert any("Epoch" in m for m in status_msgs)
        assert any("complete" in m.lower() for m in status_msgs)

    def test_done_event_fields_iris(self):
        from unittest.mock import patch
        fake_loader = _make_iris_loader(n_samples=16, batch_size=8)
        with patch(
            "classifiers.datasets.iris.plugin.IrisPlugin.get_train_loader",
            return_value=fake_loader,
        ):
            res = self.client.post(
                "/d/iris/train",
                json={"model_type": "SVM", "epochs": 1, "batch_size": 8,
                      "lr": 0.01, "name": "iris-fields"},
                content_type="application/json",
            )
        events = _parse_sse(res.data)
        done = next(e for e in events if e["type"] == "done")
        assert done["model_type"] == "SVM"
        assert done["num_params"] > 0
        assert done["stopped_early"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Model registered in registry after route training
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelRegisteredAfterTrainRoute:
    """Verify models appear in the registry after training via HTTP."""

    @pytest.fixture(autouse=True)
    def _app(self):
        self.application = create_app()
        self.application.config["TESTING"] = True
        self.client = self.application.test_client()
        self.registry = self.application.extensions["registry"]

    @pytest.mark.parametrize("model_type", ["CNN", "Linear", "SVM"])
    def test_mnist_model_in_registry(self, model_type):
        import time
        from unittest.mock import patch
        fake_loader = make_fake_train_loader(batch_size=8, n_batches=3)
        with patch(
            "classifiers.datasets.mnist.plugin.MNISTPlugin.get_train_loader",
            return_value=fake_loader,
        ):
            res = self.client.post(
                "/d/mnist/train",
                json={"model_type": model_type, "epochs": 1,
                      "batch_size": 8, "lr": 0.001, "name": f"reg-{model_type}"},
                content_type="application/json",
            )
        events = _parse_sse(res.data)
        assert any(e["type"] == "done" for e in events)
        # Give the daemon thread a moment to flush registry writes
        time.sleep(0.1)
        entry = self.registry.get("mnist", f"reg-{model_type}")
        assert entry is not None, f"{model_type} not found in registry"
        assert entry.model_type == model_type

    @pytest.mark.parametrize("model_type", ["Linear", "SVM"])
    def test_iris_model_in_registry(self, model_type):
        import time
        from unittest.mock import patch
        fake_loader = _make_iris_loader(n_samples=16, batch_size=8)
        with patch(
            "classifiers.datasets.iris.plugin.IrisPlugin.get_train_loader",
            return_value=fake_loader,
        ):
            res = self.client.post(
                "/d/iris/train",
                json={"model_type": model_type, "epochs": 1,
                      "batch_size": 8, "lr": 0.01, "name": f"reg-{model_type}"},
                content_type="application/json",
            )
        events = _parse_sse(res.data)
        assert any(e["type"] == "done" for e in events)
        time.sleep(0.1)
        entry = self.registry.get("iris", f"reg-{model_type}")
        assert entry is not None, f"{model_type} not found in registry"
        assert entry.model_type == model_type
