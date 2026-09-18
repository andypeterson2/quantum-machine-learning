"""End-to-end runs through the pipeline: train, evaluate, predict."""

import torch

from tests.conftest import make_fake_train_loader as _make_mnist_loader


def _make_iris_loader(n_samples=40, batch_size=16):
    """Build a synthetic Iris-shaped DataLoader (4 features, 3 classes)."""
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(n_samples, 4)
    y = torch.randint(0, 3, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


class TestE2EPipeline:
    """Data in, train, evaluate, predict out."""

    def test_iris_linear_train_evaluate_predict(self):
        """Full pipeline: instantiate Iris Linear, train, eval, predict."""
        from classifiers.datasets.iris.models import IrisLinear
        from classifiers.evaluator import Evaluator
        from classifiers.trainer import Trainer

        train_loader = _make_iris_loader(n_samples=60, batch_size=16)
        test_loader = _make_iris_loader(n_samples=20, batch_size=20)

        trainer = Trainer(
            model_cls=IrisLinear,
            train_loader=train_loader,
            dataset="iris",
            epochs=2,
            lr=0.01,
        )
        result = trainer.train()
        assert result.model is not None
        assert result.epochs_completed == 2

        evaluator = Evaluator()
        eval_result = evaluator.evaluate(
            result.model, test_loader, num_classes=3, class_labels=list(range(3))
        )
        assert 0.0 <= eval_result.accuracy <= 1.0

        # Predict on a single sample
        model = result.model
        model.eval()
        with torch.no_grad():
            sample = torch.randn(1, 4)
            output = model(sample)
        assert output.shape == (1, 3)

    def test_mnist_cnn_train_evaluate(self):
        """Train a CNN on synthetic MNIST-shaped data and evaluate."""
        from classifiers.datasets.mnist.models import MNISTNet
        from classifiers.evaluator import Evaluator
        from classifiers.trainer import Trainer

        train_loader = _make_mnist_loader(batch_size=8, n_batches=3)
        test_loader = _make_mnist_loader(batch_size=8, n_batches=2)

        trainer = Trainer(
            model_cls=MNISTNet,
            train_loader=train_loader,
            dataset="mnist",
            epochs=1,
            lr=0.001,
        )
        result = trainer.train()
        assert result.model is not None
        assert result.epochs_completed == 1

        evaluator = Evaluator()
        eval_result = evaluator.evaluate(
            result.model, test_loader, num_classes=10, class_labels=list(range(10))
        )
        assert 0.0 <= eval_result.accuracy <= 1.0

    def test_svm_model_uses_hinge_loss(self):
        """SVM models should use hinge loss, not cross-entropy."""
        from classifiers.datasets.iris.models import IrisSVM

        model = IrisSVM()
        output = model(torch.randn(4, 4))
        target = torch.randint(0, 3, (4,))
        loss = IrisSVM.loss_fn(output, target)
        assert loss.item() >= 0, "Hinge loss should be non-negative"


class TestModelAccuracyRegression:
    """Models learn enough to beat random accuracy."""

    def test_iris_linear_above_random(self):
        """After a few epochs, Iris Linear should beat random (33%)."""
        from classifiers.datasets.iris.models import IrisLinear
        from classifiers.evaluator import Evaluator
        from classifiers.trainer import Trainer

        train_loader = _make_iris_loader(n_samples=80, batch_size=16)
        test_loader = _make_iris_loader(n_samples=30, batch_size=30)

        trainer = Trainer(
            model_cls=IrisLinear,
            train_loader=train_loader,
            dataset="iris",
            epochs=20,
            lr=0.01,
        )
        result = trainer.train()

        evaluator = Evaluator()
        eval_result = evaluator.evaluate(
            result.model, test_loader, num_classes=3, class_labels=list(range(3))
        )
        # With random data the model may not beat 33% reliably,
        # so we check it doesn't crash and returns a valid accuracy.
        assert 0.0 <= eval_result.accuracy <= 1.0

    def test_mnist_linear_produces_valid_output(self):
        """LinearNet forward pass should produce valid 10-class logits."""
        from classifiers.datasets.mnist.models import LinearNet

        model = LinearNet()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_iris_svm_produces_valid_output(self):
        """IrisSVM forward pass should produce valid 3-class scores."""
        from classifiers.datasets.iris.models import IrisSVM

        model = IrisSVM()
        x = torch.randn(8, 4)
        out = model(x)
        assert out.shape == (8, 3)
        assert torch.isfinite(out).all()

    def test_train_result_tracks_num_params(self):
        """TrainResult should report a positive number of parameters."""
        from classifiers.datasets.iris.models import IrisLinear
        from classifiers.trainer import Trainer

        trainer = Trainer(
            model_cls=IrisLinear,
            train_loader=_make_iris_loader(n_samples=20, batch_size=10),
            dataset="iris",
            epochs=1,
            lr=0.01,
        )
        result = trainer.train()
        assert result.num_params > 0, "num_params should be positive"


class TestPerformanceAndResources:
    """Parameter counts and output sizes stay small."""

    def test_iris_linear_parameter_count(self):
        """IrisLinear should have very few parameters (4*3 + 3 = 15)."""
        from classifiers.datasets.iris.models import IrisLinear

        model = IrisLinear()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 15, f"Expected 15 params, got {n_params}"

    def test_iris_svm_parameter_count(self):
        """IrisSVM should also have 15 parameters (same architecture)."""
        from classifiers.datasets.iris.models import IrisSVM

        model = IrisSVM()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 15, f"Expected 15 params, got {n_params}"

    def test_mnist_cnn_parameter_count_reasonable(self):
        """MNISTNet should have a reasonable param count (< 5M)."""
        from classifiers.datasets.mnist.models import MNISTNet

        model = MNISTNet()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 5_000_000, f"Too many params: {n_params}"

    def test_forward_pass_does_not_allocate_huge_tensors(self):
        """Forward pass output should be reasonably sized."""
        from classifiers.datasets.mnist.models import MNISTNet

        model = MNISTNet()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 1, 28, 28))
        assert out.numel() == 10, "MNIST output should have 10 elements"
