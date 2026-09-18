"""Each architecture can learn, on synthetic data, through its own objective.

This file was called test_accuracy_claims.py, which overstated it twice over:
it trains on synthetic tensors rather than the real datasets, and it trained
every model — SVMs included — with cross-entropy, so the hinge-loss path it
appeared to cover was never executed. Models are now trained through their own
``loss_fn``, which is what the platform does.

The real accuracy claims are measured in exports/benchmarks.json
(tools/benchmark.py) and enforced by tests/test_model_docs.py. What is checked
here is narrower and still worth checking: each architecture learns something
above chance, and its outputs have the shape and range the rest of the
platform assumes.
"""

import pytest
import torch

from classifiers.datasets.iris.models import IrisLinear, IrisSVM
from classifiers.datasets.mnist.models import LinearNet, MNISTNet, SVMNet

# --- Helpers ---

def train_and_eval(model, train_loader, test_loader, epochs=3, lr=0.01):
    """Train a model briefly through its own loss and return test accuracy.

    ``type(model).loss_fn`` is the platform's own dispatch: cross-entropy by
    default, multi-class hinge for the SVMs. Using a fixed criterion here meant
    the SVM tests silently exercised cross-entropy.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = type(model).loss_fn(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0.0


@pytest.fixture
def mnist_loaders():
    """Create small but structured MNIST-like data for reproducibility checks."""
    torch.manual_seed(42)
    n_train, n_test = 200, 50
    # Create data with structure so models can learn (not pure random)
    train_data = torch.randn(n_train, 1, 28, 28)
    train_targets = torch.randint(0, 10, (n_train,))
    # Make labels slightly correlated with data for learnability
    for i in range(n_train):
        train_data[i, 0, train_targets[i].item(), :] += 2.0

    test_data = torch.randn(n_test, 1, 28, 28)
    test_targets = torch.randint(0, 10, (n_test,))
    for i in range(n_test):
        test_data[i, 0, test_targets[i].item(), :] += 2.0

    train_loader = [(train_data[j:j+32], train_targets[j:j+32])
                    for j in range(0, n_train, 32)]
    test_loader = [(test_data[j:j+32], test_targets[j:j+32])
                   for j in range(0, n_test, 32)]
    return train_loader, test_loader


@pytest.fixture
def iris_loaders():
    """Create small structured Iris-like data."""
    torch.manual_seed(42)
    n_train, n_test = 100, 30
    # 3 classes, 4 features
    train_data = torch.randn(n_train, 4)
    train_targets = torch.randint(0, 3, (n_train,))
    for i in range(n_train):
        train_data[i, train_targets[i].item()] += 3.0

    test_data = torch.randn(n_test, 4)
    test_targets = torch.randint(0, 3, (n_test,))
    for i in range(n_test):
        test_data[i, test_targets[i].item()] += 3.0

    train_loader = [(train_data[j:j+16], train_targets[j:j+16])
                    for j in range(0, n_train, 16)]
    test_loader = [(test_data[j:j+16], test_targets[j:j+16])
                   for j in range(0, n_test, 16)]
    return train_loader, test_loader


class TestMNISTTrainability:
    """MNIST architectures learn above chance on synthetic data.

    The real accuracy gate is exports/benchmarks.json, enforced by
    tests/test_model_docs.py."""

    def test_cnn_learns_above_chance(self, mnist_loaders):
        """CNN should learn well above 10% random chance."""
        train_loader, test_loader = mnist_loaders
        model = MNISTNet()
        acc = train_and_eval(model, train_loader, test_loader, epochs=15, lr=0.001)
        assert acc > 0.12, f"CNN accuracy {acc:.1%} not above chance"

    def test_linear_learns_above_chance(self, mnist_loaders):
        """Linear should learn above random chance."""
        train_loader, test_loader = mnist_loaders
        model = LinearNet()
        acc = train_and_eval(model, train_loader, test_loader, epochs=5)
        assert acc > 0.12, f"Linear accuracy {acc:.1%} not above chance"

    def test_svm_learns_above_chance(self, mnist_loaders):
        """SVM should learn above random chance."""
        train_loader, test_loader = mnist_loaders
        model = SVMNet()
        acc = train_and_eval(model, train_loader, test_loader, epochs=5)
        assert acc > 0.12, f"SVM accuracy {acc:.1%} not above chance"

    def test_cnn_produces_valid_predictions(self, mnist_loaders):
        """CNN should produce valid probability distributions."""
        _train_loader, test_loader = mnist_loaders
        model = MNISTNet()
        model.eval()
        with torch.no_grad():
            data, _ = test_loader[0]
            output = model(data)
            # Should produce logits for 10 classes
            assert output.shape[1] == 10
            # Softmax should sum to 1
            probs = torch.softmax(output, dim=1)
            sums = probs.sum(dim=1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestIrisTrainability:
    """Iris architectures learn above chance on synthetic data.

    The real accuracy gate is exports/benchmarks.json: Iris Linear measures
    90.0% (95% CI 74.4-96.5%, n=30) at the plugin's default hyper-parameters."""

    def test_linear_learns_above_chance(self, iris_loaders):
        """Iris Linear should learn above 33% chance."""
        train_loader, test_loader = iris_loaders
        model = IrisLinear()
        acc = train_and_eval(model, train_loader, test_loader, epochs=20, lr=0.01)
        assert acc > 0.40, f"Iris Linear accuracy {acc:.1%} not above chance"

    def test_svm_learns_above_chance(self, iris_loaders):
        """Iris SVM should learn above 33% chance."""
        train_loader, test_loader = iris_loaders
        model = IrisSVM()
        acc = train_and_eval(model, train_loader, test_loader, epochs=20, lr=0.01)
        assert acc > 0.40, f"Iris SVM accuracy {acc:.1%} not above chance"


class TestModelOutputShape:
    """Verify all models produce correct output dimensions."""

    def test_mnist_cnn_output_10(self):
        x = torch.randn(1, 1, 28, 28)
        assert MNISTNet()(x).shape == (1, 10)

    def test_mnist_linear_output_10(self):
        x = torch.randn(1, 1, 28, 28)
        assert LinearNet()(x).shape == (1, 10)

    def test_mnist_svm_output_10(self):
        x = torch.randn(1, 1, 28, 28)
        assert SVMNet()(x).shape == (1, 10)

    def test_iris_linear_output_3(self):
        x = torch.randn(1, 4)
        assert IrisLinear()(x).shape == (1, 3)

    def test_iris_svm_output_3(self):
        x = torch.randn(1, 4)
        assert IrisSVM()(x).shape == (1, 3)


class TestModelsTrainThroughTheirOwnObjective:
    """The dispatch this file used to bypass."""

    def test_svm_models_do_not_use_the_default_loss(self):
        from classifiers.base_model import BaseModel
        from classifiers.datasets.bb84.models import BB84SVM

        for svm in (SVMNet, IrisSVM, BB84SVM):
            assert svm.loss_fn is not BaseModel.loss_fn, f"{svm.__name__} lost its hinge loss"

    def test_hinge_and_cross_entropy_disagree(self):
        """So training through the wrong one is a real difference, not a
        stylistic one."""
        torch.manual_seed(0)
        output, target = torch.randn(8, 3), torch.randint(0, 3, (8,))
        hinge = IrisSVM.loss_fn(output, target)
        cross_entropy = torch.nn.functional.cross_entropy(output, target)
        assert not torch.isclose(hinge, cross_entropy)


class TestQvcConfidenceCeiling:
    """The QVC's documented limit: Pauli-Z expectations are bounded, and the
    models use them directly as logits."""

    def test_softmax_over_bounded_logits_cannot_reach_certainty(self):
        """The 0.79 the model docs quote, derived rather than remembered."""
        best_case = torch.tensor([1.0, -1.0, -1.0])
        assert torch.softmax(best_case, dim=0)[0].item() == pytest.approx(0.787, abs=5e-4)

    def test_qvc_outputs_stay_inside_the_bound(self):
        pytest.importorskip("pennylane", reason="pennylane not installed")
        from classifiers.datasets.iris.models import IrisQVC

        torch.manual_seed(0)
        out = IrisQVC()(torch.randn(6, 4))
        assert out.shape == (6, 3)
        assert out.abs().max().item() <= 1.0 + 1e-6
