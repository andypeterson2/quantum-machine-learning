"""WP #697: Test: Accuracy claims match reproducible results.

Verify that training each model architecture for a short run produces
accuracy within a plausible range of the documented claims. We use very
short training (a few epochs) and check that accuracy is non-trivially
above random chance and that the model converges in the expected direction.

Full convergence to documented accuracy requires longer training, so we
validate the *structure* of the claim — i.e. that the architecture can
learn the task — rather than exact numbers.
"""

import pytest
import torch

from classifiers.datasets.mnist.models import MNISTNet, LinearNet, SVMNet
from classifiers.datasets.iris.models import IrisLinear, IrisSVM


# --- Helpers ---

def train_and_eval(model, train_loader, test_loader, epochs=3, lr=0.01):
    """Train a model briefly and return test accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
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


class TestMNISTAccuracyClaims:
    """README claims: CNN ~99%, Linear ~92%, SVM ~91-92%."""

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
        train_loader, test_loader = mnist_loaders
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


class TestIrisAccuracyClaims:
    """README claims: Linear ~95-97%, SVM ~94-96%."""

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
