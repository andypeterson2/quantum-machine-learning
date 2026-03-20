"""Unit tests for classifiers.base_model ABC and model conformance."""

import pytest
import torch

from classifiers.base_model import BaseModel
from classifiers.datasets.mnist.models import MNISTNet, LinearNet, SVMNet
from classifiers.datasets.mnist.plugin import MNISTPlugin


class TestBaseModelABC:
    def test_cannot_instantiate_directly(self):
        """BaseModel is abstract — instantiating it directly should fail."""
        with pytest.raises(TypeError):
            BaseModel()

    def test_subclass_must_implement_forward(self):
        """A subclass without forward() should fail to instantiate."""

        class BadModel(BaseModel):
            name = "Bad"
            description = "Missing forward"

        with pytest.raises(TypeError):
            BadModel()


class TestPluginModelTypes:
    """Test model types returned by the MNIST plugin."""

    def test_cnn_registered(self):
        plugin = MNISTPlugin()
        types = plugin.get_model_types()
        assert "CNN" in types
        assert types["CNN"] is MNISTNet

    def test_linear_registered(self):
        plugin = MNISTPlugin()
        types = plugin.get_model_types()
        assert "Linear" in types
        assert types["Linear"] is LinearNet

    def test_svm_registered(self):
        plugin = MNISTPlugin()
        types = plugin.get_model_types()
        assert "SVM" in types
        assert types["SVM"] is SVMNet

    def test_has_at_least_five_models(self):
        """CNN, Linear, SVM, Quadratic, Polynomial (and optionally Qiskit)."""
        plugin = MNISTPlugin()
        types = plugin.get_model_types()
        assert len(types) >= 5


class TestModelConformance:
    """CNN, Linear, and SVM models should all conform to the BaseModel interface."""

    @pytest.mark.parametrize("model_cls", [MNISTNet, LinearNet, SVMNet])
    def test_is_base_model(self, model_cls):
        model = model_cls()
        assert isinstance(model, BaseModel)
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize("model_cls", [MNISTNet, LinearNet, SVMNet])
    def test_has_name_and_description(self, model_cls):
        assert isinstance(model_cls.name, str) and len(model_cls.name) > 0
        assert isinstance(model_cls.description, str) and len(model_cls.description) > 0

    @pytest.mark.parametrize("model_cls", [MNISTNet, LinearNet, SVMNet])
    def test_forward_produces_10_logits(self, model_cls):
        model = model_cls()
        x = torch.randn(1, 1, 28, 28)
        out = model(x)
        assert out.shape == (1, 10)

    @pytest.mark.parametrize("model_cls", [MNISTNet, LinearNet, SVMNet])
    def test_batch_forward(self, model_cls):
        model = model_cls()
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert out.shape == (8, 10)

    @pytest.mark.parametrize("model_cls", [MNISTNet, LinearNet, SVMNet])
    def test_has_trainable_parameters(self, model_cls):
        model = model_cls()
        count = sum(p.numel() for p in model.parameters())
        assert count > 0

    @pytest.mark.parametrize("model_cls", [MNISTNet, LinearNet, SVMNet])
    def test_loss_fn_is_callable(self, model_cls):
        out = torch.randn(4, 10)
        tgt = torch.randint(0, 10, (4,))
        loss = model_cls.loss_fn(out, tgt)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0.0  # non-negative
