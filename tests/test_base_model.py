"""Unit tests for classifiers.base_model ABC and model conformance."""

import pytest
import torch

from classifiers.base_model import BaseModel
from classifiers.datasets.mnist.models import LinearNet, MNISTNet, SVMNet
from classifiers.datasets.mnist.plugin import MNISTPlugin
from classifiers.plugin_registry import discover_plugins, list_plugins


def _plugins():
    discover_plugins()
    return list_plugins()


def _advertised():
    """(plugin, model class) for every architecture every plugin offers."""
    return [
        (plugin, model_cls)
        for plugin in _plugins().values()
        for model_cls in plugin.get_model_types().values()
    ]


def _advertised_ids():
    return [
        f"{plugin.name}-{name}"
        for plugin in _plugins().values()
        for name in plugin.get_model_types()
    ]


def _sample_input(plugin, batch: int) -> torch.Tensor:
    """A random batch shaped the way *plugin* feeds its models."""
    if plugin.input_type == "image":
        return torch.randn(batch, plugin.image_channels, *plugin.image_size)
    return torch.randn(batch, len(plugin.feature_names))


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
    """Test model types returned by the MNIST plugin.

    Which types the plugin offers when an optional backend is missing is held by
    tests/test_optional_model_gate.py; this checks what each name maps to.
    """

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("CNN", MNISTNet), ("Linear", LinearNet), ("SVM", SVMNet)],
    )
    def test_classical_names_map_to_their_architectures(self, name, expected):
        assert MNISTPlugin().get_model_types()[name] is expected


class TestModelConformance:
    """Every architecture a plugin advertises meets the BaseModel contract.

    Parametrized over what the plugins actually offer, so a new model is covered
    the moment it is registered — and so the per-model test files do not each
    re-assert the same shape and isinstance checks.
    """

    @pytest.mark.parametrize(("plugin", "model_cls"), _advertised(), ids=_advertised_ids())
    def test_is_base_model(self, plugin, model_cls):
        model = model_cls()
        assert isinstance(model, BaseModel)
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize(("plugin", "model_cls"), _advertised(), ids=_advertised_ids())
    def test_has_name_and_description(self, plugin, model_cls):
        assert isinstance(model_cls.name, str) and len(model_cls.name) > 0
        assert isinstance(model_cls.description, str) and len(model_cls.description) > 0

    @pytest.mark.parametrize(("plugin", "model_cls"), _advertised(), ids=_advertised_ids())
    def test_forward_produces_one_logit_per_class(self, plugin, model_cls):
        """Single sample and batch, at the plugin's own input shape."""
        model = model_cls()
        for batch in (1, 8):
            out = model(_sample_input(plugin, batch))
            assert out.shape == (batch, plugin.num_classes)

    @pytest.mark.parametrize(("plugin", "model_cls"), _advertised(), ids=_advertised_ids())
    def test_has_trainable_parameters(self, plugin, model_cls):
        model = model_cls()
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0

    @pytest.mark.parametrize(("plugin", "model_cls"), _advertised(), ids=_advertised_ids())
    def test_loss_fn_returns_a_non_negative_scalar(self, plugin, model_cls):
        out = torch.randn(4, plugin.num_classes)
        tgt = torch.randint(0, plugin.num_classes, (4,))
        loss = model_cls.loss_fn(out, tgt)
        assert loss.ndim == 0
        assert loss.item() >= 0.0
