"""Unit tests for classifiers.plugin_registry."""

import pytest

from classifiers.plugin_registry import (
    get_plugin,
    list_plugins,
    create_model,
    discover_plugins,
)
from classifiers.base_model import BaseModel


class TestPluginDiscovery:
    def test_discover_finds_mnist(self):
        discover_plugins()
        plugin = get_plugin("mnist")
        assert plugin is not None
        assert plugin.name == "mnist"

    def test_discover_finds_iris(self):
        discover_plugins()
        plugin = get_plugin("iris")
        assert plugin is not None
        assert plugin.name == "iris"

    def test_list_plugins_returns_all(self):
        discover_plugins()
        plugins = list_plugins()
        assert "mnist" in plugins
        assert "iris" in plugins

    def test_get_unknown_returns_none(self):
        assert get_plugin("nonexistent_dataset") is None


class TestCreateModel:
    def test_create_mnist_cnn(self):
        discover_plugins()
        model = create_model("mnist", "CNN")
        assert isinstance(model, BaseModel)
        assert model.name == "CNN"

    def test_create_iris_linear(self):
        discover_plugins()
        model = create_model("iris", "Linear")
        assert isinstance(model, BaseModel)

    def test_create_unknown_dataset_raises(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            create_model("nonexistent", "CNN")

    def test_create_unknown_model_type_raises(self):
        discover_plugins()
        with pytest.raises(KeyError, match="not available"):
            create_model("mnist", "NonexistentModel")
