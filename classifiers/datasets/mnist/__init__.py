"""MNIST dataset plugin — registers automatically on import."""

from classifiers.plugin_registry import register_plugin
from .plugin import MNISTPlugin

register_plugin(MNISTPlugin())
