"""Iris flower classification dataset plugin — registers automatically on import."""

from classifiers.plugin_registry import register_plugin
from .plugin import IrisPlugin

register_plugin(IrisPlugin())
