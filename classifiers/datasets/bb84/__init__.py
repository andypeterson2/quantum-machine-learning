"""BB84 eavesdropper-detection dataset plugin package."""

from classifiers.plugin_registry import register_plugin

from .plugin import BB84Plugin

register_plugin(BB84Plugin())
