"""Dataset plugin subpackages.

Each subdirectory here contains a self-contained dataset plugin.  The
:func:`~classifiers.plugin_registry.discover_plugins` function imports every
subpackage at startup, which triggers plugin registration automatically.
"""
