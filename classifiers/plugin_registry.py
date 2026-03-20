"""Dataset plugin registry and auto-discovery.

Plugins are registered at import time by calling :func:`register_plugin` from
their package ``__init__.py``.  The :func:`discover_plugins` function walks
``classifiers/datasets/`` and imports every subpackage, which triggers those
registrations automatically — no hard-coded list of datasets is needed.

Adding a new dataset is therefore a pure addition: create a subpackage under
``classifiers/datasets/`` and the discovery mechanism finds it at startup.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

from .base_model import BaseModel

if TYPE_CHECKING:
    from .dataset_plugin import DatasetPlugin

_PLUGINS: dict[str, DatasetPlugin] = {}


def register_plugin(plugin: DatasetPlugin) -> None:
    """Register a dataset plugin so it is available to routes and the UI.

    Typically called from ``classifiers/datasets/<name>/__init__.py``.

    Args:
        plugin: A fully constructed :class:`DatasetPlugin` instance.
    """
    _PLUGINS[plugin.name] = plugin


def get_plugin(name: str) -> DatasetPlugin | None:
    """Look up a registered plugin by its URL-safe slug.

    Args:
        name: The plugin's :attr:`~DatasetPlugin.name` attribute.

    Returns:
        The corresponding :class:`DatasetPlugin` instance, or ``None`` if
        no plugin with that name has been registered.
    """
    return _PLUGINS.get(name)


def list_plugins() -> dict[str, DatasetPlugin]:
    """Return all registered plugins keyed by name.

    Returns:
        A dict mapping plugin slug to :class:`DatasetPlugin` instance.
    """
    return dict(_PLUGINS)


def create_model(dataset_name: str, model_type_name: str) -> BaseModel:
    """Instantiate a model by dataset and architecture name.

    Used by :class:`~classifiers.persistence.ModelPersistence` when loading
    checkpoints from disk.

    Args:
        dataset_name:    Plugin slug (e.g. ``"mnist"``).
        model_type_name: Architecture display name (e.g. ``"CNN"``).

    Returns:
        A freshly constructed, untrained :class:`BaseModel` instance.

    Raises:
        KeyError: If the dataset or model type is not registered.
    """
    plugin = get_plugin(dataset_name)
    if plugin is None:
        raise KeyError(f"Unknown dataset: {dataset_name!r}")
    model_types = plugin.get_model_types()
    if model_type_name not in model_types:
        raise KeyError(
            f"Model type {model_type_name!r} not available for "
            f"dataset {dataset_name!r}. "
            f"Available: {list(model_types)}"
        )
    return model_types[model_type_name]()


def discover_plugins() -> None:
    """Import all subpackages of ``classifiers.datasets`` to trigger registration.

    This is called once during :func:`~classifiers.server.create_app`.
    """
    import classifiers.datasets as pkg

    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        importlib.import_module(info.name)
