"""classifiers — multi-dataset interactive classification platform.

This package provides a Flask web interface for training, evaluating, and
testing classifiers on multiple datasets (MNIST, Iris, and more).

Dataset plugins are auto-discovered from ``classifiers/datasets/`` at
application startup via :func:`~classifiers.plugin_registry.discover_plugins`.
Each plugin bundles dataset-specific data loading, preprocessing, and model
architectures.  Adding a new dataset requires only a new subpackage — no
existing code needs modification (Open/Closed Principle).
"""
