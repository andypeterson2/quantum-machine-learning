"""A plugin may only advertise a model it can build.

The quantum architectures live behind optional packages, and every one of them
imports its backend lazily — inside ``__init__``, or deeper still. Importing the
model class therefore succeeds whether or not the backend is installed, so a
plugin that gates on the class import advertises a model type that raises
``ImportError`` at train time and reaches the client as a 500.

Each plugin must gate on the package itself. These tests hold both directions:
the type is absent when the package cannot be imported, and present when it can.
"""

from __future__ import annotations

import pytest

from classifiers.datasets.bb84.plugin import BB84Plugin
from classifiers.datasets.iris.plugin import IrisPlugin
from classifiers.datasets.mnist.plugin import MNISTPlugin
from tests.conftest import blocked_imports

#: (plugin class, packages the models need, model types that need them).
GATED = [
    pytest.param(
        MNISTPlugin, ("qiskit", "qiskit_aer"), {"Qiskit-CNN", "Qiskit-Linear"}, id="mnist"
    ),
    pytest.param(IrisPlugin, ("pennylane",), {"QVC"}, id="iris"),
    pytest.param(BB84Plugin, ("pennylane",), {"QVC"}, id="bb84"),
]

#: Every plugin offers these whatever is installed.
ALWAYS = {
    MNISTPlugin: {"CNN", "Linear", "SVM", "Quadratic", "Polynomial"},
    IrisPlugin: {"Linear", "SVM"},
    BB84Plugin: {"Linear", "SVM"},
}


@pytest.mark.parametrize(("plugin_cls", "packages", "gated_types"), GATED)
def test_gated_types_are_hidden_when_the_package_is_missing(
    plugin_cls, packages, gated_types
) -> None:
    with blocked_imports(*packages):
        offered = set(plugin_cls().get_model_types())
    assert offered == ALWAYS[plugin_cls], (
        f"{plugin_cls.__name__} offers {sorted(offered - ALWAYS[plugin_cls])} "
        f"without {list(packages)}"
    )


@pytest.mark.parametrize(("plugin_cls", "packages", "gated_types"), GATED)
def test_gated_types_are_offered_when_the_package_is_present(
    plugin_cls, packages, gated_types
) -> None:
    for package in packages:
        pytest.importorskip(package, reason=f"{package} not installed")
    offered = set(plugin_cls().get_model_types())
    assert gated_types <= offered


@pytest.mark.parametrize(("plugin_cls", "packages", "gated_types"), GATED)
def test_every_offered_type_can_be_built(plugin_cls, packages, gated_types) -> None:
    """The point of the gate: an advertised type constructs without raising."""
    for package in packages:
        pytest.importorskip(package, reason=f"{package} not installed")
    for name, model_cls in plugin_cls().get_model_types().items():
        assert model_cls() is not None, name
