"""The README has to cover what the code exposes.

Every environment variable the code reads, every dataset plugin, every model
type and every make target has to appear in the README, and the version it
reports has to be the one in pyproject.
"""

import re
from pathlib import Path

import pytest

from classifiers.plugin_registry import discover_plugins, list_plugins

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

#: Werkzeug sets this itself; it is not ours to document.
INTERNAL_ENV = {"WERKZEUG_RUN_MAIN"}

#: Make targets for this file's own author.
UNDOCUMENTED_TARGETS = {"clean", "lint", "test", "run"}


@pytest.fixture(scope="module", autouse=True)
def _plugins() -> None:
    discover_plugins()


@pytest.fixture(scope="module")
def readme() -> str:
    assert README.is_file(), "README.md is missing"
    return README.read_text()


def _env_vars_in_code() -> set[str]:
    found: set[str] = set()
    for path in ROOT.joinpath("classifiers").rglob("*.py"):
        found |= set(re.findall(r'environ(?:\.get)?[\(\[]\s*"([A-Z_]+)"', path.read_text()))
    return found - INTERNAL_ENV


class TestTheReadmeCoversWhatTheCodeExposes:
    def test_every_environment_variable_is_documented(self, readme) -> None:
        missing = sorted(var for var in _env_vars_in_code() if f"`{var}`" not in readme)
        assert not missing, f"environment variables read but never documented: {missing}"

    def test_every_dataset_is_documented(self, readme) -> None:
        missing = [
            plugin.display_name
            for plugin in list_plugins().values()
            if plugin.display_name.split()[0] not in readme
        ]
        assert not missing, f"datasets the platform serves but the README omits: {missing}"

    def test_every_model_type_is_documented(self, readme) -> None:
        """A model the API offers and the README never mentions is invisible."""
        missing = sorted(
            {
                model_type
                for plugin in list_plugins().values()
                for model_type in plugin.get_model_types()
                if model_type not in readme
            }
        )
        assert not missing, f"model types offered but undocumented: {missing}"

    def test_every_make_target_is_documented(self, readme) -> None:
        makefile = (ROOT / "Makefile").read_text()
        targets = set(re.findall(r"^([a-z][a-z-]*):", makefile, re.MULTILINE))
        missing = sorted(t for t in targets - UNDOCUMENTED_TARGETS if f"make {t}" not in readme)
        assert not missing, f"make targets a reader cannot discover: {missing}"


class TestReadmeHonesty:
    """The README's checkable claims have to check out."""

    def test_version_single_source(self):
        """classifiers.__version__ (the /health fallback) matches pyproject."""
        import classifiers

        pyproject = (ROOT / "pyproject.toml").read_text()
        assert f'version = "{classifiers.__version__}"' in pyproject
