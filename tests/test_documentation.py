"""Documentation gates: the README's claims have to match the code.

This file used to hold about forty assertions of the form "CNN" in readme —
true of any document that mentions the word once, and unable to notice a model
nobody documented or a setting nobody wrote down. Ten environment variables had
in fact gone undocumented while every one of those tests passed.

The gates here compare the README against what the code actually exposes:
every environment variable it reads, every dataset plugin, every model type,
and every make target. Plus the honesty checks the 2026-08 audit added.
"""

import re
from pathlib import Path

import pytest

from classifiers.plugin_registry import discover_plugins, list_plugins

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

#: Werkzeug sets this itself; it is not ours to document.
INTERNAL_ENV = {"WERKZEUG_RUN_MAIN"}

#: Make targets whose audience is this file's own author, not a reader.
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
        """Ten of these were missing when this was a substring check."""
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


class TestSetupInstructionsStayRunnable:
    """One test per thing a new reader has to be able to do."""

    def test_install_step_names_the_file_it_installs(self, readme) -> None:
        assert "pip install -r requirements.txt" in readme

    def test_run_step_matches_the_module_entry_point(self, readme) -> None:
        assert "python -m classifiers" in readme
        assert (ROOT / "classifiers" / "__main__.py").is_file()

    def test_test_step_matches_the_suite(self, readme) -> None:
        assert "python -m pytest tests/" in readme

    def test_docker_steps_match_the_files_they_use(self, readme) -> None:
        assert "docker compose up" in readme
        assert (ROOT / "docker-compose.yml").is_file()
        assert (ROOT / "Dockerfile").is_file()


class TestReadmeHonesty:
    """The README's checkable claims must actually check out — the gates that
    would have caught the drift a 2026-08 audit found by hand."""

    def test_readme_test_count_matches_reality(self, readme):
        """The stated test-function count is asserted, not decorative."""
        stated = {int(n) for n in re.findall(r"\((\d+) test functions\)", readme)}
        assert stated, "README no longer states a test-function count"
        actual = sum(
            len(re.findall(r"^\s*def test_", p.read_text(), re.MULTILINE))
            for p in (ROOT / "tests").rglob("test_*.py")
        )
        assert stated == {actual}, (
            f"README says {stated} test functions; tests/ defines {actual} — update the README"
        )

    def test_readme_tree_paths_resolve(self):
        """Key paths the README documents must exist on disk."""
        for rel in (
            "classifiers/web_export.py",
            "classifiers/qsvm_export.py",
            "classifiers/wsgi.py",
            "exports/web/iris.json",
            "exports/web/qsvm-mnist.json",
            "notebooks/qsvm-iris/qsvm_iris.ipynb",
            "tests/contract/schemas",
        ):
            assert (ROOT / rel).exists(), f"README-documented path missing: {rel}"

    def test_version_single_source(self):
        """classifiers.__version__ (the /health fallback) matches pyproject."""
        import classifiers

        pyproject = (ROOT / "pyproject.toml").read_text()
        assert f'version = "{classifiers.__version__}"' in pyproject

    def test_contributing_and_license_are_present(self):
        assert (ROOT / "CONTRIBUTING.md").is_file()
        assert (ROOT / "LICENSE").is_file()
