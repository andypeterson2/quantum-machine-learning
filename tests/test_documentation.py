"""Documentation completeness and honesty gates.

Verify README has working setup instructions, architecture matches code,
all configuration options are documented, and common issues are covered.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class TestReadmeExists:
    """Basic documentation files must be present."""

    def test_readme_exists(self):
        assert (ROOT / "README.md").is_file()

    def test_readme_has_substantial_content(self):
        content = (ROOT / "README.md").read_text()
        assert len(content) > 1000, "README too short"


    def test_contributing_exists(self):
        assert (ROOT / "CONTRIBUTING.md").is_file()

    def test_license_exists(self):
        assert (ROOT / "LICENSE").is_file()


class TestReadmeSetupInstructions:
    """Setup instructions should cover essential steps."""

    @pytest.fixture
    def readme(self):
        return (ROOT / "README.md").read_text()

    def test_mentions_python_version(self, readme):
        assert "3.12" in readme or "3.11" in readme or "3.10" in readme

    def test_mentions_pip_install(self, readme):
        assert "pip install" in readme or "requirements.txt" in readme

    def test_mentions_docker(self, readme):
        assert "docker" in readme.lower() or "Docker" in readme

    def test_mentions_running_the_app(self, readme):
        assert "python -m" in readme or "flask" in readme.lower()

    def test_mentions_testing(self, readme):
        assert "pytest" in readme


class TestArchitectureDocumentation:
    """Architecture docs should match actual code structure."""

    @pytest.fixture
    def readme(self):
        return (ROOT / "README.md").read_text()

    def test_documents_flask_server(self, readme):
        assert "Flask" in readme

    def test_documents_trainer(self, readme):
        assert "Trainer" in readme

    def test_documents_evaluator(self, readme):
        assert "Evaluator" in readme

    def test_documents_predictor(self, readme):
        assert "Predictor" in readme

    def test_server_module_exists(self):
        assert (ROOT / "classifiers" / "server.py").is_file()

    def test_trainer_module_exists(self):
        assert (ROOT / "classifiers" / "trainer.py").is_file()

    def test_evaluator_module_exists(self):
        assert (ROOT / "classifiers" / "evaluator.py").is_file()

    def test_predictor_module_exists(self):
        assert (ROOT / "classifiers" / "predictor.py").is_file()


class TestModelDocumentation:
    """All model architectures should be documented."""

    @pytest.fixture
    def readme(self):
        return (ROOT / "README.md").read_text()

    def test_documents_cnn(self, readme):
        assert "CNN" in readme

    def test_documents_linear(self, readme):
        assert "Linear" in readme

    def test_documents_svm(self, readme):
        assert "SVM" in readme

    def test_documents_quadratic(self, readme):
        assert "Quadratic" in readme or "quadratic" in readme

    def test_documents_polynomial(self, readme):
        assert "Polynomial" in readme or "polynomial" in readme

    def test_documents_qiskit(self, readme):
        assert "Qiskit" in readme or "qiskit" in readme

    def test_documents_iris_dataset(self, readme):
        assert "Iris" in readme or "iris" in readme

    def test_documents_mnist_dataset(self, readme):
        assert "MNIST" in readme or "mnist" in readme


class TestConfigurationDocumentation:
    """Training parameters and configuration options should be documented."""

    @pytest.fixture
    def readme(self):
        return (ROOT / "README.md").read_text()

    def test_documents_epochs(self, readme):
        assert "epoch" in readme.lower()

    def test_documents_learning_rate(self, readme):
        assert "learning_rate" in readme or "lr" in readme.lower()

    def test_documents_batch_size(self, readme):
        assert "batch" in readme.lower()

    def test_documents_accuracy_metrics(self, readme):
        assert "accuracy" in readme.lower() or "Accuracy" in readme


class TestApiDocumentation:
    """API docs should exist and cover key endpoints."""

    def test_api_docs_exist(self):
        assert (ROOT / "docs" / "api.md").is_file()

    def test_models_docs_exist(self):
        assert (ROOT / "docs" / "models.md").is_file()

    def test_architecture_docs_exist(self):
        assert (ROOT / "docs" / "architecture.md").is_file()

    def test_api_docs_have_content(self):
        content = (ROOT / "docs" / "api.md").read_text()
        assert len(content) > 100


class TestDockerDocumentation:
    """Docker-related files should be present and documented."""

    def test_dockerfile_exists(self):
        assert (ROOT / "Dockerfile").is_file()

    def test_docker_compose_exists(self):
        assert (ROOT / "docker-compose.yml").is_file()

    def test_requirements_txt_exists(self):
        assert (ROOT / "requirements.txt").is_file()

    def test_makefile_exists(self):
        assert (ROOT / "Makefile").is_file()


class TestReadmeHonesty:
    """The README's checkable claims must actually check out — the gates that
    would have caught the drift a 2026-08 audit found by hand."""

    def test_readme_test_count_matches_reality(self):
        """The stated test-function count is asserted, not decorative."""
        import re

        readme = (ROOT / "README.md").read_text()
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
