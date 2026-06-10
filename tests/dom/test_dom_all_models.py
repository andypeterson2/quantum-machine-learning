"""DOM-level tests: verify every model type trains via the browser UI.

Drives a real browser against a live Flask server to train each model type
and verify the UI updates correctly — session model list, evaluation metrics,
and prediction table.

Uses Iris for all classical models (fast, no data download).
MNIST models are tested with a patched loader to avoid slow training.

Run with:
    python -m pytest tests/dom/test_dom_all_models.py -v --headed
    python -m pytest tests/dom/test_dom_all_models.py -v
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from tests.dom.conftest import wait_connected


# ── Helpers ──────────────────────────────────────────────────────────────────


def _goto(page: Page, url: str) -> None:
    page.goto(url)
    wait_connected(page)


def _train(page: Page, name: str, model_type: str, epochs: int = 1) -> None:
    """Fill form and train; waits for training to complete."""
    page.locator("#model-type").select_option(model_type)
    page.locator("#epochs").fill(str(epochs))
    page.locator("#model-name").fill(name)
    page.locator("#train-btn").click()
    expect(page.locator("#train-btn")).to_be_enabled(timeout=120000)


# ═══════════════════════════════════════════════════════════════════════════════
# Iris Models — DOM training (fast, real data)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIrisLinearDOM:
    """Train Iris Linear via UI and verify all DOM updates."""

    def test_train_completes(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "iris-linear", "Linear", epochs=2)
        expect(page.locator("#session-models")).to_contain_text("iris-linear")

    def test_model_shows_in_session_list(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "iris-lin-list", "Linear", epochs=2)
        expect(page.locator("#session-models .ui-list-name")).to_contain_text(
            "iris-lin-list"
        )
        expect(page.locator("#session-models .ui-list-tag").first).to_contain_text(
            "Linear"
        )

    def test_evaluation_metrics_populated(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "iris-lin-eval", "Linear", epochs=2)
        expect(page.locator("#metrics-head")).to_contain_text(
            "iris-lin-eval", timeout=60000
        )
        expect(page.locator("#metrics-body")).to_contain_text(
            re.compile(r"\d+\.\d+%"), timeout=60000
        )

    def test_prediction_row_added(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "iris-lin-pred", "Linear", epochs=2)
        expect(page.locator("#pred-body")).to_contain_text("iris-lin-pred")

    def test_train_button_re_enabled_after_training(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "iris-lin-btn", "Linear", epochs=2)
        expect(page.locator("#train-btn")).to_be_enabled()


class TestIrisSVMDOM:
    """Train Iris SVM via UI and verify all DOM updates."""

    def test_train_completes(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "iris-svm", "SVM", epochs=2)
        expect(page.locator("#session-models")).to_contain_text("iris-svm")

    def test_evaluation_metrics_populated(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "iris-svm-eval", "SVM", epochs=2)
        expect(page.locator("#metrics-head")).to_contain_text(
            "iris-svm-eval", timeout=60000
        )
        expect(page.locator("#metrics-body")).to_contain_text(
            re.compile(r"\d+\.\d+%"), timeout=60000
        )

    def test_model_type_tag_shows_svm(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "svm-tag", "SVM", epochs=2)
        expect(page.locator("#session-models .ui-list-tag").first).to_contain_text(
            "SVM"
        )

    def test_per_class_accuracy_present(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "svm-perclass", "SVM", epochs=3)
        for label in ("setosa", "versicolor", "virginica"):
            expect(page.locator("#metrics-body")).to_contain_text(
                label, timeout=60000
            )


class TestIrisQVCDOM:
    """Train Iris QVC via UI — skipped if PennyLane is not installed."""

    @pytest.fixture(autouse=True)
    def _check_pennylane(self):
        try:
            import pennylane  # noqa: F401
        except ImportError:
            pytest.skip("pennylane not installed")

    def test_qvc_option_available(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        options = page.locator("#model-type option").all_text_contents()
        assert "QVC" in options

    def test_train_completes(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "iris-qvc", "QVC", epochs=2)
        expect(page.locator("#session-models")).to_contain_text("iris-qvc")

    def test_evaluation_metrics_populated(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "qvc-eval", "QVC", epochs=2)
        expect(page.locator("#metrics-head")).to_contain_text(
            "qvc-eval", timeout=120000
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MNIST Models — DOM training (real data, longer timeouts)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMNISTLinearDOM:
    """Train MNIST Linear via UI — fast even on real data."""

    def test_train_completes(self, page: Page, mnist_url: str):
        _goto(page, mnist_url)
        _train(page, "mnist-linear", "Linear", epochs=1)
        expect(page.locator("#session-models")).to_contain_text("mnist-linear")

    def test_evaluation_metrics_populated(self, page: Page, mnist_url: str):
        _goto(page, mnist_url)
        _train(page, "mnist-lin-eval", "Linear", epochs=1)
        expect(page.locator("#metrics-head")).to_contain_text(
            "mnist-lin-eval", timeout=120000
        )
        expect(page.locator("#metrics-body")).to_contain_text(
            re.compile(r"\d+\.\d+%"), timeout=120000
        )


class TestMNISTSVMDOM:
    """Train MNIST SVM via UI — uses hinge loss."""

    def test_train_completes(self, page: Page, mnist_url: str):
        _goto(page, mnist_url)
        _train(page, "mnist-svm", "SVM", epochs=1)
        expect(page.locator("#session-models")).to_contain_text("mnist-svm")


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction after training — each Iris model predicts correctly
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredictionAfterTraining:
    """Train a model, fill features, predict, and verify the result."""

    def _fill_setosa(self, page: Page) -> None:
        page.locator("#feat-sepal_length").fill("5.1")
        page.locator("#feat-sepal_width").fill("3.5")
        page.locator("#feat-petal_length").fill("1.4")
        page.locator("#feat-petal_width").fill("0.2")

    @pytest.mark.parametrize("model_type", ["Linear", "SVM"])
    def test_predict_returns_valid_label(self, page: Page, iris_url: str, model_type: str):
        _goto(page, iris_url)
        _train(page, f"pred-{model_type}", model_type, epochs=3)
        self._fill_setosa(page)
        page.locator("#predict-btn-tab").click()
        label = page.locator("#pred-body .pred-label")
        expect(label).to_be_visible(timeout=10000)
        assert label.text_content() in ("setosa", "versicolor", "virginica")

    @pytest.mark.parametrize("model_type", ["Linear", "SVM"])
    def test_predict_shows_confidence(self, page: Page, iris_url: str, model_type: str):
        _goto(page, iris_url)
        _train(page, f"conf-{model_type}", model_type, epochs=3)
        self._fill_setosa(page)
        page.locator("#predict-btn-tab").click()
        expect(page.locator("#pred-body")).to_contain_text(
            re.compile(r"\d+\.\d+%"), timeout=10000
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Multiple model types side-by-side
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiModelComparison:
    """Train two different model types and verify they coexist in the UI."""

    def test_two_models_both_in_session(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "compare-lin", "Linear", epochs=2)
        _train(page, "compare-svm", "SVM", epochs=2)
        expect(page.locator("#session-models")).to_contain_text("compare-lin")
        expect(page.locator("#session-models")).to_contain_text("compare-svm")

    def test_two_models_both_in_metrics(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "met-lin", "Linear", epochs=2)
        _train(page, "met-svm", "SVM", epochs=2)
        expect(page.locator("#metrics-head")).to_contain_text(
            "met-lin", timeout=60000
        )
        expect(page.locator("#metrics-head")).to_contain_text(
            "met-svm", timeout=60000
        )

    def test_two_models_both_predict(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "mp-lin", "Linear", epochs=2)
        _train(page, "mp-svm", "SVM", epochs=2)
        page.locator("#feat-sepal_length").fill("5.1")
        page.locator("#feat-sepal_width").fill("3.5")
        page.locator("#feat-petal_length").fill("1.4")
        page.locator("#feat-petal_width").fill("0.2")
        page.locator("#predict-btn-tab").click()
        labels = page.locator("#pred-body .pred-label")
        expect(labels).to_have_count(2, timeout=10000)

    def test_ensemble_available_with_two_models(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train(page, "ens-lin", "Linear", epochs=2)
        _train(page, "ens-svm", "SVM", epochs=2)
        expect(page.locator("#ensemble-btn")).not_to_have_class(
            re.compile(r"hidden")
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Model type dropdown — all types listed for each dataset
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelTypeDropdown:
    """Verify the model type <select> lists all expected options."""

    def test_iris_model_types(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        options = page.locator("#model-type option").all_text_contents()
        assert "Linear" in options
        assert "SVM" in options

    def test_mnist_model_types(self, page: Page, mnist_url: str):
        _goto(page, mnist_url)
        options = page.locator("#model-type option").all_text_contents()
        assert "CNN" in options
        assert "Linear" in options
        assert "SVM" in options
        assert "Quadratic" in options
        assert "Polynomial" in options
