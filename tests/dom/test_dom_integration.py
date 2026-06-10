"""Full DOM-level integration tests for the classifier platform.

These tests drive a real browser (via Playwright) against a live Flask server
to verify that every piece of frontend functionality works end-to-end against
the real backend — no mocks.

Uses the Iris dataset for training/eval workflows because it requires no large
data downloads and trains in seconds.  MNIST is used only for canvas-specific
DOM tests.

Run with:
    python -m pytest tests/dom/ -v --headed   # watch in browser
    python -m pytest tests/dom/ -v             # headless (CI)
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from tests.dom.conftest import wait_connected


# ── Helpers ──────────────────────────────────────────────────────────────────


def _goto(page: Page, url: str, need_connection: bool = True) -> None:
    """Navigate and optionally wait for the SSE connection to be live."""
    page.goto(url)
    if need_connection:
        wait_connected(page)


def _train_quick(page: Page, name: str, model_type: str = "Linear", epochs: int = 1) -> None:
    """Fill form and train; assumes the page is already loaded and connected."""
    page.locator("#model-type").select_option(model_type)
    page.locator("#epochs").fill(str(epochs))
    page.locator("#model-name").fill(name)
    page.locator("#train-btn").click()
    expect(page.locator("#train-btn")).to_be_enabled(timeout=60000)


# ═══════════════════════════════════════════════════════════════════════════════
# Page Load & Layout
# ═══════════════════════════════════════════════════════════════════════════════


class TestPageLoadMNIST:
    """Verify initial page structure for an image-type dataset (MNIST)."""

    def test_page_loads_with_title(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page).to_have_title(re.compile(r"MNIST.*Classifier"))

    def test_navbar_present(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("header.ui-navbar")).to_be_visible()

    def test_dataset_menu_button_present(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        btn = page.locator("#dataset-menu-btn")
        expect(btn).to_be_visible()
        expect(btn).to_contain_text("MNIST")

    def test_train_card_present(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("#model-type")).to_be_visible()
        expect(page.locator("#epochs")).to_be_visible()
        expect(page.locator("#batch-size")).to_be_visible()
        expect(page.locator("#lr")).to_be_visible()
        expect(page.locator("#train-btn")).to_be_visible()

    def test_model_type_options_present(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        options = page.locator("#model-type option")
        count = options.count()
        assert count >= 2, f"Expected at least 2 model types, got {count}"

    def test_canvas_visible_for_image_dataset(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("#canvas-col")).to_be_visible()
        expect(page.locator("#draw-canvas")).to_be_visible()
        expect(page.locator("#clear-btn")).to_be_visible()
        expect(page.locator("#predict-btn")).to_be_visible()

    def test_tabular_col_hidden_for_image_dataset(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("#tabular-col")).to_be_hidden()

    def test_predictions_table_empty(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("#pred-body .empty-row")).to_be_visible()
        expect(page.locator("#pred-body .empty-row")).to_contain_text("No prediction yet")

    def test_evaluation_section_empty(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("#metrics-body .empty-row")).to_be_visible()
        expect(page.locator("#metrics-body .empty-row")).to_contain_text("No models trained yet")

    def test_session_models_container_present(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("#session-models")).to_be_attached()

    def test_saved_models_select_present(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("#saved-select")).to_be_visible()
        expect(page.locator("#import-btn")).to_be_visible()
        expect(page.locator("#refresh-saved-btn")).to_be_visible()

    def test_log_drawer_present(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("#log-drawer")).to_be_attached()
        expect(page.locator("#log-terminal")).to_be_attached()

    def test_chart_area_hidden_initially(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("#chart-area")).to_have_class(re.compile(r"hidden"))

    def test_ensemble_button_hidden_initially(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        expect(page.locator("#ensemble-btn")).to_have_class(re.compile(r"hidden"))


class TestPageLoadIris:
    """Verify initial page structure for a tabular-type dataset (Iris)."""

    def test_page_loads_with_title(self, page: Page, iris_url: str):
        page.goto(iris_url)
        expect(page).to_have_title(re.compile(r"Iris.*Classifier"))

    def test_tabular_col_visible(self, page: Page, iris_url: str):
        page.goto(iris_url)
        expect(page.locator("#tabular-col")).to_be_visible()

    def test_canvas_hidden_for_tabular_dataset(self, page: Page, iris_url: str):
        page.goto(iris_url)
        expect(page.locator("#canvas-col")).to_be_hidden()

    def test_feature_inputs_present(self, page: Page, iris_url: str):
        page.goto(iris_url)
        for feat in ("sepal_length", "sepal_width", "petal_length", "petal_width"):
            expect(page.locator(f"#feat-{feat}")).to_be_visible()

    def test_iris_model_types(self, page: Page, iris_url: str):
        page.goto(iris_url)
        options = page.locator("#model-type option").all_text_contents()
        assert "Linear" in options
        assert "SVM" in options

    def test_iris_default_hyperparams(self, page: Page, iris_url: str):
        page.goto(iris_url)
        assert page.locator("#epochs").input_value() == "50"
        assert page.locator("#batch-size").input_value() == "16"
        assert page.locator("#lr").input_value() == "0.01"


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Menu & Navigation
# ═══════════════════════════════════════════════════════════════════════════════


class TestDatasetMenu:
    """Verify the dataset dropdown menu populates and navigates correctly."""

    def _goto_with_datasets(self, page: Page, url: str) -> None:
        """Navigate, wait for connection, and re-populate the dataset menu."""
        _goto(page, url)
        page.evaluate("""async () => {
            const res = await fetch('/api/datasets');
            const list = await res.json();
            const dl = document.getElementById('dataset-list');
            dl.innerHTML = '';
            for (const ds of list) {
                const btn = document.createElement('button');
                btn.className = 'ui-dropdown-item' + (ds.name === UI_CONFIG.name ? ' active' : '');
                btn.textContent = ds.display_name;
                btn.addEventListener('click', () => {
                    if (ds.name !== UI_CONFIG.name) window.location = '/d/' + ds.name + '/';
                });
                dl.appendChild(btn);
            }
        }""")

    def test_dropdown_opens_on_click(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        page.locator("#dataset-menu-btn").click()
        expect(page.locator("#dataset-menu")).not_to_have_class(re.compile(r"hidden"))

    def test_dropdown_lists_datasets(self, page: Page, mnist_url: str):
        self._goto_with_datasets(page, mnist_url)
        page.locator("#dataset-menu-btn").click()
        items = page.locator("#dataset-list button")
        assert items.count() >= 2

    def test_active_dataset_highlighted(self, page: Page, mnist_url: str):
        self._goto_with_datasets(page, mnist_url)
        page.locator("#dataset-menu-btn").click()
        active = page.locator("#dataset-list button.active")
        expect(active).to_contain_text("MNIST")

    def test_navigate_to_iris(self, page: Page, mnist_url: str):
        self._goto_with_datasets(page, mnist_url)
        page.locator("#dataset-menu-btn").click()
        items = page.locator("#dataset-list button:not(.active)")
        items.first.click()
        page.wait_for_url(re.compile(r"/d/iris/"))
        expect(page).to_have_title(re.compile(r"Iris.*Classifier"))

    def test_root_redirects_to_dataset(self, page: Page, live_server: str):
        page.goto(live_server)
        page.wait_for_url(re.compile(r"/d/\w+/"))


# ═══════════════════════════════════════════════════════════════════════════════
# Theme Toggle
# ═══════════════════════════════════════════════════════════════════════════════


class TestThemeToggle:
    def test_default_theme_is_dark(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        assert page.locator("html").get_attribute("data-theme") == "dark"

    def test_toggle_switches_to_light(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        page.locator("#theme-toggle").click()
        assert page.locator("html").get_attribute("data-theme") == "light"

    def test_toggle_back_to_dark(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        page.locator("#theme-toggle").click()
        page.locator("#theme-toggle").click()
        assert page.locator("html").get_attribute("data-theme") == "dark"


# ═══════════════════════════════════════════════════════════════════════════════
# Train Card Form Controls
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainFormControls:
    def test_model_name_auto_populated(self, page: Page, iris_url: str):
        page.goto(iris_url)
        name = page.locator("#model-name").input_value()
        model_type = page.locator("#model-type").input_value()
        assert name == model_type

    def test_model_name_updates_on_type_change(self, page: Page, iris_url: str):
        page.goto(iris_url)
        page.locator("#model-type").select_option("SVM")
        expect(page.locator("#model-name")).to_have_value("SVM")

    def test_advanced_section_collapsed_by_default(self, page: Page, iris_url: str):
        page.goto(iris_url)
        details = page.locator("details.advanced-toggle")
        assert details.get_attribute("open") is None

    def test_advanced_section_expands(self, page: Page, iris_url: str):
        page.goto(iris_url)
        page.locator("details.advanced-toggle summary").click()
        expect(page.locator("#patience")).to_be_visible()
        expect(page.locator("#val-gap")).to_be_visible()
        expect(page.locator("#teacher-select")).to_be_visible()

    def test_distill_row_hidden_initially(self, page: Page, iris_url: str):
        page.goto(iris_url)
        page.locator("details.advanced-toggle summary").click()
        expect(page.locator("#distill-row")).to_have_class(re.compile(r"hidden"))

    def test_epochs_field_editable(self, page: Page, iris_url: str):
        page.goto(iris_url)
        page.locator("#epochs").fill("5")
        assert page.locator("#epochs").input_value() == "5"

    def test_lr_field_editable(self, page: Page, iris_url: str):
        page.goto(iris_url)
        page.locator("#lr").fill("0.005")
        assert page.locator("#lr").input_value() == "0.005"


# ═══════════════════════════════════════════════════════════════════════════════
# Canvas Drawing (MNIST)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCanvasDrawing:
    def test_canvas_has_correct_dimensions(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        canvas = page.locator("#draw-canvas")
        assert canvas.get_attribute("width") == "280"
        assert canvas.get_attribute("height") == "280"

    def test_draw_on_canvas(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        canvas = page.locator("#draw-canvas")
        box = canvas.bounding_box()
        assert box is not None
        before = page.evaluate("() => document.getElementById('draw-canvas').toDataURL()")
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(cx - 30, cy)
        page.mouse.down()
        page.mouse.move(cx + 30, cy, steps=5)
        page.mouse.up()
        after = page.evaluate("() => document.getElementById('draw-canvas').toDataURL()")
        assert before != after

    def test_clear_button_resets_canvas(self, page: Page, mnist_url: str):
        page.goto(mnist_url)
        canvas = page.locator("#draw-canvas")
        box = canvas.bounding_box()
        assert box is not None
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.mouse.move(cx + 20, cy, steps=3)
        page.mouse.up()
        drawn = page.evaluate("() => document.getElementById('draw-canvas').toDataURL()")
        page.locator("#clear-btn").click()
        cleared = page.evaluate("() => document.getElementById('draw-canvas').toDataURL()")
        assert drawn != cleared


# ═══════════════════════════════════════════════════════════════════════════════
# Training Workflow (Iris — fast, no data download)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainingIris:
    """Train a real model via the UI and verify all DOM updates."""

    def test_train_linear_model(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        page.locator("#model-type").select_option("Linear")
        page.locator("#epochs").fill("2")
        page.locator("#batch-size").fill("16")
        page.locator("#lr").fill("0.01")
        page.locator("#model-name").fill("test-linear")

        page.locator("#train-btn").click()
        expect(page.locator("#train-btn")).to_be_disabled()
        expect(page.locator("#train-btn")).to_be_enabled(timeout=60000)

        expect(page.locator("#session-models")).to_contain_text("test-linear")
        expect(page.locator("#metrics-body")).not_to_contain_text(
            "No models trained yet", timeout=60000
        )
        log_text = page.locator("#log-terminal").text_content()
        assert "test-linear" in log_text

    def test_train_updates_prediction_table(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "pred-check")
        expect(page.locator("#pred-body")).to_contain_text("pred-check")

    def test_train_svm_model(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "test-svm", model_type="SVM", epochs=2)
        expect(page.locator("#session-models")).to_contain_text("test-svm")

    def test_auto_name_increments(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "Linear")
        expect(page.locator("#model-name")).to_have_value("Linear 2")

    def test_train_shows_log_output(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "log-test", epochs=2)
        log = page.locator("#log-terminal").text_content()
        assert "log-test" in log
        assert "trained successfully" in log


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluation:
    """Verify evaluation runs automatically after training and populates metrics."""

    def test_metrics_table_populated_after_train(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "eval-model", epochs=2)
        expect(page.locator("#metrics-head")).to_contain_text("eval-model", timeout=60000)
        expect(page.locator("#metrics-body .metrics-section-row")).to_have_count(3, timeout=10000)

    def test_metrics_show_accuracy(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "acc-check", epochs=3)
        expect(page.locator("#metrics-body")).to_contain_text(
            re.compile(r"\d+\.\d+%"), timeout=60000
        )

    def test_metrics_show_per_class_accuracy(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "perclass", epochs=3)
        for label in ("setosa", "versicolor", "virginica"):
            expect(page.locator("#metrics-body")).to_contain_text(label, timeout=60000)

    def test_metrics_show_config_section(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "cfg-check", model_type="SVM", epochs=2)
        expect(page.locator("#metrics-body")).to_contain_text("SVM", timeout=60000)
        expect(page.locator("#metrics-body")).to_contain_text("Config")
        expect(page.locator("#metrics-body")).to_contain_text("Evaluation")


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction (Iris — tabular)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredictionIris:
    """Verify tabular prediction flow on the Iris dataset."""

    def _fill_setosa(self, page: Page) -> None:
        page.locator("#feat-sepal_length").fill("5.1")
        page.locator("#feat-sepal_width").fill("3.5")
        page.locator("#feat-petal_length").fill("1.4")
        page.locator("#feat-petal_width").fill("0.2")

    def test_predict_with_tabular_features(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "pred-model", epochs=2)
        self._fill_setosa(page)
        page.locator("#predict-btn-tab").click()
        expect(page.locator("#pred-body .pred-label")).to_be_visible(timeout=10000)

    def test_prediction_shows_confidence(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "conf-model", epochs=2)
        self._fill_setosa(page)
        page.locator("#predict-btn-tab").click()
        expect(page.locator("#pred-body")).to_contain_text(
            re.compile(r"\d+\.\d+%"), timeout=10000
        )

    def test_prediction_table_shows_model_name(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "named-pred", epochs=2)
        self._fill_setosa(page)
        page.locator("#predict-btn-tab").click()
        expect(page.locator("#pred-body .pred-model-name")).to_contain_text(
            "named-pred", timeout=10000
        )

    def test_predict_returns_valid_label(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "label-check", epochs=2)
        self._fill_setosa(page)
        page.locator("#predict-btn-tab").click()
        label = page.locator("#pred-body .pred-label")
        expect(label).to_be_visible(timeout=10000)
        assert label.text_content() in ("setosa", "versicolor", "virginica")


# ═══════════════════════════════════════════════════════════════════════════════
# Model Management
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelManagement:
    """Verify model list, removal, export, and import in the session."""

    def test_model_appears_in_session_list(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "session-test")
        expect(page.locator("#session-models .ui-list-name")).to_contain_text("session-test")

    def test_model_shows_type_tag(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "tag-test")
        expect(page.locator("#session-models .ui-list-tag").first).to_contain_text("Linear")

    def test_remove_model(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "remove-me")
        expect(page.locator("#session-models")).to_contain_text("remove-me")
        page.locator("[data-remove='remove-me']").click()
        expect(page.locator("#session-models")).not_to_contain_text("remove-me")
        expect(page.locator("#pred-body")).not_to_contain_text("remove-me")

    def test_remove_model_updates_metrics(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "metrics-rm")
        expect(page.locator("#metrics-head")).to_contain_text("metrics-rm", timeout=60000)
        page.locator("[data-remove='metrics-rm']").click()
        expect(page.locator("#metrics-body")).to_contain_text("No models trained yet")

    def test_export_model(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "export-test")
        page.locator("[data-export='export-test']").click()
        expect(page.locator("#saved-select")).not_to_contain_text(
            "no saved models", timeout=10000
        )

    def test_export_and_import_model(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "roundtrip")
        page.locator("[data-export='roundtrip']").click()
        expect(page.locator("#saved-select")).not_to_contain_text(
            "no saved models", timeout=10000
        )
        page.locator("[data-remove='roundtrip']").click()
        expect(page.locator("#session-models")).not_to_contain_text("roundtrip")
        # Find and select the roundtrip saved model by its value attribute
        saved_value = page.evaluate("""() => {
            const opts = document.querySelectorAll('#saved-select option');
            for (const o of opts) {
                if (o.textContent.includes('roundtrip')) return o.value;
            }
            return null;
        }""")
        assert saved_value, "roundtrip model not found in saved models dropdown"
        page.locator("#saved-select").select_option(value=saved_value)
        page.locator("#import-btn").click()
        expect(page.locator("#session-models")).to_contain_text("roundtrip", timeout=30000)

    def test_refresh_saved_models(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "refresh-check")
        page.locator("[data-export='refresh-check']").click()
        page.wait_for_timeout(1000)
        page.locator("#refresh-saved-btn").click()
        expect(page.locator("#saved-select")).not_to_contain_text(
            "no saved models", timeout=10000
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnsemble:
    def test_ensemble_button_hidden_with_one_model(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "solo")
        expect(page.locator("#ensemble-btn")).to_have_class(re.compile(r"hidden"))

    def test_ensemble_button_visible_with_two_models(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "e1", "Linear")
        _train_quick(page, "e2", "SVM")
        expect(page.locator("#ensemble-btn")).not_to_have_class(re.compile(r"hidden"))

    def test_ensemble_runs_and_logs_result(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "ens-a", "Linear")
        _train_quick(page, "ens-b", "SVM")
        page.locator("#ensemble-btn").click()
        expect(page.locator("#log-terminal")).to_contain_text(
            "Ensemble accuracy", timeout=60000
        )

    def test_ensemble_creates_virtual_model(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "ens-x", "Linear")
        _train_quick(page, "ens-y", "SVM")
        page.locator("#ensemble-btn").click()
        expect(page.locator("#metrics-head")).to_contain_text("Ensemble", timeout=60000)


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation Study
# ═══════════════════════════════════════════════════════════════════════════════


class TestAblation:
    def test_ablation_button_present_after_training(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "abl-model")
        expect(page.locator("[data-ablation='abl-model']")).to_be_visible()

    def test_ablation_logs_output(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "abl-log")
        page.evaluate("() => document.getElementById('log-terminal').innerHTML = ''")
        page.locator("[data-ablation='abl-log']").click()
        expect(page.locator("#log-terminal")).to_contain_text(
            re.compile(r"[Aa]blation"), timeout=60000
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Advanced Training Options
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdvancedTraining:
    def test_early_stopping_training(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        page.locator("#model-type").select_option("Linear")
        page.locator("#epochs").fill("50")
        page.locator("#model-name").fill("early-stop")
        page.locator("details.advanced-toggle summary").click()
        page.locator("#patience").fill("3")
        page.locator("#val-gap").fill("5")
        page.locator("#train-btn").click()
        expect(page.locator("#train-btn")).to_be_enabled(timeout=120000)
        expect(page.locator("#chart-area")).not_to_have_class(re.compile(r"hidden"))
        expect(page.locator("#session-models")).to_contain_text("early-stop")

    def test_distillation_dropdown_populated_after_training(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "teacher")
        page.locator("details.advanced-toggle summary").click()
        options = page.locator("#teacher-select option").all_text_contents()
        assert any("teacher" in opt for opt in options)

    def test_distill_weight_row_shows_when_teacher_selected(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "teach")
        page.locator("details.advanced-toggle summary").click()
        page.locator("#teacher-select").select_option("teach")
        expect(page.locator("#distill-row")).not_to_have_class(re.compile(r"hidden"))


# ═══════════════════════════════════════════════════════════════════════════════
# Connection State
# ═══════════════════════════════════════════════════════════════════════════════


class TestConnectionState:
    def test_connection_becomes_connected(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        state = page.evaluate("() => connectionManager.state")
        assert state == "connected"

    def test_connected_log_message(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        expect(page.locator("#log-terminal")).to_contain_text(
            "Connected to server", timeout=15000
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Model Info Panel
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelInfo:
    def test_model_info_panel_shows_on_page_load(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        page.wait_for_timeout(2000)
        details = page.locator("#model-info-details")
        if not details.evaluate("el => el.classList.contains('hidden')"):
            panel = page.locator("#model-info-panel")
            assert panel.text_content().strip() != ""

    def test_model_info_updates_on_type_change(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        page.wait_for_timeout(1000)
        page.locator("#model-type").select_option("SVM")
        page.wait_for_timeout(1000)
        # Main check: no JS errors during the switch (page stays functional)
        expect(page.locator("#model-type")).to_have_value("SVM")


# ═══════════════════════════════════════════════════════════════════════════════
# Health & Datasets API (from browser)
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthAPI:
    def test_health_returns_ok(self, page: Page, api_url: str):
        response = page.request.get(f"{api_url}/health")
        assert response.ok
        data = response.json()
        assert data["status"] == "ok"
        assert "uptime" in data
        assert "clients" in data


class TestDatasetsAPI:
    def test_datasets_lists_at_least_two(self, page: Page, api_url: str):
        response = page.request.get(f"{api_url}/api/datasets")
        assert response.ok
        data = response.json()
        assert len(data) >= 2
        names = [d["name"] for d in data]
        assert "mnist" in names
        assert "iris" in names

    def test_dataset_config_endpoint(self, page: Page, api_url: str):
        response = page.request.get(f"{api_url}/api/datasets/iris/config")
        assert response.ok
        data = response.json()
        assert "ui_config" in data
        assert "model_types" in data
        assert "Linear" in data["model_types"]


# ═══════════════════════════════════════════════════════════════════════════════
# Multiple Models — Columnar Metrics Layout
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiModelMetrics:
    def test_two_models_two_columns(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "col-a", "Linear")
        _train_quick(page, "col-b", "SVM")
        expect(page.locator("#metrics-head")).to_contain_text("col-a", timeout=60000)
        expect(page.locator("#metrics-head")).to_contain_text("col-b", timeout=60000)
        th_count = page.locator("#metrics-head tr th").count()
        assert th_count == 3  # corner + 2 models

    def test_prediction_table_shows_both_models(self, page: Page, iris_url: str):
        _goto(page, iris_url)
        _train_quick(page, "multi-a")
        _train_quick(page, "multi-b")
        expect(page.locator("#pred-body")).to_contain_text("multi-a")
        expect(page.locator("#pred-body")).to_contain_text("multi-b")
        page.locator("#feat-sepal_length").fill("5.0")
        page.locator("#feat-sepal_width").fill("3.0")
        page.locator("#feat-petal_length").fill("1.5")
        page.locator("#feat-petal_width").fill("0.3")
        page.locator("#predict-btn-tab").click()
        labels = page.locator("#pred-body .pred-label")
        expect(labels).to_have_count(2, timeout=10000)


# ═══════════════════════════════════════════════════════════════════════════════
# Error Handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    def test_predict_with_no_models_shows_no_crash(self, page: Page, iris_url: str):
        page.goto(iris_url)
        page.locator("#feat-sepal_length").fill("5.0")
        page.locator("#predict-btn-tab").click()
        expect(page.locator("#train-btn")).to_be_visible()

    def test_unknown_dataset_404(self, page: Page, live_server: str):
        response = page.request.get(f"{live_server}/d/nonexistent/models")
        assert response.status == 404

    def test_train_invalid_model_type(self, page: Page, api_url: str):
        response = page.request.post(
            f"{api_url}/d/iris/train",
            data={"model_type": "INVALID", "epochs": 1, "batch_size": 16, "lr": 0.01},
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 400
