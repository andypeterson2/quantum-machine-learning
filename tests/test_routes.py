"""Flask route tests — uses app.test_client() for all routes."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from classifiers.server import create_app
from classifiers.datasets.mnist.models import MNISTNet, LinearNet
from tests.conftest import make_fake_train_loader, make_fake_test_loader


# ── Helpers ─────────────────────────────────────────────────────────────────


DS = "mnist"


def _blank_png_b64(width: int = 280, height: int = 280) -> str:
    """Return a base64-encoded blank black PNG."""
    img = Image.new("L", (width, height), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _parse_sse(raw: bytes) -> list[dict]:
    """Parse raw SSE bytes into a list of event dicts."""
    events = []
    for chunk in raw.decode().split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data:"):
            continue
        events.append(json.loads(chunk[len("data:"):].strip()))
    return events


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def app():
    """Create a fresh Flask app with an empty registry for each test."""
    application = create_app()
    application.config["TESTING"] = True
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def registry(app):
    """Return the app's model registry for direct manipulation in tests."""
    return app.extensions["registry"]


# ── GET /d/mnist/ ────────────────────────────────────────────────────────────


class TestAPIDatasets:
    def test_list_datasets_returns_200(self, client):
        res = client.get("/api/datasets")
        assert res.status_code == 200

    def test_list_datasets_returns_json_array(self, client):
        res = client.get("/api/datasets")
        data = res.get_json()
        assert isinstance(data, list)
        assert len(data) >= 2  # at least mnist and iris

    def test_list_datasets_has_required_fields(self, client):
        res = client.get("/api/datasets")
        for entry in res.get_json():
            assert "name" in entry
            assert "display_name" in entry
            assert "input_type" in entry

    def test_dataset_config_returns_200(self, client):
        res = client.get(f"/api/datasets/{DS}/config")
        assert res.status_code == 200
        data = res.get_json()
        assert "ui_config" in data
        assert "model_types" in data

    def test_dataset_config_unknown_returns_404(self, client):
        res = client.get("/api/datasets/nonexistent/config")
        assert res.status_code == 404


# ── GET /d/mnist/models ──────────────────────────────────────────────────────


class TestModelsRoute:
    def test_list_models_empty(self, client):
        res = client.get(f"/d/{DS}/models")
        assert res.status_code == 200
        assert res.get_json() == {}

    def test_list_models_after_add(self, client, registry):
        registry.add(DS, "test_model", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        res = client.get(f"/d/{DS}/models")
        data = res.get_json()
        assert "test_model" in data
        assert data["test_model"]["model_type"] == "CNN"
        assert data["test_model"]["epochs"] == 1
        assert data["test_model"]["eval_result"] is None

    def test_list_models_includes_all_fields(self, client, registry):
        registry.add(DS, "m", MNISTNet(), model_type="CNN",
                      epochs=2, batch_size=64, lr=0.01)
        res = client.get(f"/d/{DS}/models")
        entry = res.get_json()["m"]
        assert set(entry.keys()) >= {"model_type", "epochs", "batch_size", "lr", "eval_result"}


# ── DELETE /d/mnist/models/<name> ────────────────────────────────────────────


class TestDeleteModel:
    def test_delete_existing_model(self, client, registry):
        registry.add(DS, "del_me", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        res = client.delete(f"/d/{DS}/models/del_me")
        assert res.status_code == 200
        assert res.get_json()["ok"] is True
        assert registry.get(DS, "del_me") is None

    def test_delete_removes_from_list(self, client, registry):
        registry.add(DS, "keep", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        registry.add(DS, "gone", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        client.delete(f"/d/{DS}/models/gone")
        names = list(client.get(f"/d/{DS}/models").get_json().keys())
        assert "gone" not in names
        assert "keep" in names

    def test_delete_nonexistent_returns_ok(self, client):
        res = client.delete(f"/d/{DS}/models/does_not_exist")
        assert res.status_code == 200


# ── POST /d/mnist/predict ────────────────────────────────────────────────────


class TestPredictRoute:
    def test_predict_no_models_returns_400(self, client):
        res = client.post(f"/d/{DS}/predict",
                          json={"image": _blank_png_b64()},
                          content_type="application/json")
        assert res.status_code == 400
        assert "error" in res.get_json()

    def test_predict_returns_results(self, client, registry):
        registry.add(DS, "cnn", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        res = client.post(f"/d/{DS}/predict",
                          json={"image": _blank_png_b64()},
                          content_type="application/json")
        assert res.status_code == 200
        data = res.get_json()
        assert "results" in data
        assert "cnn" in data["results"]

    def test_predict_result_fields(self, client, registry):
        registry.add(DS, "cnn", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        res = client.post(f"/d/{DS}/predict",
                          json={"image": _blank_png_b64()},
                          content_type="application/json")
        result = res.get_json()["results"]["cnn"]
        assert "prediction" in result
        assert "confidence" in result
        assert "probs" in result
        # prediction is now a string label ("0"-"9")
        assert result["prediction"] in [str(i) for i in range(10)]
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["probs"]) == 10
        assert abs(sum(result["probs"]) - 1.0) < 1e-4

    def test_predict_multiple_models(self, client, registry):
        registry.add(DS, "cnn", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        registry.add(DS, "lin", LinearNet(), model_type="Linear",
                      epochs=1, batch_size=32, lr=0.001)
        res = client.post(f"/d/{DS}/predict",
                          json={"image": _blank_png_b64()},
                          content_type="application/json")
        results = res.get_json()["results"]
        assert set(results.keys()) == {"cnn", "lin"}


# ── POST /d/mnist/train ──────────────────────────────────────────────────────


class TestTrainRoute:
    def test_train_cnn_sse(self, client, app):
        """Mock the plugin's train loader to avoid downloading real data."""
        from unittest.mock import patch
        fake_loader = make_fake_train_loader(batch_size=16, n_batches=3)
        with patch.object(
            app.extensions["registry"].__class__, "__len__", return_value=0
        ):
            with patch(
                "classifiers.datasets.mnist.plugin.MNISTPlugin.get_train_loader",
                return_value=fake_loader,
            ):
                res = client.post(
                    f"/d/{DS}/train",
                    json={"model_type": "CNN", "epochs": 1,
                          "batch_size": 16, "lr": 0.001, "name": "t1"},
                    content_type="application/json",
                )
        assert res.status_code == 200
        assert res.content_type.startswith("text/event-stream")
        events = _parse_sse(res.data)
        types = [e["type"] for e in events]
        assert "done" in types

    def test_train_unknown_model_type_returns_400(self, client):
        res = client.post(f"/d/{DS}/train",
                          json={"model_type": "UNKNOWN", "epochs": 1,
                                "batch_size": 16, "lr": 0.001},
                          content_type="application/json")
        assert res.status_code == 400
        assert "error" in res.get_json()


# ── POST /d/mnist/evaluate ───────────────────────────────────────────────────


class TestEvaluateRoute:
    def test_evaluate_no_models_returns_400(self, client):
        res = client.post(f"/d/{DS}/evaluate", json={})
        assert res.status_code == 400

    def test_evaluate_streams_sse(self, client, registry):
        from unittest.mock import patch
        fake_loader = make_fake_test_loader(batch_size=10, n_batches=2)
        registry.add(DS, "ev_model", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        with patch(
            "classifiers.datasets.mnist.plugin.MNISTPlugin.get_test_loader",
            return_value=fake_loader,
        ):
            res = client.post(f"/d/{DS}/evaluate", json={})
        assert res.status_code == 200
        assert res.content_type.startswith("text/event-stream")
        events = _parse_sse(res.data)
        types = [e["type"] for e in events]
        assert "done" in types

    def test_evaluate_done_has_results(self, client, registry):
        from unittest.mock import patch
        fake_loader = make_fake_test_loader(batch_size=10, n_batches=2)
        registry.add(DS, "ev2", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        with patch(
            "classifiers.datasets.mnist.plugin.MNISTPlugin.get_test_loader",
            return_value=fake_loader,
        ):
            res = client.post(f"/d/{DS}/evaluate", json={})
        events = _parse_sse(res.data)
        done = next(e for e in events if e["type"] == "done")
        assert "results" in done
        assert "ev2" in done["results"]


# ── POST /d/mnist/ensemble ───────────────────────────────────────────────────


class TestEnsembleRoute:
    def test_ensemble_requires_two_models(self, client):
        res = client.post(f"/d/{DS}/ensemble",
                          json={"model_names": ["one"]})
        assert res.status_code == 400

    def test_ensemble_with_models(self, client, registry):
        from unittest.mock import patch
        fake_loader = make_fake_test_loader(batch_size=10, n_batches=2)
        registry.add(DS, "m1", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        registry.add(DS, "m2", LinearNet(), model_type="Linear",
                      epochs=1, batch_size=32, lr=0.001)
        with patch(
            "classifiers.datasets.mnist.plugin.MNISTPlugin.get_test_loader",
            return_value=fake_loader,
        ):
            res = client.post(f"/d/{DS}/ensemble",
                              json={"model_names": ["m1", "m2"]})
        assert res.status_code == 200
        data = res.get_json()
        assert "accuracy" in data
        assert 0.0 <= data["accuracy"] <= 1.0


# ── Unknown dataset returns 404 ─────────────────────────────────────────────


class TestUnknownDataset:
    def test_unknown_dataset_returns_404(self, client):
        res = client.get("/d/doesnotexist/models")
        assert res.status_code == 404
