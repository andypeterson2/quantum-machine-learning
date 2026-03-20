"""Advanced route tests — export, disk operations, ablation, Iris routes."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from classifiers.server import create_app
from classifiers.datasets.mnist.models import MNISTNet, LinearNet
from classifiers.datasets.iris.models import IrisLinear
from tests.conftest import make_fake_test_loader


DS = "mnist"


def _blank_png_b64(width: int = 280, height: int = 280) -> str:
    img = Image.new("L", (width, height), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _parse_sse(raw: bytes) -> list[dict]:
    events = []
    for chunk in raw.decode().split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data:"):
            continue
        events.append(json.loads(chunk[len("data:"):].strip()))
    return events


@pytest.fixture()
def app(tmp_path):
    application = create_app(models_dir=tmp_path / "models")
    application.config["TESTING"] = True
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def registry(app):
    return app.extensions["registry"]


# ── Export and disk operations ────────────────────────────────────────────────


class TestExportAndLoad:
    def test_export_model(self, client, registry):
        registry.add(DS, "exportable", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        res = client.post(f"/d/{DS}/models/exportable/export")
        assert res.status_code == 200
        data = res.get_json()
        assert data["ok"] is True
        assert "filename" in data

    def test_export_nonexistent_returns_404(self, client):
        res = client.post(f"/d/{DS}/models/ghost/export")
        assert res.status_code == 404

    def test_list_disk_models_empty(self, client):
        res = client.get(f"/d/{DS}/models/disk")
        assert res.status_code == 200
        assert res.get_json() == []

    def test_export_then_list_disk(self, client, registry):
        registry.add(DS, "saved", MNISTNet(), model_type="CNN",
                      epochs=2, batch_size=64, lr=0.01)
        client.post(f"/d/{DS}/models/saved/export")
        res = client.get(f"/d/{DS}/models/disk")
        files = res.get_json()
        assert len(files) == 1
        assert files[0]["name"] == "saved"

    def test_export_then_load(self, client, registry):
        registry.add(DS, "loadme", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        export_res = client.post(f"/d/{DS}/models/loadme/export")
        filename = export_res.get_json()["filename"]

        # Remove from registry
        client.delete(f"/d/{DS}/models/loadme")
        assert registry.get(DS, "loadme") is None

        # Load from disk
        res = client.post(f"/d/{DS}/models/disk/{filename}/load")
        assert res.status_code == 200
        data = res.get_json()
        assert data["ok"] is True
        assert data["model_type"] == "CNN"


# ── Ablation route ────────────────────────────────────────────────────────────


class TestAblationRoute:
    def test_ablation_unknown_model_returns_404(self, client):
        res = client.post(f"/d/{DS}/ablation", json={"model_name": "ghost"})
        assert res.status_code == 404

    def test_ablation_streams_sse(self, client, registry):
        from unittest.mock import patch
        fake_loader = make_fake_test_loader(batch_size=10, n_batches=1)
        registry.add(DS, "abl", LinearNet(), model_type="Linear",
                      epochs=1, batch_size=32, lr=0.001)
        with patch(
            "classifiers.datasets.mnist.plugin.MNISTPlugin.get_test_loader",
            return_value=fake_loader,
        ):
            res = client.post(f"/d/{DS}/ablation", json={"model_name": "abl"})
        assert res.status_code == 200
        events = _parse_sse(res.data)
        types = {e["type"] for e in events}
        assert "done" in types


# ── Iris routes ───────────────────────────────────────────────────────────────


class TestIrisRoutes:
    def test_iris_models_empty(self, client):
        res = client.get("/d/iris/models")
        assert res.status_code == 200
        assert res.get_json() == {}

    def test_iris_predict_with_features(self, client, registry):
        registry.add("iris", "lin", IrisLinear(), model_type="Linear",
                      epochs=1, batch_size=16, lr=0.01)
        res = client.post("/d/iris/predict", json={
            "features": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        })
        assert res.status_code == 200
        data = res.get_json()
        assert "results" in data
        assert "lin" in data["results"]
        result = data["results"]["lin"]
        assert result["prediction"] in ["setosa", "versicolor", "virginica"]
        assert len(result["probs"]) == 3

    def test_iris_delete_model(self, client, registry):
        registry.add("iris", "gone", IrisLinear(), model_type="Linear",
                      epochs=1, batch_size=16, lr=0.01)
        res = client.delete("/d/iris/models/gone")
        assert res.status_code == 200
        assert registry.get("iris", "gone") is None


# ── Predict edge cases ────────────────────────────────────────────────────────


class TestPredictEdgeCases:
    def test_predict_missing_image_field_raises(self, client, registry):
        """Missing image field should raise (empty base64 is not valid)."""
        from PIL import UnidentifiedImageError
        registry.add(DS, "m", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        with pytest.raises(UnidentifiedImageError):
            client.post(f"/d/{DS}/predict", json={})

    def test_predict_with_small_image(self, client, registry):
        registry.add(DS, "m", MNISTNet(), model_type="CNN",
                      epochs=1, batch_size=32, lr=0.001)
        res = client.post(f"/d/{DS}/predict",
                          json={"image": _blank_png_b64(14, 14)})
        assert res.status_code == 200
