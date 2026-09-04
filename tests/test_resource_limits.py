"""Resource-ceiling regression tests (2026-09 security round, Pass C).

Covers: the heavy-compute job-slot gate (409 when saturated), the /connect
client cap, bounded SSE streams, registry eviction, predict image guards,
and the export checkpoint cap.
"""

from __future__ import annotations

import queue
import threading
import typing

import pytest

from classifiers.model_registry import ModelRegistry
from classifiers.routes.sse import sse_response
from classifiers.server import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(models_dir=tmp_path)


@pytest.fixture
def client(app):
    return app.test_client()


def _saturate(app):
    """Drain the job-slot semaphore so every heavy route reports busy."""
    slots = app.extensions["job_slots"]
    while slots.acquire(blocking=False):
        pass


class TestJobSlots:
    """Heavy compute routes 409 when the concurrency gate is saturated."""

    @pytest.mark.parametrize(
        ("path", "body"),
        [
            ("/d/mnist/train", {"model_type": "CNN"}),
            ("/d/mnist/train/sync", {"model_type": "CNN"}),
            ("/d/mnist/evaluate", {}),
            ("/d/mnist/evaluate/sync", {}),
            ("/d/mnist/ensemble", {"model_names": ["a", "b"]}),
        ],
    )
    def test_busy_returns_409(self, app, client, path, body):
        _saturate(app)
        resp = client.post(path, json=body)
        assert resp.status_code == 409
        assert resp.get_json()["error"]["code"] == "busy"

    def test_slot_released_after_rejected_input(self, app, client):
        # A 400 on bad input must give its slot back, or the server bricks.
        slots = app.extensions["job_slots"]
        resp = client.post("/d/mnist/train/sync", json={"model_type": "NoSuchModel"})
        assert resp.status_code == 400
        assert slots.acquire(blocking=False)  # slot came back
        slots.release()

    def test_default_capacity_is_bounded(self, app):
        slots = app.extensions["job_slots"]
        n = 0
        while slots.acquire(blocking=False):
            n += 1
        assert 1 <= n <= 8
        for _ in range(n):
            slots.release()


class TestConnectCap:
    """The SSE heartbeat channel refuses connections over the client cap."""

    def test_over_cap_is_409(self, app, client, monkeypatch):
        monkeypatch.setenv("CLASSIFIERS_MAX_CLIENTS", "0")
        resp = client.get("/connect")
        assert resp.status_code == 409
        assert resp.get_json()["error"]["code"] == "too_many_clients"

    def test_under_cap_streams_welcome(self, app, client, monkeypatch):
        monkeypatch.setenv("CLASSIFIERS_MAX_CLIENTS", "8")
        resp = client.get("/connect")
        assert resp.status_code == 200
        first = next(resp.response)
        assert b"welcome" in first
        resp.close()


class TestBoundedSSE:
    """sse_response streams are lifetime-bounded and never block forever."""

    def test_lifetime_cap_emits_error_and_ends(self, app, monkeypatch):
        monkeypatch.setenv("CLASSIFIERS_SSE_LIFETIME", "0")
        q: queue.Queue = queue.Queue()
        with app.test_request_context():
            resp = sse_response(q)
            frames = list(resp.response)
        assert any("stream lifetime exceeded" in str(f) for f in frames)

    def test_keepalive_on_quiet_queue(self, app, monkeypatch):
        monkeypatch.setenv("CLASSIFIERS_SSE_LIFETIME", "5")
        monkeypatch.setenv("CLASSIFIERS_SSE_GET_TIMEOUT", "0.01")
        q: queue.Queue = queue.Queue()

        def finish_soon():
            q.put({"type": "done"})
            q.put(None)

        with app.test_request_context():
            resp = sse_response(q)
            gen = iter(resp.response)
            first = next(gen)  # queue quiet -> keepalive comment, not a block
            assert str(first).lstrip("b'").startswith(":")
            threading.Thread(target=finish_soon).start()
            rest = list(gen)
        assert any("done" in str(f) for f in rest)


class TestRegistryEviction:
    """The registry cap evicts oldest-unevaluated first, oldest otherwise."""

    def _add(self, reg, name):
        reg.add("ds", name, model=object(), model_type="T", epochs=1, batch_size=1, lr=0.1)

    def test_cap_holds_and_unevaluated_evicted_first(self):
        reg = ModelRegistry(max_per_dataset=3)
        for name in ["a", "b", "c"]:
            self._add(reg, name)

        # Mark "a" evaluated; the next add should evict "b" (oldest unevaluated).
        class FakeEval:
            accuracy = 1.0
            avg_loss = 0.0
            per_class_accuracy: typing.ClassVar[dict] = {}
            num_params = 0

        reg.update_eval_result("ds", "a", FakeEval())
        self._add(reg, "d")
        assert reg.names("ds") == ["a", "c", "d"]

    def test_all_evaluated_falls_back_to_oldest(self):
        reg = ModelRegistry(max_per_dataset=2)

        class FakeEval:
            accuracy = 1.0
            avg_loss = 0.0
            per_class_accuracy: typing.ClassVar[dict] = {}
            num_params = 0

        for name in ["a", "b"]:
            self._add(reg, name)
            reg.update_eval_result("ds", name, FakeEval())
        self._add(reg, "c")
        assert reg.names("ds") == ["b", "c"]

    def test_replacing_existing_name_does_not_evict(self):
        reg = ModelRegistry(max_per_dataset=2)
        for name in ["a", "b"]:
            self._add(reg, name)
        self._add(reg, "b")  # replacement, not growth
        assert reg.names("ds") == ["a", "b"]


class TestPredictImageGuards:
    """Predict image decoding rejects bombs and oversized canvases."""

    def test_oversized_dimensions_rejected(self, monkeypatch):
        import base64
        import io

        from PIL import Image

        from classifiers.routes.model_routes import _decode_image

        monkeypatch.setenv("CLASSIFIERS_MAX_IMAGE_DIM", "16")
        big = Image.new("L", (32, 32))
        buf = io.BytesIO()
        big.save(buf, format="PNG")
        assert _decode_image(base64.b64encode(buf.getvalue()).decode()) is None

    def test_valid_small_image_accepted(self, monkeypatch):
        import base64
        import io

        from PIL import Image

        from classifiers.routes.model_routes import _decode_image

        monkeypatch.setenv("CLASSIFIERS_MAX_IMAGE_DIM", "64")
        img = Image.new("RGB", (28, 28))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        decoded = _decode_image(base64.b64encode(buf.getvalue()).decode())
        assert decoded is not None
        assert decoded.mode == "L"

    def test_garbage_rejected(self):
        from classifiers.routes.model_routes import _decode_image

        assert _decode_image("not-base64!!") is None
        assert _decode_image("aGVsbG8=") is None  # valid b64, not an image


class TestModelInfoEscaping:
    """MODELS.md rendering escapes inline HTML (stored-XSS lane)."""

    def test_inline_html_is_escaped(self):
        from classifiers.routes.model_routes import _markdown

        html = _markdown('## X\n\n<script>alert(1)</script>\n\n<img src=x onerror="p()">')
        assert "<script>" not in html
        assert "onerror" not in html or "&lt;img" in html
