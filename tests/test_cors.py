"""CORS header tests for the classifier Flask API."""
from __future__ import annotations

import pytest

from classifiers.server import create_app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CLASSIFIERS_CORS_ORIGINS", "https://andypeterson2.github.io")
    app = create_app()
    app.config["TESTING"] = True
    yield app.test_client()


class TestCORS:
    def test_cors_on_datasets(self, client):
        res = client.get("/api/datasets",
                         headers={"Origin": "https://andypeterson2.github.io"})
        assert res.headers.get("Access-Control-Allow-Origin") is not None

    def test_options_preflight(self, client):
        res = client.options("/api/datasets",
                             headers={"Origin": "https://andypeterson2.github.io",
                                      "Access-Control-Request-Method": "GET"})
        assert res.status_code == 200
        assert "Access-Control-Allow-Origin" in res.headers

    def test_cors_on_model_routes(self, client):
        res = client.get("/d/mnist/models",
                         headers={"Origin": "https://andypeterson2.github.io"})
        assert res.headers.get("Access-Control-Allow-Origin") is not None


class TestCORSDefaultAllowlist:
    """Regression for the unanchored-regex hole: flask-cors treats a '*' entry
    as a start-anchored-only regex, so the old ``http://localhost:*`` allowed
    the registrable origin ``http://localhostevil.com``. The default list now
    uses an explicitly anchored regex."""

    @pytest.fixture()
    def default_client(self, monkeypatch):
        monkeypatch.delenv("CLASSIFIERS_CORS_ORIGINS", raising=False)
        app = create_app()
        app.config["TESTING"] = True
        yield app.test_client()

    @pytest.mark.parametrize(
        "origin", ["http://localhost:4321", "http://localhost", "https://andypeterson.dev"]
    )
    def test_intended_origins_allowed(self, default_client, origin):
        res = default_client.get("/api/datasets", headers={"Origin": origin})
        assert res.headers.get("Access-Control-Allow-Origin") == origin

    @pytest.mark.parametrize(
        "origin",
        [
            "http://localhostevil.com",
            "http://localhost.evil.com",
            "https://localhostevil.com:4321",
            "https://evil.andypeterson.dev",
        ],
    )
    def test_lookalike_origins_rejected(self, default_client, origin):
        res = default_client.get("/api/datasets", headers={"Origin": origin})
        assert res.headers.get("Access-Control-Allow-Origin") is None
