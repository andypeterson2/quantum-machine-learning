"""CORS header tests for the classifier Flask API."""
from __future__ import annotations
import pytest
from classifiers.server import create_app


@pytest.fixture()
def client():
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
