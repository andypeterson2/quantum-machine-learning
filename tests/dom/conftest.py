"""Shared fixtures for DOM-level integration tests.

Spins up a live Flask server in a background thread so Playwright can drive
a real browser against it.  Each test session gets a fresh server; individual
tests share the same server for speed but get a fresh browser context for
isolation.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page

from classifiers.server import create_app


def _free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_server(tmp_path_factory):
    """Start a real Flask server in a daemon thread, return its base URL."""
    models_dir = tmp_path_factory.mktemp("models")
    app = create_app(models_dir=models_dir)
    app.config["TESTING"] = True
    port = _free_port()
    host = "127.0.0.1"

    server_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, use_reloader=False, threaded=True),
        daemon=True,
    )
    server_thread.start()

    # Wait for the server to be ready
    base_url = f"http://{host}:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError("Live server failed to start within 10 seconds")

    return base_url


@pytest.fixture()
def mnist_url(live_server) -> str:
    """URL for the MNIST dataset page."""
    return f"{live_server}/d/mnist/"


@pytest.fixture()
def iris_url(live_server) -> str:
    """URL for the Iris dataset page."""
    return f"{live_server}/d/iris/"


@pytest.fixture()
def api_url(live_server) -> str:
    """Base URL for API endpoints."""
    return live_server


def _inject_icons_script(route):
    """Route handler that injects ``icons.js`` before ``ui-kit.js`` in HTML responses.

    The standalone Flask server's template doesn't load ``icons.js`` (the
    parent website normally provides it).  Without it, the app-level
    ``ICONS`` const is ``undefined`` and the train button handler silently
    crashes.  This handler intercepts the dataset page HTML and injects
    the missing ``<script>`` tag so the app works identically to production.
    """
    response = route.fetch()
    body = response.text()
    if "ui-kit.js" in body:
        body = body.replace(
            '<script src="/ui-kit/ui-kit.js"></script>',
            '<script src="/ui-kit/icons.js"></script>\n'
            '<script src="/ui-kit/ui-kit.js"></script>',
        )
    route.fulfill(response=response, body=body)


@pytest.fixture(autouse=True)
def _ensure_icons(page: Page):
    """Intercept HTML pages to inject the missing icons.js script."""
    page.route("**/d/*/", _inject_icons_script)
    yield


def wait_connected(page: Page, timeout: int = 15000) -> None:
    """Wait for the connectionManager to reach 'connected' state.

    Call this after ``page.goto(...)`` in any test that needs to interact
    with the backend (training, prediction, API fetches) because the app's
    ``apiFetch`` wrapper rejects requests when disconnected.
    """
    page.wait_for_function(
        "() => typeof connectionManager !== 'undefined' && connectionManager.state === 'connected'",
        timeout=timeout,
    )
