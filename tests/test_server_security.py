"""Security-posture regression tests: debug parse fails closed, origin guard."""
from __future__ import annotations

import pytest

from classifiers.server import create_app, debug_enabled


class TestDebugParseFailsClosed:
    """CLASSIFIERS_DEBUG must only enable the Werkzeug debugger (an RCE if the
    host is exposed) for an explicit opt-in value — never for typos.

    This calls the parse the dev server calls. The matrix used to re-implement
    the expression and then grep __main__.py to check the copy still matched.
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("1", True), ("true", True), ("YES", True), (" 1 ", True),
            ("0", False), ("false", False), ("no", False), ("", False),
            ("off", False), ("disabled", False), ("n", False), ("False ", False),
        ],
    )
    def test_parse_matrix(self, value, expected, monkeypatch):
        monkeypatch.setenv("CLASSIFIERS_DEBUG", value)
        assert debug_enabled() is expected

    def test_default_is_off(self, monkeypatch):
        monkeypatch.delenv("CLASSIFIERS_DEBUG", raising=False)
        assert debug_enabled() is False


class TestOriginGuard:
    @pytest.fixture()
    def guarded_client(self, monkeypatch):
        monkeypatch.setenv("ORIGIN_SECRET", "s3cret-value")
        app = create_app()
        app.config["TESTING"] = True
        yield app.test_client()

    def test_health_stays_public(self, guarded_client):
        assert guarded_client.get("/health").status_code == 200

    def test_missing_secret_rejected(self, guarded_client):
        assert guarded_client.get("/api/datasets").status_code == 403

    def test_wrong_secret_rejected(self, guarded_client):
        res = guarded_client.get(
            "/api/datasets", headers={"X-Origin-Secret": "wrong"}
        )
        assert res.status_code == 403

    def test_correct_secret_admitted(self, guarded_client):
        res = guarded_client.get(
            "/api/datasets", headers={"X-Origin-Secret": "s3cret-value"}
        )
        assert res.status_code == 200


class TestSweeperDoesNotPinTheApp:
    """The stale-client sweeper must not keep its app alive.

    One thread is started per create_app, and the suite builds hundreds of
    apps. When the loop captured the tracker directly, every one of those apps
    stayed reachable from a sleeping thread for the life of the process.
    """

    def test_tracker_is_collected_once_the_app_goes(self, tmp_path):
        import gc
        import weakref

        from classifiers.server import create_app

        def build():
            app = create_app(models_dir=tmp_path)
            return weakref.ref(app.extensions["connections"])

        ref = build()
        gc.collect()
        assert ref() is None, "the sweeper thread is still holding the app's tracker"
