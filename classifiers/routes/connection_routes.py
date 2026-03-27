"""Connection lifecycle routes — heartbeat SSE channel, pong, and disconnect.

The ``/connect`` endpoint opens a persistent SSE stream that periodically
sends ``ping`` events.  The client responds to each ping via ``POST /pong``
so the server can track liveness.  ``POST /disconnect`` allows a graceful
teardown (also used by ``navigator.sendBeacon`` on page unload).
"""

from __future__ import annotations

import json
import time

from flask import Blueprint, Response, current_app, request, stream_with_context

from .sse import format_event

bp = Blueprint("connection", __name__)

#: Seconds between server-sent pings.
HEARTBEAT_INTERVAL = 25


@bp.get("/connect")
def connect() -> Response:
    """Open an SSE heartbeat channel for a new client.

    The stream emits a ``welcome`` event (containing the assigned
    ``client_id`` and ``heartbeat_interval``) followed by periodic ``ping``
    events.  The generator cleans up the client on exit.
    """
    tracker = current_app.extensions["connections"]
    client_id = tracker.register()

    @stream_with_context
    def _generate():
        try:
            yield format_event({
                "type": "welcome",
                "client_id": client_id,
                "heartbeat_interval": HEARTBEAT_INTERVAL,
            })
            while True:
                time.sleep(HEARTBEAT_INTERVAL)
                yield format_event({
                    "type": "ping",
                    "ts": time.time(),
                })
        except GeneratorExit:
            pass
        finally:
            tracker.unregister(client_id)

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@bp.post("/pong")
def pong() -> tuple[str, int]:
    """Acknowledge a server ping — updates the client's last-seen timestamp."""
    data = request.get_json(silent=True) or {}
    client_id = data.get("client_id", "")
    tracker = current_app.extensions["connections"]
    if not client_id or not tracker.heartbeat(client_id):
        return "", 404
    return "", 204


@bp.post("/disconnect")
def disconnect() -> tuple[str, int]:
    """Graceful client disconnect — removes the client from the tracker."""
    body = request.get_data(as_text=True)
    try:
        data = json.loads(body) if body else {}
    except (json.JSONDecodeError, ValueError):
        data = {}
    client_id = data.get("client_id", "")
    tracker = current_app.extensions["connections"]
    if client_id:
        tracker.unregister(client_id)
    return "", 204
