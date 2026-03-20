"""Server-Sent Events (SSE) helpers.

Extracting SSE formatting and response construction here keeps the train and
evaluate route handlers focused on *what* to stream rather than *how*
(Single Responsibility Principle), and provides one place to change buffering
headers or event encoding without touching individual routes (Open/Closed
Principle).
"""

from __future__ import annotations

import json
import queue

from flask import Response, stream_with_context


def format_event(obj: dict) -> str:
    """Serialise *obj* as a single SSE ``data:`` frame.

    Args:
        obj: Any JSON-serialisable mapping.

    Returns:
        A string of the form ``"data: <json>\\n\\n"`` ready to be yielded
        from a streaming Flask response.
    """
    return f"data: {json.dumps(obj)}\n\n"


def sse_response(q: "queue.Queue[dict | None]") -> Response:
    """Build a streaming :class:`~flask.Response` that reads events from *q*.

    The internal generator reads dicts from *q* and yields them as SSE frames
    until it receives the sentinel value ``None``, which signals that the
    background worker has finished.

    Args:
        q: A :class:`queue.Queue` populated by a background thread.  Each
           item should be a JSON-serialisable dict.  A ``None`` item signals
           end-of-stream.

    Returns:
        A Flask :class:`~flask.Response` with
        ``mimetype="text/event-stream"`` and headers that disable proxy
        buffering (``X-Accel-Buffering: no``) and client caching
        (``Cache-Control: no-cache``).
    """

    @stream_with_context
    def _generate():
        while True:
            event = q.get()
            if event is None:
                break
            yield format_event(event)

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
