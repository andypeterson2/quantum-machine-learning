"""Standardised error response helpers.

Every route module should use :func:`error_response` instead of constructing
error dicts inline.  This guarantees a consistent JSON shape across the API::

    {"error": "<message>"}

Centralising error formatting is an Open/Closed improvement: if we later add
fields (``code``, ``detail``, ``trace_id``) we only change this file.
"""

from __future__ import annotations

from flask import Response, jsonify


def error_response(msg: str, status: int = 400) -> tuple[Response, int]:
    """Return a JSON error response with a consistent shape.

    Args:
        msg:    Human-readable error message.
        status: HTTP status code (default 400).

    Returns:
        A ``(Response, status)`` tuple suitable for returning from a Flask
        view function.
    """
    return jsonify({"error": msg}), status
