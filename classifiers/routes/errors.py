"""Standardised JSON error envelope for the classifiers API.

Implements the cross-repo backend contract (see the website repo at
``docs/api-contract/CONTRACT.md``).  Every 4xx/5xx response body is::

    {"error": {"code": "<slug>", "message": "<human>", "details": <optional>}}

The HTTP status carries the class; the envelope never restates it.  ``code`` is a
stable snake_case slug clients may switch on; ``message`` is human text.  When a
caller omits ``code`` one is derived from the HTTP status, so the plain
``error_response(msg, status)`` call sites are compliant with no change.
"""

from __future__ import annotations

import logging

from flask import Response, jsonify
from werkzeug.exceptions import HTTPException

_log = logging.getLogger(__name__)

#: HTTP status -> stable machine code (used when a caller omits ``code``).
_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "unprocessable_entity",
    429: "rate_limited",
    500: "internal_error",
}


def error_response(
    msg: str, status: int = 400, code: str | None = None, details=None
) -> tuple[Response, int]:
    """Return a ``(Response, status)`` carrying the standard error envelope.

    Args:
        msg:     Human-readable error message.
        status:  HTTP status code (default 400).
        code:    Optional stable machine slug; derived from *status* if omitted.
        details: Optional structured context (any JSON value).

    Returns:
        A ``(Response, status)`` tuple suitable for returning from a Flask view.
    """
    body: dict = {"code": code or _STATUS_CODES.get(status, "error"), "message": msg}
    if details is not None:
        body["details"] = details
    return jsonify({"error": body}), status


def register_error_handlers(app) -> None:
    """Make framework-raised errors (404/405/500, ...) use the envelope too."""

    @app.errorhandler(HTTPException)
    def _http_exception(exc: HTTPException):
        return error_response(
            exc.description or exc.name,
            exc.code or 500,
            code=_STATUS_CODES.get(exc.code, "http_error"),
        )

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        _log.exception("Unhandled error: %s", exc)
        return error_response("Internal server error.", 500, code="internal_error")
