"""Client connection tracking for heartbeat and lifecycle management.

Provides a thread-safe registry of connected clients.  The server records
each client on initial SSE handshake and updates its *last-seen* timestamp
whenever the client responds to a ping.  A background sweep thread
periodically evicts stale entries, catching ungraceful disconnects (tab
closed without ``beforeunload``, network failure, etc.).
"""

from __future__ import annotations

import threading
import time
import uuid


class ConnectionTracker:
    """Thread-safe registry of connected clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: dict[str, float] = {}  # client_id → last-seen ts

    # ── public API ────────────────────────────────────────────────

    def register(self) -> str:
        """Create and register a new client, returning its unique id."""
        client_id = uuid.uuid4().hex
        with self._lock:
            self._clients[client_id] = time.monotonic()
        return client_id

    def heartbeat(self, client_id: str) -> bool:
        """Update *last-seen* for *client_id*.  Returns ``False`` if unknown."""
        with self._lock:
            if client_id not in self._clients:
                return False
            self._clients[client_id] = time.monotonic()
            return True

    def unregister(self, client_id: str) -> bool:
        """Remove *client_id*.  Returns ``True`` if it was present."""
        with self._lock:
            return self._clients.pop(client_id, None) is not None

    def sweep(self, timeout: float = 90.0) -> list[str]:
        """Evict clients not seen within *timeout* seconds.

        Returns the list of evicted client ids.
        """
        now = time.monotonic()
        evicted: list[str] = []
        with self._lock:
            for cid, last_seen in list(self._clients.items()):
                if now - last_seen > timeout:
                    del self._clients[cid]
                    evicted.append(cid)
        return evicted

    @property
    def count(self) -> int:
        """Number of currently tracked clients."""
        with self._lock:
            return len(self._clients)

    def active_clients(self) -> list[dict]:
        """Return a snapshot of connected clients with last-seen info."""
        now = time.monotonic()
        with self._lock:
            return [
                {"client_id": cid, "idle_seconds": round(now - ts, 1)}
                for cid, ts in self._clients.items()
            ]
