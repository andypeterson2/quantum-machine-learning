/**
 * @file ConnectionManager — client-side connection lifecycle for the
 *       classifier backend.
 *
 * State machine:
 *   idle ──► connecting ──► connected ──► degraded ──► disconnected
 *                │              │                          │
 *                └──────────────┴──────── reconnect ◄──────┘
 *
 * States and their intended dot colors:
 *   - idle         → grey  (no connection attempted)
 *   - connecting   → yellow (actively trying)
 *   - connected    → green
 *   - degraded     → yellow (missed pings / health-check retries)
 *   - disconnected → red
 *
 * Emits `connection:statechange` CustomEvent on every transition.
 *
 * Exposes a global `connectionManager` singleton.
 */
"use strict";

var connectionManager = (function () {

  // ── States ──────────────────────────────────────────────────────
  var IDLE         = "idle";
  var CONNECTING   = "connecting";
  var CONNECTED    = "connected";
  var DEGRADED     = "degraded";
  var DISCONNECTED = "disconnected";

  var _state       = IDLE;
  var _baseUrl     = "";
  var _clientId    = null;

  // SSE reader / abort
  var _abortCtrl   = null;

  // Reconnect backoff
  var _reconnectDelay    = 1000;
  var _maxReconnectDelay = 30000;
  var _reconnectTimer    = null;
  var _wantConnected     = false;

  // Ping timeout
  var _heartbeatInterval = 25;        // seconds, updated by welcome
  var _pingTimer         = null;

  // Health-check retry tracking
  var _healthFailures    = 0;
  var _maxHealthRetries  = 2;  // go degraded after this many consecutive failures

  // ── Observer — single dispatch point ──────────────────────────

  function _setState(s) {
    var prev = _state;
    _state = s;
    if (s !== prev) {
      document.dispatchEvent(new CustomEvent("connection:statechange", {
        detail: { state: s, previous: prev, clientId: _clientId },
      }));
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────

  function _clearTimers() {
    if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
    if (_pingTimer)      { clearTimeout(_pingTimer); _pingTimer = null; }
  }

  function _resetBackoff() {
    _reconnectDelay = 1000;
    _healthFailures = 0;
  }

  function _jitter(ms) {
    return ms + Math.random() * ms * 0.3;
  }

  // ── Ping timeout ────────────────────────────────────────────────

  function _startPingTimeout() {
    if (_pingTimer) clearTimeout(_pingTimer);
    // First missed ping → degraded; second → disconnected.
    var degradedTimeout = _heartbeatInterval * 2 * 1000;
    var deadTimeout     = _heartbeatInterval * 3.5 * 1000;

    _pingTimer = setTimeout(function () {
      if (_state === CONNECTED) {
        _setState(DEGRADED);
        // Give it one more interval before declaring dead.
        _pingTimer = setTimeout(function () {
          _handleDisconnect();
        }, deadTimeout - degradedTimeout);
      } else {
        _handleDisconnect();
      }
    }, degradedTimeout);
  }

  // ── SSE heartbeat channel ──────────────────────────────────────

  function _openHeartbeat() {
    _abortCtrl = new AbortController();
    fetch(_baseUrl + "/connect", { signal: _abortCtrl.signal })
      .then(function (res) {
        if (!res.ok) throw new Error("connect HTTP " + res.status);
        return _readSSE(res.body.getReader());
      })
      .catch(function (err) {
        if (err.name === "AbortError") return; // intentional close
        _handleDisconnect();
      });
  }

  function _readSSE(reader) {
    var decoder = new TextDecoder();
    var buf = "";

    function pump() {
      return reader.read().then(function (result) {
        if (result.done) { _handleDisconnect(); return; }
        buf += decoder.decode(result.value, { stream: true });
        var parts = buf.split("\n\n");
        buf = parts.pop();
        for (var i = 0; i < parts.length; i++) {
          var line = parts[i].trim();
          if (!line.startsWith("data:")) continue;
          var json = line.slice(5).trim();
          if (!json) continue;
          var event = JSON.parse(json);
          _handleEvent(event);
        }
        return pump();
      }).catch(function (err) {
        if (err.name === "AbortError") return;
        _handleDisconnect();
      });
    }
    return pump();
  }

  function _handleEvent(event) {
    if (event.type === "welcome") {
      _clientId = event.client_id;
      _heartbeatInterval = event.heartbeat_interval || 25;
      _setState(CONNECTED);
      _resetBackoff();
      _startPingTimeout();
    } else if (event.type === "ping") {
      // Any successful ping restores full connection.
      if (_state === DEGRADED) _setState(CONNECTED);
      _startPingTimeout();
      // Respond with pong
      fetch(_baseUrl + "/pong", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_id: _clientId }),
      }).catch(function () { /* best effort */ });
    }
  }

  // ── Disconnect / reconnect ─────────────────────────────────────

  function _handleDisconnect() {
    if (_state === DISCONNECTED || _state === IDLE) return;
    _closeChannel();
    _setState(DISCONNECTED);
    if (_wantConnected) _scheduleReconnect();
  }

  function _closeChannel() {
    _clearTimers();
    if (_abortCtrl) { _abortCtrl.abort(); _abortCtrl = null; }
  }

  function _scheduleReconnect() {
    if (_reconnectTimer) return;
    var delay = _jitter(_reconnectDelay);
    _reconnectDelay = Math.min(_reconnectDelay * 2, _maxReconnectDelay);
    _reconnectTimer = setTimeout(function () {
      _reconnectTimer = null;
      if (_wantConnected) _doConnect();
    }, delay);
  }

  // ── Connect flow ───────────────────────────────────────────────

  function _doConnect() {
    _setState(CONNECTING);
    // Health check first
    fetch(_baseUrl + "/health")
      .then(function (res) {
        if (!res.ok) throw new Error("health HTTP " + res.status);
        _healthFailures = 0;
        _openHeartbeat();
      })
      .catch(function () {
        _healthFailures++;
        if (_healthFailures <= _maxHealthRetries) {
          // Still might come up — show degraded/yellow, keep retrying.
          _setState(DEGRADED);
          if (_wantConnected) _scheduleReconnect();
        } else {
          // Gave up — red dot.
          _setState(DISCONNECTED);
          if (_wantConnected) _scheduleReconnect();
        }
      });
  }

  // ── Graceful unload ────────────────────────────────────────────

  window.addEventListener("beforeunload", function () {
    if (_clientId && _baseUrl) {
      navigator.sendBeacon(
        _baseUrl + "/disconnect",
        new Blob([JSON.stringify({ client_id: _clientId })], { type: "application/json" })
      );
    }
  });

  // ── Navbar integration (observer of its own events) ────────────

  document.addEventListener("navbar:connect", function (e) {
    if (e.detail.service !== "classifiers") return;
    api.disconnect();
    api.connect(e.detail.url);
  });

  document.addEventListener("navbar:disconnect", function (e) {
    if (e.detail.service !== "classifiers") return;
    api.disconnect();
  });

  // Bridge: navbar widget subscribes to connection:statechange
  (function () {
    var _navWidget = null;

    document.addEventListener("navbar:connect-ready", function (e) {
      if (e.detail.service !== "classifiers") return;
      _navWidget = e.detail.widget;
      _navWidget.setStatus(_state);
    });

    document.addEventListener("connection:statechange", function (e) {
      if (_navWidget) _navWidget.setStatus(e.detail.state);
    });
  })();

  // ── Public API ─────────────────────────────────────────────────

  var api = {
    get state() { return _state; },

    connect: function (baseUrl) {
      _closeChannel();
      _healthFailures = 0;
      _baseUrl = baseUrl.replace(/\/+$/, "");
      _wantConnected = true;
      _doConnect();
    },

    disconnect: function () {
      _wantConnected = false;
      _closeChannel();
      if (_clientId && _baseUrl) {
        fetch(_baseUrl + "/disconnect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ client_id: _clientId }),
        }).catch(function () { /* best effort */ });
      }
      _clientId = null;
      _setState(DISCONNECTED);
    },
  };

  return api;
})();
