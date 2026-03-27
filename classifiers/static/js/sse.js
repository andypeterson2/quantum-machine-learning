/**
 * @file SSE (Server-Sent Events) stream consumer utility.
 *
 * Provides a reusable function for POST-ing to an SSE endpoint and dispatching
 * parsed events to typed callbacks.  This decouples the streaming transport
 * from any specific business logic (training, evaluation, ablation).
 *
 * @example
 *   await consumeSSE("/d/mnist/train", body,
 *     msg   => log(msg),
 *     event => handleDone(event),
 *     err   => showError(err),
 *   );
 */
"use strict";

/** Per-chunk read timeout (ms).  If no data arrives within this window the
 *  stream is considered dead and an error is raised. */
var _SSE_READ_TIMEOUT = 300000; // 5 minutes — training batches can be slow

/**
 * Race a reader.read() against a timeout.
 * @param {ReadableStreamDefaultReader} reader
 * @param {number} ms
 * @returns {Promise<ReadableStreamReadResult>}
 */
function _readWithTimeout(reader, ms) {
  return new Promise(function (resolve, reject) {
    var timer = setTimeout(function () {
      reader.cancel();
      reject(new Error("SSE read timed out after " + ms + " ms"));
    }, ms);
    reader.read().then(
      function (result) { clearTimeout(timer); resolve(result); },
      function (err)    { clearTimeout(timer); reject(err); }
    );
  });
}

/**
 * POST to an SSE endpoint and dispatch parsed events.
 *
 * Uses {@link apiFetch} (from connection.js) when available so that
 * requests fail fast when the backend is unreachable.
 *
 * @param {string}   url      - Endpoint URL.
 * @param {Object}   body     - JSON request body.
 * @param {Function} onStatus - Called for each "status" or structured event.
 * @param {Function} onDone   - Called once when a "done" event arrives.
 * @param {Function} onError  - Called once when an "error" event arrives.
 */
async function consumeSSE(url, body, onStatus, onDone, onError) {
  var _fetch = typeof apiFetch === "function" ? apiFetch : fetch;
  var res;
  try {
    res = await _fetch(url, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
  } catch (e) {
    onError(e.message || "Failed to connect");
    return;
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    onError(err.error || res.statusText);
    return;
  }
  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { value, done } = await _readWithTimeout(reader, _SSE_READ_TIMEOUT);
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const json = line.slice(5).trim();
        if (!json) continue;
        const event = JSON.parse(json);
        if      (event.type === "status")  onStatus(event.msg);
        else if (event.type === "done")    onDone(event);
        else if (event.type === "error")   onError(event.msg);
        else if (event.type === "history" || event.type === "ablation_result") onStatus(event);
      }
    }
  } catch (e) {
    onError(e.message || "Stream error");
  }
}
