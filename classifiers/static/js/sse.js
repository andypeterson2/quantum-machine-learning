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

/**
 * POST to an SSE endpoint and dispatch parsed events.
 *
 * @param {string}   url      - Endpoint URL.
 * @param {Object}   body     - JSON request body.
 * @param {Function} onStatus - Called for each "status" or structured event.
 * @param {Function} onDone   - Called once when a "done" event arrives.
 * @param {Function} onError  - Called once when an "error" event arrives.
 */
async function consumeSSE(url, body, onStatus, onDone, onError) {
  const res = await fetch(url, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    onError(err.error || res.statusText);
    return;
  }
  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
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
}
