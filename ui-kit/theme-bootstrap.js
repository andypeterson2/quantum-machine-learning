/**
 * Theme Bootstrap — MUST be the first <script> in <head>.
 *
 * Reads theme preference from localStorage and applies it before
 * first paint to prevent flash-of-wrong-theme (FOWT).
 * Shared across all sub-apps on the same domain.
 *
 * Usage:
 *   <script src="/shared/theme-bootstrap.js"></script>
 *   <link rel="stylesheet" href="/shared/ui-kit/ui-kit-core.css">
 */
(function () {
  "use strict";

  // Single source of truth: UIKit.THEME_KEY (set in ui-kit.js).
  // Duplicated here because bootstrap runs before ui-kit.js loads.
  var THEME_KEY = "sm-theme";

  var stored = localStorage.getItem(THEME_KEY);
  var theme = stored || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");

  document.documentElement.dataset.theme = theme;

  /** Set theme and persist across all sub-apps. */
  window.__setTheme = function (t) {
    document.documentElement.dataset.theme = t;
    localStorage.setItem(THEME_KEY, t);
  };

  /** Get current theme. */
  window.__getTheme = function () {
    return document.documentElement.dataset.theme;
  };

  // Sync across tabs/windows on the same domain.
  window.addEventListener("storage", function (e) {
    if (e.key === THEME_KEY && e.newValue) {
      document.documentElement.dataset.theme = e.newValue;
    }
  });
})();
