/**
 * UI-KIT — Icon Library (Font Awesome 6 Free).
 *
 * All icons use Font Awesome classes wrapped in <span class="icon">.
 * Requires Font Awesome 6 CSS (loaded via CDN in ui-kit.css or HTML).
 *
 * Usage:
 *   <script src="ui-kit/icons.js"></script>
 *   element.innerHTML = UIKit.ICONS.play;
 *   // or
 *   element.innerHTML = UIKit.icon("play");
 */
(function (root) {
  "use strict";

  var UIKit = root.UIKit || {};

  // ═══════════════════════════════════════════════════════════════════════════
  // ICONS — Font Awesome 6 Free (solid unless noted)
  // ═══════════════════════════════════════════════════════════════════════════

  function fa(cls) {
    return '<span class="icon"><i class="' + cls + '"></i></span>';
  }

  UIKit.ICONS = {

    // ── App controls ──
    play:         fa("fa-solid fa-play"),
    pause:        fa("fa-solid fa-pause"),
    stop:         fa("fa-solid fa-stop"),
    refresh:      fa("fa-solid fa-arrows-rotate"),
    spinner:      '<span class="icon icon-spin"><i class="fa-solid fa-spinner"></i></span>',
    save:         fa("fa-solid fa-floppy-disk"),
    close:        fa("fa-solid fa-xmark"),
    plus:         fa("fa-solid fa-plus"),
    minus:        fa("fa-solid fa-minus"),
    hamburger:    fa("fa-solid fa-bars"),
    chevron:      '<i class="fa-solid fa-chevron-right drawer-chevron"></i>',
    gear:         fa("fa-solid fa-gear"),

    // ── Theme ──
    moon:         fa("fa-solid fa-moon"),
    sun:          fa("fa-solid fa-sun"),

    // ── Contact / social ──
    mapMarker:    fa("fa-solid fa-location-dot"),
    graduationCap: fa("fa-solid fa-graduation-cap"),
    envelope:     fa("fa-solid fa-envelope"),
    certificate:  fa("fa-solid fa-certificate"),
    download:     fa("fa-solid fa-download"),
    upload:       fa("fa-solid fa-upload"),
    lock:         fa("fa-solid fa-lock"),
    unlock:       fa("fa-solid fa-lock-open"),
    brain:        fa("fa-solid fa-brain"),
    github:       fa("fa-brands fa-github"),
    linkedin:     fa("fa-brands fa-linkedin"),
    calendar:     fa("fa-solid fa-calendar-plus"),

    // ── Media ──
    video:        fa("fa-solid fa-video"),
    videoOff:     fa("fa-solid fa-video-slash"),
    microphone:   fa("fa-solid fa-microphone"),
    microphoneOff: fa("fa-solid fa-microphone-slash"),
    wifi:         fa("fa-solid fa-wifi"),

    // ── Misc ──
    externalLink: fa("fa-solid fa-arrow-up-right-from-square"),
    copy:         fa("fa-regular fa-copy"),
    edit:         fa("fa-solid fa-pen-to-square"),
    trash:        fa("fa-solid fa-trash"),
    search:       fa("fa-solid fa-magnifying-glass"),
    info:         fa("fa-solid fa-circle-info"),
    warning:      fa("fa-solid fa-triangle-exclamation"),
    check:        fa("fa-solid fa-check"),
    arrowLeft:    fa("fa-solid fa-arrow-left"),
    arrowRight:   fa("fa-solid fa-arrow-right"),
    star:         fa("fa-regular fa-star"),
    codeBranch:   fa("fa-solid fa-code-branch"),
  };

  /**
   * Get an icon's HTML markup by name.
   *
   * @param {string} name   - Icon name (key in UIKit.ICONS).
   * @param {Object} [opts] - Options (reserved for future use: size, class).
   * @returns {string} HTML string, or empty string if not found.
   */
  UIKit.icon = function (name, opts) {
    return UIKit.ICONS[name] || "";
  };

  root.UIKit = UIKit;

})(window);
