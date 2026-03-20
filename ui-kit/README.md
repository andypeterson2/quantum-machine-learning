# UI Kit

A standalone, framework-agnostic toolkit built with CSS custom properties and vanilla JS. No build tools required — just copy the files and link them.

---

## File Structure & CSS Entry Points

```
ui-kit/
  tokens.css              CSS custom properties (dark/light themes)
  base.css                Box-sizing reset, body defaults, utilities
  components.css          Aggregator: imports all 14 component files
  components/
    buttons.css             .btn, .btn-primary, .btn-secondary, ...
    cards.css               .card, .card-header
    forms.css               .form-row, .control-row, .feature-row
    drawers.css             .drawer, .side-handle, .side-panel
    dropdowns.css           .ui-dropdown, .ui-dropdown-item
    tables.css              .ui-table, .ui-table-sticky
    progress.css            .progress-area, .progress-bar
    log-terminal.css        .log-terminal, .log-entry
    lists.css               .ui-list, .ui-list-row
    modals.css              .modal-overlay, .modal
    canvas.css              .ui-canvas, .ui-canvas-clear
    icons.css               .icon, .icon-spin
    layout.css              .app, .app-topbar, .split-layout, ...
    theme-toggle.css        .ui-theme-toggle (FAB)
  ui-kit.css              Full entry point (tokens + base + components)
  ui-kit-core.css         Minimal entry point (tokens + base only)
  icons.js                40+ inline SVG icons as UIKit.ICONS
  ui-kit.js               Behaviour helpers (theme, drawer, dropdown, ...)
  theme-bootstrap.js      Prevents flash-of-wrong-theme (FOWT)
```

### Option A — Full (all components)

```html
<link rel="stylesheet" href="ui-kit/ui-kit.css">
```

### Option B — Core + individual components

```html
<link rel="stylesheet" href="ui-kit/ui-kit-core.css">
<link rel="stylesheet" href="ui-kit/components/buttons.css">
<link rel="stylesheet" href="ui-kit/components/cards.css">
<!-- only import what you need -->
```

---

## Quick Start

Minimal boilerplate to get a page running with the full kit, theme toggle, and FOWT prevention.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <script src="ui-kit/theme-bootstrap.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap"
        rel="stylesheet">
  <link rel="stylesheet" href="ui-kit/ui-kit.css">
</head>
<body>

  <!-- your content -->

  <button id="theme-toggle" class="ui-theme-toggle"></button>
  <script src="ui-kit/icons.js"></script>
  <script src="ui-kit/ui-kit.js"></script>
  <script>
    UIKit.initThemeToggle(document.getElementById("theme-toggle"));
  </script>
</body>
</html>
```

---

## Design Tokens & Theming

All colours come from CSS custom properties in `tokens.css`. Dark is the default; add `data-theme="light"` to `<html>` for light mode.

### Surfaces

| Token | Description |
|-------|-------------|
| `--bg` | Deepest background |
| `--surface` | Panels, sidebars |
| `--surface2` | Modals, dropdowns, hover |
| `--surface-muted` | Inactive tabs, subtle separators |
| `--surface-hl` | Selection, active line |

### Text

| Token | Description |
|-------|-------------|
| `--text` | Primary body text |
| `--text-secondary` | Dimmed text |
| `--text-muted` | Placeholder, disabled |
| `--text-inverse` | On accent backgrounds |

### Accents

| Token | Description |
|-------|-------------|
| `--accent` | Primary gold/amber |
| `--accent-hover` | Hover state |
| `--accent-teal` | Teal accent |
| `--accent-olive` | Olive accent |
| `--accent-brown` | Warm brown |
| `--accent-dark` | Dark earth |

### Semantic & Borders

| Token | Description |
|-------|-------------|
| `--success` | Sage green |
| `--warn` | Amber |
| `--danger` | Muted red |
| `--border` | Panel borders, dividers |
| `--border-focus` | Focus rings |

### Usage in CSS

```css
/* Use tokens as CSS custom properties */
.my-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  color: var(--text);
}
.my-panel:focus {
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px var(--focus-ring);
}
```

---

## Components

### Buttons

Base `.btn` with variant modifiers. All share consistent sizing, font, and disabled states.

```html
<button class="btn btn-primary">Primary</button>
<button class="btn btn-secondary">Secondary</button>
<button class="btn btn-danger">Danger</button>
<button class="btn btn-icon">...</button>
```

**Disabled & loading:**

```html
<button class="btn btn-primary" disabled>Disabled</button>
<button class="btn btn-primary btn-loading" disabled>Training...</button>
```

**With icons:**

```html
<button class="btn btn-primary">
  <!-- UIKit.ICONS.play + " Train" -->
</button>
```

---

### Cards

Surface-coloured panels with border. Use `.card-header` for a title + action row. Stack cards with automatic collapsed borders.

```html
<div class="card">
  <div class="card-header">
    <h2>Model Configuration</h2>
    <button class="btn btn-secondary">Reset</button>
  </div>
  <p>Content here...</p>
</div>
```

Stacked cards automatically collapse borders between them via `.card + .card { margin-top: -1px; }`.

---

### Forms

Three form patterns: `.form-row` (label+input grid), `.control-row` (select+buttons), and `.feature-row` (inline labelled inputs).

**Form rows inside a card:**

```html
<div class="card-form">
  <div class="form-row">
    <label>Model</label>
    <select>...</select>
  </div>
  <div class="form-row">
    <label>Epochs</label>
    <input type="number" value="100">
  </div>
</div>
```

**Control row (select + action buttons):**

```html
<div class="control-row">
  <select>...</select>
  <button class="btn btn-primary">Load</button>
  <button class="btn btn-icon">...</button>
</div>
```

**Feature row (inline labelled inputs):**

```html
<div class="feature-row">
  <div class="feature-label">
    Sepal Length
    <input class="feature-input" type="number" value="5.1">
  </div>
  <!-- more inputs... -->
  <button class="btn btn-primary predict-row-btn">Predict</button>
</div>
```

---

### Tables

Two variants: `.ui-table` (simple) and `.ui-table-sticky` (sticky headers for scrollable data). Wrap in `.ui-table-wrap` for bordered container.

```html
<div class="ui-table-wrap">
  <table class="ui-table">
    <thead><tr>
      <th>Model</th>
      <th>Accuracy</th>
    </tr></thead>
    <tbody><tr>
      <td>Random Forest</td>
      <td><span class="acc-high">96.7%</span></td>
    </tr></tbody>
  </table>
</div>
```

**Empty state:**

```html
<tr class="empty-row"><td colspan="3">No models trained yet</td></tr>
```

---

### Lists

Scrollable row-based lists with name, tag badge, and action buttons.

```html
<div class="ui-list">
  <div class="ui-list-row">
    <span class="ui-list-name">rf_iris_v3.pkl</span>
    <span class="ui-list-tag">96.7%</span>
    <button class="btn btn-icon">...</button>
    <button class="btn btn-danger">...</button>
  </div>
</div>
```

---

### Progress Bar

Minimal track + fill bar. Set width via inline style or JS. Add `.hidden` to `.progress-area` when inactive.

```html
<div class="progress-area">
  <div class="progress-bar-wrap">
    <div class="progress-bar" style="width:65%"
         role="progressbar" aria-valuenow="65"></div>
  </div>
  <div class="status-text">Training epoch 65 / 100</div>
</div>
```

Update via JS:

```js
bar.style.width = pct + "%";
```

---

### Log Terminal

Monospace scrollable log area with colour-coded levels. Use `UIKit.createLogger()` to append entries via JS.

```html
<div class="log-terminal" role="log" aria-live="polite" id="log"></div>

<script>
  var log = UIKit.createLogger(document.getElementById("log"));
  log("Dataset loaded", "ok");    // green
  log("Low importance", "warn");  // amber
  log("Validation error", "err"); // red
  log("Processing...");           // default
</script>
```

---

### Dropdowns

Absolutely-positioned menus toggled via JS. Supports `.active` state on items and click-outside-to-close.

```html
<div style="position:relative">
  <button class="btn btn-secondary" id="trigger">Select dataset</button>
  <div class="ui-dropdown hidden" id="menu">
    <button class="ui-dropdown-item active">iris.csv</button>
    <button class="ui-dropdown-item">wine.csv</button>
  </div>
</div>

<script>
  UIKit.initDropdown(
    document.getElementById("trigger"),
    document.getElementById("menu")
  );
</script>
```

---

### Drawers

Collapsible side panels with a vertical handle + chevron.

```html
<div class="drawer drawer-left" id="drawer">
  <div class="side-panel side-panel-left">
    <div class="side-panel-header">
      <div class="side-panel-title">Datasets</div>
    </div>
    <!-- panel content -->
  </div>
  <button class="side-handle side-handle-left" id="handle">
    <!-- UIKit.ICONS.chevron -->
    <span>Datasets</span>
  </button>
</div>

<script>
  UIKit.initDrawer(
    document.getElementById("drawer"),
    document.getElementById("handle")
  );
</script>
```

---

### Modals

Overlay + centred dialog. Use `.hidden` on the overlay to toggle visibility. Combine with `UIKit.onEscape()` for keyboard dismissal.

```html
<div class="modal-overlay" id="modal">
  <div class="modal">
    <div class="modal-header">
      <span class="modal-title">Confirm Delete</span>
      <button class="btn btn-icon">...</button>
    </div>
    <p>Are you sure?</p>
    <div class="modal-footer">
      <button class="btn btn-secondary">Cancel</button>
      <button class="btn btn-danger">Delete</button>
    </div>
  </div>
</div>

<script>
  // Toggle with .hidden class
  document.getElementById("modal").classList.toggle("hidden");
  // Close on Escape
  UIKit.onEscape(function() { ... });
</script>
```

---

### Drawing Canvas

Black canvas with border, crosshair cursor, and an overlay clear button. Designed for digit/gesture drawing inputs.

```html
<div class="canvas-area">
  <div class="canvas-wrap">
    <canvas class="ui-canvas" width="200" height="120" id="canvas"></canvas>
    <button class="ui-canvas-clear">Clear</button>
  </div>
  <div class="canvas-hint">Draw a digit</div>
</div>
```

---

### Icons

40+ inline SVG icons (16x16, `currentColor`). Available as `UIKit.ICONS.name` strings or via `UIKit.icon("name")`. Add `.icon-spin` for animation.

```js
// Insert an icon via innerHTML
el.innerHTML = UIKit.ICONS.play;

// Or use the helper (same result)
el.innerHTML = UIKit.icon("play");

// Combine with text
btn.innerHTML = UIKit.ICONS.play + " Train Model";

// Spinning icon (e.g. loading state)
el.innerHTML = UIKit.ICONS.spinner; // has .icon-spin built-in
```

Available icons: `play`, `pause`, `stop`, `refresh`, `save`, `close`, `moon`, `sun`, `hamburger`, `chevron`, `trash`, `download`, `spinner`, and many more. Open `showcase.html` to see the full set.

---

### Layout & App Shell

Full-viewport app shell with topbar, sidebar, main area, and optional split columns with drag-to-resize.

```
+------------------------------------------+
|              .app-topbar                 |
+----------+-------------------------------+
|          |                               |
|  .app-   |         .app-main             |
|  sidebar |                               |
|          |                               |
+----------+-------------------------------+
```

```html
<div class="app">
  <div class="app-topbar">...</div>
  <div class="app-body">
    <div class="app-sidebar">...</div>
    <div class="app-main">...</div>
  </div>
</div>
```

**Split layout with resize handle:**

```html
<div class="split-layout" id="split">
  <div class="left-col" id="left">
    <!-- cards, forms -->
  </div>
  <div class="resize-handle resize-h" id="handle"></div>
  <div class="right-col">
    <!-- table, chart -->
  </div>
</div>

<script>
  UIKit.initResize(
    document.getElementById("handle"),
    document.getElementById("left"),
    document.getElementById("split"),
    { min: 200, "default": 300, key: "split-w" }
  );
</script>
```

**Responsive breakpoints:**

| Breakpoint | Behaviour |
|------------|-----------|
| `1100px` | Right column stacks below left; split layout goes vertical |
| `900px` | Drawer handle hides; body scrolls; sidebar goes full-width |
| `600px` | Feature inputs wrap to 50%; predict button goes full-width |

---

## Utilities & Colour Helpers

### Utility classes

| Class | Effect |
|-------|--------|
| `.hidden` | `display: none !important` |
| `.sr-only` | Visually hidden, accessible to screen readers |
| `.resize-dragging` | Applied to `<body>` during drag-resize (disables text selection) |

### Colour helpers for data display

```html
<span class="acc-high">96.7%</span>   <!-- green -->
<span class="acc-med">78.3%</span>    <!-- amber -->
<span class="acc-low">65.1%</span>    <!-- red -->
<span class="conf-high">Ready</span>  <!-- green -->
<span class="conf-low">...</span>     <!-- muted -->
<span class="pred-label">Setosa</span> <!-- bold accent -->
```

---

## JavaScript API

Load `icons.js` first, then `ui-kit.js`. Everything lives on `window.UIKit`. Nothing auto-initialises — call only what you need.

| Method | Purpose | Returns |
|--------|---------|---------|
| `initThemeToggle(el, key?)` | Dark/light toggle with localStorage | `{ setTheme(t) }` |
| `initDrawer(drawerEl, handleEl)` | Collapsible side drawer | `{ open(), close(), toggle() }` |
| `initDropdown(triggerEl, menuEl)` | Toggle menu + click-outside | `{ open(), close() }` |
| `onEscape(callback)` | Register Escape-key handler | — |
| `initResize(handle, target, container, opts?)` | Drag-to-resize split layout | — |
| `createLogger(terminalEl, max?)` | Log terminal appender | `addLog(msg, level?)` |
| `ICONS` | Object of inline SVG markup strings | `{ play, pause, stop, ... }` |
| `icon(name, opts?)` | Get icon HTML by name | `string` |

### theme-bootstrap.js

Prevents flash-of-wrong-theme. Must be the **first** `<script>` in `<head>`, before any CSS.

```html
<script src="ui-kit/theme-bootstrap.js"></script>
```

Reads from localStorage and sets `data-theme` before first paint. Syncs across tabs via the `storage` event. Exposes `window.__setTheme(t)` and `window.__getTheme()`.

---

## Demo

Open `showcase.html` directly in a browser (no server needed) to see every component rendered with interactive examples.
