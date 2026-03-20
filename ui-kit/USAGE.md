# UI Kit — Reusable Component Toolkit

A standalone, framework-agnostic UI toolkit built with plain CSS custom properties and vanilla JS. No build tools required — just copy the files and link them.

## What's Included

```
ui-kit/
  tokens.css        CSS custom properties (dark/light themes, colours, radius, font)
  base.css          Box-sizing reset, body defaults, utility classes
  components.css    All component styles (class-based selectors)
  ui-kit.css        Single @import entry point (imports the above three)
  ui-kit.js         IIFE exposing window.UIKit with opt-in behaviour helpers
  demo.html         Living reference page — open in a browser to see every component
```

## Quick Start

Add the Inter font and link the entry-point CSS + JS:

```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="ui-kit/ui-kit.css">
<!-- your app CSS after, so overrides win -->
<link rel="stylesheet" href="css/app.css">

<!-- at end of body -->
<script src="ui-kit/ui-kit.js"></script>
<script src="js/app.js"></script>
```

## Theming

The toolkit ships with dark (default) and light themes controlled by the `data-theme` attribute on `<html>`. All colours come from CSS custom properties defined in `tokens.css`.

Key tokens: `--bg`, `--surface`, `--surface-alt`, `--border`, `--accent`, `--accent-hover`, `--text`, `--text-muted`, `--radius`, `--font`.

To initialise the toggle button:

```html
<button id="theme-toggle" class="ui-theme-toggle"></button>
<script>
  UIKit.initThemeToggle(document.getElementById("theme-toggle"));
</script>
```

It persists the choice to `localStorage` under the key `"theme"`.

## JavaScript API — `window.UIKit`

Nothing auto-initialises. Call only what you need.

| Method | Purpose | Returns |
|--------|---------|---------|
| `UIKit.initThemeToggle(el, key?)` | Dark/light toggle with localStorage | `{ setTheme(t) }` |
| `UIKit.initDrawer(drawerEl, handleEl)` | Collapsible side drawer | `{ open(), close(), toggle() }` |
| `UIKit.initDropdown(triggerEl, menuEl)` | Toggle menu + click-outside-to-close | `{ open(), close() }` |
| `UIKit.onEscape(callback)` | Register an Escape-key handler | — |
| `UIKit.initResize(handleEl, targetEl, containerEl, opts?)` | Drag-to-resize split layout, persists to localStorage | — |
| `UIKit.createLogger(terminalEl, max?)` | Log terminal appender | `addLog(msg, level?)` |

### `UIKit.ICONS`

Object of inline SVG markup strings (16x16, `currentColor`):

`play`, `refresh`, `spinner`, `save`, `close`, `moon`, `sun`, `hamburger`, `chevron`

Usage: `element.innerHTML = UIKit.ICONS.play + " Train";`

## CSS Components

All styles use class selectors — no IDs. Apply classes to your markup:

### Layout
- `.app-shell` — full-viewport flex shell
- `.top-bar` — sticky header bar
- `.split-layout` — horizontal flex container for columns
- `.left-col`, `.right-col` — flex children
- `.resize-handle`, `.resize-h` — draggable divider between columns

### Cards & Forms
- `.card` — surface-coloured panel with border and padding
- `.card-header` — flex row for heading + action button
- `.card-form` — form inside a card (strips fieldset chrome)
- `.form-row` — label + input grid row

### Buttons
- `.btn` — base button
- `.btn-primary` — accent-coloured fill
- `.btn-secondary` — outlined
- `.btn-danger` — red
- `.btn-icon` — square icon-only button
- `.btn-loading` — disabled state with spinner

### Drawer
- `.drawer`, `.drawer-left` — off-screen panel container
- `.drawer.open` — slides in
- `.side-handle`, `.side-handle-left` — toggle tab
- `.side-panel`, `.side-panel-left` — panel content area
- `.side-panel-header`, `.side-panel-title` — panel header

### Dropdown
- `.ui-dropdown` — absolutely-positioned menu (add `.hidden` to hide)
- `.ui-dropdown-item` — clickable row inside the dropdown

### Tables
- `.ui-table` — styled table with header/row colours
- `.ui-table-sticky` — sticky-header variant for scrollable tables

### Progress
- `.progress-area` — wrapper (add `.hidden` when inactive)
- `.progress-bar-wrap` — track
- `.progress-bar` — fill (set `width` via style or JS)

### Log Terminal
- `.log-terminal` — dark scrollable terminal area
- `.log-entry`, `.log-time`, `.log-msg` — log line parts
- `.log-ok`, `.log-warn`, `.log-error` — colour levels

### Drawing Canvas
- `.ui-canvas` — bordered canvas element
- `.ui-canvas-clear` — overlay clear/close button

### Lists
- `.ui-list` — container for dynamic item rows
- `.ui-list-row` — single row with flex layout
- `.ui-list-name`, `.ui-list-tag` — name and badge inside a row

### Feature Input
- `.feature-row` — flex-wrap row of labelled numeric inputs
- `.feature-label`, `.feature-input` — individual input + label

### Icons
- `.icon` — 16x16 inline-flex container for SVGs
- `.icon-spin` — adds a rotation animation

### Utilities
- `.hidden` — `display: none`
- `.sr-only` — visually hidden, accessible to screen readers
- `.resize-dragging` — applied to `<body>` during drag-resize

### Colour Helpers
- `.acc-high`, `.acc-med`, `.acc-low` — green / amber / red for accuracy
- `.conf-high`, `.conf-low` — green / amber for confidence
- `.pred-label` — bold accent-coloured label

## Responsive Breakpoints

- **1100px**: right column stacks below left; split layout goes vertical
- **900px**: side drawer handle hides; form rows stack
- **600px**: top bar font shrinks; cards reduce padding

## Demo

Open `demo.html` directly in a browser (no server needed) to see every component with interactive examples.
