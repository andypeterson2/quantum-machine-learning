/**
 * @file Multi-Dataset Classifier App — Client-side SPA logic.
 *
 * This file drives the classifier front-end.  Reusable UI behaviours (theme
 * toggle, drawer, dropdown, resize, log terminal) are provided by
 * `UIKit` (see `ui-kit/ui-kit.js`).  This file handles application-specific
 * logic: state management, SSE streaming, training, evaluation, prediction,
 * model persistence, and canvas drawing.
 *
 * @global {Object} UI_CONFIG - Injected by the server template.
 * @global {Object} UIKit     - Provided by ui-kit.js (loaded before this file).
 */
"use strict";

// ── Shorthand ────────────────────────────────────────────────────────────────
const ICONS = UIKit.ICONS;

// ── Config injected by template ─────────────────────────────────────────────

/** @const {string} URL prefix for all API calls scoped to the active dataset. */
const BASE = API_BASE + `/d/${UI_CONFIG.name}`;

// ── State ────────────────────────────────────────────────────────────────────

/**
 * Application state — single source of truth for loaded models and predictions.
 * @namespace
 * @property {Object<string, ModelInfo>}      models
 * @property {Object<string, PredictionInfo>} predictions
 */
const state = {
  models:      {},
  predictions: {},
};

/** @typedef {Object} ModelInfo
 *  @property {string} model_type @property {number} epochs
 *  @property {number} batch_size @property {number} lr
 *  @property {number|null} num_params
 *  @property {Array} training_history
 *  @property {EvalResult|null} eval_result */

/** @typedef {Object} EvalResult
 *  @property {number} accuracy @property {number} avg_loss
 *  @property {Object<string, number>} per_class_accuracy
 *  @property {number|null} num_params */

/** @typedef {Object} PredictionInfo
 *  @property {string} prediction @property {number} confidence
 *  @property {number[]} probs */

// ── Smart naming ─────────────────────────────────────────────────────────────

/**
 * Generate the next available default name for a model type.
 * @param   {string} modelType
 * @returns {string}
 */
function defaultName(modelType) {
  const names = Object.keys(state.models);
  if (!names.includes(modelType)) return modelType;
  let n = 2;
  while (names.includes(`${modelType} ${n}`)) n++;
  return `${modelType} ${n}`;
}

// ── UIKit initialisations ────────────────────────────────────────────────────

UIKit.initThemeToggle(document.getElementById("theme-toggle"));

const logDrawer  = document.getElementById("log-drawer");
const logHandle  = document.getElementById("log-handle");
const drawer     = UIKit.initDrawer(logDrawer, logHandle);

const datasetMenuBtn = document.getElementById("dataset-menu-btn");
const datasetMenu    = document.getElementById("dataset-menu");
const dropdown       = UIKit.initDropdown(datasetMenuBtn, datasetMenu);

UIKit.onEscape(() => { drawer.close(); dropdown.close(); });

const leftCol     = document.getElementById("left-col");
const splitLayout = document.getElementById("split-layout");
const resizeH     = document.getElementById("resize-h");
UIKit.initResize(resizeH, leftCol, splitLayout, {
  min: 180, "default": 300, key: "leftColWidth_v2",
});

const logTerminal = document.getElementById("log-terminal");
const addLog      = UIKit.createLogger(logTerminal, 200);

// ── Input-type visibility ─────────────────────────────────────────────────────

const canvasCol  = document.getElementById("canvas-col");
const tabularCol = document.getElementById("tabular-col");
if (UI_CONFIG.input_type === "image") {
  canvasCol.style.display  = "";
  tabularCol.style.display = "none";
} else {
  canvasCol.style.display  = "none";
  tabularCol.style.display = "";
}

// ── DOM refs ──────────────────────────────────────────────────────────────────

const canvas           = document.getElementById("draw-canvas");
const ctx              = canvas.getContext("2d");
const trainBtn         = document.getElementById("train-btn");
const clearBtn         = document.getElementById("clear-btn");
const predictBtn       = document.getElementById("predict-btn");
const importBtn        = document.getElementById("import-btn");
const refreshSavedBtn  = document.getElementById("refresh-saved-btn");
const savedSelect      = document.getElementById("saved-select");
const datasetList      = document.getElementById("dataset-list");
const evalProgress     = document.getElementById("evaluate-progress");
const evalBar          = document.getElementById("eval-bar");
const evalStatus       = document.getElementById("eval-status");
const metricsHead      = document.getElementById("metrics-head");
const metricsBody      = document.getElementById("metrics-body");
const predBody         = document.getElementById("pred-body");
const modelNameInput   = document.getElementById("model-name");
const sessionModels    = document.getElementById("session-models");
const chartArea        = document.getElementById("chart-area");
const trainChartCanvas = document.getElementById("train-chart");
const ensembleBtn      = document.getElementById("ensemble-btn");
const teacherSelect    = document.getElementById("teacher-select");
const distillRow       = document.getElementById("distill-row");

/** @type {MiniChart|null} */
let trainChart = null;

// ── Model info panel ─────────────────────────────────────────────────────────

async function fetchModelInfo(modelType) {
  const details = document.getElementById("model-info-details");
  const panel   = document.getElementById("model-info-panel");
  try {
    const res = await fetch(`${BASE}/model-info/${encodeURIComponent(modelType)}`);
    if (!res.ok) { details.classList.add("hidden"); return; }
    const data = await res.json();
    panel.innerHTML = data.html;
    details.classList.remove("hidden");
  } catch { details.classList.add("hidden"); }
}

// ── Refresh default name when model type changes ──────────────────────────────

document.getElementById("model-type").addEventListener("change", (e) => {
  modelNameInput.value = defaultName(e.target.value);
  fetchModelInfo(e.target.value);
});

// Show/hide distillation weight when teacher is selected
teacherSelect.addEventListener("change", () => {
  distillRow.classList.toggle("hidden", !teacherSelect.value);
});

/** Update teacher select dropdown with current session models. */
function updateTeacherSelect() {
  const current = teacherSelect.value;
  teacherSelect.innerHTML = '<option value="">— none —</option>';
  for (const name of Object.keys(state.models)) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    teacherSelect.appendChild(opt);
  }
  teacherSelect.value = current;
  distillRow.classList.toggle("hidden", !teacherSelect.value);
}

/** Update ensemble button visibility (needs 2+ models). */
function updateEnsembleBtn() {
  ensembleBtn.classList.toggle("hidden", Object.keys(state.models).length < 2);
}

// ── Populate dataset menu ─────────────────────────────────────────────────────

(async function initDatasetMenu() {
  try {
    const res  = await fetch(API_BASE + "/api/datasets");
    const list = await res.json();
    datasetList.innerHTML = "";
    for (const ds of list) {
      const btn = document.createElement("button");
      btn.className = "ui-dropdown-item" + (ds.name === UI_CONFIG.name ? " active" : "");
      btn.textContent = ds.display_name;
      btn.addEventListener("click", () => {
        dropdown.close();
        if (ds.name !== UI_CONFIG.name) window.location = `/d/${ds.name}/`;
      });
      datasetList.appendChild(btn);
    }
  } catch {
    const btn = document.createElement("button");
    btn.className = "ui-dropdown-item active";
    btn.textContent = UI_CONFIG.display_name;
    datasetList.appendChild(btn);
  }
})();

// ── Canvas drawing (28×28 pixel grid for MNIST) ────────────────────────────────

const GRID = 28;
const CELL = canvas.width / GRID;  // 280 / 28 = 10px per cell
let drawing = false;

/** 28×28 grid of pixel intensities (0–255). */
const grid = new Uint8Array(GRID * GRID);

/** Render the pixel grid onto the canvas. */
function renderGrid() {
  const img = ctx.createImageData(canvas.width, canvas.height);
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const v = grid[gy * GRID + gx];
      const x0 = Math.round(gx * CELL);
      const y0 = Math.round(gy * CELL);
      const x1 = Math.round((gx + 1) * CELL);
      const y1 = Math.round((gy + 1) * CELL);
      for (let py = y0; py < y1; py++) {
        for (let px = x0; px < x1; px++) {
          const i = (py * canvas.width + px) * 4;
          img.data[i] = v; img.data[i + 1] = v; img.data[i + 2] = v; img.data[i + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  // Draw grid lines
  var gridColor = getComputedStyle(document.documentElement).getPropertyValue('--surface-muted').trim();
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 0.5;
  for (let i = 1; i < GRID; i++) {
    const pos = Math.round(i * CELL) + 0.5;
    ctx.beginPath(); ctx.moveTo(pos, 0); ctx.lineTo(pos, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(canvas.width, pos); ctx.stroke();
  }
}

function clearCanvas() {
  grid.fill(0);
  renderGrid();
}
clearCanvas();

function getGridPos(e) {
  const rect = canvas.getBoundingClientRect();
  const src  = e.touches ? e.touches[0] : e;
  const gx = Math.floor(((src.clientX - rect.left) / rect.width) * GRID);
  const gy = Math.floor(((src.clientY - rect.top) / rect.height) * GRID);
  return { gx: Math.max(0, Math.min(GRID - 1, gx)), gy: Math.max(0, Math.min(GRID - 1, gy)) };
}

function paintPixel(e) {
  if (!drawing) return;
  e.preventDefault();
  const { gx, gy } = getGridPos(e);
  // Paint a soft 3×3 brush: centre=255, neighbours=128
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const nx = gx + dx, ny = gy + dy;
      if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) continue;
      const idx = ny * GRID + nx;
      const add = (dx === 0 && dy === 0) ? 255 : 128;
      grid[idx] = Math.min(255, grid[idx] + add);
    }
  }
  renderGrid();
}

function strokeEnd() {
  drawing = false;
  if (UI_CONFIG.input_type === "image") scheduleAutoPredict();
}

canvas.addEventListener("mousedown",  (e) => { drawing = true; paintPixel(e); });
canvas.addEventListener("mousemove",  paintPixel);
canvas.addEventListener("mouseup",    strokeEnd);
canvas.addEventListener("mouseleave", () => { drawing = false; });
canvas.addEventListener("touchstart", (e) => { drawing = true; paintPixel(e); }, { passive: false });
canvas.addEventListener("touchmove",  paintPixel, { passive: false });
canvas.addEventListener("touchend",   strokeEnd);

// ── Utilities ─────────────────────────────────────────────────────────────────

function pct(v)       { return (v * 100).toFixed(1) + "%"; }
function accClass(v)  { return v >= 0.95 ? "acc-high" : v >= 0.80 ? "acc-med" : "acc-low"; }
function confClass(v) { return v >= 0.80 ? "conf-high" : "conf-low"; }

// ── Session models list (MODELS card) ────────────────────────────────────────

function buildSessionModelsList() {
  const names = Object.keys(state.models);
  if (names.length === 0) {
    sessionModels.innerHTML = '<p class="ui-list-empty">No models loaded</p>';
    updateTeacherSelect();
    updateEnsembleBtn();
    return;
  }
  sessionModels.innerHTML = "";
  for (const name of names) {
    const m   = state.models[name];
    const paramsStr = m.num_params ? `${m.num_params.toLocaleString()} params` : "";
    const row = document.createElement("div");
    row.className = "ui-list-row";
    row.innerHTML = `
      <span class="ui-list-name">${name}</span>
      <span class="ui-list-tag">${m.model_type}</span>
      ${paramsStr ? `<span class="ui-list-tag">${paramsStr}</span>` : ""}
      <button class="btn btn-icon btn-sm" data-ablation="${name}" title="Ablation study">⊘</button>
      <button class="btn btn-icon"   data-export="${name}" title="Save to disk">${ICONS.save}</button>
      <button class="btn btn-danger" data-remove="${name}" aria-label="Remove ${name}">${ICONS.close}</button>`;
    sessionModels.appendChild(row);
  }
  updateTeacherSelect();
  updateEnsembleBtn();
}

// ── Prediction table (TRY card) ───────────────────────────────────────────────

function buildPredictionTable() {
  const names = Object.keys(state.models);
  if (names.length === 0) {
    predBody.innerHTML = `<tr class="empty-row"><td colspan="3">No prediction yet</td></tr>`;
    return;
  }
  predBody.innerHTML = "";
  for (const name of names) {
    const p  = state.predictions[name];
    const tr = document.createElement("tr");
    const nameTd = document.createElement("td");
    nameTd.className  = "pred-model-name";
    nameTd.textContent = name;
    const predTd = document.createElement("td");
    predTd.innerHTML = p ? `<span class="pred-label">${p.prediction}</span>` : "—";
    const confTd = document.createElement("td");
    confTd.innerHTML = p
      ? `<span class="${confClass(p.confidence)}">${pct(p.confidence)}</span>` : "—";
    tr.appendChild(nameTd);
    tr.appendChild(predTd);
    tr.appendChild(confTd);
    predBody.appendChild(tr);
  }
}

// ── Columnar metrics table (TEST card) ───────────────────────────────────────

function buildMetricsTable() {
  const names  = Object.keys(state.models);
  const labels = UI_CONFIG.class_labels;

  if (names.length === 0) {
    metricsHead.innerHTML = "";
    metricsBody.innerHTML = `<tr class="empty-row"><td>No models trained yet</td></tr>`;
    return;
  }

  const htr    = document.createElement("tr");
  const corner = document.createElement("th");
  corner.className = "corner-cell";
  htr.appendChild(corner);
  for (const name of names) {
    const th   = document.createElement("th");
    th.scope   = "col";
    const head = document.createElement("div");
    head.className = "model-col-head";
    head.innerHTML = `<span class="col-model-name">${name}</span>`;
    th.appendChild(head);
    htr.appendChild(th);
  }
  metricsHead.innerHTML = "";
  metricsHead.appendChild(htr);

  const sections = [
    {
      label: "Config",
      rows: [
        { key: "Type",   fn: (m) => m.model_type },
        { key: "Epochs", fn: (m) => m.epochs,     cls: "cfg-cell" },
        { key: "Batch",  fn: (m) => m.batch_size, cls: "cfg-cell" },
        { key: "LR",     fn: (m) => m.lr != null
            ? parseFloat(m.lr.toPrecision(4)).toString() : "—", cls: "cfg-cell" },
        { key: "Params", fn: (m) => m.num_params
            ? m.num_params.toLocaleString() : "—", cls: "cfg-cell" },
        { key: "Early Stop", fn: (m) => m.stopped_early ? "Yes" : "—", cls: "cfg-cell" },
      ],
    },
    {
      label: "Evaluation",
      rows: [
        {
          key: "Test Acc",
          fn: (m) => m.eval_result
            ? `<span class="${accClass(m.eval_result.accuracy)}">${pct(m.eval_result.accuracy)}</span>` : "—",
          html: true,
        },
        {
          key: "Test Loss",
          fn: (m) => m.eval_result ? m.eval_result.avg_loss.toFixed(4) : "—",
        },
      ],
    },
    {
      label: "Per-Class Accuracy",
      rows: labels.map(label => ({
        key:  label,
        fn:   (m) => {
          if (!m.eval_result) return "—";
          const acc = m.eval_result.per_class_accuracy[label];
          return acc != null
            ? `<span class="${accClass(acc)}">${pct(acc)}</span>` : "—";
        },
        html: true,
      })),
    },
  ];

  metricsBody.innerHTML = "";
  const ncols = names.length + 1;

  for (const section of sections) {
    const sepTr = document.createElement("tr");
    sepTr.className = "metrics-section-row";
    const sepTd = document.createElement("td");
    sepTd.colSpan    = ncols;
    sepTd.textContent = section.label;
    sepTr.appendChild(sepTd);
    metricsBody.appendChild(sepTr);

    for (const row of section.rows) {
      const tr      = document.createElement("tr");
      const labelTh = document.createElement("th");
      labelTh.scope     = "row";
      labelTh.className = "metric-label";
      labelTh.textContent = row.key;
      tr.appendChild(labelTh);
      for (const name of names) {
        const m  = state.models[name];
        const td = document.createElement("td");
        if (row.cls) td.className = row.cls;
        const val = row.fn(m);
        if (row.html) td.innerHTML   = val ?? "—";
        else          td.textContent = val ?? "—";
        tr.appendChild(td);
      }
      metricsBody.appendChild(tr);
    }
  }
}

// ── Load models from server on page load ──────────────────────────────────────

async function loadModels() {
  try {
    const res  = await fetch(`${BASE}/models`);
    const data = await res.json();
    for (const [name, info] of Object.entries(data)) state.models[name] = info;
    buildMetricsTable();
    buildPredictionTable();
    buildSessionModelsList();
    modelNameInput.value = defaultName(document.getElementById("model-type").value);
  } catch (_) { /* silent */ }
}

// ── Evaluate all session models ───────────────────────────────────────────────

async function runEvaluate() {
  if (Object.keys(state.models).length === 0) return;
  evalProgress.classList.remove("hidden");
  evalBar.style.width = "5%";
  evalBar.setAttribute("aria-valuenow", "5");
  evalStatus.textContent = "Starting evaluation…";
  let batchesDone  = 0;
  const approxBatches = 10 * Object.keys(state.models).length;
  await consumeSSE(
    `${BASE}/evaluate`, {},
    (msg) => {
      evalStatus.textContent = msg;
      batchesDone++;
      const p = Math.min(95, (batchesDone / approxBatches) * 100);
      evalBar.style.width = p + "%";
      evalBar.setAttribute("aria-valuenow", Math.round(p));
    },
    (event) => {
      evalBar.style.width = "100%";
      evalBar.setAttribute("aria-valuenow", "100");
      evalStatus.textContent = "Evaluation complete!";
      for (const [name, result] of Object.entries(event.results)) {
        if (state.models[name]) state.models[name].eval_result = result;
      }
      buildMetricsTable();
      setTimeout(() => {
        evalProgress.classList.add("hidden");
        evalBar.style.width = "0%";
        evalBar.setAttribute("aria-valuenow", "0");
      }, 1500);
    },
    (err) => {
      evalStatus.textContent = `Error: ${err}`;
      setTimeout(() => evalProgress.classList.add("hidden"), 2000);
    }
  );
}

// ── Train ─────────────────────────────────────────────────────────────────────

/** Colours for training curve series (one per model). */
const cs = getComputedStyle(document.documentElement);
const SERIES_COLORS = [
  cs.getPropertyValue('--danger').trim() || "#ef4444",
  cs.getPropertyValue('--success').trim() || "#22c55e",
  cs.getPropertyValue('--accent-teal').trim() || "#3b82f6",
  cs.getPropertyValue('--accent').trim() || "#f59e0b",
  cs.getPropertyValue('--syntax-keyword').trim() || "#a855f7",
  cs.getPropertyValue('--accent-olive').trim() || "#ec4899",
];
let seriesColorIdx = 0;

trainBtn.addEventListener("click", async () => {
  const modelType = document.getElementById("model-type").value;
  const epochs    = parseInt(document.getElementById("epochs").value, 10);
  const batchSize = parseInt(document.getElementById("batch-size").value, 10);
  const lr        = parseFloat(document.getElementById("lr").value);
  const name      = modelNameInput.value.trim() || defaultName(modelType);

  // Advanced options
  const patienceEl = document.getElementById("patience");
  const valGapEl   = document.getElementById("val-gap");
  const patience   = patienceEl.value ? parseInt(patienceEl.value, 10) : null;
  const valGap     = parseInt(valGapEl.value, 10) || 50;
  const teacher    = teacherSelect.value || null;
  const distillW   = parseFloat(document.getElementById("distill-weight").value) || 0.5;

  const body = { model_type: modelType, epochs, batch_size: batchSize, lr, name };
  if (patience != null) body.patience = patience;
  if (patience != null || teacher) body.val_gap = valGap;
  if (teacher) { body.teacher = teacher; body.distill_weight = distillW; }

  logTerminal.innerHTML = "";
  addLog(`Training '${name}'  ·  ${modelType}  ·  ${epochs} epoch${epochs !== 1 ? "s" : ""}  ·  lr ${lr}`);
  if (patience != null) addLog(`Early stopping: patience=${patience}, val every ${valGap} batches`);
  if (teacher) addLog(`Distillation: teacher='${teacher}', α=${distillW}`);

  trainBtn.disabled = true;
  trainBtn.classList.add("btn-loading");
  trainBtn.innerHTML = `${ICONS.spinner} Training…`;

  // Prepare chart for training curves
  const useChart = patience != null || teacher;
  if (useChart) {
    if (!trainChart) {
      trainChart = new MiniChart(trainChartCanvas, {
        title: "Training Curves",
        yLabel: "Loss",
        y2Label: "Val Accuracy",
      });
    }
    const c = SERIES_COLORS[seriesColorIdx++ % SERIES_COLORS.length];
    const c2 = SERIES_COLORS[seriesColorIdx++ % SERIES_COLORS.length];
    trainChart.addSeries(`${name} loss`, c, "left");
    trainChart.addSeries(`${name} acc`, c2, "right");
    chartArea.classList.remove("hidden");
  }

  let historyStep = 0;
  let trainedName = null;

  await consumeSSE(
    `${BASE}/train`, body,
    (msg) => {
      if (typeof msg === "object" && msg.train_loss != null) {
        // History event from training
        historyStep++;
        if (useChart && trainChart) {
          trainChart.addPoint(`${name} loss`, historyStep, msg.train_loss);
          if (msg.val_accuracy != null) {
            trainChart.addPoint(`${name} acc`, historyStep, msg.val_accuracy);
          }
          trainChart.render();
        }
        addLog(`loss: ${msg.train_loss.toFixed(4)}` +
          (msg.val_accuracy != null ? `  val_acc: ${(msg.val_accuracy * 100).toFixed(1)}%` : ""));
      } else if (typeof msg === "string") {
        addLog(msg);
      }
    },
    (event) => {
      state.models[event.name] = {
        model_type:       event.model_type,
        epochs:           event.epochs,
        batch_size:       event.batch_size,
        lr:               event.lr,
        num_params:       event.num_params || null,
        training_history: event.history || [],
        stopped_early:    event.stopped_early || false,
        eval_result:      null,
      };
      trainedName = event.name;
      if (event.stopped_early) addLog(`Early stopping triggered at epoch ${event.epochs_completed}`, "ok");
      if (event.best_val_accuracy != null) addLog(`Best val accuracy: ${(event.best_val_accuracy * 100).toFixed(1)}%`, "ok");
      buildMetricsTable();
      buildPredictionTable();
      buildSessionModelsList();
      modelNameInput.value = defaultName(document.getElementById("model-type").value);
    },
    (err) => { addLog(`Error: ${err}`, "err"); }
  );

  trainBtn.disabled = false;
  trainBtn.classList.remove("btn-loading");
  trainBtn.innerHTML = `${ICONS.play} Train`;

  if (trainedName) {
    addLog(`'${trainedName}' trained successfully`, "ok");
    await runEvaluate();
  }
});

// ── Predict ───────────────────────────────────────────────────────────────────

let autoPredictTimer = null;
function scheduleAutoPredict() {
  if (Object.keys(state.models).length === 0) return;
  clearTimeout(autoPredictTimer);
  autoPredictTimer = setTimeout(runPredict, 250);
}

async function runPredict() {
  if (Object.keys(state.models).length === 0) return;
  let body;
  if (UI_CONFIG.input_type === "image") {
    const b64 = canvas.toDataURL("image/png").replace(/^data:image\/png;base64,/, "");
    body = { image: b64 };
  } else {
    const features = {};
    document.querySelectorAll(".feature-input").forEach(inp => {
      features[inp.dataset.feature] = parseFloat(inp.value) || 0;
    });
    body = { features };
  }
  try {
    const res  = await fetch(`${BASE}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) return;
    Object.assign(state.predictions, data.results);
    buildPredictionTable();
  } catch (_) { /* silent */ }
}

if (predictBtn) predictBtn.addEventListener("click", runPredict);

clearBtn.addEventListener("click", () => {
  clearCanvas();
  state.predictions = {};
  buildPredictionTable();
});

// ── Saved models on disk ──────────────────────────────────────────────────────

async function loadSavedModels() {
  try {
    const res   = await fetch(`${BASE}/models/disk`);
    const files = await res.json();
    savedSelect.innerHTML = "";
    if (files.length === 0) {
      savedSelect.innerHTML = '<option value="">— no saved models —</option>';
      importBtn.disabled = true;
    } else {
      savedSelect.innerHTML = '<option value="">— select a saved model —</option>';
      for (const f of files) {
        const opt = document.createElement("option");
        opt.value      = f.filename;
        opt.textContent = `${f.name}  (${f.model_type}, ${f.epochs} ep)`;
        savedSelect.appendChild(opt);
      }
      importBtn.disabled = false;
    }
  } catch (_) { /* silent */ }
}
savedSelect.addEventListener("change", () => { importBtn.disabled = !savedSelect.value; });
refreshSavedBtn.addEventListener("click", loadSavedModels);

// ── Import from disk ──────────────────────────────────────────────────────────

importBtn.addEventListener("click", async () => {
  const filename = savedSelect.value;
  if (!filename) return;
  importBtn.disabled = true;
  try {
    const res  = await fetch(`${BASE}/models/disk/${encodeURIComponent(filename)}/load`, { method: "POST" });
    const data = await res.json();
    if (!res.ok || data.error) return;
    state.models[data.name] = {
      model_type:  data.model_type,
      epochs:      data.epochs,
      batch_size:  data.batch_size,
      lr:          data.lr,
      eval_result: null,
    };
    buildMetricsTable();
    buildPredictionTable();
    buildSessionModelsList();
    modelNameInput.value = defaultName(document.getElementById("model-type").value);
    await runEvaluate();
  } catch (_) { /* silent */ } finally {
    importBtn.disabled = !savedSelect.value;
  }
});

// ── Export (delegated save buttons in session rows) ───────────────────────────

document.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-export]");
  const exportName = btn?.dataset?.export;
  if (!exportName) return;
  btn.disabled = true;
  try {
    const res  = await fetch(`${BASE}/models/${encodeURIComponent(exportName)}/export`, { method: "POST" });
    const data = await res.json();
    if (res.ok && !data.error) await loadSavedModels();
  } catch (_) { /* silent */ } finally {
    btn.disabled = false;
  }
});

// ── Remove model from session (delegated close buttons) ──────────────────────

document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-remove]");
  const name = btn?.dataset?.remove;
  if (!name) return;
  delete state.models[name];
  delete state.predictions[name];
  buildMetricsTable();
  buildPredictionTable();
  buildSessionModelsList();
  modelNameInput.value = defaultName(document.getElementById("model-type").value);
});

// ── Ensemble ──────────────────────────────────────────────────────────────────

ensembleBtn.addEventListener("click", async () => {
  const names = Object.keys(state.models);
  if (names.length < 2) return;
  ensembleBtn.disabled = true;
  ensembleBtn.textContent = "Running…";
  try {
    const res = await fetch(`${BASE}/ensemble`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_names: names }),
    });
    const data = await res.json();
    if (data.error) {
      addLog(`Ensemble error: ${data.error}`, "err");
      return;
    }
    addLog(`Ensemble accuracy: ${(data.accuracy * 100).toFixed(1)}%`, "ok");
    // Store as a virtual model for display
    state.models["Ensemble"] = {
      model_type: "Ensemble",
      epochs: "—",
      batch_size: "—",
      lr: null,
      num_params: null,
      training_history: [],
      eval_result: {
        accuracy: data.accuracy,
        avg_loss: data.avg_loss,
        per_class_accuracy: data.per_class_accuracy,
        num_params: null,
      },
    };
    buildMetricsTable();
  } catch (err) {
    addLog(`Ensemble error: ${err}`, "err");
  } finally {
    ensembleBtn.disabled = false;
    ensembleBtn.textContent = "Ensemble";
  }
});

// ── Ablation (delegated from session model rows) ──────────────────────────────

document.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-ablation]");
  const modelName = btn?.dataset?.ablation;
  if (!modelName) return;
  btn.disabled = true;
  addLog(`Running ablation study on '${modelName}'…`);

  await consumeSSE(
    `${BASE}/ablation`,
    { model_name: modelName },
    (msg) => {
      if (typeof msg === "object" && msg.type === "ablation_result") {
        addLog(`  ${msg.layer}: acc=${(msg.accuracy * 100).toFixed(1)}%, drop=${(msg.drop * 100).toFixed(1)}%`);
      } else if (typeof msg === "string") {
        addLog(msg);
      }
    },
    (event) => {
      addLog(`Ablation complete for '${modelName}'`, "ok");
    },
    (err) => { addLog(`Ablation error: ${err}`, "err"); }
  );
  btn.disabled = false;
});

// ── Init ──────────────────────────────────────────────────────────────────────

loadModels();
loadSavedModels();
modelNameInput.value = defaultName(document.getElementById("model-type").value);
fetchModelInfo(document.getElementById("model-type").value);
