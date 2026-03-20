/**
 * @file Minimal dual-axis line chart for training curves.
 *
 * Renders loss (left Y-axis) and accuracy (right Y-axis) on a single canvas.
 * No external dependencies — uses the Canvas 2D API directly.
 */
"use strict";

class MiniChart {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{ title?: string, yLabel?: string, y2Label?: string }} opts
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.title = opts.title || "";
    this.yLabel = opts.yLabel || "Loss";
    this.y2Label = opts.y2Label || "Accuracy";
    this.series = {};
    this.padding = { top: 30, right: 55, bottom: 30, left: 55 };
  }

  /**
   * Register a named series.
   * @param {string} name
   * @param {string} color  CSS colour string
   * @param {"left"|"right"} yAxis
   */
  addSeries(name, color, yAxis = "left") {
    this.series[name] = { color, yAxis, points: [] };
  }

  /**
   * Append a data point to a series.
   * @param {string} seriesName
   * @param {number} x
   * @param {number} y
   */
  addPoint(seriesName, x, y) {
    const s = this.series[seriesName];
    if (s) s.points.push({ x, y });
  }

  /** Remove all data points (keeps series definitions). */
  clear() {
    for (const s of Object.values(this.series)) s.points = [];
  }

  /** Redraw the chart. */
  render() {
    const { canvas, ctx, padding: p } = this;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    // Colours from CSS custom properties
    const style = getComputedStyle(document.documentElement);
    const bg = style.getPropertyValue("--surface").trim() || "#3a3830";
    const textCol = style.getPropertyValue("--text-muted").trim() || "#7c7160";
    const gridCol = style.getPropertyValue("--border").trim() || "#504d40";

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Compute ranges
    const leftSeries = Object.values(this.series).filter(s => s.yAxis === "left" && s.points.length > 0);
    const rightSeries = Object.values(this.series).filter(s => s.yAxis === "right" && s.points.length > 0);
    const allPoints = Object.values(this.series).flatMap(s => s.points);

    if (allPoints.length === 0) {
      ctx.fillStyle = textCol;
      ctx.font = "12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No data yet", W / 2, H / 2);
      return;
    }

    const xMin = Math.min(...allPoints.map(p => p.x));
    const xMax = Math.max(...allPoints.map(p => p.x));
    const leftMin = leftSeries.length ? Math.min(...leftSeries.flatMap(s => s.points.map(p => p.y))) : 0;
    const leftMax = leftSeries.length ? Math.max(...leftSeries.flatMap(s => s.points.map(p => p.y))) : 1;
    const rightMin = 0;
    const rightMax = 1;

    const plotW = W - p.left - p.right;
    const plotH = H - p.top - p.bottom;

    const scaleX = (v) => p.left + (xMax > xMin ? ((v - xMin) / (xMax - xMin)) * plotW : plotW / 2);
    const scaleYL = (v) => {
      const range = leftMax - leftMin || 1;
      return p.top + plotH - ((v - leftMin) / range) * plotH;
    };
    const scaleYR = (v) => p.top + plotH - ((v - rightMin) / (rightMax - rightMin)) * plotH;

    // Grid
    ctx.strokeStyle = gridCol;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = p.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(p.left, y);
      ctx.lineTo(W - p.right, y);
      ctx.stroke();
    }

    // Axes labels
    ctx.fillStyle = textCol;
    ctx.font = "10px Inter, sans-serif";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const y = p.top + (plotH / 4) * i;
      const range = leftMax - leftMin || 1;
      const val = leftMax - (range / 4) * i;
      ctx.fillText(val.toFixed(3), p.left - 5, y + 3);
    }
    ctx.textAlign = "left";
    for (let i = 0; i <= 4; i++) {
      const y = p.top + (plotH / 4) * i;
      const val = 1 - (i / 4);
      ctx.fillText((val * 100).toFixed(0) + "%", W - p.right + 5, y + 3);
    }

    // Title
    if (this.title) {
      ctx.fillStyle = textCol;
      ctx.font = "bold 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(this.title, W / 2, 14);
    }

    // Y-axis labels
    ctx.save();
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = textCol;
    ctx.translate(10, p.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(this.yLabel, 0, 0);
    ctx.restore();
    ctx.save();
    ctx.translate(W - 8, p.top + plotH / 2);
    ctx.rotate(Math.PI / 2);
    ctx.fillText(this.y2Label, 0, 0);
    ctx.restore();

    // Draw series
    for (const [name, s] of Object.entries(this.series)) {
      if (s.points.length === 0) continue;
      const scaleFn = s.yAxis === "left" ? scaleYL : scaleYR;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < s.points.length; i++) {
        const px = scaleX(s.points[i].x);
        const py = scaleFn(s.points[i].y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Draw dots
      ctx.fillStyle = s.color;
      for (const pt of s.points) {
        ctx.beginPath();
        ctx.arc(scaleX(pt.x), scaleFn(pt.y), 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Legend
    const legendItems = Object.entries(this.series).filter(([, s]) => s.points.length > 0);
    if (legendItems.length > 0) {
      ctx.font = "9px Inter, sans-serif";
      ctx.textAlign = "left";
      let lx = p.left + 5;
      const ly = H - 6;
      for (const [name, s] of legendItems) {
        ctx.fillStyle = s.color;
        ctx.fillRect(lx, ly - 5, 10, 3);
        lx += 14;
        ctx.fillStyle = textCol;
        ctx.fillText(name, lx, ly);
        lx += ctx.measureText(name).width + 12;
      }
    }
  }
}
