// Data panels, shared by both tabs: the log-odds view (x against z, where
// the model is the line z = w0 + w1 x) and the probability view (p against
// z, where the model is the fixed curve sigma, or against x). The weights,
// written w on the page, are c in the code.
//
// Both draw the same samples: filled by their true class y and ringed by
// their predicted class (1 when z > 0), so a mistake is a dot of one color
// in a ring of the other. Each axis has one set of grid lines, labeled on
// both scales: z lines with their p = sigma(z), p lines with their logit z.
// Two square handles sit on the model at the 25th/75th x-percentiles: drag
// one to pivot the line about the other (in the probability view it moves
// p, or z, at its x). Dragging empty space changes w0 alone. Drag a sample
// sideways to move it; click it to flip its class. A strip above them
// (DataStrip) shows the raw data (x_i, y_i) without the model.

const DP = { w: 560, h: 560, x0: 84, x1: 494, y0: 14, y1: 486 };
const P_LO = -0.08, P_HI = 1.08;
const P_TICKS = [0, 0.25, 0.5, 0.75, 1];
// Least |z| the log-odds axis always shows, so the sigmoid's bend is in view.
const Z_MIN = 6;

// Non-breaking spaces: plain ones collapse inside the rotated labels.
const Z_LABEL = '<i>z</i>&nbsp;=&nbsp;log&nbsp;odds&nbsp;=&nbsp;ln&nbsp;'
  + '<span class="frac"><span><i>P&#770;</i>(<i>Y</i>&nbsp;=&nbsp;1&nbsp;|'
  + '&nbsp;<i>x</i>)</span><span><i>P&#770;</i>(<i>Y</i>&nbsp;=&nbsp;0'
  + '&nbsp;|&nbsp;<i>x</i>)</span></span>';
const P_LABEL = '<i>p&#770;</i>&nbsp;=&nbsp;<i>P&#770;</i>(<i>Y</i>'
  + '&nbsp;=&nbsp;1&nbsp;|&nbsp;<i>x</i>)';

/** z range of weights c over x domain dom: the line, at least -/+Z_MIN. */
function zRange(c, dom) {
  const z = dom.map(v => c[0] + c[1] * v);
  const lo = Math.max(Math.min(...z, -Z_MIN), -1e4);
  const hi = Math.min(Math.max(...z, Z_MIN), 1e4);
  const pad = 0.05 * (hi - lo);
  return [lo - pad, hi + pad];
}

/** Round z ticks: every 2 on a short axis, else ticks(). */
function zTicks([lo, hi]) {
  if (hi - lo > 18) return ticks(lo, hi, 6);
  const out = [];
  for (let t = Math.ceil(lo / 2) * 2; t <= hi; t += 2) out.push(t);
  return out;
}

/** Probability for an axis label: ".88", ".018", ".998". */
function fmtP(p) { return fmt(p, p > 0.99 ? 3 : 2).replace(/^0\./, "."); }

/** HTML (fractions and all) as an SVG label centered on (cx, cy). */
function htmlLabel(parent, cx, cy, w, h, html, rotate = 0) {
  const fo = node("foreignObject", { x: cx - w / 2, y: cy - h / 2, width: w,
    height: h, class: "fo-label" }, parent);
  if (rotate) fo.setAttribute("transform", `rotate(${rotate} ${cx} ${cy})`);
  const div = document.createElementNS("http://www.w3.org/1999/xhtml", "div");
  div.className = "axis-html";
  div.innerHTML = html;
  fo.appendChild(div);
}

class DataPanel {
  /**
   * Args:
   *   svg (SVGElement): the panel's SVG
   *   opts (object): { kind: "odds" | "prob", axis: "x" | "z" (the
   *     probability view's horizontal axis), state(): the current
   *     { c, best, truth, showBest, showTruth, hover, handles, zr },
   *     onSetC(c): the model was dragged, onHover(i): the hovered sample
   *     changed (i null when none), onDataDone(): a sample drag ended,
   *     onDragEnd(): any drag ended }
   */
  constructor(svg, opts) {
    this.svg = svg;
    this.opts = opts;
    this.kind = opts.kind;
    this.axis = opts.axis || "x";
    this.drag = null;
    svg.setAttribute("viewBox", `0 0 ${DP.w} ${DP.h}`);
    this.fitX();
    this.listen();
  }

  /** Frame the samples' x range and place the line handles. */
  fitX() {
    const xs = Data.x;
    const lo = Math.min(...xs), hi = Math.max(...xs), pad = 0.08 * (hi - lo);
    this.dom = [lo - pad, hi + pad];
    this.handles();
  }

  handles() {
    const b = Data.x.slice().sort((p, q) => p - q);
    const at = p => b[Math.round(p * (b.length - 1))];
    this.hx = [at(0.25), at(0.75)];
  }

  /** Whether the horizontal axis is z (the sigmoid view). */
  get zh() { return this.kind === "prob" && this.axis === "z"; }
  hRange() { return this.zh ? this.zr : this.dom; }
  vRange() { return this.kind === "odds" ? this.zr : [P_LO, P_HI]; }

  sh(v) {
    const [lo, hi] = this.hRange();
    return DP.x0 + (v - lo) / (hi - lo) * (DP.x1 - DP.x0);
  }
  ih(px) {
    const [lo, hi] = this.hRange();
    return lo + (px - DP.x0) / (DP.x1 - DP.x0) * (hi - lo);
  }
  /** Screen y of a value on the vertical axis (z or p), unclamped. */
  sv(v) {
    const [lo, hi] = this.vRange();
    return DP.y1 - (clamp(v, -1e5, 1e5) - lo) / (hi - lo) * (DP.y1 - DP.y0);
  }
  iv(py) {
    const [lo, hi] = this.vRange();
    return lo + (DP.y1 - py) / (DP.y1 - DP.y0) * (hi - lo);
  }
  /** Screen point of a model point at feature x with log odds z. */
  at(x, z) {
    return [this.sh(this.zh ? z : x),
      this.sv(this.kind === "odds" ? z : sigmoid(z))];
  }

  render() {
    const st = this.opts.state(), svg = this.svg, odds = this.kind === "odds";
    const c = st.c, x = Data.x, y = Data.y;
    this.zr = st.zr || zRange(c, this.dom);
    svg.innerHTML = "";
    svg.classList.toggle("odds", odds);
    const defs = node("defs", {}, svg);
    const clip = node("clipPath", { id: `${svg.id}-clip` }, defs);
    node("rect", { x: DP.x0, y: DP.y0, width: DP.x1 - DP.x0,
      height: DP.y1 - DP.y0 }, clip);
    this.grid(svg);

    const g = node("g", { "clip-path": `url(#${svg.id}-clip)` }, svg);
    node("rect", { x: DP.x0, y: DP.y0, width: DP.x1 - DP.x0,
      height: DP.y1 - DP.y0, class: "dp-bg" }, g);
    this.boundary(g, c);

    const z = x.map(v => c[0] + c[1] * v);
    if (!odds) {
      x.forEach((v, i) => {
        const [px, py] = this.at(v, z[i]);
        node("line", { x1: px, x2: px, y1: this.sv(y[i]), y2: py,
          class: "err" }, g);
      });
    }
    if (this.zh) {
      this.sigmoidCurve(g);
    } else {
      if (st.showTruth) this.curve(g, st.truth, "truth");
      if (st.showBest && st.best) this.curve(g, st.best, "best");
      this.curve(g, c, "model");
    }
    if (!odds) {
      x.forEach((v, i) => {
        const [px, py] = this.at(v, z[i]);
        node("circle", { cx: px, cy: py, r: 3, class: `phat k${y[i]}` }, g);
      });
    }
    // Hovered sample last, so its ring sits on top.
    const order = x.map((_, i) => i).filter(i => i !== st.hover);
    if (st.hover !== null && st.hover < x.length) order.push(st.hover);
    for (const i of order) {
      const [px, py] = this.at(x[i], z[i]);
      const cy = odds ? py : this.sv(y[i]);
      node("circle", { cx: px, cy, r: 7,
        class: `s k${y[i]} p${z[i] > 0 ? 1 : 0}${i === st.hover ? " on" : ""}`,
        "data-i": i }, g);
    }
    // Handles above the samples, which sit on the line in the log-odds view.
    if (st.handles) {
      this.hx.forEach((hx, k) => {
        const [px, py] = this.at(hx, c[0] + c[1] * hx);
        if (px < DP.x0 || px > DP.x1 || py <= DP.y0 + 1
          || py >= DP.y1 - 1) return;
        node("rect", { x: px - 7, y: py - 7, width: 14, height: 14,
          class: `lh${this.zh ? " ew" : ""}`, "data-h": k }, g);
      });
    }
    if (st.hover !== null && st.hover < x.length) this.sampleTip(g, st, z);

    node("rect", { x: DP.x0, y: DP.y0, width: DP.x1 - DP.x0,
      height: DP.y1 - DP.y0, class: "frame" }, svg);
    this.axisLabels(svg);
  }

  /**
   * Grid lines, each labeled on both scales: z lines carry p = sigma(z)
   * on the right edge, p lines carry their logit z.
   */
  grid(svg) {
    const odds = this.kind === "odds";
    const ht = this.zh ? zTicks(this.zr) : ticks(...this.dom, 6);
    for (const t of ht) {
      const px = this.sh(t);
      node("line", { x1: px, x2: px, y1: DP.y0, y2: DP.y1,
        class: this.zh && t === 0 ? "zero" : "grid" }, svg);
      text(svg, px, DP.y1 + 17, fmt(t),
        { class: "tick", "text-anchor": "middle" });
    }
    const vt = odds ? zTicks(this.zr) : P_TICKS;
    let last = Infinity;
    for (const t of vt) {
      const py = this.sv(t);
      const mid = odds ? t === 0 : t === 0.5;
      node("line", { x1: DP.x0, x2: DP.x1, y1: py, y2: py,
        class: mid ? "zero" : "grid" }, svg);
      text(svg, DP.x0 - 6, py + 4, odds ? fmt(t) : fmtP(t),
        { class: "tick", "text-anchor": "end" });
      // The other scale: p for a z line (while it still reads), z for a p
      // line (none at p = 0 or 1).
      const other = odds ? (Math.abs(t) <= Z_MIN ? fmtP(sigmoid(t)) : null)
        : (t > 0 && t < 1 ? fmt(logit(t), 2) : null);
      if (other !== null && Math.abs(last - py) > 14) {
        text(svg, DP.x1 + 6, py + 4, other, { class: "tick ptick" });
        last = py;
      }
    }
    const my = (DP.y0 + DP.y1) / 2, mx = DP.w - 12;
    text(svg, mx, my, odds ? "p̂ = σ(z)" : "logit z = ln (p̂ / (1 − p̂))",
      { class: "axis-label small", "text-anchor": "middle",
        transform: `rotate(90 ${mx} ${my})` });
  }

  axisLabels(svg) {
    const my = (DP.y0 + DP.y1) / 2, mx = (DP.x0 + DP.x1) / 2;
    htmlLabel(svg, 24, my, DP.y1 - DP.y0, 46,
      this.kind === "odds" ? Z_LABEL : P_LABEL, -90);
    if (this.zh) {
      htmlLabel(svg, mx, DP.y1 + 50, DP.x1 - DP.x0, 46, Z_LABEL);
    } else {
      text(svg, mx, DP.y1 + 44, "x",
        { class: "axis-label sym", "text-anchor": "middle" });
    }
  }

  /** Vertical line where z = 0: the decision boundary x = -w0 / w1. */
  boundary(g, c) {
    let px;
    if (this.zh) {
      px = this.sh(0);
    } else {
      if (Math.abs(c[1]) < 1e-9) return;
      const xb = -c[0] / c[1];
      if (xb < this.dom[0] || xb > this.dom[1]) return;
      px = this.sh(xb);
      node("line", { x1: px, x2: px, y1: DP.y0, y2: DP.y1, class: "bound" },
        g);
    }
    text(g, px + 5, DP.y0 + 16, "boundary", { class: "bound-label" });
  }

  /** The model c as a line (log odds) or sigmoid (probability) over x. */
  curve(g, c, cls) {
    const n = this.kind === "odds" ? 60 : 240, pts = [];
    for (let k = 0; k <= n; k++) {
      const v = this.dom[0] + k / n * (this.dom[1] - this.dom[0]);
      const [px, py] = this.at(v, c[0] + c[1] * v);
      pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
    }
    node("polyline", { points: pts.join(" "), class: `curve ${cls}` }, g);
  }

  /** sigma(z) over the z axis: the same curve whatever the weights. */
  sigmoidCurve(g) {
    const [lo, hi] = this.zr, pts = [];
    for (let k = 0; k <= 240; k++) {
      const z = lo + k / 240 * (hi - lo);
      pts.push(`${this.sh(z).toFixed(1)},${this.sv(sigmoid(z)).toFixed(1)}`);
    }
    node("polyline", { points: pts.join(" "), class: "curve model" }, g);
  }

  /**
   * Label next to the hovered sample (its z, or its p and loss), with
   * dotted guides reading its value off the axes.
   */
  sampleTip(g, st, z) {
    const i = st.hover, x = Data.x[i], y = Data.y[i];
    const odds = this.kind === "odds";
    const p = sigmoid(z[i]);
    const [gx, gy] = this.at(x, z[i]);
    node("polyline", { points: `${gx},${DP.y1} ${gx},${gy} ${DP.x0},${gy}`,
      class: "readoff" }, g);
    const loss = y ? softplus(-z[i]) : softplus(z[i]);
    const str = odds ? `z = ${fmt(z[i])}`
      : `p̂ = ${fmt(p)}, loss = −ln ${y ? "p̂" : "(1 − p̂)"} = ${fmt(loss)}`;
    const py = odds ? gy : this.sv(y);
    const right = gx < (DP.x0 + DP.x1) / 2;
    text(g, gx + (right ? 12 : -12), py + (y || odds ? 22 : -14), str,
      { class: "tip-label", "text-anchor": right ? "start" : "end" });
  }

  // ---------------------------------------------------------- pointer

  listen() {
    const svg = this.svg;
    svg.addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      const p = svgPoint(svg, e), t = e.target;
      if (p.x < DP.x0 || p.x > DP.x1 || p.y < DP.y0 || p.y > DP.y1) return;
      if (t.classList.contains("s")) {
        this.drag = { i: +t.dataset.i, x0: p.x, y0: p.y, moved: false };
      } else if (t.classList.contains("lh")) {
        this.drag = { h: +t.dataset.h };
      } else if (this.opts.state().handles) {
        this.drag = { shift: true, at: this.kind === "odds"
          ? this.iv(p.y) : this.ih(p.x) };
      } else {
        return;
      }
      svg.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    svg.addEventListener("pointermove", e => {
      const p = svgPoint(svg, e);
      if (!this.drag) {
        const t = e.target;
        this.opts.onHover(t.classList.contains("s") ? +t.dataset.i : null);
        return;
      }
      this.move(p);
    });
    svg.addEventListener("pointerleave", () => {
      if (!this.drag) this.opts.onHover(null);
    });
    const end = () => {
      const d = this.drag;
      this.drag = null;
      if (!d) return;
      if (d.i !== undefined) {
        if (d.moved) this.opts.onDataDone();
        else Data.flip(d.i);
      }
      if (this.opts.onDragEnd) this.opts.onDragEnd();
    };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
  }

  move(p) {
    const d = this.drag, c = this.opts.state().c, [zlo, zhi] = this.zr;
    if (d.i !== undefined) {
      if (!d.moved && Math.hypot(p.x - d.x0, p.y - d.y0) < 4) return;
      d.moved = true;
      // On the z axis a sample's position is the model's, so it moves in
      // x by the same z.
      const v = this.ih(p.x);
      const xv = this.zh ? (Math.abs(c[1]) > 1e-9 ? (v - c[0]) / c[1]
        : Data.x[d.i]) : v;
      Data.move(d.i, clamp(xv, ...this.dom));
    } else if (d.h !== undefined) {
      // Pivot about the other handle, which stays on the current model.
      const xa = this.hx[d.h], xb = this.hx[1 - d.h];
      let za;
      if (this.kind === "odds") za = this.iv(p.y);
      else if (this.zh) za = this.ih(p.x);
      else za = logit(clamp(this.iv(p.y), 0.003, 0.997));
      za = clamp(za, zlo, zhi);
      const zb = c[0] + c[1] * xb, c1 = (za - zb) / (xa - xb);
      this.opts.onSetC([za - c1 * xa, c1]);
    } else if (this.kind === "odds" || this.zh) {
      // A shift in z: up/down in the log-odds view, sideways on the z axis.
      const v = this.kind === "odds" ? this.iv(p.y) : this.ih(p.x);
      this.opts.onSetC([c[0] + v - d.at, c[1]]);
      d.at = v;
    } else {
      const v = this.ih(p.x);
      this.opts.onSetC([c[0] - c[1] * (v - d.at), c[1]]);
      d.at = v;
    }
  }
}

// --------------------------------------------------------- data strip

const ST = { w: 1140, h: 124, x0: 62, x1: 1110, y0: 8, y1: 84 };

/**
 * The raw data (x_i, y_i): each sample on an x axis, in a lane for its
 * class. The model plays no part here. Drag a sample sideways to move it;
 * click it to flip its class.
 */
class DataStrip {
  /**
   * Args:
   *   svg (SVGElement): the strip's SVG
   *   opts (object): { state(): { hover }, onHover(i), onDataDone() }, as
   *     for DataPanel
   */
  constructor(svg, opts) {
    this.svg = svg;
    this.opts = opts;
    this.drag = null;
    svg.setAttribute("viewBox", `0 0 ${ST.w} ${ST.h}`);
    this.fitX();
    this.listen();
  }

  fitX() {
    const xs = Data.x;
    const lo = Math.min(...xs), hi = Math.max(...xs), pad = 0.08 * (hi - lo);
    this.dom = [lo - pad, hi + pad];
  }

  sx(v) {
    return ST.x0 + (v - this.dom[0]) / (this.dom[1] - this.dom[0])
      * (ST.x1 - ST.x0);
  }
  ix(px) {
    return this.dom[0] + (px - ST.x0) / (ST.x1 - ST.x0)
      * (this.dom[1] - this.dom[0]);
  }
  lane(y) { return y ? ST.y0 + 19 : ST.y1 - 19; }

  render() {
    const svg = this.svg, hover = this.opts.state().hover;
    const x = Data.x, y = Data.y;
    svg.innerHTML = "";
    for (const t of ticks(...this.dom, 12)) {
      node("line", { x1: this.sx(t), x2: this.sx(t), y1: ST.y0, y2: ST.y1,
        class: "grid" }, svg);
      text(svg, this.sx(t), ST.y1 + 17, fmt(t),
        { class: "tick", "text-anchor": "middle" });
    }
    node("rect", { x: ST.x0, y: ST.y0, width: ST.x1 - ST.x0,
      height: ST.y1 - ST.y0, class: "frame" }, svg);
    for (const k of [0, 1]) {
      node("line", { x1: ST.x0, x2: ST.x1, y1: this.lane(k),
        y2: this.lane(k), class: "lane" }, svg);
      text(svg, ST.x0 - 8, this.lane(k) + 5, `y = ${k}`,
        { class: "axis-label sym", "text-anchor": "end" });
    }
    text(svg, (ST.x0 + ST.x1) / 2, ST.y1 + 36, "x",
      { class: "axis-label sym", "text-anchor": "middle" });
    const order = x.map((_, i) => i).filter(i => i !== hover);
    if (hover !== null && hover < x.length) order.push(hover);
    for (const i of order) {
      node("circle", { cx: this.sx(x[i]), cy: this.lane(y[i]), r: 7,
        class: `s raw k${y[i]}${i === hover ? " on" : ""}`, "data-i": i },
        svg);
    }
  }

  listen() {
    const svg = this.svg;
    svg.addEventListener("pointerdown", e => {
      if (e.button !== 0 || !e.target.classList.contains("s")) return;
      const p = svgPoint(svg, e);
      this.drag = { i: +e.target.dataset.i, x0: p.x, moved: false };
      svg.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    svg.addEventListener("pointermove", e => {
      const d = this.drag;
      if (!d) {
        const t = e.target;
        this.opts.onHover(t.classList.contains("s") ? +t.dataset.i : null);
        return;
      }
      const p = svgPoint(svg, e);
      if (!d.moved && Math.abs(p.x - d.x0) < 4) return;
      d.moved = true;
      Data.move(d.i, clamp(this.ix(p.x), ...this.dom));
    });
    svg.addEventListener("pointerleave", () => {
      if (!this.drag) this.opts.onHover(null);
    });
    const end = () => {
      const d = this.drag;
      this.drag = null;
      if (!d) return;
      if (d.moved) this.opts.onDataDone();
      else Data.flip(d.i);
    };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
  }
}

// ------------------------------------------------------- data controls

/**
 * Wire one tab's copy of the data controls (ids prefixed pre-): the
 * separation slider, Resample and Reset. Both tabs share the data.
 */
function bindDataControls(pre) {
  const $ = id => document.getElementById(`${pre}-${id}`);
  $("sep").min = 0;
  $("sep").max = LR.sepMax;
  $("sep").step = 0.1;
  $("sep").oninput = e => Data.setSep(+e.target.value);
  $("resample").onclick = () => Data.resample();
  $("reset").onclick = () => Data.reset();
}

/** Show the shared data state in one tab's controls. */
function syncDataControls(pre) {
  const $ = id => document.getElementById(`${pre}-${id}`);
  $("sep").value = Data.sep;
  $("sepval").innerHTML = `means ${fmt(Data.sep)} sd apart`;
  const sep = separation(Data.x, Data.y);
  $("sepbadge").hidden = !sep;
  $("sepbadge").textContent = sep && !sep.side
    ? "Only one class: no best w."
    : "Separable: no sample overlaps the other class, so there is no "
      + "best w.";
}
