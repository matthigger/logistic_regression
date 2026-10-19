// Data panels, shared by both tabs: the log-odds view (x against z, where
// the model is the line z = c0 + c1 x) and the probability view (x against
// p, where it is the sigmoid p = sigma(z)).
//
// Both draw the same samples: filled by their true class y and ringed by
// their predicted class (1 when z > 0), so a mistake is a dot of one color
// in a ring of the other. Two square handles sit on the model at the
// 25th/75th x-percentiles: drag one up or down to pivot the line about the
// other. In the probability view the handle moves p at its x, which is the
// same knob through z = logit(p). Dragging empty space changes c0 alone:
// up or down in the log-odds view, left or right in the probability view.
// Drag a sample sideways to move it; click it to flip its class. A strip
// above them (DataStrip) shows the raw data (x_i, y_i) without the model.

const DP = { w: 560, h: 560, x0: 62, x1: 498, y0: 14, y1: 498 };
const Z_LIM = 6;
const P_LO = -0.08, P_HI = 1.08;
// Log odds whose probabilities the guides mark: z = -/+2, p = 0.12 / 0.88.
const Z_GUIDE = 2;

class DataPanel {
  /**
   * Args:
   *   svg (SVGElement): the panel's SVG
   *   opts (object): { kind: "odds" | "prob", state(): the current
   *     { c, best, truth, showBest, showTruth, hover, handles },
   *     onSetC(c): the model was dragged, onHover(i): the hovered sample
   *     changed (i null when none), onDataDone(): a sample drag ended }
   */
  constructor(svg, opts) {
    this.svg = svg;
    this.opts = opts;
    this.kind = opts.kind;
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

  sx(v) {
    return DP.x0 + (v - this.dom[0]) / (this.dom[1] - this.dom[0])
      * (DP.x1 - DP.x0);
  }
  ix(px) {
    return this.dom[0] + (px - DP.x0) / (DP.x1 - DP.x0)
      * (this.dom[1] - this.dom[0]);
  }
  /** Screen y of a value on this panel's vertical axis (z or p). */
  sv(v) {
    const [lo, hi] = this.kind === "odds" ? [-Z_LIM, Z_LIM] : [P_LO, P_HI];
    return DP.y1 - (clamp(v, lo, hi) - lo) / (hi - lo) * (DP.y1 - DP.y0);
  }
  iv(py) {
    const [lo, hi] = this.kind === "odds" ? [-Z_LIM, Z_LIM] : [P_LO, P_HI];
    return lo + (DP.y1 - py) / (DP.y1 - DP.y0) * (hi - lo);
  }
  /** The model's value at x on this panel's axis: z, or p = sigma(z). */
  val(c, x) {
    const z = c[0] + c[1] * x;
    return this.kind === "odds" ? z : sigmoid(z);
  }

  render() {
    const st = this.opts.state(), svg = this.svg, odds = this.kind === "odds";
    const c = st.c, x = Data.x, y = Data.y;
    svg.innerHTML = "";
    svg.classList.toggle("odds", odds);
    const defs = node("defs", {}, svg);
    const clip = node("clipPath", { id: `${svg.id}-clip` }, defs);
    node("rect", { x: DP.x0, y: DP.y0, width: DP.x1 - DP.x0,
      height: DP.y1 - DP.y0 }, clip);

    for (const t of ticks(...this.dom, 6)) {
      node("line", { x1: this.sx(t), x2: this.sx(t), y1: DP.y0, y2: DP.y1,
        class: "grid" }, svg);
      text(svg, this.sx(t), DP.y1 + 17, fmt(t),
        { class: "tick", "text-anchor": "middle" });
    }
    const vt = odds ? [-6, -4, -2, 0, 2, 4, 6] : [0, 0.25, 0.5, 0.75, 1];
    for (const t of vt) {
      node("line", { x1: DP.x0, x2: DP.x1, y1: this.sv(t), y2: this.sv(t),
        class: "grid" }, svg);
      text(svg, DP.x0 - 6, this.sv(t) + 4, fmt(t),
        { class: "tick", "text-anchor": "end" });
    }
    if (odds) this.probAxis(svg);

    const g = node("g", { "clip-path": `url(#${svg.id}-clip)` }, svg);
    node("rect", { x: DP.x0, y: DP.y0, width: DP.x1 - DP.x0,
      height: DP.y1 - DP.y0, class: "dp-bg" }, g);
    this.guides(g);
    this.boundary(g, c);

    const z = x.map(v => c[0] + c[1] * v);
    if (!odds) {
      x.forEach((v, i) => node("line", { x1: this.sx(v), x2: this.sx(v),
        y1: this.sv(y[i]), y2: this.sv(sigmoid(z[i])), class: "err" }, g));
    }
    if (st.showTruth) this.curve(g, st.truth, "truth");
    if (st.showBest && st.best) this.curve(g, st.best, "best");
    this.curve(g, c, "model");
    if (!odds) {
      x.forEach((v, i) => node("circle", { cx: this.sx(v),
        cy: this.sv(sigmoid(z[i])), r: 3, class: `phat k${y[i]}` }, g));
    }
    // Hovered sample last, so its ring sits on top.
    const order = x.map((_, i) => i).filter(i => i !== st.hover);
    if (st.hover !== null && st.hover < x.length) order.push(st.hover);
    for (const i of order) {
      const v = odds ? z[i] : y[i], pred = z[i] > 0 ? 1 : 0;
      const off = odds && Math.abs(v) > Z_LIM;
      node("circle", { cx: this.sx(x[i]), cy: this.sv(v), r: off ? 5 : 7,
        class: `s k${y[i]} p${pred}${i === st.hover ? " on" : ""}`,
        "data-i": i }, g);
    }
    // Handles above the samples, which sit on the line in the log-odds view.
    if (st.handles) {
      this.hx.forEach((hx, k) => {
        const py = this.sv(this.val(c, hx));
        if (py <= DP.y0 + 1 || py >= DP.y1 - 1) return;
        node("rect", { x: this.sx(hx) - 7, y: py - 7, width: 14, height: 14,
          class: "lh", "data-h": k }, g);
      });
    }
    if (st.hover !== null && st.hover < x.length) this.sampleTip(g, st, z);

    node("rect", { x: DP.x0, y: DP.y0, width: DP.x1 - DP.x0,
      height: DP.y1 - DP.y0, class: "frame" }, svg);
    text(svg, (DP.x0 + DP.x1) / 2, DP.y1 + 40, "x",
      { class: "axis-label sym", "text-anchor": "middle" });
    const yl = odds ? "log odds z" : "probability p̂";
    text(svg, 16, (DP.y0 + DP.y1) / 2, yl, { class: "axis-label",
      "text-anchor": "middle",
      transform: `rotate(-90 16 ${(DP.y0 + DP.y1) / 2})` });
  }

  /** Right-hand axis of the log-odds view: each z tick's probability. */
  probAxis(svg) {
    for (const t of [-4, -2, 0, 2, 4]) {
      text(svg, DP.x1 + 6, this.sv(t) + 4, fmt(sigmoid(t), 2).replace(
        /^0\./, "."), { class: "tick ptick" });
    }
    const my = (DP.y0 + DP.y1) / 2, mx = DP.w - 10;
    text(svg, mx, my, "p̂ = σ(z)", { class: "axis-label small",
      "text-anchor": "middle", transform: `rotate(90 ${mx} ${my})` });
  }

  /** z = 0 (p = 0.5) and z = -/+2 (p = 0.12 / 0.88), in both views. */
  guides(g) {
    const odds = this.kind === "odds";
    const at = z => this.sv(odds ? z : sigmoid(z));
    node("line", { x1: DP.x0, x2: DP.x1, y1: at(0), y2: at(0),
      class: "zero" }, g);
    for (const z of [-Z_GUIDE, Z_GUIDE]) {
      node("line", { x1: DP.x0, x2: DP.x1, y1: at(z), y2: at(z),
        class: "pguide" }, g);
    }
  }

  /** Vertical line where z = 0: the decision boundary x = -c0 / c1. */
  boundary(g, c) {
    if (Math.abs(c[1]) < 1e-9) return;
    const xb = -c[0] / c[1];
    if (xb < this.dom[0] || xb > this.dom[1]) return;
    node("line", { x1: this.sx(xb), x2: this.sx(xb), y1: DP.y0, y2: DP.y1,
      class: "bound" }, g);
    text(g, this.sx(xb) + 5, DP.y1 - 8, "boundary", { class: "bound-label" });
  }

  /** The model c as a line (log odds) or sigmoid (probability). */
  curve(g, c, cls) {
    const odds = this.kind === "odds", n = odds ? 60 : 240;
    const [lo, hi] = odds ? [-Z_LIM, Z_LIM] : [P_LO, P_HI];
    const pts = [];
    for (let k = 0; k <= n; k++) {
      const v = this.dom[0] + k / n * (this.dom[1] - this.dom[0]);
      // Unclamped (the clip path trims it), so a steep line keeps its
      // slope past the frame; bounded so huge c stays drawable.
      const w = clamp(this.val(c, v), -1e4, 1e4);
      const py = DP.y1 - (w - lo) / (hi - lo) * (DP.y1 - DP.y0);
      pts.push(`${this.sx(v).toFixed(1)},${py.toFixed(1)}`);
    }
    node("polyline", { points: pts.join(" "), class: `curve ${cls}` }, g);
  }

  /**
   * Label next to the hovered sample (its z, or its p and loss), with
   * dotted guides reading its value off the axes.
   */
  sampleTip(g, st, z) {
    const i = st.hover, x = Data.x[i], y = Data.y[i];
    const odds = this.kind === "odds";
    const p = sigmoid(z[i]);
    const gx = this.sx(x), gy = this.sv(odds ? z[i] : p);
    node("polyline", { points: `${gx},${DP.y1} ${gx},${gy} ${DP.x0},${gy}`,
      class: "readoff" }, g);
    const loss = y ? softplus(-z[i]) : softplus(z[i]);
    const str = odds ? `z = ${fmt(z[i])}`
      : `p̂ = ${fmt(p)}, loss = −ln ${y ? "p̂" : "(1 − p̂)"} = ${fmt(loss)}`;
    const px = this.sx(x), py = this.sv(odds ? z[i] : y);
    const right = px < (DP.x0 + DP.x1) / 2;
    text(g, px + (right ? 12 : -12), py + (y || odds ? 22 : -14), str,
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
          ? this.iv(p.y) : this.ix(p.x) };
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
    };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
  }

  move(p) {
    const d = this.drag, c = this.opts.state().c;
    if (d.i !== undefined) {
      if (!d.moved && Math.hypot(p.x - d.x0, p.y - d.y0) < 4) return;
      d.moved = true;
      Data.move(d.i, clamp(this.ix(p.x), ...this.dom));
    } else if (d.h !== undefined) {
      // Pivot about the other handle, which stays on the current model.
      const xa = this.hx[d.h], xb = this.hx[1 - d.h];
      const v = this.iv(p.y);
      const za = this.kind === "odds" ? clamp(v, -Z_LIM, Z_LIM)
        : logit(clamp(v, 0.003, 0.997));
      const zb = c[0] + c[1] * xb, c1 = (za - zb) / (xa - xb);
      this.opts.onSetC([za - c1 * xa, c1]);
    } else if (this.kind === "odds") {
      const v = this.iv(p.y);
      this.opts.onSetC([c[0] + v - d.at, c[1]]);
      d.at = v;
    } else {
      const v = this.ix(p.x);
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
    ? "Only one class: no best c."
    : "Separable: no sample overlaps the other class, so there is no "
      + "best c.";
}
