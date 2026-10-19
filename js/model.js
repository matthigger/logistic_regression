// Model tab: the weights w (c in the code) drawn twice, as the line
// z = w0 + w1 x on the log-odds scale (left) and through the sigmoid
// p = sigma(z) (right, against z or x). Dragging either moves both. The two
// share one z range, held still while anything is dragged.

const ModelTab = (function () {
  const S = { c: [0, 1], showBest: false, showTruth: false, hover: null,
    axis: "z", zr: null };
  let obj;

  const $ = id => document.getElementById(id);
  const state = () => ({ c: S.c, best: obj.cstar, truth: Data.truth(),
    showBest: S.showBest, showTruth: S.showTruth, hover: S.hover,
    handles: true, zr: S.zr });
  const opts = kind => ({
    kind, state, axis: S.axis,
    onDragEnd: () => render(),
    onSetC: c => { S.c = c; render(); },
    onHover: i => { if (i !== S.hover) { S.hover = i; render(); } },
    onDataDone: () => {
      panels.forEach(p => { if (p.handles) p.handles(); });
      render();
    },
  });
  const panels = [new DataPanel($("md-odds"), opts("odds")),
    new DataPanel($("md-prob"), opts("prob")),
    new DataStrip($("md-strip"), opts("strip"))];

  function visible() { return !$("md-odds").closest("[hidden]"); }

  Data.onChange(fresh => {
    obj = logisticLoss(Data.x, Data.y);
    if (fresh) panels.forEach(p => p.fitX());
    if (visible()) render();
  });

  function equation() {
    const sign = v => v < 0 ? "−" : "+";
    $("md-eq-odds").innerHTML = `<i>z</i> = <i>w</i>₀ + <i>w</i>₁<i>x</i> = `
      + `${fmt(S.c[0])} ${sign(S.c[1])} ${fmt(Math.abs(S.c[1]))}<i>x</i>`;
    $("md-eq-prob").innerHTML = "<i>p̂</i> = σ(<i>z</i>) = 1 / (1 + "
      + "<i>e</i><sup>−<i>z</i></sup>)";
  }

  function readout() {
    const x = Data.x, y = Data.y, n = x.length, c = S.c;
    const right = x.filter((v, i) => (c[0] + c[1] * v > 0 ? 1 : 0) === y[i])
      .length;
    const rows = [
      [`<b>w</b> = ${colVec(["<i>w</i>₀", "<i>w</i>₁"])}`, fmtCol(c)],
      ["boundary <i>x</i> = −<i>w</i>₀ / <i>w</i>₁",
        Math.abs(c[1]) > 1e-9 ? fmt(-c[0] / c[1]) : "none"],
      ["accuracy", `${right} of ${n}`],
      ["loss(<b>w</b>)", fmt(obj.f(c))],
      ["best loss", obj.cstar ? fmt(obj.fmin)
        : "none (separable)"],
    ];
    $("md-readout").innerHTML = rows.map(([k, v]) =>
      `<div class="row"><span>${k}</span><span class="val">${v}</span></div>`
    ).join("");
  }

  /** Where a sample's numbers come from: x, then z, then p̂, then loss. */
  function pipeline() {
    const i = S.hover, c = S.c, arr = '<span class="arr">→</span>';
    if (i === null || i >= Data.n) {
      $("md-pipe").innerHTML = "Each sample is a pair (<i>x<sub>i</sub></i>, "
        + "<i>y<sub>i</sub></i>). The model scores it with its log odds "
        + "<i>z<sub>i</sub></i> = <i>w</i>₀ + <i>w</i>₁<i>x<sub>i</sub></i> "
        + "(left), then turns that into a probability <i>p̂<sub>i</sub></i> "
        + "= σ(<i>z<sub>i</sub></i>) (right). Hover a sample to follow it.";
      return;
    }
    const x = Data.x[i], y = Data.y[i], z = c[0] + c[1] * x;
    const p = sigmoid(z), loss = y ? softplus(-z) : softplus(z);
    const sign = c[1] < 0 ? "−" : "+";
    $("md-pipe").innerHTML = `<i>x</i> = ${fmt(x)}${arr}`
      + `<i>z</i> = ${fmt(c[0])} ${sign} ${fmt(Math.abs(c[1]))} · `
      + `${fmt(x)} = <b>${fmt(z)}</b>${arr}`
      + `<i>p̂</i> = σ(${fmt(z)}) = <b>${fmt(p)}</b>${arr}`
      + `true <i>y</i> = ${y}, so its loss is −ln `
      + `${y ? "<i>p̂</i>" : "(1 − <i>p̂</i>)"} = <b>${fmt(loss)}</b>`;
  }

  function syncControls() {
    syncDataControls("md");
    for (const b of document.querySelectorAll("[data-md-axis]")) {
      b.setAttribute("aria-checked", b.dataset.mdAxis === S.axis);
    }
    $("md-prob").closest(".card").classList.toggle("zaxis", S.axis === "z");
    $("md-best").checked = S.showBest;
    $("md-truth").checked = S.showTruth;
  }

  function render() {
    if (!panels.some(p => p.drag)) S.zr = zRange(S.c, panels[0].dom);
    panels.forEach(p => p.render());
    equation();
    readout();
    pipeline();
    syncControls();
  }

  for (const b of document.querySelectorAll("[data-md-axis]")) {
    b.onclick = () => {
      S.axis = panels[1].axis = b.dataset.mdAxis;
      render();
    };
  }
  $("md-best").onchange = e => { S.showBest = e.target.checked; render(); };
  $("md-truth").onchange = e => { S.showTruth = e.target.checked; render(); };
  bindDataControls("md");

  obj = logisticLoss(Data.x, Data.y);

  return { step() {}, stop() {}, render };
})();
