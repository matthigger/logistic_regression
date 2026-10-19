// Model tab: the same c drawn twice, as the line z = c0 + c1 x on the
// log-odds scale (left) and as the sigmoid p = sigma(z) (right). Dragging
// either moves both.

const ModelTab = (function () {
  const S = { c: [0, 1], showBest: false, showTruth: false, hover: null };
  let obj;

  const $ = id => document.getElementById(id);
  const state = () => ({ c: S.c, best: obj.cstar, truth: Data.truth(),
    showBest: S.showBest, showTruth: S.showTruth, hover: S.hover,
    handles: true });
  const opts = kind => ({
    kind, state,
    onSetC: c => { S.c = c; render(); },
    onHover: i => { if (i !== S.hover) { S.hover = i; render(); } },
    onDataDone: () => { panels.forEach(p => p.handles()); render(); },
  });
  const panels = [new DataPanel($("md-odds"), opts("odds")),
    new DataPanel($("md-prob"), opts("prob"))];

  function visible() { return !$("md-odds").closest("[hidden]"); }

  Data.onChange(fresh => {
    obj = logisticLoss(Data.x, Data.y);
    if (fresh) panels.forEach(p => p.fitX());
    if (visible()) render();
  });

  function equation() {
    const sign = v => v < 0 ? "−" : "+";
    $("md-eq-odds").innerHTML = `<i>z</i> = <i>c</i>₀ + <i>c</i>₁<i>x</i> = `
      + `${fmt(S.c[0])} ${sign(S.c[1])} ${fmt(Math.abs(S.c[1]))}<i>x</i>`;
    $("md-eq-prob").innerHTML = "<i>p̂</i> = σ(<i>z</i>) = 1 / (1 + "
      + "<i>e</i><sup>−<i>z</i></sup>)";
  }

  function readout() {
    const x = Data.x, y = Data.y, n = x.length, c = S.c;
    const right = x.filter((v, i) => (c[0] + c[1] * v > 0 ? 1 : 0) === y[i])
      .length;
    const rows = [
      [`<b>c</b> = ${colVec(["<i>c</i>₀", "<i>c</i>₁"])}`, fmtCol(c)],
      ["boundary <i>x</i> = −<i>c</i>₀ / <i>c</i>₁",
        Math.abs(c[1]) > 1e-9 ? fmt(-c[0] / c[1]) : "none"],
      ["accuracy", `${right} of ${n}`],
      ["loss(<b>c</b>)", fmt(obj.f(c))],
      ["best loss", obj.cstar ? fmt(obj.fmin)
        : "none (separable)"],
    ];
    const i = S.hover;
    if (i !== null && i < n) {
      const z = c[0] + c[1] * x[i], p = sigmoid(z);
      const loss = y[i] ? softplus(-z) : softplus(z);
      rows.push([`sample: <i>x</i> = ${fmt(x[i])}, <i>y</i> = ${y[i]}`,
        `<i>z</i> = ${fmt(z)}, <i>p̂</i> = ${fmt(p)}`]);
      rows.push([`its loss −ln ${y[i] ? "<i>p̂</i>"
        : "(1 − <i>p̂</i>)"}`, fmt(loss)]);
    }
    $("md-readout").innerHTML = rows.map(([k, v]) =>
      `<div class="row"><span>${k}</span><span class="val">${v}</span></div>`
    ).join("");
  }

  function syncControls() {
    syncDataControls("md");
    $("md-best").checked = S.showBest;
    $("md-truth").checked = S.showTruth;
  }

  function render() {
    panels.forEach(p => p.render());
    equation();
    readout();
    syncControls();
  }

  $("md-best").onchange = e => { S.showBest = e.target.checked; render(); };
  $("md-truth").onchange = e => { S.showTruth = e.target.checked; render(); };
  bindDataControls("md");

  obj = logisticLoss(Data.x, Data.y);

  return { step() {}, stop() {}, render };
})();
