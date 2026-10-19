// Training tab: the loss surface over the weights (w0, w1; c in the code)
// on the left, the data with the current model on the right (probability
// or log-odds view, against x), linked both ways, and gradient descent
// walking downhill.

const TrainTab = (function () {
  const ETA_LO = 0.01, ETA_HI = 100;
  const START = [0, 0];

  const S = { view: "contour", log: true, kind: "prob", hover: null,
    zr: null };
  let obj, runner, stopRun = null;

  const $ = id => document.getElementById(id);
  const chart = $("tr-chart");
  const etaInput = $("tr-eta");
  const eta = etaSlider(etaInput, ETA_LO, ETA_HI);
  const view = new LossView($("tr-surf"), {
    labels: ["w₀ (intercept)", "w₁ (slope)"], fname: "loss",
    onSet: c => setC(c),
    onHover: () => render(),
  });
  const panel = new DataPanel($("tr-data"), {
    kind: S.kind,
    state: () => ({ c: shown(), best: obj.cstar, truth: Data.truth(),
      showBest: true, showTruth: false, hover: S.hover,
      handles: hovered() === null, zr: S.zr }),
    onDragEnd: () => render(),
    onSetC: c => setC(c),
    onHover: i => { if (i !== S.hover) { S.hover = i; render(); } },
    onDataDone: () => { panel.handles(); render(); },
  });

  function visible() { return !$("tr-surf").closest("[hidden]"); }

  /** Fit the surface view around the start and the minimum. */
  function frame() {
    const pts = [START];
    if (obj.cstar) {
      pts.push(obj.cstar);
    } else if (obj.gap.side) {
      // No minimum: show the ray the loss falls along, w0 = -w1 at.
      const c1 = 4 * obj.gap.side;
      pts.push([-c1 * obj.gap.at, c1]);
    }
    view.fit(pts, 2.5);
  }

  function regen() {
    stop();
    obj = logisticLoss(Data.x, Data.y);
    frame();
    runner = new Runner(obj, START, view.scale);
    panel.fitX();
  }

  Data.onChange(fresh => {
    if (fresh) {
      regen();
    } else {
      stop();
      obj = logisticLoss(Data.x, Data.y);
      runner.setObjective(obj);
    }
    if (visible()) render();
  });

  function setC(c) {
    stop();
    runner.reset(c);
    render();
  }

  function stop() {
    if (stopRun) { stopRun(); stopRun = null; }
  }

  function step() {
    stop();
    runner.step(eta.get());
    render();
  }

  function toggleRun() {
    if (stopRun) { stop(); render(); return; }
    if (runner.status) return;
    stopRun = runLoop(runner, eta.get, render, () => {
      stopRun = null;
      render();
    });
    render();
  }

  /**
   * The iterate the pointer is hovering on the surface, once the walk has
   * taken a step; null otherwise.
   */
  function hovered() {
    const h = view.hover, n = runner.trail.length;
    return h !== null && n > 1 && h < n ? h : null;
  }
  function shown() {
    const h = hovered();
    return h === null ? runner.c : runner.trail[h];
  }

  function equation() {
    const h = hovered(), c = shown(), sign = v => v < 0 ? "−" : "+";
    const z = `${fmt(c[0])} ${sign(c[1])} ${fmt(Math.abs(c[1]))}<i>x</i>`;
    $("tr-eq").innerHTML = (h === null ? "" : `step ${h}: `)
      + (S.kind === "odds" ? `<i>z</i> = ${z}` : `<i>p̂</i> = σ(${z})`);
  }

  function readout() {
    const h = hovered(), c = shown(), g = obj.grad(c);
    const x = Data.x, y = Data.y;
    const right = x.filter((v, i) => (c[0] + c[1] * v > 0 ? 1 : 0) === y[i])
      .length;
    const rows = [
      [`<b>w</b> = ${colVec(["<i>w</i>₀", "<i>w</i>₁"])}`, fmtCol(c)],
      ["loss(<b>w</b>)", fmt(obj.f(c))],
      ["∇loss(<b>w</b>)", fmtCol(g)],
      ["‖∇loss(<b>w</b>)‖", fmt(Math.hypot(...g))],
      ["accuracy", `${right} of ${x.length}`],
    ];
    if (obj.cstar) {
      rows.push(["min loss", fmt(obj.fmin)],
        ["at <b>w</b>*", fmtCol(obj.cstar)]);
    } else {
      rows.push(["min loss", "none: falls toward 0"]);
    }
    $("tr-readout").innerHTML = rows.map(([k, v]) =>
      `<div class="row"><span>${k}</span><span class="val">${v}</span></div>`
    ).join("") + statusLine(runner, h);
  }

  function syncControls() {
    syncDataControls("tr");
    for (const b of document.querySelectorAll("[data-tr-view]")) {
      b.setAttribute("aria-checked", b.dataset.trView === S.view);
    }
    for (const b of document.querySelectorAll("[data-tr-log]")) {
      b.setAttribute("aria-checked", (b.dataset.trLog === "1") === S.log);
    }
    for (const b of document.querySelectorAll("[data-tr-kind]")) {
      b.setAttribute("aria-checked", b.dataset.trKind === S.kind);
    }
    $("tr-etaval").innerHTML = `<i>η</i> = ${fmt(eta.get())}`;
    $("tr-run").innerHTML = stopRun ? "❚❚ Pause" : "&#9654; Run";
    $("tr-run").disabled = !stopRun && !!runner.status;
    $("tr-step").disabled = !!runner.status;
    $("tr-surf-hint").textContent = S.view === "3d"
      ? "Drag to orbit, scroll to zoom. Drag the point w on the floor to "
        + "move it (hold Shift to change only one parameter); the "
        + "dropline is its loss."
      : "Drag the point w, or click anywhere, to choose w. Hold Shift to "
        + "change only one parameter. After a step, hover the trail to see "
        + "that step's model.";
  }

  function render() {
    if (!panel.drag) S.zr = zRange(shown(), panel.dom);
    view.render(obj, { c: runner.c, trail: runner.trail, eta: eta.get(),
      mode: S.view });
    panel.render();
    drawLossChart(chart, runner.losses, obj.fmin, S.log, "loss", hovered());
    equation();
    readout();
    syncControls();
  }

  for (const b of document.querySelectorAll("[data-tr-view]")) {
    b.onclick = () => { S.view = b.dataset.trView; render(); };
  }
  for (const b of document.querySelectorAll("[data-tr-log]")) {
    b.onclick = () => { S.log = b.dataset.trLog === "1"; render(); };
  }
  for (const b of document.querySelectorAll("[data-tr-kind]")) {
    b.onclick = () => {
      S.kind = b.dataset.trKind;
      panel.kind = S.kind;
      render();
    };
  }
  etaInput.oninput = render;
  $("tr-step").onclick = step;
  $("tr-run").onclick = toggleRun;
  $("tr-restart").onclick = () => setC(runner.trail[0]);
  $("tr-zero").onclick = () => setC(START);
  bindDataControls("tr");

  regen();
  eta.set(2 / obj.L);

  return { step, stop, render };
})();
