// Tabs, hash routing, keyboard steps and the footer build stamp. Each tab
// (ModelTab, TrainTab) owns its own state and DOM; they share Data.

(function () {
  const TABS = { model: ModelTab, train: TrainTab };
  const HASH = { model: "#model", train: "#training" };
  let mode = "model";

  function setMode(m) {
    TABS[mode].stop();
    mode = m;
    history.replaceState(null, "", HASH[m]);
    for (const t of document.querySelectorAll(".tab")) {
      t.setAttribute("aria-selected", t.dataset.mode === m);
    }
    for (const el of document.querySelectorAll("[data-show]")) {
      el.hidden = !el.dataset.show.split(" ").includes(m);
    }
    TABS[m].render();
  }

  for (const t of document.querySelectorAll(".tab")) {
    t.onclick = () => { if (t.dataset.mode !== mode) setMode(t.dataset.mode); };
  }

  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey) return;
    if (e.key === " " || e.key === "ArrowRight") {
      e.preventDefault();
      TABS[mode].step();
    }
  });

  // Footer build stamp: the deployed commit (linked) and build time.
  const el = document.getElementById("build");
  const repo = "https://github.com/matthigger/logistic_regression";
  if (!BUILD) {
    el.textContent = "local copy";
  } else {
    const when = new Date(BUILD.time).toLocaleString("en-US", {
      dateStyle: "medium", timeStyle: "short" });
    el.innerHTML = `build <a href="${repo}/commit/${BUILD.sha}">${
      BUILD.sha.slice(0, 7)}</a>, ${when}`;
  }

  const fromHash = Object.keys(HASH).find(m => HASH[m] === location.hash);
  setMode(fromHash || "model");
})();
