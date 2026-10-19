# Logistic regression, visually: spec

Sibling of `gradient_descent`, `knn`, `cross_validation`, `binary_metrics`:
plain HTML/CSS/JS with SVG, same page shell (header + tabs, intro card with
formulas and "Try this" with `<details>` answers, plot cards + controls
sidebar, footer build stamp) and `pages.yml`. Written for lesson 11
(logistic regression: sigmoid, cross-entropy, GD update).

## Decisions

- Own repo, not a tab of `gradient_descent`: the first tab is about the
  model, not the optimizer.
- Weights are **w** = (w₀, w₁) on the page, as in all DS4400 material
  (the code calls them c, as `gradient_descent` does).
- Loss in nats on the page; one note says bits (log₂, as on the problem
  sets) scale loss and gradient by 1/ln 2, so the same walk takes η · ln 2.
- One feature x, so the loss surface over (w₀, w₁) is drawable.
- Data: two normal classes, shared sd, means a "Separation" apart, so the
  true P(Y = 1 | x) is exactly a sigmoid (shown on request). Both tabs share
  the data; each tab keeps its own w.
- Samples: fill = true class, ring = predicted class (as in
  `binary_metrics`); class colors from `binary_metrics`. The current model
  is violet, since orange is class 1.
- Log-odds panel: samples sit on the line at their own zᵢ (their labels'
  log odds would be ±∞).
- One set of grid lines per axis, labeled on both scales (z lines with
  σ(z), p̂ lines with their logit); no separate guide lines.
- The z axis is fitted to the line (at least ±6), never clamped, and held
  still during a drag so a handle cannot chase its own axis.
- Editing data: drag a sample sideways; click to flip its class.
- No step-size zones (the Hessian varies with w); a plain log η slider,
  default 2/L for the bound L = λmax(XᵀX/n)/4.
- Cut: offset/κ sliders and x scaling (covered by `gradient_descent`), an
  MSE-on-probabilities toggle, ridge.

## Tab 1: Log odds → probability (`#model`)

Intro: the problem (direct approach: P̂(Y = 1 | x) = w · x is unbounded;
doubling an 80% sample gives 160%), the approach (regress the log odds
instead), the model, the prediction rule. Four "Try this" items, the
simplest.

Strip (top): the raw data (xᵢ, yᵢ), one lane per class, no model. Hovering
a sample anywhere writes its chain x → z = w₀ + w₁x → p̂ = σ(z) → loss
under the strip and draws read-off guides in both panels.

Left: x against z, the line, decision boundary x = −w₀/w₁. Right: p̂
against z (default: the fixed curve σ, the samples sliding along it as w
changes; boundary at z = 0) or against x (the model's sigmoid, with the
best fit and truth when shown). Error segments yᵢ → p̂ᵢ. Axis labels are
HTML (z = log odds = ln P̂(Y=1|x) / P̂(Y=0|x)). Handles at the 25th/75th
x-percentiles pivot the model: vertical in log odds, in p̂ against x
(z = logit p̂), horizontal against z. Empty-space drag changes w₀ alone.
Readout: w, boundary, accuracy, loss, best loss (or "none (separable)").

## Tab 2: Training (`#training`)

Left: loss surface (sampled grid, geometric levels in loss − min; contour
or 3D), draggable w with Shift-lock, trail with hover, gradient (fixed
length) and step (true length) arrows, the level line through w, an edge
arrow when w leaves the view. Right: the data panel against x
(Probability / Log odds toggle) with the best fit dashed. Start w = 0
(every p̂ = 0.5). Readout: w, loss, ∇loss, ‖∇loss‖, accuracy, min loss and
w*, run status. Separable data: no w*, Run stops at the step cap with a
sentence saying why.
