# Logistic regression, visually: spec

Sibling of `gradient_descent`, `knn`, `cross_validation`, `binary_metrics`:
plain HTML/CSS/JS with SVG, same page shell (header + tabs, intro card with
formulas and "Try this" with `<details>` answers, plot cards + controls
sidebar, footer build stamp) and `pages.yml`. Written for lesson 11
(logistic regression: sigmoid, cross-entropy, GD update).

## Decisions

- Own repo, not a tab of `gradient_descent`: the first tab is about the
  model, not the optimizer.
- Weights are **c** = (c₀, c₁), matching the problem bank and quizzes.
- Loss in nats on the page; one note says bits (log₂, as on the problem
  sets) scale loss and gradient by 1/ln 2, so the same walk takes η · ln 2.
- One feature x, so the loss surface over (c₀, c₁) is drawable.
- Data: two normal classes, shared sd, means a "Separation" apart, so the
  true P(Y = 1 | x) is exactly a sigmoid (shown on request). Both tabs share
  the data; each tab keeps its own c.
- Samples: fill = true class, ring = predicted class (as in
  `binary_metrics`); class colors from `binary_metrics`. The current model
  is violet, since orange is class 1.
- Log-odds panel: samples sit on the line at their own zᵢ (their labels'
  log odds would be ±∞).
- Editing data: drag a sample sideways; click to flip its class.
- No step-size zones (the Hessian varies with c); a plain log η slider,
  default 2/L for the bound L = λmax(XᵀX/n)/4.
- Cut: offset/κ sliders and x scaling (covered by `gradient_descent`), an
  MSE-on-probabilities toggle, ridge.

## Tab 1: Log odds → probability (`#model`)

Intro opens with the problem (direct approach: P̂(Y = 1 | x) = c · x is
unbounded; doubling an 80% sample gives 160%) and the approach (regress
the log odds instead). Four "Try this" items, the simplest. Above the two
panels, a strip of the raw data (xᵢ, yᵢ) in one lane per class, no model;
hovering a sample (anywhere) writes its chain x → z = c₀ + c₁x → p̂ = σ(z)
→ loss under the strip and draws read-off guides in both panels.

Left: x against z with the line, guides at z = 0 and ±2, right-edge axis
giving σ of each z tick, decision boundary x = −c₀/c₁ as a vertical line.
Right: x against p with the sigmoid, error segments yᵢ → p̂ᵢ, and the same
guides (p = 0.5, 0.12, 0.88) and boundary. Handles at the 25th/75th
x-percentiles pivot the model; in the probability panel a handle sets p at
its x (z = logit p). Empty-space drag changes c₀ alone (vertical in log
odds, horizontal in probability). Hover a sample: its z, p̂ and loss in
both panels and the readout. Readout: c, boundary, accuracy, loss, best
loss (or "none (separable)").

## Tab 2: Training (`#training`)

Left: loss surface (sampled grid, geometric levels in loss − min; contour
or 3D), draggable c with Shift-lock, trail with hover, gradient (fixed
length) and step (true length) arrows, the level line through c, an edge
arrow when c leaves the view. Right: the data panel (Probability / Log odds
toggle) with the best fit dashed. Start c = 0 (every p̂ = 0.5). Readout:
c, loss, ∇loss, ‖∇loss‖, accuracy, min loss and c*, run status. Separable
data: no c*, Run stops at the step cap with a sentence saying why.
