# logistic_regression

**Live demo:** <https://matthigger.github.io/logistic_regression/>

Interactive teaching demo of logistic regression with one feature x and
labels y ∈ {0, 1}, in the style of the
[gradient descent demo](https://matthigger.github.io/gradient_descent/).
Two tabs:

- **Log odds → probability** (`#model`). The intro motivates the model:
  regressing the probability directly is unbounded, so regress the log
  odds instead. A strip shows the raw data (xᵢ, yᵢ); hovering a sample
  walks it through xᵢ → zᵢ → p̂ᵢ → its loss, with read-off guides in both
  panels below. The model is drawn twice, side by side: the line z = c₀ + c₁x on the log-odds scale, and the sigmoid
  p̂ = σ(z) on the probability scale. Drag a square handle in either panel
  to pivot the model, or empty space to change c₀ alone; both panels
  follow. Guides mark z = 0 ↔ p = 0.5 and z = ±2 ↔ p = 0.12 / 0.88, and the
  log-odds panel's right edge reads each z tick as a probability. Each
  sample is filled by its true class and ringed by its predicted class;
  hover one for its z, p̂ and loss. Toggles show the best fit (min loss)
  and the true P(Y = 1 | x).
- **Training** (`#training`). Gradient descent on the mean cross-entropy
  (nats): the loss surface over (c₀, c₁), as a contour map or orbitable 3D
  mesh, linked both ways to the data panel (probability or log-odds view).
  Gradient step / Run / Reset start, a log-scale step size η, a loss-by-step
  chart, and a readout of c, the loss, ∇loss and the minimum. Hover the
  trail to see any step's model.

Both tabs share the data: 15 samples per class, each class's x normal with
sd 1 around means a set distance apart (the Separation slider). With equal
variances the true P(Y = 1 | x) is exactly a sigmoid of a line (derivation:
`notes/day11_logistic/sigmoid_from_normals.pdf` in the course repo). Drag a
sample sideways, or click it to flip its class. When the classes stop
overlapping the loss has no minimum: a badge says so, there is no best fit,
and gradient descent walks off along a ray until it hits the step cap.

Plain HTML/CSS/JS with SVG: no build step and no dependencies.

## Run locally

Open `index.html` in a browser, or serve it:

```sh
python3 -m http.server 8000   # then visit http://localhost:8000
```

## Publish on GitHub Pages

Pages is built by `.github/workflows/pages.yml` on every push to `main`. It
stamps `js/version.js` with the commit and build time, which the footer
shows ("local copy" when run locally).

## Layout

| file | role |
|------|------|
| `index.html` | page, explanation text, controls |
| `style.css` | layout, palette, plot styles |
| `js/util.js` | SVG helpers, formatting, seeded RNG, sigmoid, eta slider |
| `js/data.js` | shared data set; logistic loss, its minimum, separability |
| `js/gd.js` | gradient-descent runner and run loop |
| `js/surface.js` | sampled contour + 3D loss surface with the draggable c |
| `js/chart.js` | loss-by-step chart |
| `js/plots.js` | log-odds and probability panels; shared data controls |
| `js/model.js` | Log odds → probability tab |
| `js/training.js` | Training tab |
| `js/app.js` | tabs, hash routing, keyboard, footer build stamp |
| `js/version.js` | build stamp, overwritten at deploy |
| `design/spec.md` | the design spec |
| `.github/workflows/pages.yml` | Pages deploy |
