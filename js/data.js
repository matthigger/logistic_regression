// The data set both tabs share, and the logistic loss over c = (c0, c1).
//
// Each class's x is normal with a shared sd s, around means mid -/+ sep/2.
// With equal variances the true P(Y=1 | x) is exactly sigma(c0 + c1 x),
// with c1 = sep / s^2 and c0 = -sep mid / s^2 + log(n1 / n0) (Bayes' rule;
// see sigmoid_from_normals.pdf in the day 11 notes).

const LR = { n0: 15, n1: 15, mid: 1, s: 1, sepMax: 6 };

const Data = (function () {
  const S = { sep: 2, seed: 1 };
  // x: (n,) feature; y: (n,) labels in {0, 1}. Edits (drags, flips) change
  // them in place until the next generate.
  let x = [], y = [];
  const listeners = [];

  /**
   * Draw the sample. The seed fixes the standardized draws, so moving sep
   * slides the same two clouds apart rather than drawing new ones.
   */
  function generate() {
    const r = rng(S.seed * 7919 + 31);
    x = [];
    y = [];
    for (const [k, n] of [[0, LR.n0], [1, LR.n1]]) {
      const z = [];
      for (let i = 0; i < n; i++) z.push(randn(r));
      const m = z.reduce((t, v) => t + v, 0) / n;
      const sd = Math.sqrt(z.reduce((t, v) => t + (v - m) ** 2, 0) / n);
      const mu = LR.mid + (k ? 1 : -1) * S.sep / 2;
      for (const v of z) {
        x.push(mu + LR.s * (v - m) / sd);
        y.push(k);
      }
    }
    changed(true);
  }

  function changed(fresh) {
    for (const fn of listeners) fn(fresh);
  }

  return {
    get x() { return x; },
    get y() { return y; },
    get sep() { return S.sep; },
    get n() { return x.length; },
    setSep(v) { S.sep = v; generate(); },
    resample() { S.seed += 1; generate(); },
    reset: generate,
    /** Move sample i to feature value v. */
    move(i, v) { x[i] = v; changed(false); },
    /** Swap sample i's label. */
    flip(i) { y[i] = 1 - y[i]; changed(false); },
    /**
     * Subscribe to changes: fn(fresh), fresh true for a new draw (sep,
     * resample, reset) and false for an edit.
     */
    onChange(fn) { listeners.push(fn); },
    /** The c of the true P(Y=1 | x) that generated the sample. */
    truth() {
      const s2 = LR.s * LR.s;
      return [-S.sep * LR.mid / s2 + Math.log(LR.n1 / LR.n0), S.sep / s2];
    },
  };
})();

/**
 * Mean cross-entropy (in nats) of p_i = sigma(c0 + c1 x_i) against y.
 *
 * loss(c) = (1/n) sum [log(1 + e^z_i) - y_i z_i], which is
 * -(1/n) sum [y_i log p_i + (1 - y_i) log(1 - p_i)], with gradient
 * (1/n) sum (p_i - y_i) [1, x_i]. Its Hessian is
 * (1/n) sum p_i (1 - p_i) [1, x_i][1, x_i]', at most a quarter of
 * (1/n) X'X, whose top eigenvalue gives the smoothness bound L.
 *
 * When one class lies wholly on one side of the other (separable, ties
 * included) the loss has no minimum: it falls toward 0 as c runs off along
 * a ray, so cstar is null and fmin is 0.
 *
 * Args:
 *   x (number[]): (n,) feature
 *   y (number[]): (n,) labels in {0, 1}
 */
function logisticLoss(x, y) {
  const n = x.length;
  const f = c => {
    let t = 0;
    for (let i = 0; i < n; i++) {
      const z = c[0] + c[1] * x[i];
      t += softplus(z) - y[i] * z;
    }
    return t / n;
  };
  const grad = c => {
    let g0 = 0, g1 = 0;
    for (let i = 0; i < n; i++) {
      const e = sigmoid(c[0] + c[1] * x[i]) - y[i];
      g0 += e;
      g1 += e * x[i];
    }
    return [g0 / n, g1 / n];
  };
  const mx = x.reduce((t, v) => t + v, 0) / n;
  const mxx = x.reduce((t, v) => t + v * v, 0) / n;
  const L = eigMax(1, mx, mxx) / 4;

  const sep = separation(x, y);
  let cstar = null, fmin = 0;
  if (!sep) {
    cstar = newton(f, grad, x);
    fmin = f(cstar);
  }
  return {
    f, grad, L, cstar, fmin,
    separable: !!sep,
    gap: sep,
    excess: c => f(c) - fmin,
    converged(c, start) {
      if (!cstar) return false;
      const e0 = f(start) - fmin;
      return f(c) - fmin <= Math.max(1e-4 * e0, 1e-10);
    },
  };
}

/**
 * Whether the classes separate along x, ties included.
 *
 * Returns null when they overlap, else { side, at }: side +1 when class 1
 * sits to the right (-1 left, 0 when a class is empty) and at, the middle
 * of the gap.
 */
function separation(x, y) {
  const x0 = x.filter((_, i) => !y[i]), x1 = x.filter((_, i) => y[i]);
  if (!x0.length || !x1.length) return { side: 0, at: null };
  const max0 = Math.max(...x0), min0 = Math.min(...x0);
  const max1 = Math.max(...x1), min1 = Math.min(...x1);
  if (max0 <= min1) return { side: 1, at: (max0 + min1) / 2 };
  if (max1 <= min0) return { side: -1, at: (max1 + min0) / 2 };
  return null;
}

/** Minimizer of the (strictly convex, overlapping-class) loss by Newton. */
function newton(f, grad, x) {
  const n = x.length;
  let c = [0, 0], fc = f(c);
  for (let it = 0; it < 100; it++) {
    const g = grad(c);
    let h00 = 0, h01 = 0, h11 = 0;
    for (let i = 0; i < n; i++) {
      const p = sigmoid(c[0] + c[1] * x[i]), w = p * (1 - p);
      h00 += w; h01 += w * x[i]; h11 += w * x[i] * x[i];
    }
    h00 /= n; h01 /= n; h11 /= n;
    const det = h00 * h11 - h01 * h01;
    if (!(det > 0)) break;
    const d = [(h11 * g[0] - h01 * g[1]) / det,
      (h00 * g[1] - h01 * g[0]) / det];
    // Backtrack so a far start cannot overshoot into a worse point.
    let t = 1, next, fn;
    for (let k = 0; k < 40; k++) {
      next = [c[0] - t * d[0], c[1] - t * d[1]];
      fn = f(next);
      if (fn <= fc) break;
      t /= 2;
    }
    const moved = Math.hypot(c[0] - next[0], c[1] - next[1]);
    c = next;
    fc = fn;
    if (moved < 1e-12) break;
  }
  return c;
}

Data.reset();
