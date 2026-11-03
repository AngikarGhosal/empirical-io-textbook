"""Verify the two-phase (stick-and-carrot) punishment results of Section 12.3.

Linear Cournot duopoly, constant marginal cost c, inverse demand a - Q.
Units: a - c = 1 and slope 1 (thresholds on delta are scale free).

Part A. Inverse demand linear for every Q >= 0 (the price may fall below zero).
        The minimal delta over stick outputs x is 9/32, attained at x = 5/12.
Part B. Price floored at zero, P = max{a - Q, 0}. The answer depends on c.
Part C. Individual-rationality bound delta >= 1/9 for any SPE sustaining the
        monopoly split (deviation payoff 9/64 against minmax 0).

Deviation payoffs are computed numerically (grid search followed by a bounded
scalar refinement), so the script does not rely on the closed forms in the text.
"""
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import minimize_scalar

QMAX = 3.0
COARSE = np.linspace(0.0, QMAX, 3001)


def profit(qi, qj, a, c, floor):
    """Stage profit of a firm producing qi when the rival produces qj."""
    p = a - qi - qj
    if floor:
        p = np.maximum(p, 0.0)
    return (p - c) * qi


def best_dev(qj, a, c, floor):
    """Best one-shot deviation profit against rival output qj."""
    vals = profit(COARSE, qj, a, c, floor)
    k = int(np.argmax(vals))
    lo, hi = COARSE[max(k - 1, 0)], COARSE[min(k + 1, len(COARSE) - 1)]
    res = minimize_scalar(lambda q: -profit(q, qj, a, c, floor),
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-12})
    return max(vals[k], -res.fun)


def ratio(x, a, c, floor, pic, gain_c):
    """Smallest delta satisfying both constraints for stick output x."""
    pix = profit(x, x, a, c, floor)
    loss = pic - pix
    if loss <= 1e-14:
        return np.inf
    gain_p = best_dev(x, a, c, floor) - pix
    return max(gain_c / loss, gain_p / loss)


def min_delta(a, c, floor, xs):
    qm = (a - c) / 4.0                      # per-firm monopoly-sharing output
    pic = profit(qm, qm, a, c, floor)        # collusive profit per firm
    gain_c = best_dev(qm, a, c, floor) - pic
    vals = np.array([ratio(x, a, c, floor, pic, gain_c) for x in xs])
    k = int(np.argmin(vals))
    lo, hi = xs[max(k - 1, 0)], xs[min(k + 1, len(xs) - 1)]
    res = minimize_scalar(lambda x: ratio(x, a, c, floor, pic, gain_c),
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    if res.fun < vals[k]:
        return res.fun, res.x
    return vals[k], xs[k]


def main():
    # ------------------------------------------------------------------ A
    print("Part A: linear inverse demand on all Q (no price floor)")
    xs = np.linspace(0.26, QMAX, 2741)
    for c in (0.0, 0.1, 0.5, 2.0):
        a = 1.0 + c
        d, x = min_delta(a, c, False, xs)
        print(f"  c = {c:4.2f}: min delta = {d:.6f} at x = {x:.5f}"
              f"   (9/32 = {9/32:.6f}, 5/12 = {5/12:.5f})")
    x = Fr(5, 12)
    pic, pix = Fr(1, 8), x * (1 - 2 * x)
    dev_c, dev_x = Fr(9, 64), ((1 - x) / 2) ** 2
    print("  exact at x = 5/12: pi(x) =", pix, " pi^d(x) =", dev_x,
          " pi^d(x) - pi(x) =", dev_x - pix,
          " carrot ratio =", (dev_c - pic) / (pic - pix),
          " stick ratio =", (dev_x - pix) / (pic - pix))
    print("  roots of 48x^2-32x+5:", np.roots([48, -32, 5]))
    for xv in (0.34, 0.375, 5 / 12, 0.5, 0.6):
        dc = 1 / (8 * (4 * xv - 1) ** 2)
        ds = 2 * (3 * xv - 1) ** 2 / (4 * xv - 1) ** 2
        print(f"  x = {xv:.4f}: delta_C = {dc:.4f}, delta_S = {ds:.4f}")
    print("  Nash reversion threshold 9/17 =", round(9 / 17, 5))
    print("  stick price minus cost at x = 5/12:", 1 - 2 * 5 / 12)
    # ------------------------------------------------------------------ B
    print("Part B: price floored at zero, P = max{a - Q, 0}")
    xs = np.linspace(0.26, QMAX, 2741)
    for c in (0.0, 0.005, 0.01, 1 / 64, 0.03, 0.05, 0.1, 0.5):
        a = 1.0 + c
        d, x = min_delta(a, c, True, xs)
        print(f"  c = {c:6.4f}: min delta = {d:.5f} at x = {x:.4f}")
    # cutoff cost above which flooding sticks cannot beat 9/32. In the region
    # where the stick drives the price to the floor, pi(x) = -c x and the
    # stick constraint binds, so the best flooding stick gives min_x h(x).
    def min_flood(c):
        x1 = np.linspace((1 + c) / 2, 1, 200001)
        h1 = ((1 - x1) ** 2 / 4 + c * x1) / (1 / 8 + c * x1)
        x2 = np.linspace(1, 5, 100001)
        h2 = c * x2 / (1 / 8 + c * x2)
        return min(h1.min(), h2.min())
    from scipy.optimize import brentq
    cstar = brentq(lambda c: min_flood(c) - 9 / 32, 0.04, 0.09, xtol=1e-10)
    print(f"  cutoff: flooding sticks beat 9/32 iff c < {cstar:.4f} (a - c)")
    print("  c <= 1/64 =", round(1 / 64, 5), "attains the bound 1/9 =",
          round(1 / 9, 5))
    # ------------------------------------------------------------------ C
    print("Part C: minmax bound 1 - (1/8)/(9/64) =", 1 - Fr(1, 8) / Fr(9, 64))


if __name__ == "__main__":
    main()
