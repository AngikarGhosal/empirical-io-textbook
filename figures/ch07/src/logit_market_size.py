"""Market-size sensitivity of the logit inversion (Chapter 7, Example ex:07-market-size).

Simulates logit demand with single-product Bertrand-Nash pricing in many markets, then
re-estimates the inverted logit regression
    ln s_jt - ln s_0t = b0 + bx x_jt - alpha p_jt + xi_jt
by two-stage least squares after the econometrician scales the true market size M_t by a
factor kappa.  Reports the price coefficient with and without market fixed effects, the
implied mean own-price elasticity and the mean diversion ratio to the outside good.

Only numpy is used.  Fixed seed; numbers are printed and pasted into the text.
"""
import numpy as np

rng = np.random.default_rng(20261022)
T, J = 400, 6
alpha, b0, bx = 1.0, 1.0, 1.0
g0, gx = 1.0, 0.5

x = rng.normal(size=(T, J))
xi = rng.normal(size=(T, J))
eta = rng.normal(scale=0.5, size=(T, 1))            # market-level cost shock
nu = rng.normal(scale=0.5, size=(T, J))             # product-level cost shock
w = eta + nu                                        # observed cost shifter
mc = g0 + gx * x + w

# Bertrand-Nash prices: p = mc + 1 / (alpha (1 - s)), solved by damped iteration.
p = mc + 1.0
for it in range(5000):
    d = b0 + bx * x - alpha * p + xi
    e = np.exp(d)
    s = e / (1.0 + e.sum(axis=1, keepdims=True))
    p_new = mc + 1.0 / (alpha * (1.0 - s))
    if np.max(np.abs(p_new - p)) < 1e-13:
        break
    p = 0.5 * p + 0.5 * p_new
d = b0 + bx * x - alpha * p + xi
e = np.exp(d)
s = e / (1.0 + e.sum(axis=1, keepdims=True))
s0 = 1.0 - s.sum(axis=1, keepdims=True)
print(f"converged after {it} iterations; outside share range {s0.min():.3f}-{s0.max():.3f}, "
      f"mean {s0.mean():.3f}; inside share range {s.min():.4f}-{s.max():.3f}")


def tsls(y, X, Z):
    """Two-stage least squares coefficients."""
    PZ_X = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return np.linalg.lstsq(PZ_X, y, rcond=None)[0]


def demean_by_market(a):
    return a - a.mean(axis=1, keepdims=True)


print(f"{'kappa':>6} {'alpha (no FE)':>14} {'alpha (FE)':>11} {'mean own el.':>13} "
      f"{'mean D_j0':>10} {'mean s0':>8}")
for kappa in [0.95, 1.0, 1.5, 2.0, 4.0]:
    sk = s / kappa                                   # q / (kappa M)
    s0k = 1.0 - sk.sum(axis=1, keepdims=True)
    y = np.log(sk) - np.log(s0k)
    # (a) pooled 2SLS: regressors [1, x, p], instruments [1, x, w]
    X = np.column_stack([np.ones(T * J), x.ravel(), p.ravel()])
    Z = np.column_stack([np.ones(T * J), x.ravel(), w.ravel()])
    a_pool = -tsls(y.ravel(), X, Z)[2]
    # (b) market fixed effects (within transformation), instruments [x, w]
    Xf = np.column_stack([demean_by_market(x).ravel(), demean_by_market(p).ravel()])
    Zf = np.column_stack([demean_by_market(x).ravel(), demean_by_market(w).ravel()])
    a_fe = -tsls(demean_by_market(y).ravel(), Xf, Zf)[1]
    # implied elasticities and diversion to the outside good, using the FE estimate
    own = a_fe * p * (1.0 - sk)
    Dj0 = s0k / (1.0 - sk)
    print(f"{kappa:6.2f} {a_pool:14.3f} {a_fe:11.3f} {own.mean():13.3f} {Dj0.mean():10.3f} "
          f"{s0k.mean():8.3f}")
