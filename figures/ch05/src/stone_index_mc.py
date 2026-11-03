"""Monte Carlo illustration of the Stone-index approximation (Chapter 5, Section 5.6.4).

Data are generated from the exact AIDS with the running-example parameters
(alpha_0 = 0, alpha = (0.5, 0.3, 0.2), beta = (-0.10, 0.04, 0.06), Gamma as in
aids_example.py).  Log prices are ln p_jt = pi_t + s_e * e_jt with a common
component pi_t ~ N(0, 0.3^2) and relative-price noise e_jt ~ N(0, 1); log total
expenditure is ln I_t = 1 + pi_t + 0.3 g_t with g_t ~ N(0, 1).  Share errors are
xi_t = M v_t with M = I - (1/3) 1 1' (so they sum to zero) and v_t ~ N(0, s_v^2 I).

Four estimators, all imposing adding up, homogeneity and symmetry and all
minimizing the total sum of squared residuals over the three share equations
(the Gaussian ML estimator when the error covariance is proportional to M):
  (1) exact (nonlinear) AIDS with alpha_0 fixed at its true value 0;
  (2) LA-AIDS with the Stone index ln P^S = sum_k w_kt ln p_kt;
  (3) the same after measuring good 1 in units 100 times larger (p_1 -> 100 p_1);
  (4) LA-AIDS with a fixed-weight (Laspeyres-type) index sum_k wbar_k ln p_kt,
      whose estimates do not depend on units.
Elasticities are evaluated at the base point (p = 1, ln I = 1, shares
w0 = (0.40, 0.34, 0.26)).  Design A has nearly collinear prices (s_e = 0.02),
design B has substantial relative-price variation (s_e = 0.20).
Output: stone_index_mc_numbers.txt and stone_index_mc_table.tex (booktabs rows).
"""
import os
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stone_index_mc_numbers.txt")
TEX = os.path.join(HERE, "stone_index_mc_table.tex")

alpha0 = 0.0
alpha = np.array([0.5, 0.3, 0.2])
beta = np.array([-0.10, 0.04, 0.06])
Gamma = np.array([[-0.20, 0.12, 0.08],
                  [0.12, -0.22, 0.10],
                  [0.08, 0.10, -0.18]])
w0 = np.array([0.40, 0.34, 0.26])
T, R = 100, 500
s_v = 0.005
LAMBDA = np.log(100.0)


def unpack(th):
    a1, a2, b1, b2, g11, g12, g22 = th
    a = np.array([a1, a2, 1 - a1 - a2])
    b = np.array([b1, b2, -b1 - b2])
    g13 = -g11 - g12
    g23 = -g12 - g22
    G = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, -g13 - g23]])
    return a, b, G


def lnP_mat(lnp, a, G, a0=0.0):
    return a0 + lnp @ a + 0.5 * np.einsum("tj,jk,tk->t", lnp, G, lnp)


def gen(rng, s_e):
    pi = rng.normal(0, 0.3, T)
    lnp = pi[:, None] + s_e * rng.normal(size=(T, 3))
    lnI = 1 + pi + 0.3 * rng.normal(size=T)
    lnPt = lnP_mat(lnp, alpha, Gamma, alpha0)
    W = alpha + lnp @ Gamma.T + np.outer(lnI - lnPt, beta)
    M = np.eye(3) - np.ones((3, 3)) / 3
    W = W + (rng.normal(0, s_v, (T, 3)) @ M.T)
    return lnp, lnI, W, lnPt


def resid_exact(th, lnp, lnI, W):
    a, b, G = unpack(th)
    lnPt = lnP_mat(lnp, a, G, alpha0)
    fit = a + lnp @ G.T + np.outer(lnI - lnPt, b)
    return (W - fit).ravel()


def fit_linear(lnp, y, W):
    """Restricted LS of w_i = a_i + sum_j g_ij ln p_j + b_i y, stacked over 3 equations.

    Parameters th = (a1, a2, b1, b2, g11, g12, g22) as in unpack().  Each fitted
    share is affine in th: fit = c + D th, so we build c and D numerically.
    """
    def fitted(th):
        a, b, G = unpack(th)
        return (a + lnp @ G.T + np.outer(y, b)).ravel()
    c = fitted(np.zeros(7))
    D = np.column_stack([fitted(np.eye(7)[k]) - c for k in range(7)])
    th, *_ = np.linalg.lstsq(D, W.ravel() - c, rcond=None)
    return th


def elast(th, exact):
    a, b, G = unpack(th)
    if exact:
        # at the base point ln p = 0, ln(I/P) = 1 for the exact model
        e = -1 + (np.diag(G) - b * (w0 - b * 1.0)) / w0
    else:
        e = -1 + (np.diag(G) - b * w0) / w0
    return e


true_th = np.array([0.5, 0.3, -0.10, 0.04, -0.20, 0.12, -0.22])
true_e = elast(true_th, exact=True)

designs = {"A": 0.02, "B": 0.20}
rng = np.random.default_rng(20261022)
res = {}
for dname, s_e in designs.items():
    store = {k: [] for k in ("exact", "stone", "stone100", "lasp")}
    gap = []
    for r in range(R):
        lnp, lnI, W, lnPt = gen(rng, s_e)
        # (1) exact nonlinear AIDS
        sol = least_squares(resid_exact, true_th + 0.01, args=(lnp, lnI, W), method="lm")
        store["exact"].append(sol.x)
        # (2) Stone index, original units
        lnPS = np.sum(W * lnp, axis=1)
        gap.append(np.max(np.abs((lnPS - lnPS.mean()) - (lnPt - lnPt.mean()))))
        store["stone"].append(fit_linear(lnp, lnI - lnPS, W))
        # (3) Stone index, good 1 measured in units 100 times larger
        lnp2 = lnp.copy()
        lnp2[:, 0] += LAMBDA
        lnPS2 = np.sum(W * lnp2, axis=1)
        store["stone100"].append(fit_linear(lnp2, lnI - lnPS2, W))
        # (4) fixed-weight index with sample-mean shares (units-invariant)
        wbar = W.mean(axis=0)
        store["lasp"].append(fit_linear(lnp, lnI - lnp @ wbar, W))
    res[dname] = (store, np.mean(gap), np.max(gap))

labels = {"exact": "Exact AIDS (nonlinear)", "stone": "LA-AIDS, Stone index",
          "stone100": "LA-AIDS, Stone index, $p_1$ in new units",
          "lasp": "LA-AIDS, fixed-weight index"}
lines = [f"T = {T}, R = {R}, s_v = {s_v}, units change ln(lambda) = {LAMBDA:.3f}",
         "true beta_1 = -0.100, gamma_11 = -0.200; true own elasticities = " +
         ", ".join(f"{x:.3f}" for x in true_e)]
tex = []
for dname in designs:
    store, mgap, xgap = res[dname]
    lines.append(f"Design {dname} (s_e = {designs[dname]}): mean max |demeaned ln P^S - ln P| = {mgap:.4f}, worst = {xgap:.4f}")
    for k in ("exact", "stone", "stone100", "lasp"):
        th = np.array(store[k])
        ex = (k == "exact")
        E = np.array([elast(t, ex) for t in th])
        mb, sb = th[:, 2].mean(), th[:, 2].std()
        mg, sg = th[:, 4].mean(), th[:, 4].std()
        me, se = E.mean(axis=0), E.std(axis=0)
        lines.append(f"  {k:9s}: beta1 {mb: .3f} ({sb:.3f}); gamma11 {mg: .3f} ({sg:.3f}); "
                     f"eps11 {me[0]: .3f} ({se[0]:.3f}); eps22 {me[1]: .3f} ({se[1]:.3f}); eps33 {me[2]: .3f} ({se[2]:.3f})")
        tex.append(f"{dname} & {labels[k]} & ${mb:.3f}$ & $({sb:.3f})$ & ${mg:.3f}$ & $({sg:.3f})$ & "
                   f"${me[0]:.2f}$ & ${me[2]:.2f}$ \\\\")
open(OUT, "w").write("\n".join(lines) + "\n")
open(TEX, "w").write("\n".join(tex) + "\n")
print("\n".join(lines))
