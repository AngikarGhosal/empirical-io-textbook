"""Merger simulation with the running AIDS example (Chapter 5, Section 5.8).

Hypothetical illustration, not an estimate of any real market.

Demand: the chapter's running three-good AIDS (alpha_0 = 0, alpha = (0.5, 0.3,
0.2), beta = (-0.10, 0.04, 0.06), Gamma symmetric with zero row sums), base
prices p = (1, 1, 1) and ln X_G = 1, where X_G is expenditure on the segment.

Two treatments of segment expenditure:
  (A) X_G fixed (demand conditional on segment expenditure);
  (B) X_G responds to the segment price index through a constant-elasticity
      upper level, ln X_G = 1 + (1 + eps_B) ln P_G(p), with eps_B = -1.36.

Supply: three single-product firms, constant marginal cost, Bertrand-Nash in
prices.  Marginal costs are recovered from the pre-merger first-order
conditions at observed prices.  Firms 1 and 2 merge; the post-merger Nash
equilibrium is solved exactly, with merger-specific marginal-cost savings of 0,
5 and 10 percent for goods 1 and 2.  The script also computes the
"linearized" prediction of Hausman, Leonard and Zona (1994, eqs. 8-12), which
holds elasticities and rival prices at their pre-merger values.

For treatment (A) it computes the exact compensating variation of the merger
from the AIDS expenditure function (segment level).
Output: figures/ch05/src/aids_merger_sim_numbers.txt
"""
import os
import numpy as np
from scipy.optimize import root

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "aids_merger_sim_numbers.txt")

alpha0 = 0.0
alpha = np.array([0.5, 0.3, 0.2])
beta = np.array([-0.10, 0.04, 0.06])
Gamma = np.array([[-0.20, 0.12, 0.08],
                  [0.12, -0.22, 0.10],
                  [0.08, 0.10, -0.18]])
J = 3
lnX0 = 1.0


def lnP(l):
    return alpha0 + alpha @ l + 0.5 * l @ Gamma @ l


def b_of(l):
    return np.exp(beta @ l)


def shares(l, lnX):
    return alpha + Gamma @ l + beta * (lnX - lnP(l))


def ln_e(l, u):
    return lnP(l) + u * b_of(l)


def make_demand(epsB):
    """Return q(l) and the elasticity matrix E(l) (E[k, j] = dln q_k / dln p_j)."""
    def lnX_of(l):
        if epsB is None:
            return lnX0
        return lnX0 + (1.0 + epsB) * lnP(l)

    def q(l):
        lx = lnX_of(l)
        return shares(l, lx) * np.exp(lx) / np.exp(l)

    def E(l):
        lx = lnX_of(l)
        w = shares(l, lx)
        dlnP = alpha + Gamma @ l
        dw_cond = Gamma - np.outer(beta, dlnP)              # holding X_G fixed
        eps_cond = -np.eye(J) + dw_cond / w[:, None]
        if epsB is None:
            return eps_cond
        eta = 1.0 + beta / w
        dlnX = (1.0 + epsB) * dlnP                          # dln X_G / dln p_j
        return eps_cond + np.outer(eta, dlnX)
    return q, E


def foc(l, c, Om, q, E):
    """r_j + sum_k Om[j,k] * lerner_k * r_k * E[k,j] = 0 (elasticity form)."""
    p = np.exp(l)
    R = p * q(l)
    lern = (p - c) / p
    return R + (Om * E(l).T) @ (lern * R)


def solve_eq(c, Om, q, E, start):
    sol = root(lambda l: foc(l, c, Om, q, E), start, method="hybr", tol=1e-13)
    if not sol.success:
        sol = root(lambda l: foc(l, c, Om, q, E), start, method="lm", tol=1e-13)
    return sol


def profit(f_goods, l, c, q):
    p = np.exp(l)
    return np.sum(((p - c) * q(l))[f_goods])


def check_soc(owners, l, c, q):
    """Each firm's profit Hessian in its own log prices, holding rivals fixed."""
    out = []
    h = 1e-4
    for goods in owners:
        n = len(goods)
        H = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                da = np.zeros(J); da[goods[a]] = h
                db = np.zeros(J); db[goods[b]] = h
                H[a, b] = (profit(goods, l + da + db, c, q) - profit(goods, l + da - db, c, q)
                           - profit(goods, l - da + db, c, q) + profit(goods, l - da - db, c, q)) / (4 * h * h)
        out.append(np.linalg.eigvalsh(H).max())
    return out


lines = []
out = lambda s="": lines.append(s)
l0 = np.zeros(J)
Om_pre = np.eye(J)
Om_post = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)

for label, epsB in [("A: segment expenditure fixed", None), ("B: upper level, eps_B = -1.36", -1.36)]:
    q, E = make_demand(epsB)
    E0 = E(l0)
    w0 = shares(l0, lnX0)
    c = np.exp(l0) * (1.0 + 1.0 / np.diag(E0))
    out(label)
    out("  pre-merger elasticity matrix (row k = quantity, column j = price):")
    for row in E0:
        out("   " + "  ".join(f"{x: .3f}" for x in row))
    out(f"  marginal costs c = {np.round(c, 4)}; Lerner = {np.round(1 - c, 4)}")
    out(f"  pre-merger FOC residual = {np.abs(foc(l0, c, Om_pre, q, E)).max():.1e}")
    # HLZ linearized prediction: theta^M from pre-merger elasticities, rival prices fixed
    R0 = np.exp(l0) * q(l0)
    Ef = E0[:2, :2]
    m = -np.linalg.solve(Ef.T, R0[:2])
    thetaM = m / R0[:2]
    pmc = np.diag(E0)[:2] / (1.0 + np.diag(E0)[:2])          # p/mc pre-merger
    lin = {}
    for eff in [0.0, 0.05, 0.10]:
        lin[eff] = 100 * ((1 - eff) / (pmc * (1 - thetaM)) - 1)
    out(f"  HLZ linearized post-merger Lerner theta^M = {np.round(thetaM, 4)}")
    for eff in [0.0, 0.05, 0.10]:
        out(f"    efficiency {int(100 * eff)}%: linearized price change goods 1, 2 = {np.round(lin[eff], 2)} percent")
    start = l0.copy()
    for eff in [0.0, 0.05, 0.10]:
        cc = c.copy(); cc[:2] *= (1 - eff)
        sol = solve_eq(cc, Om_post, q, E, start)
        lp = sol.x
        p = np.exp(lp)
        socs = check_soc([[0, 1], [2]], lp, cc, q)
        wpost = shares(lp, lnX0 if epsB is None else lnX0 + (1 + epsB) * lnP(lp))
        out(f"  efficiency {int(100 * eff)}%: converged = {sol.success}, max |FOC| = {np.abs(foc(lp, cc, Om_post, q, E)).max():.1e}")
        out(f"    post-merger prices = {np.round(p, 4)}; percent changes = {np.round(100 * (p - 1), 2)}")
        out(f"    post-merger shares = {np.round(wpost, 4)}; max Hessian eigenvalues (merged, firm 3) = {np.round(socs, 5)}")
        out(f"    post-merger own elasticities = {np.round(np.diag(E(lp)), 3)}")
        if epsB is None:
            u0 = (lnX0 - lnP(l0)) / b_of(l0)
            CV = np.exp(ln_e(lp, u0)) - np.exp(lnX0)
            out(f"    exact CV (segment level) = {CV:.4f} = {100 * CV / np.exp(lnX0):.2f} percent of X_G")
            pi0 = (np.exp(l0) - c) * q(l0)
            pi1 = (p - cc) * q(lp)
            out(f"    profits pre = {np.round(pi0, 4)}, post = {np.round(pi1, 4)}; merged firm gain = {pi1[:2].sum() - pi0[:2].sum():.4f}, firm 3 gain = {pi1[2] - pi0[2]:.4f}")
        start = lp
    # robustness of the solution: several starting values
    sols = []
    for s0 in [np.zeros(J), np.array([0.5, 0.5, 0.1]), np.array([1.0, 1.0, 0.3]), np.array([0.2, 0.8, 0.0])]:
        s = solve_eq(c, Om_post, q, E, s0)
        if s.success:
            sols.append(np.round(np.exp(s.x), 4))
    out(f"  equilibria found from four starting values (0% efficiency): {[list(x) for x in sols]}")
    out()

text = "\n".join(lines)
print(text)
with open(OUT, "w") as fh:
    fh.write(text + "\n")
