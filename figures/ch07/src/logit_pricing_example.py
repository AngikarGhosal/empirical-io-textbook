"""Logit Bertrand equilibrium with single- and multi-product firms (ex:07-pricing).

A market with 25 products and five owners. Demand: logit with
delta_j = X_j beta - alpha p_j + xi_j, beta = (0.1, 0.25, 0.1), alpha = 2.
Marginal cost: mc_j = X_j gamma + omega_j, gamma = (0.2, 0.25, 0.3).
Product data (X, xi, omega, owner) are read from a CSV file with columns
prod_id, owner, X1, X2, X3, xi, omega.

The equilibrium is computed by best-response iteration on firm-level common
markups with the Lambert-W formula
    m_f = (1/alpha) [1 + W(E_f e^{-1} / D_{-f})],
and checked against the aggregative characterization of uniqueness.
Also computes pass-through of a firm-wide cost increase at fixed rival prices.
Writes figures/ch07/src/logit_pricing_numbers.txt.
"""
import os
import sys

import numpy as np
from scipy.special import lambertw

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "..", "ProblemSet_1", "data_simulations.csv")

BETA = np.array([0.1, 0.25, 0.1])
ALPHA = 2.0
GAMMA = np.array([0.2, 0.25, 0.3])


def load(path=DATA):
    d = np.genfromtxt(path, delimiter=",", names=True)
    X = np.column_stack([d["X1"], d["X2"], d["X3"]])
    return X, d["xi"], d["omega"], d["owner"].astype(int)


def equilibrium(v, owners, tol=1e-14, maxit=10000):
    """Common-markup best-response iteration. v_j = X_j beta + xi_j - alpha mc_j."""
    firms = np.unique(owners)
    m = np.full(v.shape, 1.0 / ALPHA)
    for it in range(maxit):
        e = np.exp(v - ALPHA * m)
        m_new = m.copy()
        for f in firms:
            own = owners == f
            Ef = np.exp(v[own]).sum()
            Dminus = 1.0 + e[~own].sum()
            m_new[own] = (1.0 + lambertw(Ef * np.exp(-1.0) / Dminus).real) / ALPHA
        if np.max(np.abs(m_new - m)) < tol:
            return m_new, it + 1
        m = m_new
    raise RuntimeError("no convergence")


def shares(v, m):
    e = np.exp(v - ALPHA * m)
    tot = 1.0 + e.sum()
    return e / tot, 1.0 / tot, tot


def main():
    X, xi, om, owner = load()
    mc = X @ GAMMA + om
    v = X @ BETA + xi - ALPHA * mc
    out = []
    res = {}
    for label, own in [("single", np.arange(len(v))), ("multi", owner)]:
        m, nit = equilibrium(v, own)
        s, s0, Lam = shares(v, m)
        p = mc + m
        # FOC residual: s_j [1 - alpha m_j + alpha sum_{k in f} m_k s_k]
        foc = []
        for j in range(len(v)):
            same = own == own[j]
            foc.append(s[j] * (1 - ALPHA * m[j] + ALPHA * np.sum(m[same] * s[same])))
        cs = np.log(Lam) / ALPHA
        res[label] = dict(m=m, s=s, s0=s0, p=p, cs=cs, profit=np.sum(m * s), Lam=Lam)
        out.append(f"=== {label}-product ownership: {nit} iterations, max |FOC| {np.max(np.abs(foc)):.2e}")
        out.append(f"markups: min {m.min():.3f} max {m.max():.3f}; 1/alpha = {1 / ALPHA:.3f}")
        out.append(f"s0 = {s0:.4f}, Lambda = {Lam:.3f}, CS = {cs:.4f}, total profit = {np.sum(m * s):.4f}, "
                   f"share-weighted mean price = {np.sum(s * p) / np.sum(s):.4f}")
        for f in np.unique(owner):
            sel = owner == f
            Sf = s[sel].sum()
            out.append(f"  firm {f}: {sel.sum()} products, S_f = {100 * Sf:.2f}%, "
                       f"markups {m[sel].min():.3f}-{m[sel].max():.3f}, "
                       f"1/(alpha(1-S_f)) = {1 / (ALPHA * (1 - Sf)):.3f}, profit x100 = {100 * np.sum(m[sel] * s[sel]):.3f}")
    dp = res["multi"]["p"] - res["single"]["p"]
    out.append(f"price increase single->multi: min {dp.min():.3f} max {dp.max():.3f}")
    out.append(f"total profit change: {100 * (res['multi']['profit'] / res['single']['profit'] - 1):.1f}%")
    # uniqueness check: restarts
    rng = np.random.default_rng(3)
    devs = []
    for label, own in [("single", np.arange(len(v))), ("multi", owner)]:
        firms = np.unique(own)
        mstar = res[label]["m"]
        maxdev = 0.0
        for _ in range(200):
            m = rng.uniform(0, 5, size=len(v))
            for it in range(10000):
                e = np.exp(v - ALPHA * m)
                m_new = m.copy()
                for f in firms:
                    sel = own == f
                    Ef = np.exp(v[sel]).sum()
                    m_new[sel] = (1.0 + lambertw(Ef * np.exp(-1.0) / (1.0 + e[~sel].sum())).real) / ALPHA
                if np.max(np.abs(m_new - m)) < 1e-13:
                    break
                m = m_new
            maxdev = max(maxdev, np.max(np.abs(m_new - mstar)))
        devs.append(maxdev)
    out.append(f"200 random restarts: max deviation single {devs[0]:.1e}, multi {devs[1]:.1e}")
    # pass-through at fixed rival prices: firm with the largest S_f under multi
    s = res["multi"]["s"]
    Sf = {f: s[owner == f].sum() for f in np.unique(owner)}
    fbig = max(Sf, key=Sf.get)
    h = 1e-6
    mc2 = mc.copy()
    mc2[owner == fbig] += h
    v2 = X @ BETA + xi - ALPHA * mc2
    e_riv = np.exp(v - ALPHA * res["multi"]["m"])[owner != fbig].sum()
    Ef2 = np.exp(v2[owner == fbig]).sum()
    m_f2 = (1.0 + lambertw(Ef2 * np.exp(-1.0) / (1.0 + e_riv)).real) / ALPHA
    m_f1 = res["multi"]["m"][owner == fbig][0]
    rho = 1 + (m_f2 - m_f1) / h
    out.append(f"pass-through, firm {fbig} (S_f = {Sf[fbig]:.4f}), common cost rise, rivals fixed: "
               f"{rho:.4f} vs 1 - S_f = {1 - Sf[fbig]:.4f}")
    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(HERE, "logit_pricing_numbers.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
