"""Numerical check of the aggregative uniqueness argument (thm:04-logit-unique) in the 25-product market.

Market (Section 4.2.7 of the book; data file ProblemSet_1/data_simulations.csv):
  * 25 products with characteristics x_j = (1, X2_j, X3_j), unobserved quality xi_j, cost shock omega_j.
  * Mean utility delta_j = x_j beta - alpha p_j + xi_j with beta = (0.1, 0.25, 0.1), alpha = 2.
  * Marginal cost mc_j = x_j gamma + omega_j with gamma = (0.2, 0.25, 0.3). Market size 1.
  * Ownership: single-product firms (S) or the five owners in the data (M).

Checks:
  1. Equilibrium by best-response iteration with the Lambert-W formula (multi-product version
     m_f = (1/alpha) [1 + W(E_f e^{-1} / Phi_{-f})]).
  2. The aggregate equation Psi(Phi) = sum_f S_f(Phi) + 1/Phi = 1, where S_f(Phi) solves
     h(S) = S exp(1/(1-S)) = E_f / Phi, E_f = sum_{k in f} exp(v_k), v_k = x_k beta + xi_k - alpha mc_k.
     We record the sign changes of Psi - 1 on a 2,000-point log grid over [1, 1e4].
  3. 500 random starting points p0 = mc + U[0, 5]^25 (plus p0 = mc and p0 = mc + 20).
  4. The sup-norm Lipschitz bound of the best-response map, max_f S_f (1 - S_f - s_0)/(1 - S_f),
     at the equilibrium.

Output: figures/ch04/src/logit_uniqueness_check_output.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import lambertw

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / "ProblemSet_1" / "data_simulations.csv"
OUT_TXT = HERE / "logit_uniqueness_check_output.txt"

BETA = np.array([0.1, 0.25, 0.1])
GAMMA = np.array([0.2, 0.25, 0.3])
ALPHA = 2.0

df = pd.read_csv(DATA)
X = df[["X1", "X2", "X3"]].to_numpy()
XI = df["xi"].to_numpy()
OMEGA = df["omega"].to_numpy()
MC = X @ GAMMA + OMEGA
V = X @ BETA + XI - ALPHA * MC          # mean utility at marginal-cost pricing
J = len(df)


def firms_of(structure):
    if structure == "S":
        return [np.array([j]) for j in range(J)]
    owner = df["owner"].to_numpy()
    return [np.where(owner == f)[0] for f in sorted(set(owner))]


def br_iterate(p0, firms, tol=1e-13, maxit=10_000):
    p = p0.copy()
    for it in range(maxit):
        delta = X @ BETA + XI - ALPHA * p
        ed = np.exp(delta)
        pn = p.copy()
        for f in firms:
            phi_minus = 1.0 + ed.sum() - ed[f].sum()
            ef = np.exp(V[f]).sum()
            m = (1.0 + lambertw(ef * np.exp(-1.0) / phi_minus).real) / ALPHA
            pn[f] = MC[f] + m
        if np.max(np.abs(pn - p)) < tol:
            return pn, it + 1
        p = pn
    raise RuntimeError("no convergence")


def h(S):
    with np.errstate(over="ignore"):
        return S * np.exp(1.0 / (1.0 - S))


def S_of_phi(ef, phi):
    target = ef / phi
    return brentq(lambda s: h(s) - target, 0.0, 1.0 - 1e-15, xtol=1e-15)


def psi(phi, firms):
    return sum(S_of_phi(np.exp(V[f]).sum(), phi) for f in firms) + 1.0 / phi


def main():
    rng = np.random.default_rng(257)
    out = []
    for st in ["S", "M"]:
        firms = firms_of(st)
        p_star, nit = br_iterate(MC + 1.0 / ALPHA, firms)
        delta = X @ BETA + XI - ALPHA * p_star
        phi_star = 1.0 + np.exp(delta).sum()
        s = np.exp(delta) / phi_star
        s0 = 1.0 / phi_star
        # FOC residual: s + (Omega o Delta^T)(p - mc) = 0 with Delta_jk = ds_j/dp_k
        Om = np.zeros((J, J))
        for f in firms:
            Om[np.ix_(f, f)] = 1.0
        Dl = ALPHA * np.outer(s, s)
        np.fill_diagonal(Dl, -ALPHA * s * (1.0 - s))
        res = s + (Om * Dl.T) @ (p_star - MC)
        # aggregate equation on a log grid
        grid = np.logspace(0, 4, 2000)
        vals = np.array([psi(g, firms) - 1.0 for g in grid])
        sc = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
        roots = [brentq(lambda x: psi(x, firms) - 1.0, grid[i], grid[i + 1], xtol=1e-12) for i in sc]
        # random starts
        maxdev = 0.0
        starts = [MC.copy(), MC + 20.0] + [MC + rng.uniform(0, 5, J) for _ in range(500)]
        for p0 in starts:
            pr, _ = br_iterate(p0, firms)
            maxdev = max(maxdev, np.max(np.abs(pr - p_star)))
        # Lipschitz bound of the BR map at the equilibrium
        Sf = np.array([s[f].sum() for f in firms])
        lip = np.max(Sf * (1.0 - Sf - s0) / (1.0 - Sf))
        out.append(f"[{st}] BR iteration from mc + 1/alpha converged in {nit} iterations; "
                   f"max |FOC residual| = {np.max(np.abs(res)):.2e}")
        out.append(f"[{st}] Phi* from equilibrium = {phi_star:.4f} = 1/s0 (s0 = {s0:.4f})")
        out.append(f"[{st}] sign changes of Psi(Phi) - 1 on the grid: {len(sc)}; roots: "
                   f"{[round(r, 4) for r in roots]}; Psi(1) - 1 = {psi(1.0, firms) - 1.0:.4f}")
        out.append(f"[{st}] 502 starting points: max deviation from p* = {maxdev:.1e}")
        out.append(f"[{st}] max_f S_f (1 - S_f - s0)/(1 - S_f) at p* = {lip:.4f}; largest S_f = {Sf.max():.4f}")
        out.append(f"[{st}] markups: min {np.min(p_star - MC):.4f}, max {np.max(p_star - MC):.4f}")
    txt = "\n".join(out)
    print(txt)
    OUT_TXT.write_text(txt + "\n")


if __name__ == "__main__":
    main()
