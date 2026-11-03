"""Aggregation example for Chapter 5, Section 5.4 (almost exact aggregation of AIDS).

Running-example parameters (identical to aids_example.py):
alpha_0 = 0, alpha = (0.5, 0.3, 0.2), beta = (-0.10, 0.04, 0.06),
Gamma symmetric with zero row and column sums, prices p = (1, 1, 1).

Two households with total expenditure I_1 = 1 and I_2 = 3.  The script computes
household budget shares, the aggregate (expenditure-weighted) shares, mean
expenditure, the representative budget level I_0, the Theil index, the
Deaton-Muellbauer index kappa = Z / H, and the shares of a consumer at I_0 and
at mean expenditure.  It also compares the sum of household compensating
variations with the representative-consumer approximation for a 10 percent and
a 50 percent rise in the price of good 3, and reports the share-validity range
of ln I at unit prices.  Output: aggregation_example_numbers.txt.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "aggregation_example_numbers.txt")

alpha0 = 0.0
alpha = np.array([0.5, 0.3, 0.2])
beta = np.array([-0.10, 0.04, 0.06])
Gamma = np.array([[-0.20, 0.12, 0.08],
                  [0.12, -0.22, 0.10],
                  [0.08, 0.10, -0.18]])


def lnP(lnp):
    return alpha0 + alpha @ lnp + 0.5 * lnp @ Gamma @ lnp


def b_of(lnp):
    return np.exp(beta @ lnp)


def shares(lnp, lnI):
    return alpha + Gamma @ lnp + beta * (lnI - lnP(lnp))


def ln_e(lnp, u):
    return lnP(lnp) + u * b_of(lnp)


def indirect_u(lnp, lnI):
    return (lnI - lnP(lnp)) / b_of(lnp)


lines = []
lnp0 = np.zeros(3)
I = np.array([1.0, 3.0])
H = len(I)
X = I.sum()
Ibar = X / H
lnI0 = np.sum(I / X * np.log(I))
I0 = np.exp(lnI0)
theil = np.mean(I / Ibar * np.log(I / Ibar))
lnZ = -np.sum(I / X * np.log(I / X))
Z = np.exp(lnZ)
kappa = Z / H

W = np.array([shares(lnp0, np.log(Ih)) for Ih in I])
wbar = (I[:, None] * W).sum(axis=0) / X
w_rep = shares(lnp0, lnI0)
w_mean = shares(lnp0, np.log(Ibar))

lines.append("Two-household aggregation at unit prices")
lines.append(f"  incomes I = {I}, H = {H}, X = {X:.4f}, mean Ibar = {Ibar:.4f}, ln Ibar = {np.log(Ibar):.5f}")
lines.append(f"  ln I0 = {lnI0:.5f}, I0 = {I0:.5f}")
lines.append(f"  Theil index T = {theil:.5f}; ln Ibar + T = {np.log(Ibar) + theil:.5f}")
lines.append(f"  ln Z = {lnZ:.5f}, Z = {Z:.5f}, kappa = Z/H = {kappa:.5f}, exp(-T) = {np.exp(-theil):.5f}, Ibar/kappa = {Ibar / kappa:.5f}")
for h in range(H):
    lines.append(f"  household {h + 1} (I = {I[h]:.0f}, ln I = {np.log(I[h]):.5f}): shares = " +
                 ", ".join(f"{x:.4f}" for x in W[h]) +
                 "; quantities = " + ", ".join(f"{x:.4f}" for x in W[h] * I[h]))
lines.append("  aggregate quantities = " + ", ".join(f"{x:.4f}" for x in (W * I[:, None]).sum(axis=0)))
lines.append("  aggregate shares wbar = " + ", ".join(f"{x:.4f}" for x in wbar))
lines.append("  shares of consumer at I0 = " + ", ".join(f"{x:.4f}" for x in w_rep))
lines.append("  shares of consumer at Ibar = " + ", ".join(f"{x:.4f}" for x in w_mean))
lines.append(f"  max |wbar - w(I0)| = {np.max(np.abs(wbar - w_rep)):.2e}; max |wbar - w(Ibar)| = {np.max(np.abs(wbar - w_mean)):.4f}")

# Welfare: sum of household CVs versus representative approximation
for rise in (0.10, 0.50):
    lnp1 = np.array([0.0, 0.0, np.log(1 + rise)])
    cvs = []
    for Ih in I:
        u = indirect_u(lnp0, np.log(Ih))
        cvs.append(np.exp(ln_e(lnp1, u)) - Ih)
    cvs = np.array(cvs)
    u0 = indirect_u(lnp0, lnI0)
    cv0 = np.exp(ln_e(lnp1, u0)) - I0
    approx = X * cv0 / I0
    r = b_of(lnp1) / b_of(lnp0)
    lines.append(f"Welfare, p3 rises by {100 * rise:.0f} percent: b(p')/b(p) = {r:.6f}")
    lines.append("  household CVs = " + ", ".join(f"{c:.5f}" for c in cvs) +
                 "; as share of budget = " + ", ".join(f"{c / Ih:.5f}" for c, Ih in zip(cvs, I)))
    lines.append(f"  sum of household CVs = {cvs.sum():.6f}")
    lines.append(f"  representative CV at I0 = {cv0:.6f} = {cv0 / I0:.6f} of I0; scaled to X: {approx:.6f}")
    lines.append(f"  difference sum - scaled = {cvs.sum() - approx:.3e} (relative {100 * (cvs.sum() - approx) / cvs.sum():.4f} percent)")

# share-validity range at unit prices (u = ln I because ln a = 0 and ln b - ln a = 1)
a_i = alpha.copy()
b_i = alpha + beta
up, lo = [], []
for ai, bi in zip(a_i, b_i):
    if bi > ai:
        up.append((1 - ai) / (bi - ai))
        lo.append(-ai / (bi - ai))
    elif bi < ai:
        up.append(ai / (ai - bi))
        lo.append(-(1 - ai) / (ai - bi))
lines.append(f"PIGLOG validity at unit prices: a_i = {a_i}, b_i = {b_i}")
lines.append(f"  upper crossings = {np.round(up, 3)}, lower crossings = {np.round(lo, 3)}")
lines.append(f"  valid u = ln I in [{max(lo):.3f}, {min(up):.3f}]")

open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
