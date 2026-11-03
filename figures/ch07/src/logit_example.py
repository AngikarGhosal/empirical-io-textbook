"""Illustrative four-product logit market for Chapter 7 (Example ex:07-logit-numbers).

Prints market shares, the full price-elasticity matrix (including the outside-good row),
diversion ratios, and the category elasticities, and writes the LaTeX rows that are
pasted into Table tab:07-logit-elast.  Also locates the unit-elastic point on the
single-product demand curve drawn in Figure fig:07-logit-demand.

Conventions
-----------
* delta_j = x_j beta - alpha p_j + xi_j is the mean utility, delta_0 = 0.
* Elasticity matrix E[j, k] = (d s_j / d p_k) * p_k / s_j, signed (own entries negative).
  Rows index the product whose share responds, columns the product whose price changes.
* Diversion D[j, k] = -(d s_k / d p_j) / (d s_j / d p_j): share of the sales lost by j
  that go to k (k = 0 is the outside good).
"""
import numpy as np
from scipy.optimize import brentq

alpha = 1.0
delta = np.array([1.0, 0.5, 0.1, -0.5])
price = np.array([2.0, 2.5, 3.0, 4.0])
J = len(delta)

expd = np.exp(delta)
Lam = 1.0 + expd.sum()
s = expd / Lam
s0 = 1.0 / Lam
quality = delta + alpha * price          # x beta + xi implied by delta and price

# Share derivatives with respect to prices: ds_j/dp_k
dsdp = np.empty((J, J))
for j in range(J):
    for k in range(J):
        dsdp[j, k] = -alpha * s[j] * (1 - s[j]) if j == k else alpha * s[j] * s[k]
ds0dp = alpha * s0 * s                  # ds_0/dp_k

E = dsdp * price[None, :] / s[:, None]
E0 = ds0dp * price / s0

# Diversion ratios
D = np.empty((J, J + 1))                # column 0 = outside good
for j in range(J):
    D[j, 0] = ds0dp[j] / (-dsdp[j, j])
    for k in range(J):
        D[j, k + 1] = np.nan if k == j else dsdp[k, j] / (-dsdp[j, j])

# Category elasticities
cat_semi = alpha * s0                    # -d ln(1-s0) / d Delta (common absolute rise)
pbar = (s * price).sum() / (1 - s0)      # share-weighted mean inside price
cat_prop = alpha * s0 * pbar             # -d ln(1-s0) / d ln(1+lambda) (proportional rise)

np.set_printoptions(precision=4, suppress=True)
print("quality (x beta + xi):", quality)
print("shares s_j:", s, " s_0 =", round(s0, 4), " sum inside =", round(1 - s0, 4))
print("own elasticities (positive convention):", -np.diag(E))
print("elasticity matrix E[j,k]:\n", E)
print("outside-good row E[0,k]:", E0)
print("diversion D[j, (0,1..J)]:\n", D)
print("row sums of diversion:", np.nansum(D, axis=1))
print("category semi-elasticity alpha*s0 =", round(cat_semi, 4))
print("share-weighted mean inside price =", round(pbar, 4))
print("category elasticity (proportional) alpha*s0*pbar =", round(cat_prop, 4))

print("\nLaTeX rows for the elasticity block:")
for j in range(J):
    cells = " & ".join(f"${E[j, k]:.3f}$" for k in range(J))
    print(f"${j+1}$ & ${price[j]:.1f}$ & ${s[j]:.3f}$ & {cells} \\\\")
cells = " & ".join(f"${E0[k]:.3f}$" for k in range(J))
print(f"$0$ & & ${s0:.3f}$ & {cells} \\\\")
print("\nLaTeX rows for the diversion block (columns 0,1,..,J):")
for j in range(J):
    cells = " & ".join("" if np.isnan(D[j, k]) else f"${D[j, k]:.3f}$" for k in range(J + 1))
    print(f"${j+1}$ & {cells} \\\\")

# Unit-elastic point of the demand curve in Figure fig:07-logit-demand:
# s(p) = exp(2 - p) / (2 + exp(2 - p)), alpha = 1, elasticity p (1 - s(p)).
sfun = lambda p: np.exp(2 - p) / (2 + np.exp(2 - p))
pu = brentq(lambda p: p * (1 - sfun(p)) - 1.0, 0.1, 5.0)
print(f"\nFigure: unit-elastic price p = {pu:.4f}, share there = {sfun(pu):.4f}")
for p in [0.5, 1, 2, 3, 4, 5, 6]:
    print(f"  p = {p}: s = {sfun(p):.4f}, elasticity = {p * (1 - sfun(p)):.4f}")
