"""Affiliated private values example used in Sections 14.8-14.9 (Example B).

Two bidders, private values V_i = S_i, joint density f(s1, s2) = (4/5)(1 + s1 s2) on [0,1]^2.
The script
  1. verifies symbolically that the closed-form first-price bid
        beta_FP(s) = s - ((2 + s^2)^{3/2} - 2^{3/2}) / (3 s sqrt(2 + s^2))
     solves the Milgrom-Weber differential equation beta' = (s - beta) f(s|s) / F(s|s);
  2. computes expected revenue in the second-price (= English, n = 2) and first-price auctions,
     and the bidders' expected surplus, by numerical integration and by Monte Carlo;
  3. computes the same numbers for the independent-uniform benchmark;
  4. draws figures/ch14/apv_bids.pdf.

Run from any directory: python apv_example.py
"""
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "apv_bids.pdf"

# ---------------------------------------------------------------- symbolic checks
s, y, a, t = sp.symbols("s y alpha t", positive=True)
f_joint = sp.Rational(4, 5) * (1 + s * y)
f_marg = sp.integrate(f_joint, (y, 0, 1))                 # marginal density of S_1
f_cond = sp.simplify(f_joint / f_marg)                    # density of Y_1 = S_2 given S_1 = s
F_cond = sp.simplify(sp.integrate(f_cond.subs(y, t), (t, 0, y)))
ratio = sp.simplify((f_cond / F_cond).subs(y, s))         # f(s|s) / F(s|s)
print("marginal density of S_1:", sp.simplify(f_marg))
print("f(y|s) =", f_cond, "   F(y|s) =", F_cond)
print("f(s|s)/F(s|s) =", ratio)
print("partial fractions:", sp.apart(ratio, s))

L = sp.simplify(sp.exp(-sp.integrate(ratio.subs(s, t), (t, a, s))))
print("L(alpha|s) =", L)

beta_fp = s - ((2 + s**2) ** sp.Rational(3, 2) - 2 ** sp.Rational(3, 2)) / (3 * s * sp.sqrt(2 + s**2))
ode_resid = sp.simplify(sp.diff(beta_fp, s) - (s - beta_fp) * ratio)
print("ODE residual (should be 0):", ode_resid)
beta_from_L = sp.simplify(s - sp.integrate(L, (a, 0, s)))
print("beta from v(s,s) - int L d alpha minus closed form:", sp.simplify(beta_from_L - beta_fp))
print("limit of beta_FP as s -> 0:", sp.limit(beta_fp, s, 0))
print("limit of beta_FP / s as s -> 0:", sp.limit(beta_fp / s, s, 0))
print("beta_FP(1) =", sp.nsimplify(beta_fp.subs(s, 1)), "=", float(beta_fp.subs(s, 1)))

# ---------------------------------------------------------------- revenues (exact / numerical)
s1, s2 = sp.symbols("s1 s2", positive=True)
dens = sp.Rational(4, 5) * (1 + s1 * s2)
rev_sp = 2 * sp.integrate(sp.integrate(s2 * dens, (s2, 0, s1)), (s1, 0, 1))   # E[min]
e_max = 2 * sp.integrate(sp.integrate(s1 * dens, (s2, 0, s1)), (s1, 0, 1))    # E[max]
print("SPA revenue E[min(S1,S2)] =", rev_sp, "=", float(rev_sp))
print("E[max(S1,S2)] =", e_max, "=", float(e_max))

beta_np = sp.lambdify(s, beta_fp, "numpy")


def beta_num(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 1e-6
    out[small] = x[small] / 2.0
    out[~small] = beta_np(x[~small])
    return out


def dens_np(x1, x2):
    return 0.8 * (1.0 + x1 * x2)


# FPA revenue = E[beta(max)] = 2 * int_0^1 int_0^{s1} beta(s1) f(s1, s2) ds2 ds1
rev_fp = 2 * integrate.dblquad(lambda x2, x1: beta_num(np.array([x1]))[0] * dens_np(x1, x2),
                               0, 1, lambda x1: 0.0, lambda x1: x1, epsabs=1e-12, epsrel=1e-12)[0]
print("FPA revenue E[beta_FP(max)] = %.6f" % rev_fp)
print("SPA - FPA = %.6f" % (float(rev_sp) - rev_fp))
print("bidder surplus SPA (E[max]-E[min]) = %.6f, FPA = %.6f" % (float(e_max - rev_sp), float(e_max) - rev_fp))

# independent uniform benchmark (n = 2)
print("independent uniform benchmark: revenue 1/3 = %.6f in both formats, E[max] = 2/3" % (1 / 3))

# exact correlation of (S1, S2)
m1 = sp.integrate(sp.integrate(s1 * dens, (s2, 0, 1)), (s1, 0, 1))
m2 = sp.integrate(sp.integrate(s1**2 * dens, (s2, 0, 1)), (s1, 0, 1))
m12 = sp.integrate(sp.integrate(s1 * s2 * dens, (s2, 0, 1)), (s1, 0, 1))
corr = sp.simplify((m12 - m1**2) / (m2 - m1**2))
print("E[S] =", m1, " Cov =", sp.simplify(m12 - m1**2), " Corr =", corr, "=", float(corr))

# independent benchmark with the same marginal density (2/5)(2 + s): FPA bid s - int_0^s F / F(s)
Fm = sp.integrate(sp.Rational(2, 5) * (2 + t), (t, 0, s))
beta_ind = sp.simplify(s - sp.integrate(Fm.subs(s, t), (t, 0, s)) / Fm)
print("same-marginal independent FPA bid:", beta_ind)
for val in (0.25, 0.5, 0.75, 1.0):
    print("   s = %.2f: affiliated %.4f, same-marginal independent %.4f"
          % (val, float(beta_fp.subs(s, val)), float(beta_ind.subs(s, val))))

# ---------------------------------------------------------------- Monte Carlo check
rng = np.random.default_rng(14082026)
N = 2_000_000
# accept-reject from the uniform proposal; density bounded by 8/5
x1 = rng.random(3 * N)
x2 = rng.random(3 * N)
u = rng.random(3 * N)
keep = u * 1.6 <= dens_np(x1, x2)
x1, x2 = x1[keep][:N], x2[keep][:N]
mx, mn = np.maximum(x1, x2), np.minimum(x1, x2)
print("Monte Carlo (%d draws): SPA %.4f, FPA %.4f, corr(S1,S2) %.4f"
      % (len(mx), mn.mean(), beta_num(mx).mean(), np.corrcoef(x1, x2)[0, 1]))

# a few bid values for the text
for val in (0.25, 0.5, 0.75, 1.0):
    print("s = %.2f: beta_FP = %.4f, s/2 = %.4f" % (val, beta_num(np.array([val]))[0], val / 2))

# ---------------------------------------------------------------- figure
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})

grid = np.linspace(0.0, 1.0, 401)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(grid, grid, color=BLUE, lw=1.8, label=r"second price: $\beta^{SP}(s)=s$")
ax.plot(grid, beta_num(grid), color=ORANGE, lw=1.8,
        label=r"first price, affiliated: $\beta^{FP}(s)$")
ax.plot(grid, grid / 2, color=GREEN, lw=1.6, ls=(0, (5, 3)),
        label=r"first price, independent: $\frac{s}{2}$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r"signal (value) $s$")
ax.set_ylabel(r"bid")
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
