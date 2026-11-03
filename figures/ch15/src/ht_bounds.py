"""Haile-Tamer bounds on simulated open-outcry auctions (Section 15.2.6, fig:15-ht).

Design (our simulation):
  T = 1,000 auctions, 500 with n = 3 and 500 with n = 5 bidders,
  values i.i.d. lognormal(mu = 0, sigma = 0.5), minimum increment Delta = 0.05.
  Losing bidders record bids b_i = v_i * U_i with U_i ~ U[0.8, 1] (they stop short of their values).
  The winner's bid (the transaction price) is
      P = max{ v^(n-1:n) - Delta * U', max of losing bids },  U' ~ U[0, 1],
  so every bid is at most the bidder's value (assumption HT1) and the second-highest value is at most
  P + Delta (assumption HT2).

Bounds (Athey and Haile 2007, eqs. 4.10-4.11, Theorem 4.3):
  F+(v) = min_{i,n} Lambda_{i:n}^{-1}( G_hat_{B^(i:n)}(v) ),
  F-(v) = max_n   Lambda_{n-1:n}^{-1}( G_hat_{B^(n:n) + Delta}(v) ).
In finite samples a candidate is used at v only if its empirical CDF lies in [TRIM, 1 - TRIM], TRIM = 0.01
(an order statistic with no observations on one side of v carries no information there).
Naive estimate: treat the highest losing bid B^(n-1:n) as the second-highest value and invert,
averaging the two estimates for n = 3 and n = 5.
Reserve bounds (Athey and Haile 2007, Theorem 4.4) with seller value v0 = 0:
  pi1(r) = r (1 - F+(r)),  pi2(r) = r (1 - F-(r)),  r- = sup{r < r1*: pi2(r) <= pi1*},
  r+ = inf{r > r1*: pi2(r) <= pi1*}.

Output: figures/ch15/ht_bounds.pdf and figures/ch15/src/ht_bounds_numbers.txt
Seed: 15202.
"""
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize_scalar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_FIG = HERE.parent / "ht_bounds.pdf"
OUT_TXT = HERE / "ht_bounds_numbers.txt"

SEED = 15202
T_PER_N, NS, MU, SIGMA, DELTA = 500, (3, 5), 0.0, 0.5, 0.05
TRIM = 0.01

BLUE, ORANGE, AQUA, GRAY = "#2A78D6", "#EB6834", "#1BAF7A", "#52514E"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": "#222222",
    "xtick.color": GRAY,
    "ytick.color": GRAY,
})


def Lam(x, i, n):
    """CDF of the i-th lowest of n i.i.d. draws as a function of x = F(v)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for j in range(i, n + 1):
        out += comb(n, j) * x ** j * (1 - x) ** (n - j)
    return out


def Lam_inv(y, i, n, iters=60):
    y = np.asarray(y, dtype=float)
    lo, hi = np.zeros_like(y), np.ones_like(y)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = Lam(mid, i, n) < y
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def ecdf(sample, grid):
    s = np.sort(sample)
    return np.searchsorted(s, grid, side="right") / s.size


rng = np.random.default_rng(SEED)
dist = lognorm(s=SIGMA, scale=np.exp(MU))
grid = np.linspace(0.2, 3.0, 561)
F_true = dist.cdf(grid)

upper_cands, lower_cands, naive_cands = [], [], []
for n in NS:
    v = np.sort(dist.rvs(size=(T_PER_N, n), random_state=rng), axis=1)
    losing = v[:, : n - 1] * rng.uniform(0.8, 1.0, size=(T_PER_N, n - 1))
    price = np.maximum(v[:, n - 2] - DELTA * rng.uniform(0, 1, size=T_PER_N), losing.max(axis=1))
    bids = np.sort(np.column_stack([losing, price]), axis=1)      # bids[:, i-1] = B^(i:n)
    assert np.all(bids[:, n - 1] == price)
    assert np.all(np.sort(np.column_stack([losing, price]), axis=1) <= v + 1e-12)   # HT1
    assert np.all(v[:, n - 2] <= price + DELTA + 1e-12)                            # HT2
    for i in range(1, n + 1):
        G = ecdf(bids[:, i - 1], grid)
        # use a candidate only where its empirical CDF is informative (TRIM of the sample on each side)
        upper_cands.append(np.where((G >= TRIM) & (G <= 1 - TRIM), Lam_inv(G, i, n), np.nan))
    G = ecdf(price + DELTA, grid)
    lower_cands.append(np.where((G >= TRIM) & (G <= 1 - TRIM), Lam_inv(G, n - 1, n), np.nan))
    naive_cands.append(Lam_inv(ecdf(bids[:, n - 2], grid), n - 1, n))

upper_cands, lower_cands = np.array(upper_cands), np.array(lower_cands)
F_plus = np.where(np.all(np.isnan(upper_cands), axis=0), 1.0, np.nanmin(np.where(np.isnan(upper_cands), np.inf, upper_cands), axis=0))
F_minus = np.where(np.all(np.isnan(lower_cands), axis=0), 0.0, np.nanmax(np.where(np.isnan(lower_cands), -np.inf, lower_cands), axis=0))
F_naive = np.mean(naive_cands, axis=0)

# informative range: both bounds use at least one candidate estimator
ok = (~np.all(np.isnan(upper_cands), axis=0)) & (~np.all(np.isnan(lower_cands), axis=0))
lo_v, hi_v = grid[ok].min(), grid[ok].max()

# reserve-price bounds, seller value v0 = 0, computed on the informative range
pi_true = grid * (1 - F_true)
pi1 = np.where(ok, grid * (1 - F_plus), np.nan)
pi2 = np.where(ok, grid * (1 - F_minus), np.nan)
k1 = np.nanargmax(pi1)
r1, pi1_star = grid[k1], pi1[k1]
left = np.where(ok & (grid < r1) & (pi2 <= pi1_star))[0]
right = np.where(ok & (grid > r1) & (pi2 <= pi1_star))[0]
r_minus = grid[left.max()] if left.size else lo_v
r_plus = grid[right.min()] if right.size else hi_v
r_star = minimize_scalar(lambda r: -r * dist.sf(r), bounds=(0.2, 3.0), method="bounded").x

inr = ok
share_inside = np.mean((F_true[inr] >= F_minus[inr] - 1e-12) & (F_true[inr] <= F_plus[inr] + 1e-12))
share_cross = np.mean(F_minus[inr] > F_plus[inr])
lines = [
    f"seed={SEED}, n in {NS}, {T_PER_N} auctions each, lognormal(mu={MU}, sigma={SIGMA}), Delta={DELTA}, TRIM={TRIM}",
    f"informative range of v: [{lo_v:.3f}, {hi_v:.3f}]",
    f"on that range: share with F- <= F <= F+: {share_inside:.3f}; share with F- > F+: {share_cross:.3f}",
    f"mean gap F+ - F- on that range: {np.mean(F_plus[inr] - F_minus[inr]):.3f}; max gap: {np.max(F_plus[inr] - F_minus[inr]):.3f}",
    f"max (F_naive - F_true) on that range: {np.max(F_naive[inr] - F_true[inr]):.3f}",
    f"true optimal reserve r* (v0 = 0): {r_star:.3f}, true max pi = {r_star*dist.sf(r_star):.3f}",
    f"r1* = {r1:.3f}, pi1* = {pi1_star:.3f}, r- = {r_minus:.3f}, r+ = {r_plus:.3f}",
]
for vv in (0.6, 0.8, 1.0, 1.2, 1.5):
    k = np.argmin(np.abs(grid - vv))
    lines.append(f"v={grid[k]:.3f}: F-={F_minus[k]:.3f} F={F_true[k]:.3f} F+={F_plus[k]:.3f} naive={F_naive[k]:.3f}")
OUT_TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ----------------------------------------------------------------- figure
Fp = np.where(ok, F_plus, np.nan)
Fm = np.where(ok, F_minus, np.nan)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

ax = axes[0]
ax.fill_between(grid, Fm, Fp, where=ok, color=BLUE, alpha=0.15, linewidth=0)
ax.plot(grid, Fp, color=BLUE, lw=1.5)
ax.plot(grid, Fm, color=BLUE, lw=1.5, ls="--")
ax.plot(grid, F_true, color=GRAY, lw=2.0)
ax.plot(grid, np.where(ok, F_naive, np.nan), color=ORANGE, lw=1.5, ls=":")
ax.set_xlim(0.3, 2.3)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel(r"value $v$")
ax.set_ylabel(r"$F(v)$")
k = np.argmin(np.abs(grid - 0.7))
ax.annotate(r"$\hat F^{+}$", xy=(grid[k], F_plus[k]), xytext=(0.38, 0.45), color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
k = np.argmin(np.abs(grid - 1.25))
ax.annotate(r"$\hat F^{-}$", xy=(grid[k], F_minus[k]), xytext=(1.55, 0.42), color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
k = np.argmin(np.abs(grid - 1.05))
ax.annotate(r"true $F$", xy=(grid[k], F_true[k]), xytext=(1.4, 0.22), color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
k = np.argmin(np.abs(grid - 0.9))
ax.annotate("naive: losing\nbids as values", xy=(grid[k], F_naive[k]), xytext=(0.35, 0.78), color=ORANGE,
            fontsize=8.5, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
ax.set_title("(a) Bounds on the value distribution", fontsize=10)

ax = axes[1]
ax.axvspan(r_minus, r_plus, color=ORANGE, alpha=0.10, linewidth=0)
ax.plot(grid, pi2, color=BLUE, lw=1.5, ls="--")
ax.plot(grid, pi1, color=BLUE, lw=1.5)
ax.plot(grid, pi_true, color=GRAY, lw=2.0)
ax.axhline(pi1_star, color=GRAY, lw=0.7, ls=(0, (2, 2)))
for rr in (r_minus, r_plus):
    ax.axvline(rr, color=ORANGE, lw=1.0, ls=(0, (4, 2)))
ax.plot([r_star], [r_star * dist.sf(r_star)], "o", color=GRAY, ms=5)
ax.set_xlim(0.3, 2.3)
ax.set_ylim(0, 0.75)
ax.set_xlabel(r"reserve price $r$")
ax.set_ylabel(r"$\pi(r)=(r-v_0)(1-F(r))$")
ax.text(r_minus, 0.70, r"$r^{-}$", color=ORANGE, ha="right", va="center")
ax.text(r_plus, 0.70, r"$r^{+}$", color=ORANGE, ha="left", va="center")
k = np.argmin(np.abs(grid - 1.45))
ax.annotate(r"$\hat\pi_2$", xy=(grid[k], pi2[k]), xytext=(1.65, 0.50), color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
k = np.argmin(np.abs(grid - 1.6))
ax.annotate(r"$\hat\pi_1$", xy=(grid[k], pi1[k]), xytext=(1.6, 0.12), color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
k = np.argmin(np.abs(grid - 0.45))
ax.annotate(r"true $\pi$", xy=(grid[k], pi_true[k]), xytext=(0.32, 0.6), color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
ax.set_title(r"(b) Bounds on the optimal reserve ($v_0=0$)", fontsize=10)

fig.tight_layout()
fig.savefig(OUT_FIG)
fig.savefig(HERE / "ht_bounds_preview.png", dpi=130)
print("saved", OUT_FIG)
