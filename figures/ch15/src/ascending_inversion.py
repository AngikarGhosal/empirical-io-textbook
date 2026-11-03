"""Price-only inversion in simulated button auctions (Section 15.2.2, fig:15-asc-sim).

Design (our simulation):
  T = 500 ascending (button) auctions, n = 4 symmetric bidders,
  values i.i.d. lognormal with mu = 0, sigma = 0.5, no reserve price.
  The econometrician observes only the transaction price, the second-highest value.

Estimator (Athey and Haile 2002, Theorem 1; Athey and Haile 2007, eqs. 4.3-4.6):
  G_hat(v) = empirical CDF of prices,  F_hat(v) = Lambda_n^{-1}(G_hat(v)),
  Lambda_n(x) = n x^(n-1) - (n-1) x^n, inverted by bisection.

Inference: nonparametric bootstrap that resamples whole auctions (B = 500 draws),
pointwise 5th and 95th percentiles.  Delta-method standard error:
  Var(F_hat(v)) ~ G(1-G) / (T Lambda_n'(F)^2),  Lambda_n'(x) = n(n-1) x^(n-2) (1-x).
Benchmark: standard error of the empirical CDF of all nT values, sqrt(F(1-F)/(nT)).

Output: figures/ch15/asc_inversion.pdf and figures/ch15/src/asc_inversion_numbers.txt
Seed: 15201.
"""
from pathlib import Path

import numpy as np
from scipy.stats import lognorm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_FIG = HERE.parent / "asc_inversion.pdf"
OUT_TXT = HERE / "asc_inversion_numbers.txt"

SEED = 15201
T, N, MU, SIGMA, B = 500, 4, 0.0, 0.5, 500

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


def Lam(x, n):
    return n * x ** (n - 1) - (n - 1) * x ** n


def dLam(x, n):
    return n * (n - 1) * x ** (n - 2) * (1 - x)


def Lam_inv(y, n, iters=60):
    """Vectorized bisection for Lambda_n^{-1} on [0, 1]."""
    y = np.asarray(y, dtype=float)
    lo = np.zeros_like(y)
    hi = np.ones_like(y)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = Lam(mid, n) < y
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def ecdf(sample, grid):
    s = np.sort(sample)
    return np.searchsorted(s, grid, side="right") / s.size


rng = np.random.default_rng(SEED)
dist = lognorm(s=SIGMA, scale=np.exp(MU))
values = dist.rvs(size=(T, N), random_state=rng)
prices = np.sort(values, axis=1)[:, N - 2]          # second-highest value

grid = np.linspace(0.25, 3.0, 221)
F_true = dist.cdf(grid)
G_hat = ecdf(prices, grid)
F_hat = Lam_inv(G_hat, N)

boot = np.empty((B, grid.size))
for b in range(B):
    idx = rng.integers(0, T, size=T)
    boot[b] = Lam_inv(ecdf(prices[idx], grid), N)
lo_band, hi_band = np.percentile(boot, [5, 95], axis=0)
se_boot = boot.std(axis=0, ddof=1)

G_true = Lam(F_true, N)
with np.errstate(divide="ignore", invalid="ignore"):
    se_delta = np.sqrt(G_true * (1 - G_true) / (T * dLam(F_true, N) ** 2))
se_allbids = np.sqrt(F_true * (1 - F_true) / (N * T))

# coverage of the true CDF by the pointwise band, and summary numbers
inside = np.mean((F_true >= lo_band) & (F_true <= hi_band))
sup_err = np.max(np.abs(F_hat - F_true))
lines = [
    f"seed={SEED}, T={T}, n={N}, lognormal(mu={MU}, sigma={SIGMA}), bootstrap B={B}",
    f"share of grid points where true F lies in the 90% pointwise band: {inside:.3f}",
    f"sup |F_hat - F| on grid [0.25, 3]: {sup_err:.4f}",
]
for v in (0.5, 0.75, 1.0, 1.5, 2.0):
    k = np.argmin(np.abs(grid - v))
    lines.append(
        f"v={grid[k]:.3f}: F={F_true[k]:.3f} F_hat={F_hat[k]:.3f} se_boot={se_boot[k]:.4f} "
        f"se_delta={se_delta[k]:.4f} se_allbids={se_allbids[k]:.4f} ratio_delta/all={se_delta[k]/se_allbids[k]:.2f}"
    )
OUT_TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

ax = axes[0]
ax.fill_between(grid, lo_band, hi_band, color=BLUE, alpha=0.18, linewidth=0)
ax.plot(grid, F_true, color=GRAY, lw=2.0)
ax.plot(grid, F_hat, color=BLUE, lw=1.6, ls="--")
ax.set_xlabel(r"value $v$")
ax.set_ylabel(r"$F(v)$")
ax.set_ylim(-0.02, 1.02)
ax.set_xlim(grid[0], grid[-1])
ax.annotate(r"true $F$", xy=(1.55, dist.cdf(1.55)), xytext=(1.9, 0.62), color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
ax.annotate(r"$\hat F=\Lambda_4^{-1}(\hat H_4)$", xy=(0.95, F_hat[np.argmin(np.abs(grid - 0.95))]),
            xytext=(1.35, 0.25), color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
ax.text(2.1, 0.86, "90% bootstrap\nband", color=BLUE, fontsize=8.5, ha="left", va="top")
ax.set_title("(a) Estimate from transaction prices", fontsize=10)

ax = axes[1]
mask = (grid >= 0.6)
ax.plot(grid[mask], se_boot[mask], color=BLUE, lw=1.6)
ax.plot(grid[mask], se_delta[mask], color=ORANGE, lw=1.6, ls="--")
ax.plot(grid[mask], se_allbids[mask], color=AQUA, lw=1.6, ls=":")
ax.set_xlabel(r"value $v$")
ax.set_ylabel("standard error")
ax.set_xlim(0.6, grid[-1])
ax.set_ylim(0, 0.03)
kb = np.argmin(np.abs(grid - 0.8))
ax.annotate("bootstrap", xy=(grid[kb], se_boot[kb]), xytext=(1.05, 0.024), color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
kd = np.argmin(np.abs(grid - 1.6))
ax.annotate(r"delta method", xy=(grid[kd], se_delta[kd]), xytext=(1.85, 0.018), color=ORANGE,
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
ka = np.argmin(np.abs(grid - 1.2))
ax.annotate("all bids observed", xy=(grid[ka], se_allbids[ka]), xytext=(1.3, 0.0012), color="#127a55",
            arrowprops=dict(arrowstyle="-", color=AQUA, lw=0.8))
ax.set_title("(b) Pointwise standard errors", fontsize=10)

fig.tight_layout()
fig.savefig(OUT_FIG)
fig.savefig(HERE / "asc_inversion_preview.png", dpi=130)
print("saved", OUT_FIG)
