"""Figure fig:15-wc-n (Chapter 15, Section 15.6.2).

Illustrates the logic of the Haile-Hong-Shum test (Athey and Haile 2007,
Lemma 8.1 and Corollary 8.1) in a Gaussian mineral rights model:
common value V ~ N(0, 1), signals S_i = V + e_i with e_i ~ N(0, sigma^2)
independent, sigma = 1. Adding bidders leaves the joint law of the incumbent
bidders' signals and the value unchanged, so participation is exogenous.

The pivotal expected value is
    v(s, s; n) = E[V | S_1 = s, max_{j != 1} S_j = s]
               = E[V | S_1 = S_2 = s, S_j <= s for j = 3..n].
Given S_1 = S_2 = s, V is normal with mean 2 s / (2 + sigma^2) and variance
sigma^2 / (2 + sigma^2). The event {S_j <= s, j >= 3} reweights this
posterior by Phi((s - V) / sigma)^(n - 2). We integrate on a fine grid.

Panel (a): v(s, s; n) against s for n = 2, 3, 5, 10, with E[V | S_1 = s].
Panel (b): the CDF of the random variable v(S_i, S_i; n), the object whose
law the Haile-Hong-Shum test compares across n. Under private values these
CDFs would coincide for all n.

No random numbers are used. Output: figures/ch15/cv_winners_curse.pdf and
cv_winners_curse_numbers.txt.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "cv_winners_curse.pdf"
NUM = HERE / "cv_winners_curse_numbers.txt"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.8,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SIG = 1.0
NS = (2, 3, 5, 10)


def pivotal_value(s, n, sig=SIG):
    m = 2 * s / (2 + sig ** 2)
    sd = np.sqrt(sig ** 2 / (2 + sig ** 2))
    v = np.linspace(m - 8 * sd, m + 8 * sd, 4001)
    w = norm.pdf(v, m, sd) * norm.cdf((s - v) / sig) ** (n - 2)
    return np.trapezoid(v * w, v) / np.trapezoid(w, v)


s_grid = np.linspace(-3.5, 3.5, 281)
vals = {n: np.array([pivotal_value(s, n) for s in s_grid]) for n in NS}
post_mean = s_grid / (1 + SIG ** 2)

lines = ["Gaussian mineral rights model, sigma = 1"]
for n in NS:
    for s in (-1.0, 0.0, 1.0):
        lines.append(f"n={n:2d}, s={s:+.1f}: v(s,s;n) = {pivotal_value(s, n):+.4f}")
    lines.append(f"n={n:2d}: v(s,s;n) strictly increasing in s: "
                 f"{bool(np.all(np.diff(vals[n]) > 0))}")
for n0, n1 in zip(NS[:-1], NS[1:]):
    lines.append(f"v(s,s;{n0}) > v(s,s;{n1}) on the grid: {bool(np.all(vals[n0] > vals[n1]))}")

# CDF of v(S_i, S_i; n): S_i ~ N(0, 1 + sigma^2), v increasing in s.
t_grid = np.linspace(-3.0, 2.5, 400)
cdfs = {}
for n in NS:
    s_of_t = np.interp(t_grid, vals[n], s_grid)
    cdfs[n] = norm.cdf(s_of_t / np.sqrt(1 + SIG ** 2))
for n in NS:
    lines.append(f"n={n:2d}: median of v(S,S;n) = {np.interp(0.5, cdfs[n], t_grid):+.4f}")
NUM.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.8), constrained_layout=True)
for k, n in enumerate(NS):
    ax1.plot(s_grid, vals[n], color=COLORS[k], lw=2, label=f"$n={n}$")
ax1.plot(s_grid, post_mean, color="#52514e", lw=1.2, ls=":",
         label=r"$\mathrm{E}[V\mid S_i=s]$")
ax1.set_xlim(-3, 3)
ax1.set_ylim(-2.2, 2.2)
ax1.set_xlabel(r"signal $s$")
ax1.set_ylabel(r"$v(s,s;n)$")
ax1.set_title("(a) pivotal expected value", loc="left")
ax1.legend(frameon=False, loc="upper left")

for k, n in enumerate(NS):
    ax2.plot(t_grid, cdfs[n], color=COLORS[k], lw=2, label=f"$n={n}$")
ax2.set_xlim(-3, 2.5)
ax2.set_ylim(0, 1)
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$\Pr\left(v(S_i,S_i;n)\leq t\right)$")
ax2.set_title(r"(b) distribution of $v(S_i,S_i;n)$", loc="left")
ax2.legend(frameon=False, loc="upper left")
fig.savefig(OUT)
print("saved", OUT)
