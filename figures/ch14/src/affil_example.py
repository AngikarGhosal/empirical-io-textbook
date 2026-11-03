"""Figure fig:14-affil-example (Chapter 14, Section 14.3.3).

Panel (a): the affiliated density f(s1, s2) = (4/5)(1 + s1 s2) on [0, 1]^2.
Panel (b): the conditional CDF F(s2 | s1) = s2 (2 + s1 s2) / (2 + s1) for
s1 in {0, 1/2, 1}. Higher s1 shifts the conditional distribution of s2 up in
the sense of first-order stochastic dominance, as affiliation implies.

Output: figures/ch14/affil_example.pdf (run from any directory).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "affil_example.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.8,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})

# validated categorical slots 1-3 (blue, orange, aqua)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def density(s1, s2):
    return 0.8 * (1.0 + s1 * s2)


def cond_cdf(s2, s1):
    return s2 * (2.0 + s1 * s2) / (2.0 + s1)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.85), constrained_layout=True)
# widen the gap between the colorbar label of (a) and the y-axis label of (b)
fig.get_layout_engine().set(wspace=0.12)

# Panel (a): filled contours of the joint density (single-hue sequential map)
g = np.linspace(0, 1, 301)
S1, S2 = np.meshgrid(g, g)
levels = np.linspace(0.8, 1.6, 9)
cf = ax1.contourf(S1, S2, density(S1, S2), levels=levels, cmap="Blues")
cs = ax1.contour(S1, S2, density(S1, S2), levels=levels[1:-1], colors="white", linewidths=0.6)
cb = fig.colorbar(cf, ax=ax1, shrink=0.9, pad=0.02)
cb.set_label(r"$f(s_1,s_2)$")
cb.outline.set_linewidth(0.5)
ax1.set_xlabel(r"$s_1$")
ax1.set_ylabel(r"$s_2$")
ax1.set_title(r"(a) Joint density $f(s_1,s_2)=\frac{4}{5}(1+s_1s_2)$")
ax1.set_aspect("equal")
ax1.set_xticks([0, 0.5, 1])
ax1.set_yticks([0, 0.5, 1])

# Panel (b): conditional CDFs
labels = [r"$s_1=0$", r"$s_1=\frac{1}{2}$", r"$s_1=1$"]
for s1, col, lab in zip([0.0, 0.5, 1.0], COLORS, labels):
    ax2.plot(g, cond_cdf(g, s1), color=col, lw=1.8, label=lab)
ax2.plot([0, 1], [0, 1], color="#c3c2b7", lw=0.6, ls=":", zorder=0)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.0)
ax2.set_xlabel(r"$s_2$")
ax2.set_ylabel(r"$F(s_2\mid s_1)$")
ax2.set_title(r"(b) Conditional CDF of $S_2$ given $S_1=s_1$")
ax2.grid(True, color="#e6e5e0", lw=0.5)
ax2.legend(frameon=False, loc="upper left")
ax2.set_xticks([0, 0.5, 1])
ax2.set_yticks([0, 0.5, 1])

fig.savefig(OUT)
print("wrote", OUT)
