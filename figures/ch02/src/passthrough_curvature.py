"""Pass-through with flat marginal cost as a function of demand curvature.

With a specific tax t, constant marginal cost and conduct theta, the supply relation
P(Q) + theta Q P'(Q) = c + t implies

    rho = dP/dt = 1 / (1 + theta (1 - chi)),     chi = -Q P''(Q) / P'(Q).

The figure plots rho against chi in [-1, 1.9] for theta in {0, 1/4, 1/2, 1}. Each curve is
restricted to the region where the denominator is positive (the stability condition), and the
vertical axis is clipped at 3.5. Vertical guides mark linear demand (chi = 0), logarithmic
inverse demand P = a - b ln Q (chi = 1) and constant elasticity with epsilon = 2 (chi = 1.5).

Output: figures/ch02/passthrough_curvature.pdf (run from any directory).
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "passthrough_curvature.pdf"

# Chapter palette (fixed categorical order, validated for colour-vision deficiency);
# line styles give a second encoding so identity never relies on colour alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STYLES = ["-", "--", "-.", (0, (1, 1.2))]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})

YMAX = 3.5
chi = np.linspace(-1.0, 1.9, 1200)
thetas = [0.0, 0.25, 0.5, 1.0]
labels = [r"$\theta = 0$", r"$\theta = \frac{1}{4}$", r"$\theta = \frac{1}{2}$", r"$\theta = 1$"]

fig, ax = plt.subplots(figsize=(6.0, 3.7))

# reference guides
for x0, txt in [(0.0, "linear"), (1.0, r"$P = a - b\,\ln Q$"), (1.5, "constant elasticity\n" + r"$\varepsilon = 2$")]:
    ax.axvline(x0, color=GRID, lw=1.0, zorder=0)
    ax.text(x0, YMAX + 0.05, txt, ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.axhline(1.0, color=INK2, lw=0.8, ls=":", zorder=0)

for th, col, ls in zip(thetas, COLORS, STYLES):
    den = 1.0 + th * (1.0 - chi)
    rho = np.where(den > 0, 1.0 / np.where(den > 0, den, 1.0), np.nan)
    rho = np.where(rho <= YMAX, rho, np.nan)
    ax.plot(chi, rho, color=col, ls=ls, lw=2.0, zorder=3)

# direct labels at the right edge of the plotting region (text in ink, marks carry colour)
right_y = {0.0: 1.0, 0.25: 1.0 / (1 + 0.25 * (1 - 1.9)), 0.5: 1.0 / (1 + 0.5 * (1 - 1.9))}
for th, lab in zip(thetas[:3], labels[:3]):
    ax.text(1.93, right_y[th], lab, color=INK, fontsize=9, ha="left", va="center", clip_on=False)
ax.text(1.76, 3.3, labels[3], color=INK, fontsize=9, ha="left", va="center", clip_on=False)

# mark the monopoly points on the three guides
for x0 in [0.0, 1.0, 1.5]:
    ax.plot([x0], [1.0 / (2.0 - x0)], marker="o", ms=5, color=COLORS[3], mec="white", mew=1.0, zorder=4)
ax.annotate(r"monopoly, linear: $\rho = \frac{1}{2}$", xy=(0.0, 0.5), xytext=(0.2, 0.22),
            fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
ax.annotate(r"monopoly, $\varepsilon = 2$: $\rho = 2$", xy=(1.5, 2.0), xytext=(0.35, 2.55),
            fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))

ax.set_xlim(-1.0, 1.9)
ax.set_ylim(0.0, YMAX)
ax.set_xlabel(r"curvature of inverse demand $\chi = -\frac{QP''(Q)}{P'(Q)}$")
ax.set_ylabel(r"pass-through $\rho = \frac{dP}{dt}$")
ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
