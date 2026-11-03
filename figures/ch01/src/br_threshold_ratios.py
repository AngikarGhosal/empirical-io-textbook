"""Redraw Figure 4 of Bresnahan and Reiss (1991) from the entry thresholds in their Table 4.

The thresholds are the total market sizes (thousands of people) needed to support N firms,
as printed in Table 4 of the working-paper version of the article. The per-firm threshold is
tau_N = M_N / N, and the figure plots tau_5 / tau_N against N for the five industries.

Output: figures/ch01/br_threshold_ratios.pdf (run from any directory).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "br_threshold_ratios.pdf"

# Table 4, entry thresholds M_1, ..., M_5 in thousands of people.
THRESHOLDS = {
    "Druggists": [0.53, 2.12, 5.04, 7.67, 9.39],
    "Tire dealers": [0.49, 1.78, 3.41, 4.74, 6.10],
    "Doctors": [0.88, 3.49, 5.78, 7.72, 9.14],
    "Dentists": [0.71, 2.54, 4.18, 5.43, 6.41],
    "Plumbers": [1.43, 3.02, 4.53, 6.20, 7.47],
}

# Reference categorical palette (first five slots, fixed order) plus distinct markers,
# so identity never relies on colour alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["v", "D", "x", "s", "+"]
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

fig, ax = plt.subplots(figsize=(5.6, 3.5))
ns = [1, 2, 3, 4, 5]
for (name, m), col, mk in zip(THRESHOLDS.items(), COLORS, MARKERS):
    tau = [m[i] / (i + 1) for i in range(5)]
    ratio = [tau[4] / t for t in tau]
    ax.plot(ns, ratio, color=col, lw=1.6, marker=mk, ms=7,
            mfc="white" if mk not in ("x", "+") else col, mew=1.6, label=name, zorder=3)
    ax.annotate(name, xy=(1, ratio[0]), xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=9, color=INK)

ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.set_xlim(-0.6, 5.4)
ax.set_xticks(ns)
ax.set_ylim(0.8, 3.7)
ax.set_xlabel(r"Number of firms $N$")
ax.set_ylabel(r"$\frac{\tau_5}{\tau_N}$", rotation=0, labelpad=14, fontsize=14)
ax.grid(axis="y", color=GRID, lw=0.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
