"""Redraw Table III of Ashenfelter and Sullivan (1987, JIE 35(4), p. 492).

For each candidate numbers equivalent N_e = 1, ..., 9 (and infinity), the table reports the
percentage of tax-increase episodes whose price and quantity changes satisfy the finite-difference
bound N_e >= q0 (p1 - p0) / ((q0 - q1)(p1 - t0)). Two samples: skip-year changes with
dq <= 0 and dp >= 0 (702 cases) and flat-jump-flat changes with dq <= 0 and dp >= 0 (133 cases;
the printed column header reads "Jump-Flat-Jump", the text calls the sample "Flat-Jump-Flat").
Numbers typed from the PDF (printed p. 492, PDF p. 11) and checked twice.

Output: figures/ch03/as_numbers_equivalent.pdf (run from any directory).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "as_numbers_equivalent.pdf"

N_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SKIP_YEAR = [24.4, 45.2, 60.4, 69.7, 75.1, 79.3, 82.9, 85.2, 86.5]   # 702 cases
FLAT_JUMP_FLAT = [15.8, 39.8, 55.6, 70.7, 76.7, 81.2, 83.5, 85.7, 88.0]  # 133 cases
X_INF = 10.6  # plotting position for N_e = infinity (100 percent in both samples)

# Reference categorical palette, slots 1-2 in fixed order, plus distinct markers so that
# identity never relies on colour alone (the book is also printed in grayscale).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})

fig, ax = plt.subplots(figsize=(5.6, 3.4))
series = [
    (SKIP_YEAR, BLUE, "o", "Skip-year changes (702 cases)"),
    (FLAT_JUMP_FLAT, ORANGE, "s", "Flat-jump-flat changes (133 cases)"),
]
for ys, col, mk, lab in series:
    ax.plot(N_VALUES, ys, color=col, lw=2, marker=mk, ms=6, label=lab, zorder=3,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.plot([9, X_INF], [ys[-1], 100.0], color=col, lw=1.2, ls=":", zorder=2)
    ax.plot([X_INF], [100.0], color=col, marker=mk, ms=6, zorder=3,
            markeredgecolor="white", markeredgewidth=1.0)

ax.set_xticks(N_VALUES + [X_INF])
ax.set_xticklabels([str(n) for n in N_VALUES] + [r"$\infty$"])
ax.set_xlim(0.5, 11.1)
ax.set_ylim(0, 105)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel(r"Numbers equivalent of firms, $N_e$")
ax.set_ylabel("Percent of episodes consistent")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# Selective direct labels at N_e = 1 and N_e = 5 only.
ax.annotate("24.4", (1, 24.4), xytext=(-15, 9), textcoords="offset points", color=INK2, fontsize=8.5)
ax.annotate("15.8", (1, 15.8), xytext=(6, -11), textcoords="offset points", color=INK2, fontsize=8.5)
ax.annotate("75.1", (5, 75.1), xytext=(-8, -14), textcoords="offset points", color=INK2, fontsize=8.5)
ax.annotate("76.7", (5, 76.7), xytext=(-12, 7), textcoords="offset points", color=INK2, fontsize=8.5)

leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
for txt in leg.get_texts():
    txt.set_color(INK)

fig.tight_layout()
fig.savefig(OUT)
fig.savefig(HERE / "as_numbers_equivalent_preview.png", dpi=150)
print("wrote", OUT)
