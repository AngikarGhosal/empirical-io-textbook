"""UPP screening thresholds in (Lerner margin, diversion) space for Chapter 11 (fig:11-uppregion).

Symmetric single-product Bertrand merger (Farrell and Shapiro 2010):
  simple net UPP test, eq. (2):       D > E (1 - L) / L
  feedback (Werden 1996), eq. (4):    D / (1 - D) > E (1 - L) / L, i.e. D > k / (1 + k), k = E (1 - L) / L
  NUPPI test, symmetric form of (6):  D L - E (1 - L) > G / rho, i.e. D > [G / rho + E (1 - L)] / L
The script prints the threshold table used in the text (tab:11-thresholds) and draws the figure.

Output: figures/ch11/upp_thresholds.pdf
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "upp_thresholds.pdf"
OUT_PNG = HERE / "upp_thresholds_preview.png"


def simple(L, E):
    return E * (1 - L) / L


def feedback(L, E):
    k = E * (1 - L) / L
    return k / (1 + k)


def nuppi(L, E, G, rho):
    return (G / rho + E * (1 - L)) / L


print("L | simple E=0.05 | simple E=0.10 | feedback E=0.10 | NUPPI E=0.10,G=0.05,rho=0.5 | GUPPI at threshold")
for L in (0.2, 0.25, 1 / 3, 0.4, 0.5, 0.6):
    print(f"{L:.4f} | {simple(L, 0.05):.4f} | {simple(L, 0.10):.4f} | {feedback(L, 0.10):.4f} | "
          f"{nuppi(L, 0.10, 0.05, 0.5):.4f} | {simple(L, 0.10) * L:.4f}")

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
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

Lg = np.linspace(0.1, 0.8, 400)
fig, ax = plt.subplots(figsize=(6.0, 3.9))
ax.fill_between(Lg, simple(Lg, 0.10), 0.8, color=COLORS[0], alpha=0.08, lw=0)
ax.plot(Lg, simple(Lg, 0.05), color=COLORS[2], lw=2.0, ls="-.", label=r"net UPP, $E=0.05$")
ax.plot(Lg, simple(Lg, 0.10), color=COLORS[0], lw=2.0, ls="-", label=r"net UPP, $E=0.10$")
ax.plot(Lg, feedback(Lg, 0.10), color=COLORS[1], lw=2.0, ls="--", label=r"with feedback, $E=0.10$")
ax.plot(Lg, nuppi(Lg, 0.10, 0.05, 0.5), color=COLORS[3], lw=2.0, ls=(0, (1, 1.2)),
        label=r"NUPPI, $E=0.10$, $G=0.05$, $\rho=\frac{1}{2}$")
pts = [(1 / 3, 0.20), (0.4, 0.15), (0.5, 0.10)]
for L, d in pts:
    ax.plot([L], [d], marker="o", ms=5, color=COLORS[0], mec="white", mew=1.0, zorder=4)
ax.annotate(r"$L=\frac{1}{3}$, $D=0.20$", (1 / 3, 0.20), xytext=(8, 4), textcoords="offset points", fontsize=8,
            color=INK2)
ax.annotate(r"$D=0.15$", (0.4, 0.15), xytext=(6, 3), textcoords="offset points", fontsize=8, color=INK2)
ax.annotate(r"$D=0.10$", (0.5, 0.10), xytext=(6, 3), textcoords="offset points", fontsize=8, color=INK2)
for L, d in [(0.4, 0.40), (0.5, 0.30)]:
    ax.plot([L], [d], marker="s", ms=5, color=COLORS[3], mec="white", mew=1.0, zorder=4)
ax.text(0.6, 0.47, "shaded: flagged when\n" + r"$E=0.10$ (net UPP $>0$)", fontsize=8.5, color=INK2, ha="left",
        va="center")
ax.text(0.115, 0.04, "not flagged", fontsize=8.5, color=INK2, ha="left", va="center")
ax.set_xlim(0.1, 0.8)
ax.set_ylim(0, 0.8)
ax.set_xlabel(r"Lerner margin $L=\frac{p-c}{p}$")
ax.set_ylabel(r"diversion ratio $D$")
ax.grid(color=GRID, lw=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=150)
