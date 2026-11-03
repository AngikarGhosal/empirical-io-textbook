"""Figure fig:02-stigler-detect (Chapter 2, Section 9.2).

Left panel: probability that a rival hears of at least one of n secret price
cuts, 1 - (1 - h)^n, for h in {0.005, 0.01, 0.02} (Stigler 1964, Section II).
Right panel: Stigler's Criterion I (largest undetected gain from one rival, in
percent of average sales) against the number of sellers for n_o = 50 old buyers
and four repeat probabilities phi. Curves come from the formula; markers are the
printed entries of Stigler's Table 1 (2010 reprint, PDF pp. 9-10).

Output: ../stigler_detection.pdf (vector) and a PNG preview next to this script.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "stigler_detection.pdf")

# Validated categorical slots 1-4 (blue, orange, aqua, yellow); see the dataviz palette.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def crit_I(phi, n_o, n_s):
    t = math.sqrt(phi / (1 - phi) * n_s / n_o)
    root = math.sqrt(max(n_s * (n_s - 2), 0.0) * (1 - phi) * (1 + t) / n_o)
    return 100 * (t * (1 - phi) + root) / (n_s - 1)


TABLE1_NO50 = {
    0.95: [4.4, 6.6, 6.5, 6.4, 6.5, 6.8],
    0.90: [6.0, 8.8, 8.6, 8.4, 8.3, 8.6],
    0.80: [8.0, 11.6, 11.2, 11.0, 10.6, 10.8],
    0.70: [9.2, 13.5, 13.1, 12.8, 12.3, 12.4],
}
SELLERS = [2, 3, 4, 5, 10, 20]


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.9))

    # ---------------------------------------------------------------- left panel
    n = np.logspace(0, 3, 400)
    hs = [0.005, 0.01, 0.02]
    styles = [(0, (6, 2, 1.5, 2)), "-", (0, (4, 2))]
    for h, col, ls in zip(hs, C[:3], styles):
        ax1.plot(n, 1 - (1 - h) ** n, color=col, lw=1.8, ls=ls, label=rf"$h = {h}$")
    ax1.legend(loc="lower right", frameon=False, fontsize=8, handlelength=2.6)

    for x in (100, 1000):
        y = 1 - 0.99 ** x
        ax1.plot([x], [y], "o", ms=5, color=C[1], mec="white", mew=1.0, zorder=5)
    ax1.annotate(r"$(100,\ 0.634)$", xy=(100, 0.634), xytext=(4.5, 0.80), fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax1.annotate(r"$(1000,\ 0.99996)$", xy=(1000, 0.99996), xytext=(70, 1.06), fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax1.set_xscale("log")
    ax1.set_xlim(1, 1000)
    ax1.set_ylim(0, 1.12)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xlabel(r"Number of buyers given the secret cut, $n$")
    ax1.set_ylabel(r"Detection probability $1 - (1-h)^{n}$")
    ax1.grid(True, color=GRID, lw=0.6)
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Detection of secret price cuts", fontsize=9, color=INK)

    # ---------------------------------------------------------------- right panel
    ns = np.arange(2, 21)
    phis = [0.95, 0.90, 0.80, 0.70]
    markers = ["o", "s", "^", "D"]
    for phi, col, mk in zip(phis, C, markers):
        ax2.plot(ns, [crit_I(phi, 50, x) for x in ns], color=col, lw=1.4, marker=".", ms=3.5)
        ax2.plot(SELLERS, TABLE1_NO50[phi], mk, ms=5.5, mfc="white", mec=col, mew=1.2, zorder=5)
        ax2.text(20.6, crit_I(phi, 50, 20), rf"$\phi = {phi:.2f}$", color=INK, fontsize=8, va="center")
    ax2.set_xlim(1.5, 24.5)
    ax2.set_xticks([2, 5, 10, 15, 20])
    ax2.set_ylim(0, 15)
    ax2.set_xlabel(r"Number of sellers $n_s$")
    ax2.set_ylabel(r"Criterion I, % of average sales")
    ax2.grid(True, color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.set_title(r"(b) Undetected gain from one rival, $n_o = 50$", fontsize=9, color=INK)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT)
    fig.savefig(os.path.join(HERE, "stigler_detection_preview.png"), dpi=170)


if __name__ == "__main__":
    main()
