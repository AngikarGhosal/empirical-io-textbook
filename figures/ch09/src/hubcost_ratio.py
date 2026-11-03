"""Hub-and-spoke versus point-to-point costs (ex:09-hubcost, fig:09-hubcost).

n outer cities plus a hub city, A round-trip passengers in every city pair, leg cost a Q^gamma.
Point to point: n(n+1)/2 legs with density A.  Hub and spoke: n spokes with density nA.
  (a) total cost ratio  C_HS / C_PP = 2 n^gamma / (n + 1)
  (b) marginal cost of a connecting passenger relative to a nonstop passenger: 2 n^(gamma - 1)

Output: figures/ch09/hubcost_ratio.pdf and printed thresholds.
"""
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "hubcost_ratio.pdf"

GAMMAS = [1.0, 0.9, 0.8, 0.7, 0.6]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DASHES = ["-", (0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2)), (0, (3, 1.5))]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def total_ratio(n, g):
    return 2.0 * n ** g / (n + 1.0)


def mc_ratio(n, g):
    return 2.0 * n ** (g - 1.0)


def main():
    for g in GAMMAS:
        ns = np.arange(1, 2001)
        cheaper = ns[total_ratio(ns, g) < 1.0]
        mc_cheaper = ns[mc_ratio(ns, g) < 1.0]
        n_tot = cheaper.min() if cheaper.size else None
        n_mc = mc_cheaper.min() if mc_cheaper.size else None
        print(f"gamma={g}: hub total cost lower for n >= {n_tot}; connecting MC below nonstop MC for n >= {n_mc}")
    print(f"n=10, gamma=0.7: total ratio = {total_ratio(10, 0.7):.4f}, 2n^g = {2 * 10 ** 0.7:.3f}")
    print(f"n=10, gamma=0.7: MC ratio = {mc_ratio(10, 0.7):.4f}; hub-endpoint MC ratio n^(g-1) = {10 ** (-0.3):.4f}")
    print(f"n=20, gamma=0.7: total ratio = {total_ratio(20, 0.7):.4f}, MC ratio = {mc_ratio(20, 0.7):.4f}")

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
        "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=True)
    n = np.arange(1, 31)
    for g, col, ls in zip(GAMMAS, COLORS, DASHES):
        axes[0].plot(n, total_ratio(n, g), color=col, lw=1.8, ls=ls, label=rf"$\gamma={g:.1f}$")
        axes[1].plot(n, mc_ratio(n, g), color=col, lw=1.8, ls=ls, label=rf"$\gamma={g:.1f}$")
    axes[0].set_ylabel("Ratio to point-to-point")
    axes[0].set_title(r"(a) Total cost, $\frac{2n^{\gamma}}{n+1}$", fontsize=10)
    axes[1].set_title(r"(b) Connecting marginal cost, $2n^{\gamma-1}$", fontsize=10)
    for ax in axes:
        ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xlabel(r"Number of outer cities $n$")
        ax.set_xlim(1, 30)
        ax.set_ylim(0.4, 2.1)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
