"""Best responses in a symmetric logit duopoly with an outside good (fig:07-best-response).

Two single-product firms, logit demand s_j = e^{a - alpha p_j} / (1 + sum_k e^{a - alpha p_k}),
quality a = 3, alpha = 1, marginal cost mc = 1 for both firms. Best response
    BR_j(p_k) = mc + (1/alpha) [1 + W(e^{v - 1} / (1 + e^{a - alpha p_k}))],  v = a - alpha mc.
Bounds: mc + 1/alpha (rival price -> -infinity) and the monopoly price
mc + (1/alpha)[1 + W(e^{v - 1})] (rival price -> +infinity).
Writes figures/ch07/logit_best_response.pdf and prints the equilibrium.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)

A, ALPHA, MC = 3.0, 1.0, 1.0
V = A - ALPHA * MC


def br(p_rival):
    d_minus = 1.0 + np.exp(A - ALPHA * p_rival)
    return MC + (1.0 + lambertw(np.exp(V - 1.0) / d_minus).real) / ALPHA


def main():
    lower = MC + 1.0 / ALPHA
    pm = MC + (1.0 + lambertw(np.exp(V - 1.0)).real) / ALPHA
    pstar = brentq(lambda p: br(p) - p, lower, pm)
    s = np.exp(A - ALPHA * pstar) / (1 + 2 * np.exp(A - ALPHA * pstar))
    slope = (br(pstar + 1e-6) - br(pstar - 1e-6)) / 2e-6
    print(f"lower bound {lower:.4f}, monopoly price {pm:.4f}, equilibrium price {pstar:.4f}, "
          f"share {s:.4f}, markup {pstar - MC:.4f} = 1/(alpha(1-s)) = {1 / (ALPHA * (1 - s)):.4f}, "
          f"BR slope at equilibrium {slope:.4f}, s0 {1 - 2 * s:.4f}")
    grid = np.linspace(0.0, 8.0, 600)
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot(grid, br(grid), color="#2a78d6", linewidth=2, label=r"$BR_1(p_2)$")
    ax.plot(br(grid), grid, color="#eb6834", linewidth=2, label=r"$BR_2(p_1)$")
    ax.axhline(lower, color="#777777", linestyle=":", linewidth=1)
    ax.axhline(pm, color="#777777", linestyle="--", linewidth=1)
    ax.text(3.98, lower - 0.03, r"$mc + \frac{1}{\alpha}$", ha="right", va="top", fontsize=10, color="#444444")
    ax.text(3.98, pm + 0.03, r"monopoly price $p^m$", ha="right", va="bottom", fontsize=10, color="#444444")
    ax.plot([pstar], [pstar], "o", color="#222222", markersize=6)
    ax.annotate(r"$(p^*, p^*)$", xy=(pstar, pstar), xytext=(pstar + 0.3, pstar - 0.35),
                fontsize=9, arrowprops=dict(arrowstyle="-", color="#444444", linewidth=0.8))
    ax.set_xlim(1.5, 4.0)
    ax.set_ylim(1.5, 4.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Price of firm 2, $p_2$", fontsize=10)
    ax.set_ylabel(r"Price of firm 1, $p_1$", fontsize=10)
    ax.tick_params(labelsize=9)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "logit_best_response.pdf"))
    fig.savefig(os.path.join(HERE, "logit_best_response_preview.png"), dpi=130)


if __name__ == "__main__":
    main()
