"""Consumer-surplus gain from adding products: models with and without idiosyncratic tastes
(Figure fig:10-variety-value, Section sec:10-charspace, Proposition prop:10-clones).

All gains are per consumer, relative to a market with one product, in units of 1/alpha
(the price coefficient is normalized to alpha = 1).

(i)   Logit clones. J identical products with mean utility delta plus an outside good (delta_0 = 0):
          CS(J) = ln(1 + J e^delta),  gain(J) = ln[(1 + J e^delta) / (1 + e^delta)].
(ii)  Nested-logit clones. The J products form one nest, the outside good its own nest:
          D = J e^{delta/(1-sigma)},  CS(J) = ln(1 + D^{1-sigma}) = ln(1 + J^{1-sigma} e^delta).
(iii) Pure characteristics (Hotelling line). Consumers uniform on [0, 1], J products at
      (2k - 1) / (2J), equal fixed prices, quadratic transport cost t d^2, everyone buys:
          CS(J) = v - p - t / (12 J^2),  gain(J) = (t / 12)(1 - 1 / J^2)  <= t / 12.
(iv)  Hotelling line plus logit errors with scale mu (u = -t (x - l_j)^2 + mu * eps_j):
          CS(x; J) = mu ln sum_j exp(-t (x - l_j)^2 / mu) + const, averaged over x in [0, 1]
      by numerical quadrature.

Parameters: delta = 0 (one product has share 1/2 in (i)-(ii)), sigma in {0.5, 0.8}, t = 6, mu = 0.25.
Outputs: figures/ch10/variety_value.pdf, figures/ch10/src/variety_value_numbers.txt,
         figures/ch10/src/variety_value_preview.png
Run from anywhere: python variety_value.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "variety_value.pdf"
TXT = HERE / "variety_value_numbers.txt"

DELTA, T_COST, MU = 0.0, 6.0, 0.25
SIGMAS = (0.5, 0.8)
J_GRID = np.arange(1, 51)

BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})


def logit_gain(J, delta=DELTA):
    return np.log((1 + J * np.exp(delta)) / (1 + np.exp(delta)))


def nested_gain(J, sigma, delta=DELTA):
    return np.log((1 + J ** (1 - sigma) * np.exp(delta)) / (1 + np.exp(delta)))


def hotelling_gain(J, t=T_COST):
    return t / 12.0 * (1 - 1.0 / J ** 2)


def hotelling_cs_closed(J, t=T_COST):
    """Average transport cost on the line, closed form (for checking the quadrature)."""
    return -t / (12.0 * J ** 2)


X = np.linspace(0.0, 1.0, 20001)


def hotelling_logit_cs(J, t=T_COST, mu=MU):
    loc = (2 * np.arange(1, J + 1) - 1) / (2.0 * J)
    z = -t * (X[:, None] - loc[None, :]) ** 2 / mu
    zmax = z.max(axis=1, keepdims=True)
    cs_x = mu * (zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1)))
    return np.trapezoid(cs_x, X)


def hotelling_cs_numeric(J, t=T_COST):
    loc = (2 * np.arange(1, J + 1) - 1) / (2.0 * J)
    d2 = ((X[:, None] - loc[None, :]) ** 2).min(axis=1)
    return np.trapezoid(-t * d2, X)


def main():
    g_logit = logit_gain(J_GRID)
    g_nl = {s: nested_gain(J_GRID, s) for s in SIGMAS}
    g_hot = hotelling_gain(J_GRID)
    cs_hl = np.array([hotelling_logit_cs(J) for J in J_GRID])
    g_hl = cs_hl - cs_hl[0]

    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    ax.plot(J_GRID, g_logit, color=BLUE, lw=1.8, label=r"Logit clones")
    ax.plot(J_GRID, g_nl[0.5], color=BLUE, lw=1.4, ls=(0, (5, 2)),
            label=r"Nested-logit clones, $\sigma = 0.5$")
    ax.plot(J_GRID, g_nl[0.8], color=BLUE, lw=1.4, ls=(0, (1.5, 1.5)),
            label=r"Nested-logit clones, $\sigma = 0.8$")
    ax.plot(J_GRID, g_hl, color=ORANGE, lw=1.8,
            label=r"Hotelling line plus logit errors ($\mu = 0.25$)")
    ax.plot(J_GRID, g_hot, color=GREEN, lw=2.0,
            label=r"Hotelling line, pure characteristics ($t = 6$)")
    ax.axhline(T_COST / 12, color=GREEN, lw=0.8, ls=(0, (4, 3)))
    ax.text(50, T_COST / 12 + 0.04, r"bound $\frac{t}{12}$", ha="right", va="bottom",
            fontsize=8.5, color=GREEN)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_xlim(1, 52)
    ax.set_ylim(0, 3.4)
    ax.set_xlabel(r"Number of products $J$ (log scale)")
    ax.set_ylabel(r"Gain in $CS$ relative to $J = 1$ (units of $\frac{1}{\alpha}$)")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=2.4)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(HERE / "variety_value_preview.png", dpi=130, bbox_inches="tight")

    lines = ["Gains relative to J = 1 (units of 1/alpha); delta = 0, t = 6, mu = 0.25"]
    for J in (2, 5, 10, 20, 50):
        k = J - 1
        lines.append(
            f"J={J:2d}: logit {g_logit[k]:.4f} | NL0.5 {g_nl[0.5][k]:.4f} | NL0.8 {g_nl[0.8][k]:.4f}"
            f" | Hotelling {g_hot[k]:.4f} | Hotelling+logit {g_hl[k]:.4f}")
    lines.append(f"Hotelling bound t/12 = {T_COST / 12:.4f}")
    lines.append(f"mu ln 50 = {MU * np.log(50):.4f} (asymptotic slope check for (iv))")
    for J in (1, 3, 10):
        lines.append(f"quadrature check J={J}: numeric {hotelling_cs_numeric(J):.6f} "
                     f"closed {hotelling_cs_closed(J):.6f}")
    TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
