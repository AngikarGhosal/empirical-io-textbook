"""Logit Bertrand equilibrium with a dominant carrier (Figure fig:09-dominance).

Two single-product carriers and an outside good on one route. Utility
u_ij = delta_j - alpha p_j + eps_ij (delta_j is the non-price part of mean utility).
Prices are in $100, alpha = 1, c_1 = c_2 = 1, delta_2 = 1, delta_1 runs from 1 to 3.

The equilibrium is computed exactly as in the proof of prop:09-dominance:
  h(m) = (1 - 1/(alpha m)) exp(alpha m) = s_0 a_j,   a_j = exp(delta_j - alpha c_j),
  s_j = 1 - 1/(alpha m_j),  and s_0 solves  s_0 + sum_j s_j(s_0) = 1.

Output: figures/ch09/logit_dominance.pdf and a printed table of checks.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "logit_dominance.pdf"

ALPHA = 1.0


def h(m, alpha=ALPHA):
    return (1.0 - 1.0 / (alpha * m)) * np.exp(alpha * m)


def h_inv(y, alpha=ALPHA):
    """Inverse of h on (1/alpha, infinity); h is strictly increasing there."""
    lo = 1.0 / alpha + 1e-14
    hi = lo + 1.0
    while h(hi, alpha) < y:
        hi *= 2.0
    return brentq(lambda m: h(m, alpha) - y, lo, hi, xtol=1e-14)


def equilibrium(delta, cost, alpha=ALPHA):
    delta = np.asarray(delta, float)
    cost = np.asarray(cost, float)
    a = np.exp(delta - alpha * cost)

    def shares_given_s0(s0):
        m = np.array([h_inv(s0 * aj, alpha) for aj in a])
        return 1.0 - 1.0 / (alpha * m), m

    def excess(s0):
        s, _ = shares_given_s0(s0)
        return s0 + s.sum() - 1.0

    s0 = brentq(excess, 1e-12, 1.0 - 1e-12, xtol=1e-14)
    s, m = shares_given_s0(s0)
    p = cost + m
    # Check: logit shares at these prices reproduce s, and the FOC holds.
    v = np.exp(delta - alpha * p)
    s_check = v / (1.0 + v.sum())
    assert np.allclose(s_check, s, atol=1e-10)
    assert np.allclose(m, 1.0 / (alpha * (1.0 - s)), atol=1e-10)
    return p, s, s0


def main():
    d1_grid = np.linspace(1.0, 3.0, 201)
    res = [equilibrium([d1, 1.0], [1.0, 1.0]) for d1 in d1_grid]
    p1 = np.array([r[0][0] for r in res])
    p2 = np.array([r[0][1] for r in res])
    s1 = np.array([r[1][0] for r in res])
    s2 = np.array([r[1][1] for r in res])
    s0 = np.array([r[2] for r in res])

    print("delta1    p1      p2      s1      s2      s0     HHI(inside)  ln p1 - ln p2")
    for d1 in (1.0, 1.5, 2.0, 2.5, 3.0):
        p, s, z = equilibrium([d1, 1.0], [1.0, 1.0])
        inside = s / s.sum()
        hhi = (inside ** 2).sum()
        print(f"{d1:5.2f}  {p[0]:.3f}  {p[1]:.3f}  {s[0]:.3f}  {s[1]:.3f}  {z:.3f}  {hhi:.3f}"
              f"   {np.log(p[0]) - np.log(p[1]):.4f}")

    # Cost advantage instead of quality advantage: c1 from 1 down to 0.5 with delta1 = 1.
    print("\nCost advantage (delta1 = delta2 = 1):")
    for c1 in (1.0, 0.75, 0.5):
        p, s, z = equilibrium([1.0, 1.0], [c1, 1.0])
        print(f"c1={c1:4.2f}: p1={p[0]:.3f} p2={p[1]:.3f} m1={p[0]-c1:.3f} s1={s[0]:.3f} s2={s[1]:.3f}")

    # Slope d ln p / d s = L / (1 - s) at a few points.
    for d1 in (1.0, 3.0):
        p, s, z = equilibrium([d1, 1.0], [1.0, 1.0])
        L = (p - 1.0) / p
        print(f"delta1={d1}: Lerner={L.round(3)}, slope L/(1-s)={(L / (1 - s)).round(3)}")

    # ------------------------------------------------------------------ figure
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
        "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))

    ax = axes[0]
    ax.plot(d1_grid, p1, color=BLUE, lw=2.0)
    ax.plot(d1_grid, p2, color=ORANGE, lw=2.0, ls=(0, (5, 2)))
    ax.annotate(r"$p_1$ (dominant)", xy=(2.6, p1[160]), xytext=(-6, 8),
                textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.annotate(r"$p_2$ (rival)", xy=(2.6, p2[160]), xytext=(0, -14),
                textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_xlabel(r"Quality of carrier 1, $\delta_1$")
    ax.set_ylabel(r"Price (\$100)")
    ax.set_title("(a) Equilibrium prices", fontsize=10)
    ax.set_ylim(2.0, 3.0)

    ax = axes[1]
    ax.plot(d1_grid, s1, color=BLUE, lw=2.0)
    ax.plot(d1_grid, s2, color=ORANGE, lw=2.0, ls=(0, (5, 2)))
    ax.plot(d1_grid, s0, color=GREY, lw=1.4, ls=(0, (1, 1.5)))
    ax.annotate(r"$s_1$", xy=(2.9, s1[190]), xytext=(0, 6), textcoords="offset points",
                ha="center", color=INK, fontsize=9)
    ax.annotate(r"$s_2$", xy=(2.9, s2[190]), xytext=(0, 6), textcoords="offset points",
                ha="center", color=INK, fontsize=9)
    ax.annotate(r"$s_0$ (outside)", xy=(1.6, s0[60]), xytext=(4, 4), textcoords="offset points",
                ha="left", color=INK, fontsize=9)
    ax.set_xlabel(r"Quality of carrier 1, $\delta_1$")
    ax.set_ylabel("Market share")
    ax.set_title("(b) Equilibrium shares", fontsize=10)
    ax.set_ylim(0.0, 0.7)

    for ax in axes:
        ax.set_xlim(1.0, 3.0)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
