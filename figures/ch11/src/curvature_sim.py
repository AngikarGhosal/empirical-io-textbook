"""Demand curvature and simulated merger price effects (Chapter 11, fig:11-curvature).

Two symmetric single-product firms merge; there is no other inside product. Every demand system is calibrated
to the same pre-merger point: price p = 1, marginal cost c = 0.6 (Lerner margin L = 0.4), own-price elasticity
1/L = 2.5 (the single-product Bertrand condition), and diversion ratio D between the two products. Only the
curvature of demand differs.

  linear:               q_j = a - b p_j + d p_k, with b = 1/L, d = D b, a = 1 + b - d (so q_j = 1 at p = 1)
  logit (outside good): s_j = exp(a - alpha p_j) / (1 + sum_k exp(a - alpha p_k)), s = D/(1+D), alpha = 1/(L(1-s))
  constant elasticity:  q_j = k p_j^(-eps) p_k^(eta), eps = 1/L, eta = D eps

For each D the script solves the post-merger joint-profit-maximizing prices (no efficiencies) and also reports the
"frozen shares and slopes" first-iteration approximation, which in the symmetric case equals the Werden
compensating marginal cost reduction D (p - c) / (1 - D).

Output: figures/ch11/curvature_sim.pdf, figures/ch11/src/curvature_sim_output.txt
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "curvature_sim.pdf"
OUT_PNG = HERE / "curvature_sim_preview.png"
OUT_TXT = HERE / "curvature_sim_output.txt"

P = 1.0
L = 0.4
C = P * (1 - L)
EPS = 1.0 / L


def linear_effect(D):
    b = EPS
    d = D * b
    a = 1.0 + b - d
    # symmetric merged FOC: a - (b-d) p - (b-d)(p - c) = 0
    p = (a + (b - d) * C) / (2 * (b - d))
    return p / P - 1


def logit_effect(D):
    s = D / (1 + D)
    alpha = EPS / (P * (1 - s))
    a = np.log(s / (1 - 2 * s)) + alpha * P

    def gap(p):
        e = np.exp(a - alpha * p)
        sh = e / (1 + 2 * e)
        # merged firm common markup: p - c = 1 / (alpha (1 - 2 s(p)))
        return (p - C) - 1.0 / (alpha * (1 - 2 * sh))

    p = brentq(gap, C + 1e-9, 10.0)
    return p / P - 1


def ce_effect(D):
    eta = D * EPS
    if EPS - eta <= 1:
        return np.nan
    # symmetric merged FOC: 1 - (eps - eta)(p - c)/p = 0
    p = C * (EPS - eta) / (EPS - eta - 1)
    return p / P - 1


def frozen_effect(D):
    return D * (P - C) / (1 - D) / P


def check_calibration(D):
    """Own elasticity and diversion at p = 1 for each system (numerical derivatives)."""
    h = 1e-6
    out = {}
    # linear
    b = EPS
    d = D * b
    a = 1 + b - d
    q = lambda p1, p2: a - b * p1 + d * p2
    out["linear"] = (-(q(1 + h, 1) - q(1 - h, 1)) / (2 * h) / q(1, 1),
                     (q(1, 1 + h) - q(1, 1 - h)) / (q(1 + h, 1) - q(1 - h, 1)) * -1)
    # logit
    s = D / (1 + D)
    alpha = EPS / (1 - s)
    al = np.log(s / (1 - 2 * s)) + alpha

    def sh(p1, p2):
        e1, e2 = np.exp(al - alpha * p1), np.exp(al - alpha * p2)
        return np.array([e1, e2]) / (1 + e1 + e2)

    ds_dp1 = (sh(1 + h, 1) - sh(1 - h, 1)) / (2 * h)
    out["logit"] = (-ds_dp1[0] / sh(1, 1)[0], -ds_dp1[1] / ds_dp1[0])
    # constant elasticity
    eta = D * EPS
    qc = lambda p1, p2: np.array([p1 ** -EPS * p2 ** eta, p2 ** -EPS * p1 ** eta])
    dq = (qc(1 + h, 1) - qc(1 - h, 1)) / (2 * h)
    out["CE"] = (-dq[0] / qc(1, 1)[0], -dq[1] / dq[0])
    return out


lines = []
for D in (0.1, 0.2, 0.3):
    lines.append(f"calibration check D={D}: " + ", ".join(
        f"{k}: own elast {v[0]:.4f}, diversion {v[1]:.4f}" for k, v in check_calibration(D).items()))
lines.append("D | linear | logit | const. elasticity | frozen approx | UPP/p = D L | CE/linear")
for D in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
    li, lo, ce, fr = linear_effect(D), logit_effect(D), ce_effect(D), frozen_effect(D)
    lines.append(f"{D:.2f} | {100 * li:.2f} | {100 * lo:.2f} | {100 * ce:.2f} | {100 * fr:.2f} | {100 * D * L:.2f} | "
                 f"{ce / li:.2f}")
print("\n".join(lines))
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

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

Dg = np.linspace(0.0, 0.35, 141)
li = np.array([linear_effect(d) for d in Dg])
lo = np.array([logit_effect(d) if d > 0 else 0.0 for d in Dg])
ce = np.array([ce_effect(d) for d in Dg])
fr = np.array([frozen_effect(d) for d in Dg])

fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(Dg, 100 * ce, color=COLORS[1], lw=2.0, ls="-", label="constant elasticity")
ax.plot(Dg, 100 * fr, color=COLORS[3], lw=2.0, ls=(0, (1, 1.2)), label="frozen shares and slopes (pass-through 1)")
ax.plot(Dg, 100 * li, color=COLORS[0], lw=2.0, ls="--", label="linear")
ax.plot(Dg, 100 * lo, color=COLORS[2], lw=2.0, ls="-.", label="logit with outside good")
ax.set_xlim(0, 0.35)
ax.set_ylim(0, 60)
ax.set_xlabel(r"diversion ratio $D$ between the merging products")
ax.set_ylabel("predicted price increase (percent)")
ax.grid(color=GRID, lw=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=150)
