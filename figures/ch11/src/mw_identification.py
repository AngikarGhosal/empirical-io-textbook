"""Identification of (kappa, merged-firm cost) in the logit illustration of Miller and Weinberg (2017).

Calibration (same as ex:11-logit in Section 11.5 and the paper's Section 5.2 example): three symmetric
single-product firms, logit demand, p = 1.00, mc = 0.70, inside shares 0.20, outside share 0.40, so that
alpha = 1 / (0.3 * 0.8). Firms 1 and 2 merge and both products get marginal cost c_M. The merged firm and
firm 3 put weight kappa on each other's profits (Omega(kappa)).

For each (kappa, c_M) on a grid we solve the post-merger first-order conditions
    s + (Omega(kappa) .* Delta') (p - c) = 0
and record the merging-firm price p_M (= p_1 = p_2) and the rival price p_3. The figure draws iso-price
contours of p_M and p_3 in the (kappa, c_M) plane: an observed pair (p_M, p_3) is the intersection of one
contour of each family, which is how the two post-merger prices identify the conduct parameter and the
efficiency jointly.

Outputs: figures/ch11/mw_identification.pdf, src/mw_identification_preview.png, src/mw_identification_output.txt
Run: python mw_identification.py
"""
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "mw_identification.pdf"
OUT_PNG = HERE / "mw_identification_preview.png"
OUT_TXT = HERE / "mw_identification_output.txt"

ALPHA = 1.0 / (0.3 * 0.8)
A = np.log(0.2 / 0.4) + ALPHA * 1.0 * np.ones(3)  # exp(A - alpha p) / (1 + sum) = 0.2 at p = 1

lines = []


def say(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    lines.append(s)


def shares(p):
    e = np.exp(A - ALPHA * p)
    return e / (1.0 + e.sum())


def jac(p):
    s = shares(p)
    return -ALPHA * (np.diag(s) - np.outer(s, s))  # (j,k) = d s_j / d p_k


def omega(kappa):
    return np.array([[1.0, 1.0, kappa], [1.0, 1.0, kappa], [kappa, kappa, 1.0]])


def foc(p, kappa, c):
    return shares(p) + (omega(kappa) * jac(p).T) @ (p - c)


def equilibrium(kappa, cm, p0=None):
    c = np.array([cm, cm, 0.7])
    p0 = np.array([1.05, 1.05, 1.02]) if p0 is None else p0
    p, info, ier, msg = fsolve(foc, p0, args=(kappa, c), full_output=True, xtol=1e-12)
    if ier != 1:
        raise RuntimeError(msg)
    return p


# --------------------------------------------------------------- checks used in the text
say("alpha =", round(ALPHA, 6))
say("pre-merger FOC residual at p = 1 (Nash, diagonal Omega):",
    np.abs(shares(np.ones(3)) + (np.eye(3) * jac(np.ones(3)).T) @ (np.ones(3) - 0.7)).max())
for kap, cm in [(0.0, 0.70), (0.5, 0.70), (0.5, 0.50), (0.25, 0.70), (0.25, 0.60), (1.0, 0.70)]:
    p = equilibrium(kap, cm)
    say(f"kappa={kap:.2f}, c_M={cm:.3f}: p = {np.round(p, 4)}")

# comparative statics of the rival price in c_M at kappa = 0 and kappa = 0.5
for kap in [0.0, 0.5]:
    h = 1e-5
    d = (equilibrium(kap, 0.6 + h)[2] - equilibrium(kap, 0.6 - h)[2]) / (2 * h)
    say(f"d p_3 / d c_M at kappa={kap}, c_M=0.6: {d:.4f}")


# invert two observed prices for (kappa, c_M)
def invert(pm_obs, p3_obs, guess=(0.4, 0.65)):
    def g(x):
        p = equilibrium(x[0], x[1])
        return [p[0] - pm_obs, p[2] - p3_obs]
    x = fsolve(g, guess, xtol=1e-12)
    return x


for pm_obs, p3_obs in [(1.09, 1.08), (1.0637, 1.0062), (1.104, 1.0731)]:
    k, c = invert(pm_obs, p3_obs)
    say(f"observed (p_M, p_3) = ({pm_obs}, {p3_obs}) -> kappa = {k:.4f}, c_M = {c:.4f}; check p = "
        f"{np.round(equilibrium(k, c), 4)}")

# --------------------------------------------------------------- grid
kap_grid = np.linspace(0.0, 1.0, 81)
c_grid = np.linspace(0.45, 0.75, 61)
PM = np.zeros((c_grid.size, kap_grid.size))
P3 = np.zeros_like(PM)
for i, cm in enumerate(c_grid):
    p_prev = None
    for j, kap in enumerate(kap_grid):
        p = equilibrium(kap, cm, p_prev)
        p_prev = p
        PM[i, j] = p[0]
        P3[i, j] = p[2]
say("grid ranges: p_M in [%.4f, %.4f], p_3 in [%.4f, %.4f]" % (PM.min(), PM.max(), P3.min(), P3.max()))

# --------------------------------------------------------------- figure
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "mathtext.fontset": "cm",
})
fig, ax = plt.subplots(figsize=(5.6, 3.9))
lev_m = [0.95, 1.00, 1.05, 1.10, 1.15]
lev_3 = [1.02, 1.05, 1.08, 1.11, 1.14]
cs_m = ax.contour(kap_grid, c_grid, PM, levels=lev_m, colors=BLUE, linewidths=1.6)
cs_3 = ax.contour(kap_grid, c_grid, P3, levels=lev_3, colors=ORANGE, linewidths=1.6, linestyles="--")
ax.clabel(cs_m, fmt=lambda v: f"$p_M={v:.2f}$", fontsize=7.5, inline_spacing=3)
ax.clabel(cs_3, fmt=lambda v: f"$p_3={v:.2f}$", fontsize=7.5, inline_spacing=3,
          manual=[(0.11, 0.62), (0.30, 0.62), (0.46, 0.62), (0.62, 0.60), (0.75, 0.60)])

pts = [(0.0, 0.70, "A"), (0.5, 0.70, "B"), (0.5, 0.50, "C")]
for k, c, lab in pts:
    ax.plot(k, c, "o", ms=6, color=INK, zorder=5)
    ax.annotate(lab, (k, c), xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
ax.axhline(0.70, color=MUTED, lw=0.6, ls=":")
ax.set_xlabel(r"internalization weight $\kappa$")
ax.set_ylabel(r"merged-firm marginal cost $c_M$")
ax.set_xlim(0, 1)
ax.set_ylim(0.45, 0.75)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=BLUE, lw=1.6, label=r"iso-price curve, merging firms ($p_M$)"),
           Line2D([], [], color=ORANGE, lw=1.6, ls="--", label=r"iso-price curve, rival ($p_3$)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=160)
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
say("wrote", OUT_PDF)
