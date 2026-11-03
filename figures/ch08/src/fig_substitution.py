"""Figure fig:08-substitution (Chapter 8, section 8.3).

Panel (a). Three inside products A, B, C and an outside good. A and B are close in the
characteristic x ("size"); B and C have the same market share. Both models are calibrated to the
same shares (logit by the analytic inversion, RC logit by the BLP contraction) and the same mean
price coefficient. We raise p_A by 1 percent and record the percentage change in the shares of B,
C and the outside good.
  RC logit: random coefficients on the constant (sd 1.0) and on x (sd 2.5); alpha = 2 for all.
  x = (1.0, 0.9, -1.0), p = (1, 1, 1), s = (0.20, 0.10, 0.10), s_0 = 0.60.

Panels (b) and (c). Six products: three small cheap cars (S1-S3) and three large expensive cars
(L1-L3), each with share 0.05. Matrix of price elasticities eps_jk = (d s_j / d p_k)(p_k / s_j)
under logit and under the RC logit (random coefficients on the constant, sd 1.0; on size, sd 2.0;
on price, sd 0.5, so alpha_i = 2 - 0.5 nu_ip). Diagonal cells report the own elasticity as a
positive number.

All integrals use Gauss-Hermite product rules (no simulation error).
Output: figures/ch08/fig_substitution.pdf, src/substitution_numbers.txt
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from rclogit_p1 import shares, contraction, gauss_hermite_nodes, derivative_matrix

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "fig_substitution.pdf"
PNG = HERE / "fig_substitution_preview.png"
TXT = HERE / "substitution_numbers.txt"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

lines = []


def out(s=""):
    print(s)
    lines.append(s)


# ----------------------------------------------------------------------------- panel (a)
ALPHA = 2.0
xA = np.array([1.0, 0.9, -1.0])
pA = np.array([1.0, 1.0, 1.0])
sA_obs = np.array([0.20, 0.10, 0.10])
xrA = np.column_stack([np.ones(3), xA])          # constant and size carry random coefficients
sigA = np.array([1.0, 2.5])
nodesA, wA = gauss_hermite_nodes(2, 60)

delta_rc, it = contraction(sA_obs, xrA, sigA, nodesA, wA)
s_chk, s0_chk = shares(delta_rc, xrA, sigA, nodesA, wA)
delta_lg = np.log(sA_obs) - np.log(1 - sA_obs.sum())
out("Panel (a): three-product market")
out(f"  RC contraction iterations {it}; max share error {np.max(np.abs(s_chk - sA_obs)):.2e}")
out(f"  delta RC = {np.round(delta_rc, 4)}, delta logit = {np.round(delta_lg, 4)}")


def logit_shares(delta):
    e = np.exp(delta)
    return e / (1 + e.sum()), 1 / (1 + e.sum())


def pct_changes(model):
    """Percentage change in (s_B, s_C, s_0) and in s_A after a 1 percent rise in p_A."""
    if model == "logit":
        s_old, s0_old = logit_shares(delta_lg)
        d = delta_lg.copy()
        d[0] -= ALPHA * 0.01 * pA[0]
        s_new, s0_new = logit_shares(d)
    else:
        s_old, s0_old = shares(delta_rc, xrA, sigA, nodesA, wA)
        d = delta_rc.copy()
        d[0] -= ALPHA * 0.01 * pA[0]
        s_new, s0_new = shares(d, xrA, sigA, nodesA, wA)
    ch = 100 * (np.append(s_new, s0_new) / np.append(s_old, s0_old) - 1)
    lost = s_old[0] - s_new[0]
    div = np.append(s_new[1:] - s_old[1:], s0_new - s0_old) / lost
    return ch, div


res = {}
for m in ("logit", "rc"):
    ch, div = pct_changes(m)
    res[m] = (ch, div)
    out(f"  {m:5s}: % change s_A {ch[0]:.3f}, s_B {ch[1]:.3f}, s_C {ch[2]:.3f}, s_0 {ch[3]:.3f}")
    out(f"         diversion from A to B {100*div[0]:.1f}%, to C {100*div[1]:.1f}%, "
        f"to outside {100*div[2]:.1f}%")

# Analytic derivatives for the RC model (check against the finite differences)
D_rc = derivative_matrix(delta_rc, xrA, sigA, lambda nd: np.full(nd.shape[0], ALPHA), nodesA, wA)
out(f"  RC elasticities w.r.t. p_A (analytic): own {-D_rc[0,0]*pA[0]/sA_obs[0]:.3f}, "
    f"B {D_rc[1,0]*pA[0]/sA_obs[1]:.3f}, C {D_rc[2,0]*pA[0]/sA_obs[2]:.3f}")
out(f"  Logit elasticities w.r.t. p_A: own {ALPHA*pA[0]*(1-sA_obs[0]):.3f}, "
    f"cross {ALPHA*pA[0]*sA_obs[0]:.3f}")

# ----------------------------------------------------------------------------- panels (b), (c)
names = ["S1", "S2", "S3", "L1", "L2", "L3"]
size = np.array([-1.0, -0.9, -0.8, 0.8, 0.9, 1.0])
price = np.array([1.0, 1.1, 1.2, 2.0, 2.2, 2.4])
s6 = np.full(6, 0.05)
SIG_P = 0.5
xr6 = np.column_stack([np.ones(6), size, price])
sig6 = np.array([1.0, 2.0, SIG_P])
nodes6, w6 = gauss_hermite_nodes(3, 30)
delta6, it6 = contraction(s6, xr6, sig6, nodes6, w6)
D6 = derivative_matrix(delta6, xr6, sig6, lambda nd: ALPHA - SIG_P * nd[:, 2], nodes6, w6)
E_rc = D6 * price[None, :] / s6[:, None]
E_lg = np.outer(np.ones(6), ALPHA * price * s6)
np.fill_diagonal(E_lg, -ALPHA * price * (1 - s6))
out()
out("Panels (b)-(c): six-product market, shares 0.05 each")
out(f"  RC contraction iterations {it6}")
out("  Logit elasticity matrix (rows j, columns k):")
for j in range(6):
    out("   " + " ".join(f"{E_lg[j, k]:8.3f}" for k in range(6)))
out("  RC elasticity matrix (rows j, columns k):")
for j in range(6):
    out("   " + " ".join(f"{E_rc[j, k]:8.3f}" for k in range(6)))
own_rc = -np.diag(E_rc)
own_lg = -np.diag(E_lg)
out("  own elasticities logit: " + ", ".join(f"{v:.3f}" for v in own_lg))
out("  own elasticities RC   : " + ", ".join(f"{v:.3f}" for v in own_rc))
# diversion to the outside good, D_j0 = (d s_0 / d p_j) / (- d s_j / d p_j)
ds0 = -D6.sum(axis=0)            # d s_0 / d p_k = - sum_j d s_j / d p_k
D_j0 = ds0 / (-np.diag(D6))
out("  diversion to outside good RC: " + ", ".join(f"{100*v:.1f}%" for v in D_j0))
out("  diversion to outside good logit: " + ", ".join(f"{100*0.7/(1-0.05):.1f}%" for _ in range(6)))

TXT.write_text("\n".join(lines) + "\n")

# ----------------------------------------------------------------------------- figure
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral",
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
})
fig = plt.figure(figsize=(6.3, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.12)
ax = fig.add_subplot(gs[0, :])
cats = ["Product B\n(close to A)", "Product C\n(far from A)", "Outside good"]
xpos = np.arange(3)
wbar = 0.34
vals_lg = res["logit"][0][1:]
vals_rc = res["rc"][0][1:]
b1 = ax.bar(xpos - wbar / 2 - 0.01, vals_lg, wbar, color=BLUE, label="Logit", zorder=3)
b2 = ax.bar(xpos + wbar / 2 + 0.01, vals_rc, wbar, color=ORANGE, label="Random coefficients logit",
            zorder=3)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=7.5, color=INK)
ax.set_xticks(xpos)
ax.set_xticklabels(cats)
ax.set_ylabel("Percentage change in share")
ax.set_ylim(0, max(vals_rc.max(), vals_lg.max()) * 1.22)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, loc="upper right")
ax.set_title(r"(a) Share gains after a 1 percent rise in $p_A$", loc="left")

vmax = max(np.max(E_rc - np.diag(np.diag(E_rc))), np.max(E_lg - np.diag(np.diag(E_lg))))
norm_ = Normalize(vmin=0.0, vmax=vmax)
cmap = plt.get_cmap("Blues")
for col, (E, title) in enumerate([(E_lg, "(b) Logit"), (E_rc, "(c) Random coefficients logit")]):
    axh = fig.add_subplot(gs[1, col])
    rgb = np.zeros((6, 6, 3))
    for j in range(6):
        for k in range(6):
            if j == k:
                rgb[j, k] = (0.90, 0.90, 0.89)
            else:
                rgb[j, k] = cmap(0.08 + 0.85 * norm_(E[j, k]))[:3]
    axh.imshow(rgb, aspect="equal")
    for j in range(6):
        for k in range(6):
            v = -E[j, k] if j == k else E[j, k]
            shade = 0.08 + 0.85 * norm_(E[j, k]) if j != k else 0.0
            colr = "white" if shade > 0.6 else INK
            axh.text(k, j, f"{v:.2f}", ha="center", va="center", fontsize=6.8, color=colr,
                     fontweight="bold" if j == k else "normal")
    axh.set_xticks(range(6))
    axh.set_xticklabels(names)
    axh.set_yticks(range(6))
    axh.set_yticklabels(names if col == 0 else [])
    axh.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    axh.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    axh.grid(which="minor", color="white", lw=1.5)
    axh.tick_params(which="minor", length=0)
    axh.tick_params(which="major", length=0)
    for sp in axh.spines.values():
        sp.set_visible(False)
    axh.set_title(title, loc="left")
    axh.set_xlabel(r"Price of product $k$")
    if col == 0:
        axh.set_ylabel(r"Share of product $j$")

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(PNG, dpi=150, bbox_inches="tight")
print("saved", OUT)
