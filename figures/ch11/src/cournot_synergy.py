"""Price, profit and welfare effects of a Cournot merger as a function of the merged firm's cost.

Linear inverse demand P(Q) = a - Q with a = 10. Three firms with constant marginal costs
c = (2, 3, 3). Firms 1 and 2 merge and the merged firm has constant marginal cost c_M.
With n active firms and costs c_i, linear Cournot gives P = (a + sum_i c_i) / (n + 1) and
q_i = P - c_i.

Pre-merger:  P0 = 4.5, q = (2.5, 1.5, 1.5), insiders' joint profit 8.5.
Post-merger: P1(c_M) = (13 + c_M) / 3, q_M = (13 - 2 c_M) / 3, q_3 = (4 + c_M) / 3.

Thresholds printed by the script:
  price-neutral cost   c_hat = c_1 + c_2 - P0 = 0.5   (Farrell-Shapiro synergy condition)
  break-even cost      c_pi  solves q_M = sqrt(8.5)
  welfare-neutral cost c_W   solves Delta W(c_M) = 0

Output: figures/ch11/cournot_synergy.pdf
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "cournot_synergy.pdf"

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

a = 10.0
c = np.array([2.0, 3.0, 3.0])


def cournot(costs):
    n = len(costs)
    P = (a + costs.sum()) / (n + 1)
    q = P - costs
    assert np.all(q > 0)
    return P, q


P0, q0 = cournot(c)
Q0 = q0.sum()
CS0 = 0.5 * Q0 ** 2
pi0 = q0 ** 2
PI_I0 = pi0[0] + pi0[1]
PI_O0 = pi0[2]


def post(cM):
    P, q = cournot(np.array([cM, c[2]]))
    Q = q.sum()
    return dict(P=P, dCS=0.5 * Q ** 2 - CS0, dPI=q[0] ** 2 - PI_I0, dPO=q[1] ** 2 - PI_O0)


def dW(cM):
    r = post(cM)
    return r["dCS"] + r["dPI"] + r["dPO"]


c_hat = c[0] + c[1] - P0
c_pi = (13 - 3 * np.sqrt(PI_I0)) / 2
c_W = brentq(dW, 0.0, 3.0)
print(f"P0 = {P0}, q0 = {q0}, CS0 = {CS0}, insiders' profit = {PI_I0}")
print(f"c_hat = {c_hat:.4f}, c_pi = {c_pi:.4f}, c_W = {c_W:.4f}")
for cm in [0.5, 1.0, 2.0, 3.0]:
    r = post(cm)
    print(f"c_M = {cm}: P1 = {r['P']:.4f}, dCS = {r['dCS']:.4f}, dPI_I = {r['dPI']:.4f}, "
          f"dPI_O = {r['dPO']:.4f}, dW = {dW(cm):.4f}")

grid = np.linspace(0.0, 3.0, 601)
res = [post(x) for x in grid]
P1 = np.array([r["P"] for r in res])
dCS = np.array([r["dCS"] for r in res])
dPI = np.array([r["dPI"] for r in res])
dPO = np.array([r["dPO"] for r in res])
dWv = dCS + dPI + dPO

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

ax = axes[0]
ax.axhline(P0, color=INK2, lw=0.8, ls=":")
ax.plot(grid, P1, color=COLORS[0], lw=2.0)
ax.axvline(c_hat, color=GRID, lw=1.0, zorder=0)
ax.axvline(min(c[0], c[1]), color=GRID, lw=1.0, zorder=0)
ax.text(c_hat + 0.05, 4.02, r"$\hat c = c_1 + c_2 - P^0$", fontsize=8.5, color=INK2, ha="left")
ax.text(min(c[0], c[1]) + 0.05, 4.02, r"$\min\{c_1, c_2\}$", fontsize=8.5, color=INK2, ha="left")
ax.text(2.95, P0 - 0.04, r"$P^0 = 4.5$", fontsize=8.5, color=INK2, ha="right", va="top")
ax.text(0.15, 4.62, "price falls", fontsize=8.5, color=INK2)
ax.text(1.45, 4.98, "price rises", fontsize=8.5, color=INK2)
ax.set_xlim(0, 3)
ax.set_ylim(3.95, 5.4)
ax.set_xlabel(r"merged firm's marginal cost $c_M$")
ax.set_ylabel(r"post-merger price $P^1$")
ax.set_title("(a) Price", fontsize=10)

ax = axes[1]
ax.axhline(0.0, color=INK2, lw=0.8)
ax.axvspan(c_hat, c_pi, color=GRID, alpha=0.45, lw=0, zorder=0)
ax.plot(grid, dPI, color=COLORS[0], lw=2.0, ls="-", label=r"insiders' profit $\Delta \pi_I$")
ax.plot(grid, dCS, color=COLORS[1], lw=2.0, ls="--", label=r"consumer surplus $\Delta CS$")
ax.plot(grid, dPO, color=COLORS[2], lw=2.0, ls="-.", label=r"outsider's profit $\Delta \pi_O$")
ax.plot(grid, dWv, color=INK, lw=1.4, ls=(0, (1, 1.2)), label=r"total surplus $\Delta W$")
ax.legend(loc="upper right", fontsize=8, frameon=False, handlelength=2.6)
ax.text(0.5 * (c_hat + c_pi), -4.6, "profitable and\nprice-raising", fontsize=8, color=INK2, ha="center")
ax.set_xlim(0, 3)
ax.set_ylim(-5.2, 11.5)
ax.set_xlabel(r"merged firm's marginal cost $c_M$")
ax.set_ylabel("change from pre-merger level")
ax.set_title("(b) Profits and surplus", fontsize=10)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT)
fig.savefig(HERE / "cournot_synergy_preview.png", dpi=150)
print("saved", OUT)
