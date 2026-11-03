"""Asymmetric first-price auction with uniform values (Section 14.10).

Weak bidder:   V_w ~ U[0, vw],  strong bidder: V_s ~ U[0, vs],  vs > vw, no reserve.
Closed form (Griesmer, Levitan and Shubik 1967; Plum 1992):
    phi_i(b) = 2 b / (1 + k_i b^2),  k_i = 1/vbar_i^2 - 1/vbar_j^2,
    top bid  bbar = vw vs / (vw + vs),
    beta_i(v) = (1 - sqrt(1 - k_i v^2)) / (k_i v).

The script
  1. verifies symbolically (sympy) that phi_i solves the ODE system
     phi_i'(b) = phi_i(b) / (phi_j(b) - b) and the boundary conditions;
  2. computes revenue, allocation and surplus numbers for vw = 1, vs = 2,
     by quadrature and by Monte Carlo as a cross-check;
  3. draws the two-panel figure figures/ch14/asym_bids.pdf.

Outputs: figures/ch14/asym_bids.pdf and figures/ch14/src/asym_uniform_numbers.txt
"""
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import integrate

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_FIG = HERE.parent / "asym_bids.pdf"
OUT_TXT = HERE / "asym_uniform_numbers.txt"

# ----------------------------------------------------------------- symbolic check
b, vw_s, vs_s = sp.symbols("b v_w v_s", positive=True)
kw = 1 / vw_s**2 - 1 / vs_s**2
ks = -kw
phi_w = 2 * b / (1 + kw * b**2)
phi_s = 2 * b / (1 + ks * b**2)
ode_w = sp.simplify(sp.diff(phi_w, b) - phi_w / (phi_s - b))
ode_s = sp.simplify(sp.diff(phi_s, b) - phi_s / (phi_w - b))
bbar_s = vw_s * vs_s / (vw_s + vs_s)
top_w = sp.simplify(phi_w.subs(b, bbar_s) - vw_s)
top_s = sp.simplify(phi_s.subs(b, bbar_s) - vs_s)
first_integral = sp.simplify((phi_s - b) * (phi_w - b) - b**2)
assert ode_w == 0 and ode_s == 0, (ode_w, ode_s)
assert top_w == 0 and top_s == 0, (top_w, top_s)
assert first_integral == 0, first_integral

# Exact FPA revenue for vw = 1, vs = 2 (derived in the text).
R_fp_exact = sp.Rational(2, 3) - 4 / (3 * sp.sqrt(3)) * sp.log(2 + sp.sqrt(3)) \
    + 4 * sp.pi / (9 * sp.sqrt(3))

# ----------------------------------------------------------------- numbers
VW, VS = 1.0, 2.0
KW = 1 / VW**2 - 1 / VS**2
KS = -KW
BBAR = VW * VS / (VW + VS)


def beta(v, k):
    v = np.asarray(v, dtype=float)
    out = np.where(v > 0, (1 - np.sqrt(np.maximum(1 - k * v**2, 0))) / (k * np.where(v > 0, v, 1)), 0.0)
    return out


def phi(bid, k):
    return 2 * bid / (1 + k * bid**2)


def G_w(bid):  # CDF of the weak bidder's bid
    return np.minimum(phi(bid, KW) / VW, 1.0)


def G_s(bid):  # CDF of the strong bidder's bid
    return np.minimum(phi(bid, KS) / VS, 1.0)


# FPA revenue: E[max bid] = int_0^bbar (1 - G_w G_s) db
R_fp = integrate.quad(lambda x: 1 - G_w(x) * G_s(x), 0, BBAR)[0]
# SPA revenue: E[min(V_w, V_s)] = int_0^vw (1 - t/vw)(1 - t/vs) dt
R_sp = integrate.quad(lambda t: (1 - t / VW) * (1 - t / VS), 0, VW)[0]
# Probability that the weak bidder wins
P_w_fp = integrate.quad(lambda v: G_s(beta(v, KW)) / VW, 0, VW)[0]
P_w_eff = integrate.quad(lambda v: min(v / VS, 1.0) / VW, 0, VW)[0]
P_ineff = P_w_fp - P_w_eff


def s_cut(v):  # strong value that ties with weak value v: phi_s(beta_w(v)) = v beta_w / (v - beta_w)
    bw = beta(v, KW)
    return v * bw / (v - bw) if v > 0 else 0.0


# Expected total surplus
S_eff = integrate.quad(lambda t: 1 - min(t / VW, 1) * min(t / VS, 1), 0, VS)[0]
loss = integrate.quad(lambda v: (s_cut(v) - v) ** 2 / 2 / VS / VW, 0, VW)[0]
S_fp = S_eff - loss
# Bidder payoffs
U_w_fp = integrate.quad(lambda v: (v - beta(v, KW)) * G_s(beta(v, KW)) / VW, 0, VW)[0]
U_s_fp = integrate.quad(lambda v: (v - beta(v, KS)) * G_w(beta(v, KS)) / VS, 0, VS)[0]
U_w_sp = integrate.dblquad(lambda x, v: max(v - x, 0) / (VW * VS), 0, VW, 0, VS)[0]
U_s_sp = integrate.dblquad(lambda x, v: max(v - x, 0) / (VW * VS), 0, VS, 0, VW)[0]

# Monte Carlo cross-check
rng = np.random.default_rng(14)
M = 2_000_000
vw = rng.uniform(0, VW, M)
vs = rng.uniform(0, VS, M)
bw_ = beta(vw, KW)
bs_ = beta(vs, KS)
w_wins = bw_ > bs_
mc = {
    "R_fp": np.mean(np.maximum(bw_, bs_)),
    "R_sp": np.mean(np.minimum(vw, vs)),
    "P_w_fp": np.mean(w_wins),
    "P_ineff": np.mean(w_wins & (vs > vw)),
    "S_fp": np.mean(np.where(w_wins, vw, vs)),
    "S_eff": np.mean(np.maximum(vw, vs)),
    "U_w_fp": np.mean(np.where(w_wins, vw - bw_, 0)),
    "U_s_fp": np.mean(np.where(~w_wins, vs - bs_, 0)),
}

lines = [
    "Asymmetric uniform example, vw = 1, vs = 2 (no reserve)",
    f"symbolic checks: ODEs, boundary conditions, first integral all satisfied",
    f"k_w = {KW:.4f}, k_s = {KS:.4f}, bbar = {BBAR:.6f}",
    f"FPA revenue (quad)        = {R_fp:.6f}   exact = {float(R_fp_exact):.6f}   MC = {mc['R_fp']:.5f}",
    f"SPA revenue (quad)        = {R_sp:.6f}   (5/12 = {5/12:.6f})          MC = {mc['R_sp']:.5f}",
    f"P(weak wins), FPA         = {P_w_fp:.6f}   MC = {mc['P_w_fp']:.5f}",
    f"P(weak wins), efficient   = {P_w_eff:.6f}",
    f"P(inefficient), FPA       = {P_ineff:.6f}   MC = {mc['P_ineff']:.5f}",
    f"Total surplus, efficient  = {S_eff:.6f}   (13/12 = {13/12:.6f})       MC = {mc['S_eff']:.5f}",
    f"Total surplus, FPA        = {S_fp:.6f}   MC = {mc['S_fp']:.5f}",
    f"Efficiency loss, FPA      = {loss:.6f}",
    f"Weak payoff:  FPA = {U_w_fp:.6f} (MC {mc['U_w_fp']:.5f}),  SPA = {U_w_sp:.6f} (1/12 = {1/12:.6f})",
    f"Strong payoff: FPA = {U_s_fp:.6f} (MC {mc['U_s_fp']:.5f}),  SPA = {U_s_sp:.6f} (7/12 = {7/12:.6f})",
    f"Check FPA: surplus - payoffs = {S_fp - U_w_fp - U_s_fp:.6f} vs revenue {R_fp:.6f}",
    f"beta_w(0.5) = {float(beta(0.5, KW)):.4f}, beta_s(0.5) = {float(beta(0.5, KS)):.4f}, beta_s(1) = {float(beta(1.0, KS)):.4f}",
]
OUT_TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ----------------------------------------------------------------- figure
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))

ax = axes[0]
v_w = np.linspace(1e-6, VW, 300)
v_s = np.linspace(1e-6, VS, 300)
ax.plot(v_s, v_s / 2, color=INK2, lw=1.2, ls="--", label=r"symmetric benchmark $\frac{v}{2}$")
ax.plot(v_w, beta(v_w, KW), color=ORANGE, lw=2, label=r"weak, $\beta_w(v)$")
ax.plot(v_s, beta(v_s, KS), color=BLUE, lw=2, label=r"strong, $\beta_s(v)$")
ax.axhline(BBAR, color=GRID, lw=0.8, zorder=0)
ax.text(0.05, BBAR - 0.075, r"$\bar b=\frac{2}{3}$", color=INK2, fontsize=9)
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 1.0)
ax.set_xlabel(r"value $v$")
ax.set_ylabel(r"bid $b$")
ax.set_title("(a) Equilibrium bids", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[1]
vv = np.linspace(1e-6, VW, 300)
cut = np.array([s_cut(x) for x in vv])
ax.fill_between(vv, vv, cut, color=ORANGE, alpha=0.30, lw=0, label="weak wins, $v_s>v_w$")
ax.plot(vv, vv, color=INK2, lw=1.2, ls="--", label=r"$v_s=v_w$")
ax.plot(vv, cut, color=ORANGE, lw=2, label=r"$v_s=\phi_s(\beta_w(v_w))$")
ax.text(0.62, 0.22, "weak wins\n(efficient)", fontsize=8, color=INK2, ha="center")
ax.text(0.30, 1.05, "strong wins", fontsize=8, color=INK2, ha="center")
ax.set_xlim(0, 1)
ax.set_ylim(0, 2)
ax.set_xlabel(r"weak value $v_w$")
ax.set_ylabel(r"strong value $v_s$")
ax.set_title("(b) Allocation in value space", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(OUT_FIG)
fig.savefig(HERE / "asym_bids_preview.png", dpi=150)
print("wrote", OUT_FIG)
