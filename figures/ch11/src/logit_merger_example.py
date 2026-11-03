"""Worked logit merger example for Chapter 11, Section 11.5 (ex:11-logit, tab:11-logitex, fig:11-kappa).

Calibration (the illustration used by Miller and Weinberg 2017 to explain the internalization parameter):
three symmetric single-product firms, price 1.00, marginal cost 0.70, inside shares 0.20 each, outside share
0.40, logit demand, Bertrand-Nash pricing before the merger. Firms 1 and 2 merge. The merged firm and firm 3
may partially internalize each other's profits with weight kappa (Omega(kappa)).

The script prints every number used in the text and in tab:11-logitex, and draws fig:11-kappa.

Outputs: figures/ch11/logit_kappa.pdf and figures/ch11/src/logit_merger_example_output.txt
Run: python logit_merger_example.py (from any directory).
"""
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve, brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "logit_kappa.pdf"
OUT_PNG = HERE / "logit_kappa_preview.png"
OUT_TXT = HERE / "logit_merger_example_output.txt"

lines = []


def say(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    lines.append(s)


# ------------------------------------------------------------------ calibration
P0 = np.ones(3)
C0 = 0.7 * np.ones(3)
S0 = 0.2 * np.ones(3)
s_out = 1.0 - S0.sum()
# single-product logit Lerner condition p - c = 1 / (alpha (1 - s))
ALPHA = 1.0 / ((P0[0] - C0[0]) * (1.0 - S0[0]))
# mean utilities delta_j = a_j - alpha p_j with exp(delta_j) = s_j / s_0
A = np.log(S0 / s_out) + ALPHA * P0


def shares(p):
    e = np.exp(A - ALPHA * p)
    return e / (1.0 + e.sum())


def jac(p):
    """Delta with (j,k) element d s_j / d p_k."""
    s = shares(p)
    return -ALPHA * (np.diag(s) - np.outer(s, s))


def omega(kappa):
    return np.array([[1.0, 1.0, kappa], [1.0, 1.0, kappa], [kappa, kappa, 1.0]])


OMEGA_PRE = np.eye(3)


def foc(p, om, c):
    """s + (Omega o Delta^T)(p - c), zero at an equilibrium."""
    return shares(p) + (om * jac(p).T) @ (p - c)


def solve(om, c, start=None):
    x0 = P0.copy() if start is None else start
    if np.max(np.abs(foc(x0, om, c))) < 1e-12:
        return x0.copy()
    p, info, ier, msg = fsolve(foc, x0, args=(om, c), xtol=1e-13, full_output=True)
    # fsolve may report slow progress when it starts at (or next to) the solution; accept tiny residuals
    assert ier == 1 or np.max(np.abs(foc(p, om, c))) < 1e-10, msg
    assert np.max(np.abs(foc(p, om, c))) < 1e-10
    return p


def cs(p):
    """Consumer surplus per consumer, log-sum formula (market size 1), up to a constant."""
    return np.log(1.0 + np.exp(A - ALPHA * p).sum()) / ALPHA


def profits(p, c):
    return (p - c) * shares(p)


say("alpha =", f"{ALPHA:.6f}", " (1/(0.3*0.8))")
say("pre-merger FOC residual at p = 1:", np.max(np.abs(foc(P0, OMEGA_PRE, C0))))
p_pre = solve(OMEGA_PRE, C0)
say("pre-merger equilibrium (check):", np.round(p_pre, 6))

# ------------------------------------------------------------------ screening statistics
D = S0[1] / (1.0 - S0[0])
UPP = D * (P0[1] - C0[1])
GUPPI = UPP / P0[0]
L = (P0[0] - C0[0]) / P0[0]
say(f"diversion D12 = s2/(1-s1) = {D:.4f}; diversion to outside = {s_out / (1 - S0[0]):.4f}")
say(f"UPP = D (p2 - c2) = {UPP:.4f}; GUPPI = {100 * GUPPI:.2f} percent")
say(f"single-product pass-through holding rivals fixed: 1 - s = {1 - S0[0]:.2f}")
say(f"UPP x (1 - s) = {UPP * (1 - S0[0]):.4f}")
# Werden compensating marginal cost reduction (symmetric)
cmcr_abs = (D * (P0[1] - C0[1]) + D * D * (P0[0] - C0[0])) / (1 - D * D)
cmcr_frac = (D * L * P0[1] / P0[0] + D * D * L) / ((1 - L) * (1 - D * D))
say(f"CMCR (Werden): absolute {cmcr_abs:.6f}, as fraction of c {cmcr_frac:.6f}")
say(f"symmetric net-UPP test with E=0.10: D > E(1-L)/L = {0.1 * (1 - L) / L:.4f}")
say(f"feedback test with E=0.10: D/(1-D) = {D / (1 - D):.4f} vs E(1-L)/L = {0.1 * (1 - L) / L:.4f}")
say(f"default efficiency that exactly offsets simple UPP: E = D L/(1-L) = {D * L / (1 - L):.4f}")

# Nevo (2000) eq. (11) approximation: post-merger FOC with shares and Jacobian frozen at pre-merger values
Dl = jac(P0)
om_post = omega(0.0)
p_approx = C0 - np.linalg.solve(om_post * Dl.T, S0)
say("first-iteration approximation (frozen s, Delta):", np.round(p_approx, 6))


# ------------------------------------------------------------------ first-order approach (Jaffe-Weyl)
def f_upp_form(p, om):
    """Per-unit form of the FOCs: f_j = (p_j-c_j) - s_j/(-ds_j/dp_j) - sum_{k != j} Om_jk D_jk (p_k - c_k)."""
    s = shares(p)
    Dm = jac(p)
    out = np.empty(3)
    for j in range(3):
        own = -Dm[j, j]
        div = Dm[:, j] / own  # D_jk = (ds_k/dp_j) / (-ds_j/dp_j), k != j
        extra = sum(om[j, k] * div[k] * (p[k] - C0[k]) for k in range(3) if k != j)
        out[j] = (p[j] - C0[j]) - s[j] / own - extra
    return out


def num_jac(fun, p, h=1e-6):
    J = np.empty((3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        J[:, k] = (fun(p + e) - fun(p - e)) / (2 * h)
    return J


f_post0 = f_upp_form(P0, om_post)
say("f_post(p0) = -UPP vector:", np.round(f_post0, 6))
J_post = num_jac(lambda p: f_upp_form(p, om_post), P0)
J_pre = num_jac(lambda p: f_upp_form(p, OMEGA_PRE), P0)
rho_pre = np.linalg.inv(J_pre)
rho_post = np.linalg.inv(J_post)
say("pre-merger cost pass-through matrix (dp/dc):\n", np.round(rho_pre, 4))
say("post-merger pass-through matrix:\n", np.round(rho_post, 4))
dp_foa_post = -rho_post @ f_post0
dp_foa_pre = -rho_pre @ f_post0
say("first-order approach, post-merger pass-through: p =", np.round(P0 + dp_foa_post, 4))
say("first-order approach, pre-merger pass-through:  p =", np.round(P0 + dp_foa_pre, 4))

# ------------------------------------------------------------------ Farrell-Shapiro iterative tax procedure
# (a) impose internal taxes t1, t2 on the merging products, (b) re-solve single-product Bertrand pricing with
# costs c + t, (c) recompute t_j = D_jk(p) (p_k - c_k) at the new prices, (d) repeat.
p_it = P0.copy()
say("iterative tax procedure (taxes evaluated at previous prices):")
for it in range(1, 7):
    Dm = jac(p_it)
    t = np.zeros(3)
    t[0] = Dm[1, 0] / (-Dm[0, 0]) * (p_it[1] - C0[1])
    t[1] = Dm[0, 1] / (-Dm[1, 1]) * (p_it[0] - C0[0])
    p_it = solve(OMEGA_PRE, C0 + t, p_it)
    if True:
        say(f"  iteration {it}: taxes {np.round(t[:2], 5)}, prices {np.round(p_it, 5)}")

# ------------------------------------------------------------------ full simulations
scen = {}


def run(name, kappa, cm):
    c = C0.copy()
    c[:2] = cm
    p = solve(omega(kappa), c)
    pr = profits(p, c)
    scen[name] = dict(kappa=kappa, cm=cm, p=p, s=shares(p), cs=cs(p), pi=pr)
    return p


run("pre", 0.0, 0.7)
scen["pre"]["p"] = P0.copy()
scen["pre"]["s"] = shares(P0)
scen["pre"]["cs"] = cs(P0)
scen["pre"]["pi"] = profits(P0, C0)
run("k0", 0.0, 0.7)
run("k0_cmcr", 0.0, 0.7 - cmcr_abs)
run("k05", 0.5, 0.7)
run("k05_mc05", 0.5, 0.5)
run("k1", 1.0, 0.7)
for cm in (0.67, 0.675, 0.68):
    run(f"k05_mc{cm}", 0.5, cm)

base = scen["pre"]
say("\nscenario | kappa | merging mc | p1 p2 p3 | s1 s3 s0 | dCS | dPi(1+2) | dPi3 | dW  (market size 1)")
for name, d in scen.items():
    p, s = d["p"], d["s"]
    dcs = d["cs"] - base["cs"]
    dpi_m = d["pi"][:2].sum() - base["pi"][:2].sum()
    dpi_3 = d["pi"][2] - base["pi"][2]
    say(f"{name:10s} | {d['kappa']:.2f} | {d['cm']:.4f} | {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} | "
        f"{s[0]:.4f} {s[2]:.4f} {1 - s.sum():.4f} | {dcs:+.5f} | {dpi_m:+.5f} | {dpi_3:+.5f} | "
        f"{dcs + dpi_m + dpi_3:+.5f}")
say("percent price changes, kappa=0:", np.round(100 * (scen["k0"]["p"] - 1), 2))
say("percent price changes, kappa=0.5:", np.round(100 * (scen["k05"]["p"] - 1), 2))
say("percent CS change, kappa=0:", round(100 * (scen["k0"]["cs"] / base["cs"] - 1), 3),
    "; kappa=0.5:", round(100 * (scen["k05"]["cs"] / base["cs"] - 1), 3))
say(f"pre-merger CS = {base['cs']:.5f}, merging profits = {base['pi'][:2].sum():.5f}, firm 3 = {base['pi'][2]:.5f}")


# merging-firm cost that keeps consumer surplus unchanged
def cs_gap(cm, kappa):
    c = C0.copy()
    c[:2] = cm
    return cs(solve(omega(kappa), c)) - base["cs"]


for kappa in (0.0, 0.5):
    cm_star = brentq(cs_gap, 0.2, 0.7, args=(kappa,), xtol=1e-12)
    c = C0.copy()
    c[:2] = cm_star
    p = solve(omega(kappa), c)
    say(f"kappa={kappa}: CS-neutral merging mc = {cm_star:.4f} (reduction {100 * (1 - cm_star / 0.7):.2f} percent),"
        f" prices {np.round(p, 4)}")


# merging-firm cost at which the merging firms' price equals 1.09 with kappa = 0.5
def p1_gap(cm, target):
    c = C0.copy()
    c[:2] = cm
    return solve(omega(0.5), c)[0] - target


for target in (1.085, 1.09, 1.095):
    cm_t = brentq(p1_gap, 0.55, 0.7, args=(target,), xtol=1e-12)
    c = C0.copy()
    c[:2] = cm_t
    say(f"kappa=0.5: merging price {target} needs merging mc {cm_t:.4f}; prices {np.round(solve(omega(0.5), c), 4)}")

# multiproduct logit common markup check: m_f = 1/(alpha (1 - S_f))
p = scen["k0"]["p"]
s = scen["k0"]["s"]
say(f"common-markup check (merged firm): p-c = {p[0] - 0.7:.6f}, 1/(alpha(1-S_f)) = {1 / (ALPHA * (1 - s[0] - s[1])):.6f}")
say(f"rival markup check: p-c = {p[2] - 0.7:.6f}, 1/(alpha(1-s3)) = {1 / (ALPHA * (1 - s[2])):.6f}")

# ------------------------------------------------------------------ figure: prices against kappa
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

kgrid = np.linspace(0.0, 1.0, 101)
curves = {}
for cm in (0.7, 0.6):
    pm, pr = [], []
    start = None
    for k in kgrid:
        c = C0.copy()
        c[:2] = cm
        p = solve(omega(k), c, start)
        start = p
        pm.append(p[0])
        pr.append(p[2])
    curves[cm] = (np.array(pm), np.array(pr))

fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.axhline(1.0, color=INK2, lw=0.8, ls=":", zorder=0)
ax.plot(kgrid, curves[0.7][0], color=COLORS[0], lw=2.0, ls="-", label="merging firms, $c_{1}=c_{2}=0.7$")
ax.plot(kgrid, curves[0.7][1], color=COLORS[1], lw=2.0, ls="-", label="rival (firm 3), $c_{1}=c_{2}=0.7$")
ax.plot(kgrid, curves[0.6][0], color=COLORS[0], lw=2.0, ls="--", label="merging firms, $c_{1}=c_{2}=0.6$")
ax.plot(kgrid, curves[0.6][1], color=COLORS[1], lw=2.0, ls="--", label="rival (firm 3), $c_{1}=c_{2}=0.6$")
for k in (0.0, 0.5):
    i = int(round(k * 100))
    for arr, col in ((curves[0.7][0], COLORS[0]), (curves[0.7][1], COLORS[1])):
        ax.plot([k], [arr[i]], marker="o", ms=5, color=col, mec="white", mew=1.0, zorder=4)
ax.annotate(f"{curves[0.7][0][0]:.4f}", (0.0, curves[0.7][0][0]), xytext=(8, -14), textcoords="offset points",
            fontsize=8, color=INK2)
ax.annotate(f"{curves[0.7][1][0]:.4f}", (0.0, curves[0.7][1][0]), xytext=(0.11, 0.991), textcoords="data",
            fontsize=8, color=INK2, va="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=3))
ax.set_ylim(0.985, 1.18)
ax.annotate(f"{curves[0.7][0][50]:.4f}", (0.5, curves[0.7][0][50]), xytext=(-8, 8), textcoords="offset points",
            fontsize=8, color=INK2, ha="right")
ax.annotate(f"{curves[0.7][1][50]:.4f}", (0.5, curves[0.7][1][50]), xytext=(6, -12), textcoords="offset points",
            fontsize=8, color=INK2)
ax.set_xlabel(r"internalization weight $\kappa$")
ax.set_ylabel("post-merger price")
ax.set_xlim(0, 1)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=150)
say("kappa = 1 prices (mc 0.7):", np.round([curves[0.7][0][-1], curves[0.7][1][-1]], 4))
say("kappa = 1 prices (mc 0.6):", np.round([curves[0.6][0][-1], curves[0.6][1][-1]], 4))
say("kappa = 0.5 prices (mc 0.6):", np.round([curves[0.6][0][50], curves[0.6][1][50]], 4))

OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
