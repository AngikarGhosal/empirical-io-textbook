"""Demand curvature and simulated merger price effects (Chapter 11, fig:11-curvature).

Two symmetric single-product firms merge; there is no other inside product. Every demand system is calibrated
to the same pre-merger point: price p = 1, marginal cost c = 0.6 (Lerner margin L = 0.4), own-price elasticity
1/L = 2.5 (the single-product Bertrand condition), and diversion ratio D between the two products. Only the
curvature of demand differs.

  linear:               q_j = a - b p_j + d p_k, with b = 1/L, d = D b, a = 1 + b - d (so q_j = 1 at p = 1)
  logit (outside good): s_j = exp(a - alpha p_j) / (1 + sum_k exp(a - alpha p_k)), s = D/(1+D), alpha = 1/(L(1-s))
  constant-expenditure logit (CES with an outside good):
                        q_j = B a p_j^(-sigma) / (1 + a p_1^(1-sigma) + a p_2^(1-sigma)), sigma = (1 + D)/L,
                        expenditure share w = a/(1 + 2a) = D/(L (sigma - 1)), B = 1/w
  constant elasticity:  q_j = k p_j^(-eps) p_k^(eta), eps = 1/L, eta = D eps

For each D the script solves the post-merger joint-profit-maximizing prices (no efficiencies) and also reports the
"frozen shares and slopes" first-iteration approximation, which in the symmetric case equals the Werden
compensating marginal cost reduction D (p - c) / (1 - D).

Linear, logit and constant-expenditure logit demand give a bounded joint profit with a symmetric global maximum,
which the script checks on a price grid. Constant-elasticity demand does not. With p_1 > c fixed, q_1 grows without
bound as p_2 rises, so joint profit has no maximum for any D > 0. The constant-elasticity curve is therefore the
merged firm's optimum when it must charge one common price for both products. That price is also the symmetric
solution of the unrestricted first-order conditions. The script checks that this solution is a strict local maximum
of unrestricted joint profit for small D and a saddle point for D above about 0.254.

Output: figures/ch11/curvature_sim.pdf, figures/ch11/src/curvature_sim_output.txt
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
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


def ces_demand(D):
    """Constant-expenditure logit (CES with an outside good) calibrated at p = 1."""
    sig = (1 + D) * EPS
    w = D * EPS / (sig - 1)
    a = w / (1 - 2 * w)
    B = 1 / w

    def q(p1, p2):
        den = 1 + a * p1 ** (1 - sig) + a * p2 ** (1 - sig)
        return B * a * p1 ** (-sig) / den, B * a * p2 ** (-sig) / den

    return q


def ces_effect(D):
    q = ces_demand(D)
    res = minimize_scalar(lambda p: -2 * (p - C) * q(p, p)[0], bounds=(C, 10.0), method="bounded",
                          options=dict(xatol=1e-12))
    return res.x / P - 1


def ce_effect(D):
    eta = D * EPS
    if EPS - eta <= 1:
        return np.nan
    # merged firm restricted to a common price p: joint profit 2 (p - c) k p^(eta - eps)
    p = C * (EPS - eta) / (EPS - eta - 1)
    return p / P - 1


def ce_profit(D, p1, p2):
    eta = D * EPS
    return (p1 - C) * p1 ** -EPS * p2 ** eta + (p2 - C) * p2 ** -EPS * p1 ** eta


def joint_profit(D, system):
    """Joint profit of the merged firm as a function of (p1, p2) for the bounded systems."""
    if system == "linear":
        b = EPS
        d = D * b
        a = 1.0 + b - d
        return lambda p1, p2: (p1 - C) * (a - b * p1 + d * p2) + (p2 - C) * (a - b * p2 + d * p1)
    if system == "logit":
        s = D / (1 + D)
        alpha = EPS / (P * (1 - s))
        al = np.log(s / (1 - 2 * s)) + alpha * P

        def f(p1, p2):
            e1, e2 = np.exp(al - alpha * p1), np.exp(al - alpha * p2)
            den = 1 + e1 + e2
            return (p1 - C) * e1 / den + (p2 - C) * e2 / den
        return f
    q = ces_demand(D)
    return lambda p1, p2: (p1 - C) * q(p1, p2)[0] + (p2 - C) * q(p1, p2)[1]


def global_check(D, system, p_sym):
    """Grid search over (p1, p2) in [c, 4]^2: returns the best grid profit and the symmetric optimum's profit."""
    f = joint_profit(D, system)
    g = np.linspace(C, 4.0, 681)
    X, Y = np.meshgrid(g, g)
    with np.errstate(all="ignore"):
        Z = f(X, Y)
    i = np.unravel_index(np.nanargmax(Z), Z.shape)
    return f(p_sym, p_sym), Z[i], X[i], Y[i]


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
    # constant-expenditure logit
    qs = ces_demand(D)
    dqs = (np.array(qs(1 + h, 1)) - np.array(qs(1 - h, 1))) / (2 * h)
    out["CES"] = (-dqs[0] / qs(1, 1)[0], -dqs[1] / dqs[0])
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
lines.append("percent price increases")
lines.append("D | linear | logit | const.-expend. logit | const. elasticity (common price) | frozen approx | "
             "UPP/p = D L | CES/linear | CE/linear")
for D in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
    li, lo, cs, ce, fr = linear_effect(D), logit_effect(D), ces_effect(D), ce_effect(D), frozen_effect(D)
    lines.append(f"{D:.2f} | {100 * li:.2f} | {100 * lo:.2f} | {100 * cs:.2f} | {100 * ce:.2f} | {100 * fr:.2f} | "
                 f"{100 * D * L:.2f} | {cs / li:.2f} | {ce / li:.2f}")
lines.append("global check on [c, 4]^2 grid: profit at symmetric optimum vs best grid point (p1, p2)")
for D in (0.05, 0.20, 0.35):
    for system, eff in (("linear", linear_effect), ("logit", logit_effect), ("CES", ces_effect)):
        ps = P * (1 + eff(D))
        fs, fg, x1, x2 = global_check(D, system, ps)
        assert fg <= fs + 1e-9, (D, system)
        lines.append(f"D={D:.2f} {system}: symmetric optimum p={ps:.4f}, profit {fs:.6f}; "
                     f"best grid point ({x1:.3f}, {x2:.3f}) profit {fg:.6f}")
lines.append("constant elasticity: no global maximum of unrestricted joint profit")
for D in (0.05, 0.20, 0.35):
    ps = P * (1 + ce_effect(D))
    h = 1e-4
    x0 = np.array([ps, ps])
    E = np.eye(2) * h
    f = lambda x: ce_profit(D, x[0], x[1])
    H = np.array([[(f(x0 + E[i] + E[j]) - f(x0 + E[i] - E[j]) - f(x0 - E[i] + E[j]) + f(x0 - E[i] - E[j]))
                   / (4 * h * h) for j in range(2)] for i in range(2)])
    eig = np.linalg.eigvalsh(H)
    kind = "strict local maximum" if eig.max() < 0 else "saddle point"
    lines.append(f"D={D:.2f}: common-price optimum p={ps:.4f}, profit {ce_profit(D, ps, ps):.6f}, "
                 f"Hessian eigenvalues {eig[0]:.4f}, {eig[1]:.4f} ({kind}); "
                 f"profit at (1, 100) {ce_profit(D, 1.0, 100.0):.6f}, at (1, 10000) {ce_profit(D, 1.0, 1e4):.6f}")
# At the symmetric stationary point the Hessian eigenvalue in the direction (1, -1) has the sign of
# -eps^2 + 3 eta^2 + 2 eps eta + eps - eta, with eta = D eps. Its root in D is the saddle threshold.
D_saddle = brentq(lambda D: -EPS ** 2 + 3 * (D * EPS) ** 2 + 2 * EPS * D * EPS + EPS - D * EPS, 0.0, 1.0)
lines.append(f"constant elasticity: the symmetric stationary point is a saddle point for D > {D_saddle:.4f}")
print("\n".join(lines))
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
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
cs = np.array([ces_effect(d) if d > 0 else 0.0 for d in Dg])
ce = np.array([ce_effect(d) for d in Dg])
fr = np.array([frozen_effect(d) for d in Dg])

fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(Dg, 100 * ce, color=COLORS[1], lw=2.0, ls="-", label="constant elasticity, common price imposed")
ax.plot(Dg, 100 * cs, color=COLORS[4], lw=2.0, ls=(0, (7, 2)), label="constant-expenditure logit")
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
