"""Model-generated recreation of the design of Bresnahan (1987), Figure 2,
plus a companion figure for the close-neighbor limit (Proposition
prop:06-bertrand-limit).

Model (Bresnahan 1987, eqs. 1-21, in the notation of Chapter 6):
  utility  v*delta_j + y - p_j  for a new car, v*delta_u + y - p_u otherwise;
  v uniform on [0, vbar] with density kappa;
  marginal cost mc(delta) = psi * exp(a * delta); used-car price p_u = mc(delta_u);
  demand q_j = kappa * (v_{j,j+1} - v_{j-1,j}),  v_{h,i} = (p_i - p_h)/(delta_i - delta_h),
  top product q_n = kappa * (vbar - v_{n-1,n}).
Because demand is linear, q(p) = q(mc) + D (p - mc) with D symmetric
tridiagonal, and the first-order conditions (20) are
  q + (Omega o D') m = 0   <=>   A(Omega) m = q(mc),   A(Omega) = -(D + Omega o D).
Collusion: Omega = all ones.  Bertrand: Omega_jk = 1 iff same owner.

The script
  (1) solves both equilibria for the seven-product design and draws
      figures/ch06/bresnahan_fig2_sim.pdf;
  (2) checks the closed form of the collusive markups,
      m_j = [vbar (delta_j - delta_u) - (mc_j - mc_u)] / 2,
      the monotonicity of every markup in Omega, the M-matrix properties
      of A(Omega), and the "top block" property (a single firm owning the
      top consecutive products sets the same price increments as a
      monopolist);
  (3) varies the gap delta_3 - delta_2 and draws
      figures/ch06/bresnahan_limit.pdf.
No random numbers are used. Numbers are printed and written to
figures/ch06/src/bresnahan_fig2_sim_numbers.txt.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


# ----------------------------------------------------------------- model
def build(delta, kappa, vbar, psi, a, du):
    """Return mc, used-car cost, gaps, Jacobian D and q(mc)."""
    n = len(delta)
    c = psi * np.exp(a * delta)
    cu = psi * np.exp(a * du)
    d0 = np.r_[du, delta]
    c0 = np.r_[cu, c]
    g = np.diff(d0)                      # g_j = delta_j - delta_{j-1}, g_1 uses delta_u
    D = np.zeros((n, n))
    for j in range(n):
        D[j, j] = -kappa / g[j] - (kappa / g[j + 1] if j < n - 1 else 0.0)
        if j < n - 1:
            D[j, j + 1] = D[j + 1, j] = kappa / g[j + 1]
    chord = np.diff(c0) / g              # chord slopes of mc between neighbors
    qmc = kappa * (np.r_[chord[1:], vbar] - chord)
    return c, cu, g, D, qmc


def omega(owners, collude):
    n = len(owners)
    if collude:
        return np.ones((n, n))
    return np.array([[1.0 if owners[i] == owners[k] else 0.0 for k in range(n)]
                     for i in range(n)])


def equilibrium(delta, owners, kappa, vbar, psi, a, du, Om):
    c, cu, g, D, qmc = build(delta, kappa, vbar, psi, a, du)
    A = -(D + Om * D)
    m = np.linalg.solve(A, qmc)
    q = qmc + D @ m
    p = c + m
    cut = np.diff(np.r_[cu, p]) / g      # v_{j-1,j}, j = 1..n
    return dict(c=c, cu=cu, g=g, D=D, A=A, qmc=qmc, m=m, q=q, p=p, cut=cut)


def is_m_matrix(A):
    off = A - np.diag(np.diag(A))
    z = (off <= 1e-12).all() and (np.diag(A) > 0).all()
    Ainv = np.linalg.inv(A)
    return z and (Ainv > 0).all()


# ----------------------------------------------------------------- design
DELTA = np.array([1.0, 2.0, 2.15, 2.8, 4.3, 5.2, 6.1])
OWN = "GFGFGGG"
KAPPA, PSI, A_EXP, DU = 1.0, 1.0, 0.7, 0.85
_c = PSI * np.exp(A_EXP * DELTA)
VBAR = 1.3 * (_c[-1] - _c[-2]) / (DELTA[-1] - DELTA[-2])
VBAR = float(np.round(VBAR, 1))

say("Design: delta = %s, owners = %s" % (DELTA, OWN))
say("kappa = %.1f, psi = %.1f, a = %.2f, delta_u = %.2f, vbar = %.1f"
    % (KAPPA, PSI, A_EXP, DU, VBAR))

eqC = equilibrium(DELTA, OWN, KAPPA, VBAR, PSI, A_EXP, DU, omega(OWN, True))
eqB = equilibrium(DELTA, OWN, KAPPA, VBAR, PSI, A_EXP, DU, omega(OWN, False))

say("\n j own  delta     mc   p_C    m_C   q_C    p_B    m_B   q_B   m_B/m_C")
for j in range(len(DELTA)):
    say("%2d  %s  %5.2f %6.2f %6.2f %6.2f %5.3f %6.2f %6.2f %5.3f  %5.3f" % (
        j + 1, OWN[j], DELTA[j], eqC["c"][j], eqC["p"][j], eqC["m"][j], eqC["q"][j],
        eqB["p"][j], eqB["m"][j], eqB["q"][j], eqB["m"][j] / eqC["m"][j]))
say("q(mc) = %s" % np.round(eqC["qmc"], 3))
say("cutoffs C: %s (vbar %.1f)" % (np.round(eqC["cut"], 3), VBAR))
say("cutoffs B: %s" % np.round(eqB["cut"], 3))

# Checks.
mC_closed = (VBAR * (DELTA - DU) - (eqC["c"] - eqC["cu"])) / 2.0
say("\nmax |collusive markup - closed form| = %.2e" % np.max(np.abs(mC_closed - eqC["m"])))
say("A(Omega) nonsingular M-matrix with positive inverse: C %s, B %s"
    % (is_m_matrix(eqC["A"]), is_m_matrix(eqB["A"])))
say("All quantities positive: C %s, B %s; q(mc) > 0: %s"
    % ((eqC["q"] > 0).all(), (eqB["q"] > 0).all(), (eqC["qmc"] > 0).all()))
say("Every markup higher under collusion: %s" % (eqC["m"] > eqB["m"]).all())
dC = np.diff(eqC["p"][4:])
dB = np.diff(eqB["p"][4:])
say("Top block (products 5-7, one owner): price increments C %s, B %s"
    % (np.round(dC, 4), np.round(dB, 4)))
say("Markup drop C - B for products 5, 6, 7: %s" % np.round((eqC["m"] - eqB["m"])[4:], 4))

# Monotonicity: switch each neighbor pair from 0 to 1 starting from Bertrand.
OmB = omega(OWN, False)
base = eqB["m"]
for i in range(len(DELTA) - 1):
    if OmB[i, i + 1] == 1:
        continue
    Om2 = OmB.copy()
    Om2[i, i + 1] = Om2[i + 1, i] = 1.0
    e2 = equilibrium(DELTA, OWN, KAPPA, VBAR, PSI, A_EXP, DU, Om2)
    say("Switch Omega_{%d,%d} to 1: all markups rise: %s (min rise %.4f)"
        % (i + 1, i + 2, (e2["m"] > base).all(), np.min(e2["m"] - base)))

# ----------------------------------------------------------------- figure 2
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral"})

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
xx = np.linspace(DU - 0.15, DELTA[-1] + 0.35, 400)
ymax = 1.08 * max(eqC["p"].max(), eqB["p"].max())
for ax, eq, title in [(axes[0], eqC, "(a) Collusion: all products cooperate"),
                      (axes[1], eqB, "(b) Bertrand-Nash: firms G and F compete")]:
    ax.plot(xx, PSI * np.exp(A_EXP * xx), color=INK, lw=1.3)
    ax.text(xx[-1] + 0.06, PSI * np.exp(A_EXP * xx[-1]), r"$mc(\delta)$",
            fontsize=8, ha="left", va="center", color=INK)
    for j in range(len(DELTA)):
        ax.plot([DELTA[j], DELTA[j]], [eq["c"][j], eq["p"][j]], color=BLUE, lw=0.8,
                ls=(0, (1.5, 1.5)))
    ax.plot(DELTA, eq["p"], "o", ms=4.5, color=BLUE, mec="white", mew=0.6,
            label=r"equilibrium price $p_j^*$")
    ax.plot(DU, eq["cu"], "s", ms=4, color=ORANGE, mec="white", mew=0.5,
            label=r"used car $(\delta_{\mathrm{u}}, p_{\mathrm{u}})$")
    ax.set_title(title, loc="left")
    ax.set_xlim(DU - 0.3, DELTA[-1] + 0.9)
    ax.set_ylim(-0.12 * ymax, ymax)
    ax.set_xticks([])
    for j in range(len(DELTA)):
        ax.plot([DELTA[j], DELTA[j]], [0, -0.018 * ymax], color=INK2, lw=0.6)
    for j in range(len(DELTA)):
        dx = -0.06 if j == 1 else (0.06 if j == 2 else 0.0)
        ax.text(DELTA[j] + dx, -0.035 * ymax, str(j + 1), ha="center", va="top", fontsize=7.5)
        ax.text(DELTA[j] + dx, -0.085 * ymax, OWN[j], ha="center", va="top", fontsize=7.5,
                color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_bounds(0, ymax)
    ax.set_xlabel(r"quality $\delta$ (product number and owner below the axis)")
    ax.grid(axis="y", color=GRID, lw=0.5)
axes[0].set_ylabel(r"price $p^*$ and marginal cost $mc$")
axes[0].legend(loc="upper left", frameon=False)
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(FIGDIR, "bresnahan_fig2_sim.pdf"))
fig.savefig(os.path.join(HERE, "bresnahan_fig2_sim_preview.png"), dpi=170)
plt.close(fig)

# ----------------------------------------------------------------- limit figure
gaps = np.geomspace(0.6, 0.002, 60)
rows = []
for gp in gaps:
    d = DELTA.copy()
    d[2] = d[1] + gp
    eC = equilibrium(d, OWN, KAPPA, VBAR, PSI, A_EXP, DU, omega(OWN, True))
    eB = equilibrium(d, OWN, KAPPA, VBAR, PSI, A_EXP, DU, omega(OWN, False))
    OmP = omega(OWN, False)
    OmP[1, 2] = OmP[2, 1] = 1.0
    eP = equilibrium(d, OWN, KAPPA, VBAR, PSI, A_EXP, DU, OmP)
    ok = all((e["q"] > 0).all() for e in (eC, eB, eP))
    rows.append((gp, eB["m"][1], eB["m"][2], eP["m"][1], eP["m"][2], eC["m"][1],
                 eC["m"][2], ok))
rows = np.array(rows, dtype=float)
say("\nLimit check (gap = delta_3 - delta_2): gap, m2_B, m3_B, m2_pair, m3_pair, m2_C, m3_C, interior")
for r in rows[::10]:
    say("  %.4f  %.4f %.4f  %.4f %.4f  %.4f %.4f  %d" % tuple(r))
say("  %.4f  %.4f %.4f  %.4f %.4f  %.4f %.4f  %d" % tuple(rows[-1]))
say("All solutions interior along the path: %s" % bool(rows[:, 7].all()))

fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.plot(rows[:, 0], rows[:, 1], color=BLUE, lw=1.6, label="Bertrand, product 2 (F)")
ax.plot(rows[:, 0], rows[:, 2], color=BLUE, lw=1.2, ls=(0, (4, 2)),
        label="Bertrand, product 3 (G)")
ax.plot(rows[:, 0], rows[:, 3], color=ORANGE, lw=1.6,
        label=r"only $\Omega_{23} = 1$ added, product 2")
ax.plot(rows[:, 0], rows[:, 4], color=ORANGE, lw=1.2, ls=(0, (4, 2)),
        label=r"only $\Omega_{23} = 1$ added, product 3")
ax.plot(rows[:, 0], rows[:, 5], color=INK2, lw=1.0, ls=(0, (1, 1.5)),
        label="collusion, products 2 and 3")
ax.plot(rows[:, 0], rows[:, 6], color=INK2, lw=1.0, ls=(0, (1, 1.5)))
ax.set_xscale("log")
ax.set_yscale("log")
ax.invert_xaxis()
ax.set_xlabel(r"quality gap $\delta_3 - \delta_2$ (log scale, shrinking to the right)")
ax.set_ylabel(r"markup $p_j - mc_j$ (log scale)")
ax.grid(color=GRID, lw=0.5)
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "bresnahan_limit.pdf"))
fig.savefig(os.path.join(HERE, "bresnahan_limit_preview.png"), dpi=170)
plt.close(fig)

with open(os.path.join(HERE, "bresnahan_fig2_sim_numbers.txt"), "w") as fh:
    fh.write("\n".join(OUT) + "\n")
