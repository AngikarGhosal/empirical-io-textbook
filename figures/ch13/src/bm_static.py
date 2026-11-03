"""Figure fig:13-bm-cdfs (Chapter 13, Section 13.7.2): the Bernheim-Madsen model, uniform example.

Primitives: customer reservation value v ~ U[0, 1], so D(p) = 1 - p on [0, 1];
home marginal cost c_H = 0.1, away marginal cost c_A = 0.2, fixed cost of quoting phi = 0.05.

On-path behavior in a market with price floor p_f in [p_A_low, p_H*] (Section 13.7.2):
  home firm:  Psi_H(p) = 1 - phi / (D(p)(p - c_A))   on [p_f, p_H*), atom at p_f of mass
              1 - phi / (D(p_f)(p_f - c_A)), atom at p_H* of mass phi / (D(p_H*)(p_H* - c_A)).
  away firm:  lambda_A Psi_A(p) = 1 - D(p_f)(p_f - c_H) / (D(p)(p - c_H))  on [p_f, p_H*],
              lambda_A = 1 - D(p_f)(p_f - c_H) / (D(p_H*)(p_H* - c_H)).
The stage-game Nash equilibrium (Bernheim and Madsen, Theorem 1) is p_f = p_A_low.
Optimal stationary collusion (their Theorems 4-5) has the floor p_f(delta) solving
  (1 - 2 delta) [D(p)(p - c_H) - phi] = (1 - delta) Delta_c D(p)
for delta in [delta_low, delta^M], where delta_low = 1 / (2 + Delta_c D(p_H*) / phi) (Theorem 3)
and delta^M = (Pi^M - Delta_c D(p_H*)) / (2 Pi^M - Delta_c D(p_H*)) (Theorem 4).

Outputs: figures/ch13/bm_cdfs.pdf and figures/ch13/src/bm_static_output.txt
Run from any directory:  python bm_static.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
OUT_TXT = os.path.join(HERE, "bm_static_output.txt")
OUT_PDF = os.path.join(FIGDIR, "bm_cdfs.pdf")

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}",
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.8,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

lines = []


def say(s=""):
    print(s)
    lines.append(s)


# ---------------------------------------------------------------- primitives
cH, cA, phi = 0.1, 0.2, 0.05
dc = cA - cH


def D(p):
    return np.clip(1.0 - p, 0.0, 1.0)


def profit(p, c):
    """Single-firm stage profit net of the quoting cost."""
    return D(p) * (p - c) - phi


# break-even price of the away firm: lowest root of (1 - p)(p - cA) = phi
pA_low = (1.0 + cA - np.sqrt((1.0 + cA) ** 2 - 4.0 * (cA + phi))) / 2.0
pH_low = (1.0 + cH - np.sqrt((1.0 + cH) ** 2 - 4.0 * (cH + phi))) / 2.0
pH_star = (1.0 + cH) / 2.0
pA_star = (1.0 + cA) / 2.0
PiC = dc * D(pA_low)
PiM = profit(pH_star, cH)
x_M = dc * D(pH_star)
deltaM = (PiM - x_M) / (2.0 * PiM - x_M)
delta_low = 1.0 / (2.0 + x_M / phi)

say("Bernheim-Madsen uniform example: v ~ U[0,1], c_H = 0.1, c_A = 0.2, phi = 0.05")
say(f"  closed form p_A_low = (1.2 - sqrt(0.44))/2 = {(1.2 - np.sqrt(0.44)) / 2:.6f}")
say(f"  break-even price away firm  p_A_low = {pA_low:.6f}")
say(f"  break-even price home firm  p_H_low = {pH_low:.6f}")
say(f"  monopoly prices             p_H* = {pH_star:.4f},  p_A* = {pA_star:.4f}")
say(f"  assumption p_A_low < p_H*:  {pA_low < pH_star}")
say(f"  Pi^C = Delta_c D(p_A_low)      = {PiC:.6f}")
say(f"  Pi^M = D(p_H*)(p_H* - c_H) - phi = {PiM:.6f}")
say(f"  Delta_c D(p_H*)                = {x_M:.6f}")
say(f"  Theorem 4 condition Pi^M > phi + Delta_c D(p_H*): {PiM:.4f} > {phi + x_M:.4f} -> {PiM > phi + x_M}")
say(f"  delta^M   = {deltaM:.6f}")
say(f"  delta_low = 1/(2 + Delta_c D(p_H*)/phi) = {delta_low:.6f}")


def onpath(pf, n=4001):
    """On-path CDFs for a floor pf. Returns grid, Psi_H (left limits), lambda_A Psi_A, atoms, lambda_A."""
    p = np.linspace(pf, pH_star, n)
    PsiH = 1.0 - phi / (D(p) * (p - cA))         # valid on [pf, pH*)
    atom_low = 1.0 - phi / (D(pf) * (pf - cA))
    atom_top = phi / (D(pH_star) * (pH_star - cA))
    lamPsiA = 1.0 - D(pf) * (pf - cH) / (D(p) * (p - cH))
    lamA = 1.0 - D(pf) * (pf - cH) / (D(pH_star) * (pH_star - cH))
    return p, PsiH, lamPsiA, atom_low, atom_top, lamA


def summary(pf, label):
    p, PsiH, lamPsiA, atom_low, atom_top, lamA = onpath(pf, n=200001)
    # density of the away firm's quotes (with respect to p), unconditional on entry
    dens = np.gradient(lamPsiA, p)
    surv_home = 1.0 - PsiH                     # Pr(home price > p) for p in (pf, pH*)
    pr_steal = np.trapezoid(surv_home * dens, p)
    # expected lowest quote: home price if away absent or above; away price otherwise
    # E[min] = E[home price * 1{away absent or away > home}] + E[away price * 1{away < home}]
    # home price distribution: atom at pf, density on (pf, pH*), atom at pH*
    dPsiH = np.gradient(PsiH, p)
    surv_away = 1.0 - lamPsiA                  # Pr(away absent or away price > p)
    e_home = pf * atom_low * 1.0 + np.trapezoid(p * dPsiH * surv_away, p) + pH_star * atom_top * (1.0 - lamA)
    e_away = np.trapezoid(p * dens * surv_home, p)
    e_min = e_home + e_away
    home_profit = profit(pf, cH)
    say(f"--- {label}: floor p_f = {pf:.6f}")
    say(f"  home profit Pi = D(p_f)(p_f - c_H) - phi = {home_profit:.6f}")
    say(f"  home atom at floor = {atom_low:.6f}; home atom at p_H* = {atom_top:.6f}")
    say(f"  away entry probability lambda_A = {lamA:.6f}")
    say(f"  Pr(away firm wins the customer) = {pr_steal:.6f}")
    say(f"  expected lowest quoted price    = {e_min:.6f}")
    say(f"  check total probability of a quote below p_H* by home: {atom_low + PsiH[-2] - PsiH[0]:.6f} + atom_top {atom_top:.6f}")
    return dict(pf=pf, Pi=home_profit, atom_low=atom_low, atom_top=atom_top, lamA=lamA,
                steal=pr_steal, emin=e_min)


nash = summary(pA_low, "stage-game Nash equilibrium (p_f = p_A_low)")


def floor(delta):
    f = lambda p: (1.0 - 2.0 * delta) * profit(p, cH) - (1.0 - delta) * dc * D(p)
    if f(pH_star) <= 0:
        return pH_star
    return brentq(f, pA_low, pH_star)


delta_ex = 0.38
coll = summary(floor(delta_ex), f"optimal stationary collusion at delta = {delta_ex}")
say(f"  IC check at delta = {delta_ex}: Pi = {coll['Pi']:.6f}, "
    f"(1-delta)(2 Pi - Delta_c D(p_f)) = {(1 - delta_ex) * (2 * coll['Pi'] - dc * D(coll['pf'])):.6f}")
for d in [delta_low, 0.36, 0.38, 0.40, deltaM]:
    pf = floor(d)
    say(f"  delta = {d:.4f}: floor = {pf:.4f}, Pi* = {profit(pf, cH):.4f}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.45), constrained_layout=True)


def draw_cdfs(ax, pf, title):
    p, PsiH, lamPsiA, atom_low, atom_top, lamA = onpath(pf)
    # home CDF: jump at pf (if any), continuous part, jump to 1 at pH*
    ax.plot([0.2, pf], [0, 0], color=BLUE, lw=1.6)
    ax.plot(p[:-1], PsiH[:-1], color=BLUE, lw=1.6, label=r"home $\Psi_H(p)$")
    ax.plot([pH_star, 0.62], [1, 1], color=BLUE, lw=1.6)
    if atom_low > 1e-6:
        ax.plot([pf, pf], [0, atom_low], color=BLUE, lw=0.8, ls=":")
        ax.plot(pf, atom_low, "o", ms=4, color=BLUE)
        ax.plot(pf, 0, "o", ms=4, mfc="white", mec=BLUE)
    ax.plot([pH_star, pH_star], [PsiH[-2], 1], color=BLUE, lw=0.8, ls=":")
    ax.plot(pH_star, PsiH[-2], "o", ms=4, mfc="white", mec=BLUE)
    ax.plot(pH_star, 1, "o", ms=4, color=BLUE)
    # away: unconditional CDF of quotes lambda_A Psi_A
    ax.plot([0.2, pf], [0, 0], color=ORANGE, lw=1.6, ls="--")
    ax.plot(p, lamPsiA, color=ORANGE, lw=1.6, ls="--", label=r"away $\lambda_A\Psi_A(p)$")
    ax.plot([pH_star, 0.62], [lamA, lamA], color=ORANGE, lw=1.6, ls="--")
    ax.set_xlim(0.2, 0.62)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"price $p$")
    ax.set_title(title)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)


draw_cdfs(axes[0], pA_low, "(a) stage game")
axes[0].set_xticks([pA_low, 0.4, pH_star])
axes[0].set_xticklabels([r"$\underline{p}_A$", "0.4", r"$p_H^*$"])
axes[0].set_ylabel("cumulative probability")
axes[0].legend(loc="upper left", frameon=False, handlelength=1.8)

pf_ex = coll["pf"]
draw_cdfs(axes[1], pf_ex, r"(b) collusion, $\delta = 0.38$")
axes[1].set_xticks([pA_low, pf_ex, pH_star])
axes[1].set_xticklabels([r"$\underline{p}_A$", r"$p^{\mathrm{fl}}$", r"$p_H^*$"])

# panel (c): floor as a function of delta
ax = axes[2]
ds = np.linspace(delta_low, deltaM, 200)
fl = np.array([floor(d) for d in ds])
ax.plot(ds, fl, color=AQUA, lw=1.8, label=r"floor $p^{\mathrm{fl}}(\delta)$")
ax.plot([deltaM, 0.5], [pH_star, pH_star], color=GREY, lw=1.8, label=r"monopoly $p_H^*$")
ax.axhline(pA_low, color=GREY, lw=0.8, ls=":")
ax.axvline(deltaM, color=GREY, lw=0.6, ls=":")
ax.set_xlim(0.33, 0.5)
ax.set_ylim(0.24, 0.58)
ax.set_xticks([delta_low, deltaM, 0.5])
ax.set_xticklabels([r"$\underline{\delta}$", r"$\delta^M$", r"$\frac{1}{2}$"])
ax.set_yticks([pA_low, 0.4, pH_star])
ax.set_yticklabels([r"$\underline{p}_A$", "0.4", r"$p_H^*$"])
ax.set_xlabel(r"discount factor $\delta$")
ax.set_title("(c) price floor")
ax.legend(loc="center right", frameon=False, handlelength=1.8)
ax.grid(axis="y", color="#e4e3df", lw=0.6)

fig.savefig(OUT_PDF)
fig.savefig(os.path.join(HERE, "bm_cdfs_preview.png"), dpi=150)
say(f"saved {OUT_PDF}")

with open(OUT_TXT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
