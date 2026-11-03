"""Markups and consumer surplus as products are added: logit versus vertical.

Model (i), logit with tastes for products. J identical single-product firms,
utility u_ij = delta - alpha p_j + eps_ij (eps i.i.d. standard Gumbel), outside
good u_i0 = eps_i0, marginal cost c. Symmetric Bertrand-Nash price solves
    p - c = 1 / (alpha (1 - s)),   s = exp(delta - alpha p) / (1 + J exp(delta - alpha p)).
Expected consumer surplus (dropping the constant) is
    CS = (1/alpha) ln(1 + J exp(delta - alpha p)).

Model (ii), pure vertical model (Bresnahan's form). Utility v*delta_j - P_j,
taste for quality v ~ U[0, vbar] with density kappa = 1/vbar (unit mass),
outside good delta_0 = P_0 = 0. The J products have qualities at the midpoints
of J equal bins of [delta_L, delta_H], marginal cost mc(delta) = psi exp(a delta),
and each product has its own owner. Demand is linear in prices; the Bertrand
first-order conditions form a linear system that we solve exactly. Consumer
surplus is the integral over v of max(0, max_j v delta_j - P_j).

No random numbers are used. Output: figures/ch06/clones.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)

# ----------------------------------------------------------------- logit
ALPHA, DELTA, C_LOGIT = 1.0, 2.0, 1.0


def logit_eq(J):
    """Symmetric Nash price, markup, share and CS with J identical products."""
    p = C_LOGIT + 1.0 / ALPHA
    for _ in range(10000):
        ev = np.exp(DELTA - ALPHA * p)
        s = ev / (1.0 + J * ev)
        p_new = C_LOGIT + 1.0 / (ALPHA * (1.0 - s))
        if abs(p_new - p) < 1e-13:
            p = p_new
            break
        p = 0.5 * p + 0.5 * p_new
    ev = np.exp(DELTA - ALPHA * p)
    s = ev / (1.0 + J * ev)
    cs = np.log(1.0 + J * ev) / ALPHA
    return p, p - C_LOGIT, s, cs


# -------------------------------------------------------------- vertical
VBAR, DL, DH, PSI, A = 1.0, 1.0, 3.0, 0.05, 1.0
KAPPA = 1.0 / VBAR


def mc(d):
    return PSI * np.exp(A * d)


def vertical_eq(J):
    """Bertrand-Nash prices with single-product firms and linear vertical demand."""
    d = DL + (np.arange(1, J + 1) - 0.5) * (DH - DL) / J
    c = mc(d)
    dfull = np.concatenate(([0.0], d))
    g = 1.0 / np.diff(dfull)            # g[j-1] = 1/(delta_j - delta_{j-1}), j = 1..J
    Amat = np.zeros((J, J))
    b = np.zeros(J)
    for j in range(J):                  # python index j is product j+1
        gj = g[j]
        if j < J - 1:
            gn = g[j + 1]
            # g_{j+1}(P_{j+1} - P_j) - g_j(P_j - P_{j-1}) - (g_j + g_{j+1})(P_j - c_j) = 0
            Amat[j, j] = -gn - gj - (gj + gn)
            Amat[j, j + 1] = gn
            if j > 0:
                Amat[j, j - 1] = gj
            b[j] = -(gj + gn) * c[j]
        else:
            # vbar - g_J(P_J - P_{J-1}) - g_J(P_J - c_J) = 0
            Amat[j, j] = -2.0 * gj
            if j > 0:
                Amat[j, j - 1] = gj
            b[j] = -VBAR - gj * c[j]
    P = np.linalg.solve(Amat, b)
    Pfull = np.concatenate(([0.0], P))
    cut = g * np.diff(Pfull)            # cutoff v_j between j-1 and j
    upper = np.concatenate((cut[1:], [VBAR]))
    q = KAPPA * (upper - cut)
    ok = bool(np.all(np.diff(cut) > 0) and cut[0] > 0 and cut[-1] < VBAR and np.all(q > 0))
    vgrid = np.linspace(0.0, VBAR, 200001)
    util = np.max(np.outer(vgrid, d) - P[None, :], axis=1)
    cs = np.trapezoid(np.maximum(util, 0.0), vgrid) * KAPPA
    markup = np.sum(q * (P - c)) / np.sum(q)
    return P, c, q, markup, cs, ok


def vertical_limit_cs():
    """CS when every quality in [DL, DH] is sold at marginal cost."""
    vgrid = np.linspace(0.0, VBAR, 20001)
    dgrid = np.linspace(DL, DH, 4001)
    util = np.max(np.outer(vgrid, dgrid) - mc(dgrid)[None, :], axis=1)
    return np.trapezoid(np.maximum(util, 0.0), vgrid) * KAPPA


Js = np.unique(np.round(np.logspace(0, np.log10(200), 40)).astype(int))
lg = np.array([logit_eq(J) for J in Js])
vt = [vertical_eq(J) for J in Js]
v_markup = np.array([r[3] for r in vt])
v_cs = np.array([r[4] for r in vt])
v_ok = all(r[5] for r in vt)
cs_lim = vertical_limit_cs()

print("Logit: alpha=%.1f delta=%.1f c=%.1f" % (ALPHA, DELTA, C_LOGIT))
for J in (1, 2, 5, 10, 50, 200):
    p, m, s, cs = logit_eq(J)
    print("  J=%3d  p=%.4f  markup=%.4f  share=%.4f  CS=%.4f" % (J, p, m, s, cs))
print("Vertical: vbar=%.1f delta in [%.1f,%.1f] psi=%.2f a=%.1f; all products sell: %s"
      % (VBAR, DL, DH, PSI, A, v_ok))
for J in (1, 2, 5, 10, 50, 200):
    P, c, q, m, cs, ok = vertical_eq(J)
    print("  J=%3d  avg markup=%.5f  CS=%.5f  total sales=%.4f  ok=%s"
          % (J, m, cs, q.sum(), ok))
print("  competitive-limit CS = %.5f" % cs_lim)

# index each series to its J = 1 value
lm0, lc0 = lg[0, 1], lg[0, 3]
vm0, vc0 = v_markup[0], v_cs[0]
print("Index bases: logit markup %.4f, logit CS %.4f, vertical markup %.4f, vertical CS %.4f"
      % (lm0, lc0, vm0, vc0))
print("Logit markup floor index = %.4f; vertical CS limit index = %.4f"
      % ((1.0 / ALPHA) / lm0, cs_lim / vc0))

# ----------------------------------------------------------------- figure
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral"})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.7))

ax1.plot(Js, lg[:, 1] / lm0, color=BLUE, lw=1.6, label="Logit (tastes for products)")
ax1.plot(Js, v_markup / vm0, color=ORANGE, lw=1.6, ls=(0, (4, 2)),
         label="Vertical (pure characteristics)")
ax1.axhline((1.0 / ALPHA) / lm0, color=INK2, lw=0.7, ls=(0, (1, 1.5)))
ax1.annotate(r"logit floor $\frac{1}{\alpha}$", xy=(60, (1.0 / ALPHA) / lm0),
             xytext=(25, 0.52), fontsize=7.2, color=INK,
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax1.set_xscale("log")
ax1.set_ylim(0.0, 1.08)
ax1.set_xlim(1, 200)
ax1.set_xlabel(r"Number of products $J$ (log scale)")
ax1.set_ylabel(r"Index, value at $J = 1$ equals 1")
ax1.set_title("(a) Average Nash markup", loc="left")
ax1.legend(loc="upper right", frameon=False, handlelength=2.4, borderaxespad=0.2)

ax2.plot(Js, lg[:, 3] / lc0, color=BLUE, lw=1.6, label="Logit")
ax2.plot(Js, v_cs / vc0, color=ORANGE, lw=1.6, ls=(0, (4, 2)), label="Vertical")
ax2.axhline(cs_lim / vc0, color=INK2, lw=0.7, ls=(0, (1, 1.5)))
ax2.annotate("vertical limit:\nall qualities at cost", xy=(40, cs_lim / vc0),
             xytext=(1.25, cs_lim / vc0 + 2.3), fontsize=7.2, color=INK,
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax2.text(22, 9.0, "Logit", color=INK, fontsize=7.2, ha="right")
ax2.text(60, cs_lim / vc0 - 0.9, "Vertical", color=INK, fontsize=7.2, ha="center")
ax2.set_xscale("log")
ax2.set_xlim(1, 200)
ax2.set_xlabel(r"Number of products $J$ (log scale)")
ax2.set_ylabel(r"Index, value at $J = 1$ equals 1")
ax2.set_title("(b) Expected consumer surplus", loc="left")

from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
for a_ in (ax1, ax2):
    a_.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200]))
    a_.xaxis.set_major_formatter(FixedFormatter(["1", "2", "5", "10", "20", "50", "100", "200"]))
    a_.xaxis.set_minor_locator(NullLocator())
    a_.grid(True, color=GRID, lw=0.5)
    a_.set_axisbelow(True)

fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIGDIR, "clones.pdf"))
fig.savefig(os.path.join(HERE, "clones_preview.png"), dpi=150)
print("saved")
