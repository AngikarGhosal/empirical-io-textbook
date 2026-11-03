"""Binding reserve price and unknown number of potential bidders (Section 15.4.7, fig:15-gpv-reserve).

Design (our simulation, following Guerre, Perrigne and Vuong 2000, Section 4):
  T = 500 auctions, N = 5 potential bidders (unknown to the econometrician),
  values i.i.d. lognormal(0, 1) truncated to [0.055, 2.5] as in GPV Section 2.4,
  reserve price r = 0.6 in every auction.  Only bidders with V >= r bid, and they bid
  beta_r(v) = v - int_r^v F(u)^{N-1} du / F(v)^{N-1}  (Riley-Samuelson with a reserve).
  Bidders do not observe how many rivals are actual bidders.

Estimator (GPV Section 4.2, no covariates):
  N_hat     = max_t n*_t,
  F_hat(r)  = 1 - mean(n*) / N_hat                        (binomial mean),
  B_dag     = sqrt(B* - r)                                 (removes the unbounded density at r),
  G_dag, g_dag: empirical CDF and triweight kernel density of B_dag,
                h = 1.06 sd(B_dag) (#bids)^(-1/5), trimming within h of both ends,
  V_hat     = r + B_dag^2 + 2 B_dag/(N_hat-1) [G_dag/g_dag + F_hat(r)/(1-F_hat(r)) / g_dag]  (GPV eq. 27),
  F_hat(v)  = F_hat(r) + (1 - F_hat(r)) G*_hat(xi_hat^{-1}(v)) for v >= r.
  A naive version drops the term in F(r), as if the truncated bid distribution were the bid
  distribution of N_hat bidders without a reserve.

Outputs: figures/ch15/gpv_reserve.pdf and figures/ch15/src/gpv_reserve_numbers.txt
Seed: 15404.
"""
from pathlib import Path

import numpy as np
from scipy.stats import norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_FIG = HERE.parent / "gpv_reserve.pdf"
OUT_TXT = HERE / "gpv_reserve_numbers.txt"

BLUE, ORANGE, AQUA, GRAY = "#2A78D6", "#EB6834", "#1BAF7A", "#52514E"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": "#222222",
    "xtick.color": GRAY,
    "ytick.color": GRAY,
})

SEED = 15404
T, NPOT, RES = 500, 5, 0.6
VL, VH = 0.055, 2.5
PL, PH = norm.cdf(np.log(VL)), norm.cdf(np.log(VH))
MASS = PH - PL


def F_true(v):
    v = np.clip(v, VL, VH)
    return (norm.cdf(np.log(v)) - PL) / MASS


def draw_values(rng, size):
    u = rng.uniform(size=size)
    return np.exp(norm.ppf(PL + u * MASS))


FR = F_true(RES)
VGRID = np.linspace(RES, VH, 200001)
FV = F_true(VGRID)
INTEG = np.concatenate([[0.0], np.cumsum(0.5 * (FV[1:] ** (NPOT - 1) + FV[:-1] ** (NPOT - 1))
                                          * np.diff(VGRID))])
BETA = VGRID - INTEG / FV ** (NPOT - 1)
B_TOP = BETA[-1]


def beta(v):
    return np.interp(v, VGRID, BETA)


def xi_true(b):
    return np.interp(b, BETA, VGRID)


def gstar_true(b):
    v = xi_true(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return F_true(v) / ((NPOT - 1) * (v - b) * (1 - FR))


def K(u):
    return np.where(np.abs(u) <= 1.0, 35.0 / 32.0 * (1.0 - u ** 2) ** 3, 0.0)


rng = np.random.default_rng(SEED)
vals = draw_values(rng, (T, NPOT))
active = vals >= RES
nstar = active.sum(axis=1)
bstar = beta(vals[active])                  # all actual bids, pooled
vact = vals[active]

N_hat = nstar.max()
FR_hat = 1 - nstar.mean() / N_hat
bdag = np.sqrt(bstar - RES)
nb = bdag.size
h = 1.06 * bdag.std(ddof=1) * nb ** (-0.2)
s = np.sort(bdag)
Gd = np.searchsorted(s, bdag, side="right") / nb
gd = K((bdag[:, None] - bdag[None, :]) / h).sum(axis=1) / (nb * h)
keep = (bdag >= s[0] + h) & (bdag <= s[-1] - h)
V_hat = RES + bdag ** 2 + 2 * bdag / (N_hat - 1) * (Gd / gd + FR_hat / (1 - FR_hat) / gd)
V_naive = RES + bdag ** 2 + 2 * bdag / (N_hat - 1) * (Gd / gd)
Gstar_hat = np.searchsorted(np.sort(bstar), bstar, side="right") / nb
F_at_V = FR_hat + (1 - FR_hat) * Gstar_hat

err = np.abs(V_hat[keep] - vact[keep])
err_naive = np.abs(V_naive[keep] - vact[keep])
near = keep & (vact < RES + 0.3)
lines = [
    f"seed={SEED}, T={T}, N={NPOT}, r={RES}, truncated lognormal(0,1) on [{VL},{VH}]",
    f"true F(r) = {FR:.4f}; true E[n*] = N(1-F(r)) = {NPOT*(1-FR):.4f}",
    f"sample mean n* = {nstar.mean():.4f}; share of auctions with no bid = {np.mean(nstar==0):.4f}",
    f"N_hat = max n* = {N_hat}; F_hat(r) = {FR_hat:.4f}",
    f"number of bids = {nb}; bandwidth on sqrt scale h = {h:.4f}; kept = {keep.sum()}",
    f"top bid beta(v_bar) = {B_TOP:.4f}",
    f"mean |V_hat - V| (kept): GPV reserve formula {err.mean():.4f}, naive {err_naive.mean():.4f}",
    f"mean (V_hat - V) for kept values in [r, r+0.3]: GPV {np.mean(V_hat[near]-vact[near]):.4f}, "
    f"naive {np.mean(V_naive[near]-vact[near]):.4f}",
]
OUT_TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
ax = axes[0]
ax.hist(bstar, bins=40, density=True, color=BLUE, alpha=0.35, edgecolor="white", linewidth=0.4)
bb = np.linspace(RES + 1e-4, B_TOP, 600)
ax.plot(bb, gstar_true(bb), color=GRAY, lw=1.6)
ax.set_ylim(0, 6)
ax.set_xlim(RES - 0.02, B_TOP + 0.02)
ax.set_xlabel(r"bid $b^*$")
ax.set_ylabel("density")
ax.annotate(r"$g^*(b)\to\infty$ as $b\downarrow r$", xy=(0.61, 5.0), xytext=(0.75, 4.6), color=GRAY,
            fontsize=8.5, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
ax.set_title("(a) Bids above the reserve", fontsize=9.5)

ax = axes[1]
ax.hist(bdag, bins=40, density=True, color=BLUE, alpha=0.35, edgecolor="white", linewidth=0.4)
dd = np.linspace(1e-4, np.sqrt(B_TOP - RES), 600)
ax.plot(dd, 2 * dd * gstar_true(RES + dd ** 2), color=GRAY, lw=1.6)
ax.set_xlabel(r"$b^\dagger=(b^*-r)^{\frac{1}{2}}$")
ax.set_title("(b) After the transformation", fontsize=9.5)

ax = axes[2]
vv = np.linspace(RES, VH, 400)
ax.plot(vv, F_true(vv), color=GRAY, lw=2.0)
o = np.argsort(V_hat[keep])
ax.plot(V_hat[keep][o], F_at_V[keep][o], color=BLUE, lw=1.3, ls="--")
o2 = np.argsort(V_naive[keep])
ax.plot(V_naive[keep][o2], F_at_V[keep][o2], color=ORANGE, lw=1.3, ls=":")
ax.set_xlim(RES - 0.05, VH)
ax.set_ylim(FR - 0.05, 1.02)
ax.set_xlabel(r"value $v$")
ax.set_ylabel(r"$F(v)$")
ax.text(1.45, 0.62, "true $F$\nand estimate", color=GRAY, fontsize=8.5)
ax.text(0.95, 0.93, r"naive (no $F(r)$ term)", color=ORANGE, fontsize=8.5, ha="center")
ax.set_title(r"(c) $F$ above the reserve", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT_FIG)
fig.savefig(HERE / "gpv_reserve_preview.png", dpi=130)
print("saved", OUT_FIG)
