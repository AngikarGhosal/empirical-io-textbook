"""
Replication of the Monte Carlo design of Guerre, Perrigne and Vuong (2000, Section 2.4) for
Appendix A, Section A.10.7 (fig:A-gpv).

Design (as stated in GPV, PDF pp. 9-10, journal pp. 532-533):
    L = 200 auctions, N = 5 bidders (GPV's I), 1,000 replications;
    values i.i.d. log-normal(0, 1) truncated to [0.055, 2.5]; nonbinding reserve (p0 = lower bound);
    equilibrium bids s(v) = v - int_{v_lo}^{v} F(u)^{N-1} du / F(v)^{N-1};
    step 1: G-hat = empirical cdf of the N L bids, g-hat = triweight kernel density,
            h_g = 1.06 sd(bids) (N L)^(-1/5);
            pseudo-values V-hat = B + G-hat(B) / ((N-1) g-hat(B)) for B in
            [B_min + h_g, B_max - h_g], and +infinity otherwise (GPV eq. 6 with rho_g = 2);
    step 2: f-hat(v) = (1/(N L h_f)) sum K((v - V-hat)/h_f), triweight kernel,
            h_f = 1.06 sd(finite V-hat) (N L_T)^(-1/5), where N L_T is the number of untrimmed bids
            (our reading of "L_T is the number of auctions remaining after the trimming").
    Both functions evaluated at 500 equally spaced points, as in GPV.
Also computed: the infeasible density estimate from the true values (same kernel and rule),
GPV's interior region (footnote 9): [mean of xi-hat(B_min + h_g) + h_f, mean of xi-hat(B_max - h_g) - h_f].

Output: figures/appA/gpv.pdf, figures/appA/src/gpv_mc.json, figures/appA/src/gpv_preview.png
"""
import os
import json
import time
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
SEED = 20261022
rng = np.random.default_rng(SEED)

BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7b5cd6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral"})

L, N, REPS = 200, 5, 1000
VLO, VHI = 0.055, 2.5
PA, PB = stats.norm.cdf(np.log(VLO)), stats.norm.cdf(np.log(VHI))


def F(v):
    return (stats.norm.cdf(np.log(v)) - PA) / (PB - PA)


def f(v):
    return stats.norm.pdf(np.log(v)) / (v * (PB - PA))


def draw_values(size):
    uu = rng.uniform(size=size)
    return np.exp(stats.norm.ppf(PA + uu * (PB - PA)))


# Equilibrium bid function on a fine grid (cumulative trapezoid for int F^{N-1}).
vg = np.linspace(VLO, VHI, 200001)
Fg = F(vg)
integrand = Fg ** (N - 1)
cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(vg))])
sg = vg.copy()
sg[1:] = vg[1:] - cum[1:] / Fg[1:] ** (N - 1)
assert np.all(np.diff(sg) > 0)


def bid(v):
    return np.interp(v, vg, sg)


def inv_bid(b):
    return np.interp(b, sg, vg)


def triweight(z):
    return np.where(np.abs(z) <= 1, 35.0 / 32.0 * (1 - z**2) ** 3, 0.0)


def kde(points, data, h, denom):
    out = np.empty(points.size)
    for start in range(0, points.size, 250):
        p = points[start:start + 250]
        out[start:start + 250] = triweight((p[:, None] - data[None, :]) / h).sum(axis=1)
    return out / (denom * h)


bgrid = np.linspace(bid(VLO), bid(VHI), 500)
vgrid = np.linspace(VLO, VHI, 500)
f_true = f(vgrid)
xi_true = inv_bid(bgrid)

XI = np.empty((REPS, bgrid.size))
FH = np.empty((REPS, vgrid.size))
FT = np.empty((REPS, vgrid.size))
hg_all, hf_all, trim_share, lo_line, hi_line = [], [], [], [], []
bmin_l, bmax_l = [], []
first = None
t0 = time.time()
for r in range(REPS):
    V = draw_values(N * L)
    B = bid(V)
    nb = B.size
    sd_b = B.std(ddof=1)
    hg = 1.06 * sd_b * nb ** (-0.2)
    Bs = np.sort(B)
    Ghat_B = np.searchsorted(Bs, B, side="right") / nb
    ghat_B = kde(B, B, hg, nb)
    keep = (B >= Bs[0] + hg) & (B <= Bs[-1] - hg)
    Vhat = np.full(nb, np.inf)
    Vhat[keep] = B[keep] + Ghat_B[keep] / ((N - 1) * ghat_B[keep])
    fin = Vhat[np.isfinite(Vhat)]
    hf = 1.06 * fin.std(ddof=1) * fin.size ** (-0.2)
    FH[r] = kde(vgrid, fin, hf, nb)                 # GPV eq. (7): divide by N L
    ht = 1.06 * V.std(ddof=1) * nb ** (-0.2)
    FT[r] = kde(vgrid, V, ht, nb)                   # infeasible: true values observed
    Ghat_grid = np.searchsorted(Bs, bgrid, side="right") / nb
    ghat_grid = kde(bgrid, B, hg, nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        XI[r] = bgrid + Ghat_grid / ((N - 1) * ghat_grid)
    # GPV footnote 9 lines: xi-hat(B_min + h_g) + h_f and xi-hat(B_max - h_g) - h_f
    def xi_at(b):
        G = np.searchsorted(Bs, b, side="right") / nb
        g = kde(np.array([b]), B, hg, nb)[0]
        return b + G / ((N - 1) * g)
    lo_line.append(xi_at(Bs[0] + hg) + hf)
    hi_line.append(xi_at(Bs[-1] - hg) - hf)
    hg_all.append(hg)
    hf_all.append(hf)
    trim_share.append(1 - keep.mean())
    bmin_l.append(Bs[0] + hg)
    bmax_l.append(Bs[-1] - hg)
    if r == 0:
        first = dict(B=B[keep], Vhat=Vhat[keep], Btrim=B[~keep], bmin=Bs[0] + hg, bmax=Bs[-1] - hg)
print(f"{REPS} replications in {time.time() - t0:.1f}s")

lo, hi = float(np.mean(lo_line)), float(np.mean(hi_line))
inner = (vgrid >= lo) & (vgrid <= hi)
dv = vgrid[1] - vgrid[0]
mean_fh, q05, q95 = FH.mean(0), np.quantile(FH, 0.05, axis=0), np.quantile(FH, 0.95, axis=0)
mean_ft = FT.mean(0)
ise_h = ((FH[:, inner] - f_true[inner]) ** 2).sum(1) * dv
ise_t = ((FT[:, inner] - f_true[inner]) ** 2).sum(1) * dv
cover = np.mean((q05[inner] <= f_true[inner]) & (f_true[inner] <= q95[inner]))
xi_mean = np.nanmean(XI, 0)
xi_q05, xi_q95 = np.nanquantile(XI, 0.05, axis=0), np.nanquantile(XI, 0.95, axis=0)
bmin_avg, bmax_avg = float(np.mean(bmin_l)), float(np.mean(bmax_l))

res = {
    "seed": SEED, "L": L, "N": N, "reps": REPS,
    "mass_kept_by_truncation": float(PB - PA),
    "bid_support": [float(bid(VLO)), float(bid(VHI))],
    "mean_hg": float(np.mean(hg_all)), "mean_hf": float(np.mean(hf_all)),
    "mean_trim_share": float(np.mean(trim_share)),
    "trim_bid_lo": bmin_avg, "trim_bid_hi": bmax_avg,
    "interior_lo": lo, "interior_hi": hi,
    "mean_ise_twostep_interior": float(ise_h.mean()),
    "mean_ise_infeasible_interior": float(ise_t.mean()),
    "ise_ratio": float(ise_h.mean() / ise_t.mean()),
    "max_abs_meanbias_interior": float(np.max(np.abs(mean_fh[inner] - f_true[inner]))),
    "max_f_true_interior": float(f_true[inner].max()),
    "coverage_5_95_band_interior": float(cover),
    "f_true_at_lo_end": float(f_true[0]), "mean_fh_at_lo_end": float(mean_fh[0]),
    "f_true_at_hi_end": float(f_true[-1]), "mean_fh_at_hi_end": float(mean_fh[-1]),
    "mean_ft_at_lo_end": float(mean_ft[0]), "mean_ft_at_hi_end": float(mean_ft[-1]),
}
with open(os.path.join(HERE, "gpv_mc.json"), "w") as fh:
    json.dump(res, fh, indent=2)
print(json.dumps(res, indent=2))

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.75))
a = ax[0]
a.axvspan(bgrid[0], bmin_avg, color=GRID, lw=0)
a.axvspan(bmax_avg, bgrid[-1], color=GRID, lw=0)
a.fill_between(bgrid, xi_q05, xi_q95, color=BLUE, alpha=0.18, lw=0, label="5 to 95 percent band")
a.scatter(first["B"], first["Vhat"], s=3, color=ORANGE, alpha=0.55, lw=0,
          label=r"pseudo-values $(B_{il}, \hat V_{il})$")
a.plot(bgrid, xi_mean, color=BLUE, lw=1.4, label=r"mean of $\hat\xi(b)$")
a.plot(bgrid, xi_true, color=INK, lw=1.0, ls="--", label=r"true $\xi(b) = \beta^{-1}(b)$")
a.set_xlim(bgrid[0], bgrid[-1])
a.set_ylim(0, 3.2)
a.set_xlabel("bid $b$")
a.set_ylabel("value $v$")
a.set_title("(a) Inverse bid function and pseudo-values", loc="left")
a.legend(frameon=False, loc="upper left")
a.grid(color=GRID, lw=0.5)

a = ax[1]
a.axvspan(VLO, lo, color=GRID, lw=0)
a.axvspan(hi, VHI, color=GRID, lw=0)
a.fill_between(vgrid, q05, q95, color=BLUE, alpha=0.18, lw=0, label="5 to 95 percent band")
a.plot(vgrid, mean_fh, color=BLUE, lw=1.4, label=r"mean of two-step $\hat f$")
a.plot(vgrid, mean_ft, color=ORANGE, lw=1.0, ls=":", label="mean, values observed")
a.plot(vgrid, f_true, color=INK, lw=1.0, ls="--", label="true $f$")
a.set_xlim(0, VHI)
a.set_ylim(0, 1.0)
a.set_xlabel("value $v$")
a.set_ylabel("density")
a.set_title("(b) Density of private values", loc="left")
a.legend(frameon=False, loc="upper right")
a.grid(color=GRID, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "gpv.pdf"))
fig.savefig(os.path.join(HERE, "gpv_preview.png"), dpi=170)
print("saved")
