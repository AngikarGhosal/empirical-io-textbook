"""
Kernel density estimation: bandwidth choice and the bias-variance trade-off.
Appendix A, Section A.10 (fig:A-kde).

True density: two-component normal mixture f = 0.6 N(0, 1) + 0.4 N(2.5, 0.5^2).
Gaussian kernel, sample size n = 200.

Panel (a): one sample with estimates at h*/2, h*, 2h*, where h* minimizes the exact MISE.
Panel (b): integrated squared bias, integrated variance and MISE as functions of h.
           Lines: exact finite-sample formulas for a normal mixture with a Gaussian kernel
           (E f-hat = f convolved with K_h; Var f-hat = [E K_h(x-X)^2 - (E f-hat)^2] / n).
           Markers: Monte Carlo averages over 1,000 samples at 12 bandwidths.
           Dashed: the AMISE approximation h^4 mu2^2 R(f'') / 4 + kappa2 / (n h).
Bandwidth selectors (Monte Carlo, 1,000 samples): Silverman's rule 1.06 sd n^(-1/5) and
least-squares cross-validation (LSCV) minimized over a fine grid; the integrated squared
error of each is recorded.

Output: figures/appA/kde.pdf, figures/appA/src/kde_demo.json, figures/appA/src/kde_preview.png
"""
import os
import json
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

W = np.array([0.6, 0.4])
MU = np.array([0.0, 2.5])
SD = np.array([1.0, 0.5])
n = 200
REPS = 1000
x = np.linspace(-5.0, 5.5, 1051)
dx = x[1] - x[0]


def mix_pdf(xx, var_add=0.0):
    s = np.sqrt(SD**2 + var_add)
    return (W[None, :] * stats.norm.pdf(xx[:, None], MU[None, :], s[None, :])).sum(1)


def mix_d2(xx):
    z = (xx[:, None] - MU[None, :]) / SD[None, :]
    return (W[None, :] * (z**2 - 1) * stats.norm.pdf(z) / SD[None, :] ** 3).sum(1)


f = mix_pdf(x)
Rf2 = float(np.sum(mix_d2(x) ** 2) * dx)          # int f''^2
mu2, kappa2 = 1.0, 1.0 / (2 * np.sqrt(np.pi))


def exact_terms(h):
    Ef = mix_pdf(x, h**2)
    EK2 = mix_pdf(x, h**2 / 2) / (2 * np.sqrt(np.pi) * h)
    ib2 = np.sum((Ef - f) ** 2) * dx
    iv = np.sum((EK2 - Ef**2) / n) * dx
    return ib2, iv


hgrid = np.exp(np.linspace(np.log(0.04), np.log(1.5), 200))
IB2, IV = np.array([exact_terms(h) for h in hgrid]).T
MISE = IB2 + IV
h_star = float(hgrid[np.argmin(MISE)])
h_amise = float((kappa2 / (mu2**2 * Rf2 * n)) ** 0.2)
AMISE = hgrid**4 * mu2**2 * Rf2 / 4 + kappa2 / (n * hgrid)
sd_mix = float(np.sqrt(np.sum(W * (SD**2 + MU**2)) - np.sum(W * MU) ** 2))


def draw(m):
    comp = rng.uniform(size=m) < W[0]
    return np.where(comp, rng.normal(MU[0], SD[0], m), rng.normal(MU[1], SD[1], m))


def kde(xx, data, h):
    return stats.norm.pdf((xx[:, None] - data[None, :]) / h).sum(1) / (data.size * h)


def lscv(data, hs):
    d = data[:, None] - data[None, :]
    m = data.size
    out = []
    for h in hs:
        term1 = stats.norm.pdf(d / h, scale=np.sqrt(2)).sum() / (m**2 * h)   # int f-hat^2
        K = stats.norm.pdf(d / h)
        loo = (K.sum(1) - K.diagonal()) / ((m - 1) * h)                    # leave-one-out
        out.append(term1 - 2 * loo.mean())
    return np.array(out)


# Monte Carlo
hmc = np.exp(np.linspace(np.log(0.06), np.log(1.2), 12))
EST = np.zeros((hmc.size, x.size))
EST2 = np.zeros((hmc.size, x.size))
hs_cv_grid = np.exp(np.linspace(np.log(0.05), np.log(1.5), 60))
h_silv, h_cv, ise_silv, ise_cv, ise_star = [], [], [], [], []
for r in range(REPS):
    X = draw(n)
    for k, h in enumerate(hmc):
        e = kde(x, X, h)
        EST[k] += e
        EST2[k] += e**2
    hs = 1.06 * X.std(ddof=1) * n ** (-0.2)
    hc = hs_cv_grid[np.argmin(lscv(X, hs_cv_grid))]
    h_silv.append(hs)
    h_cv.append(hc)
    ise_silv.append(np.sum((kde(x, X, hs) - f) ** 2) * dx)
    ise_cv.append(np.sum((kde(x, X, hc) - f) ** 2) * dx)
    ise_star.append(np.sum((kde(x, X, h_star) - f) ** 2) * dx)
EST /= REPS
EST2 /= REPS
mc_ib2 = np.sum((EST - f[None, :]) ** 2, 1) * dx
mc_iv = np.sum(EST2 - EST**2, 1) * dx

res = {
    "seed": SEED, "n": n, "reps": REPS, "Rf2": Rf2, "sd_mixture": sd_mix,
    "h_star_exact_mise": h_star, "h_amise": h_amise,
    "mise_at_h_star": float(MISE.min()),
    "silverman_population": float(1.06 * sd_mix * n ** (-0.2)),
    "h_silverman_mean": float(np.mean(h_silv)),
    "h_lscv_mean": float(np.mean(h_cv)), "h_lscv_sd": float(np.std(h_cv)),
    "h_lscv_q10": float(np.quantile(h_cv, 0.1)), "h_lscv_q90": float(np.quantile(h_cv, 0.9)),
    "mean_ise_silverman": float(np.mean(ise_silv)),
    "mean_ise_lscv": float(np.mean(ise_cv)),
    "mean_ise_hstar": float(np.mean(ise_star)),
    "exact_ib2_iv_at_hstar": [float(v) for v in exact_terms(h_star)],
    "mc_vs_exact_max_rel_dev_mise": float(np.max(np.abs((mc_ib2 + mc_iv) -
                                                       np.interp(hmc, hgrid, MISE)) /
                                                np.interp(hmc, hgrid, MISE))),
}
with open(os.path.join(HERE, "kde_demo.json"), "w") as fh:
    json.dump(res, fh, indent=2)
print(json.dumps(res, indent=2))

# ------------------------------------------------------------------ figure
# Display sample: among 101 fresh samples, the one whose ISE at h* is the median.
cands = [draw(n) for _ in range(101)]
ises = np.array([np.sum((kde(x, c, h_star) - f) ** 2) * dx for c in cands])
X1 = cands[int(np.argsort(ises)[50])]
res["display_sample_ise_at_hstar"] = float(np.sort(ises)[50])
with open(os.path.join(HERE, "kde_demo.json"), "w") as fh:
    json.dump(res, fh, indent=2)
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
a = ax[0]
a.plot(x, f, color=INK, lw=1.0, ls="--", label="true $f$")
a.plot(x, kde(x, X1, h_star / 2), color=ORANGE, lw=0.9, label=r"$h = \frac{1}{2}h^{*}$")
a.plot(x, kde(x, X1, h_star), color=BLUE, lw=1.4, label=r"$h = h^{*}$")
a.plot(x, kde(x, X1, 2 * h_star), color=AQUA, lw=1.2, label=r"$h = 2h^{*}$")
a.plot(X1, np.full(n, -0.012), "|", color=INK2, ms=4, mew=0.5)
a.set_xlim(-3.5, 4.5)
a.set_ylim(-0.025, 0.42)
a.set_xlabel("$x$")
a.set_ylabel("density")
a.set_title(f"(a) One sample, $n = {n}$", loc="left")
a.legend(frameon=False, loc="upper left")
a.grid(color=GRID, lw=0.5)

a = ax[1]
a.plot(hgrid, IB2, color=ORANGE, lw=1.2, label="integrated squared bias")
a.plot(hgrid, IV, color=AQUA, lw=1.2, label="integrated variance")
a.plot(hgrid, MISE, color=BLUE, lw=1.6, label="MISE")
a.plot(hgrid, AMISE, color=BLUE, lw=0.8, ls="--", label="AMISE approximation")
a.plot(hmc, mc_ib2, "o", color=ORANGE, ms=3.2, mfc="white", mew=0.8)
a.plot(hmc, mc_iv, "s", color=AQUA, ms=3.2, mfc="white", mew=0.8)
a.plot(hmc, mc_ib2 + mc_iv, "D", color=BLUE, ms=3.2, mfc="white", mew=0.8,
       label="Monte Carlo (markers)")
a.axvline(h_star, color=INK2, lw=0.6, ls=":")
a.text(h_star * 1.04, 0.012, r"$h^{*}$", color=INK2, fontsize=7.5)
a.set_xscale("log")
a.set_ylim(0, 0.032)
a.set_xlim(0.04, 1.5)
a.set_xlabel("bandwidth $h$ (log scale)")
a.set_ylabel("integrated error")
a.set_title("(b) Bias, variance and MISE", loc="left")
a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.0), ncol=1)
a.grid(color=GRID, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "kde.pdf"))
fig.savefig(os.path.join(HERE, "kde_preview.png"), dpi=170)
print("saved")
