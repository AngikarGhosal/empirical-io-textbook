"""Replication of the Guerre, Perrigne and Vuong (2000, Section 2.4) Monte Carlo design
(Section 15.4.6, fig:15-gpv-mc, fig:15-gpv-trim, tab:15-gpv-mc).

Design (GPV 2000, p. 532):
  T = 200 auctions, n = 5 bidders, values i.i.d. lognormal(0, 1) truncated to [0.055, 2.5],
  no reserve price (r equals the lower bound of the support).  Bids are computed from the
  Riley-Samuelson bid function beta(v) = v - int_{v_low}^v F(u)^{n-1} du / F(v)^{n-1}
  by numerical integration on a fine grid.

Estimator (GPV eqs. (4)-(7)):
  G_hat(b)  = empirical CDF of all nT bids,
  g_hat(b)  = triweight kernel density, h_g = c * 1.06 * sd(bids) * (nT)^(-1/5),
  trimming  : keep bids in [B_min + h_g, B_max - h_g]  (kernel support length rho = 2),
  V_hat     = B + G_hat(B) / ((n-1) g_hat(B)),
  f_hat(v)  = (1/(nT h_f)) sum_kept K((v - V_hat)/h_f),
              h_f = c * 1.06 * sd(kept V_hat) * (n_kept)^(-1/5).
  GPV use c = 1 (rule of thumb with the factor 1.06 only).  Later work (Li, Perrigne and Vuong
  2002; Campo, Perrigne and Vuong 2003; Krasnokutskaya 2011) uses c = 2.978, the canonical
  bandwidth ratio between the triweight and Gaussian kernels.  Both are reported.

Outputs:
  figures/ch15/gpv_mc.pdf     (panel a: inverse bid function, panel b: value density)
  figures/ch15/gpv_trim.pdf   (one sample: boundary bias of g_hat and untrimmed pseudo-values)
  figures/ch15/src/gpv_montecarlo_numbers.txt
Seeds: 15401 (main experiment), 15402 (sample-size experiment), 15403 (single sample).
"""
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_MC = HERE.parent / "gpv_mc.pdf"
OUT_TRIM = HERE.parent / "gpv_trim.pdf"
OUT_TXT = HERE / "gpv_montecarlo_numbers.txt"

BLUE, ORANGE, AQUA, GRAY = "#2A78D6", "#EB6834", "#1BAF7A", "#52514E"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": "#222222",
    "xtick.color": GRAY,
    "ytick.color": GRAY,
})

N_BID = 5
T_MAIN = 200
REPS = 1000
VL, VH = 0.055, 2.5
SEED_MAIN, SEED_SIZE, SEED_ONE = 15401, 15402, 15403

# ----------------------------------------------------------------- true model
PL, PH = norm.cdf(np.log(VL)), norm.cdf(np.log(VH))
MASS = PH - PL


def F_true(v):
    v = np.clip(v, VL, VH)
    return (norm.cdf(np.log(v)) - PL) / MASS


def f_true(v):
    v = np.asarray(v, dtype=float)
    out = norm.pdf(np.log(v)) / (v * MASS)
    return np.where((v >= VL) & (v <= VH), out, 0.0)


VGRID = np.linspace(VL, VH, 200001)
FV = F_true(VGRID)
FV[0] = 0.0
INTEG = np.concatenate([[0.0], np.cumsum(0.5 * (FV[1:] ** (N_BID - 1) + FV[:-1] ** (N_BID - 1))
                                          * np.diff(VGRID))])
with np.errstate(divide="ignore", invalid="ignore"):
    BETA = VGRID - INTEG / FV ** (N_BID - 1)
BETA[0] = VL
B_LOW, B_HIGH = BETA[0], BETA[-1]


def beta(v):
    return np.interp(v, VGRID, BETA)


def xi_true(b):
    return np.interp(b, BETA, VGRID)


def G_bid_true(b):
    return F_true(xi_true(b))


def g_bid_true(b):
    v = xi_true(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = F_true(v) / ((N_BID - 1) * (v - b))
    # at the lower bound the limit is n/(n-1) f(v_low)
    return np.where(v - b > 1e-10, out, N_BID / (N_BID - 1) * f_true(VL))


def draw_values(rng, size):
    u = rng.uniform(size=size)
    return np.exp(norm.ppf(PL + u * MASS))


def K(u):
    return np.where(np.abs(u) <= 1.0, 35.0 / 32.0 * (1.0 - u ** 2) ** 3, 0.0)


def ksum(points, data, h, chunk=2000):
    """sum_j K((points_i - data_j)/h) for each i.

    The triweight kernel has support [-1, 1], so only data within h of a point matter.  We sort
    the data, find each point's window with searchsorted, and evaluate the kernel on a padded
    index matrix (exact, and much faster than the full n-by-n matrix)."""
    points = np.atleast_1d(np.asarray(points, dtype=float))
    d = np.sort(np.asarray(data, dtype=float))
    lo = np.searchsorted(d, points - h, side="left")
    hi = np.searchsorted(d, points + h, side="right")
    out = np.zeros(points.size)
    for a in range(0, points.size, chunk):
        l, u = lo[a:a + chunk], hi[a:a + chunk]
        W = int((u - l).max()) if l.size else 0
        if W == 0:
            continue
        idx = l[:, None] + np.arange(W)[None, :]
        valid = idx < u[:, None]
        idx = np.minimum(idx, d.size - 1)
        z = (points[a:a + chunk, None] - d[idx]) / h
        kz = 35.0 / 32.0 * np.clip(1.0 - z * z, 0.0, None) ** 3
        out[a:a + chunk] = np.where(valid, kz, 0.0).sum(axis=1)
    return out


def gpv(bids, n, c=1.0, bgrid=None, vgrid=None, trim=True):
    """Two-step GPV estimator for one sample of bids (flattened)."""
    nb = bids.size
    s = np.sort(bids)
    hg = c * 1.06 * bids.std(ddof=1) * nb ** (-0.2)
    Ghat = np.searchsorted(s, bids, side="right") / nb
    ghat = ksum(bids, bids, hg) / (nb * hg)
    V = bids + Ghat / ((n - 1) * ghat)
    lo, hi = s[0] + hg, s[-1] - hg
    keep = (bids >= lo) & (bids <= hi) if trim else np.ones(nb, bool)
    Vk = V[keep]
    hf = c * 1.06 * Vk.std(ddof=1) * Vk.size ** (-0.2)
    out = dict(hg=hg, hf=hf, lo=lo, hi=hi, keep=keep, V=V, Ghat=Ghat, ghat=ghat)
    if bgrid is not None:
        Gg = np.searchsorted(s, bgrid, side="right") / nb
        gg = ksum(bgrid, bids, hg) / (nb * hg)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["xi"] = bgrid + Gg / ((n - 1) * gg)
        out["g_grid"] = gg
    if vgrid is not None:
        out["f"] = ksum(vgrid, Vk, hf) / (nb * hf)
    # boundary of the reliable region for f (GPV footnote 9)
    lo_v = bgrid_eval(bids, s, hg, n, lo) + hf
    hi_v = bgrid_eval(bids, s, hg, n, hi) - hf
    out["lo_v"], out["hi_v"] = lo_v, hi_v
    return out


def bgrid_eval(bids, s, hg, n, b):
    nb = bids.size
    G = np.searchsorted(s, b, side="right") / nb
    g = K((b - bids) / hg).sum() / (nb * hg)
    return b + G / ((n - 1) * g)


def kde(x, grid, h, nb):
    return ksum(grid, x, h) / (nb * h)


# ----------------------------------------------------------------- main experiment
bgrid = np.linspace(B_LOW, B_HIGH, 500)
vgrid = np.linspace(VL, VH, 500)
lines = []
lines.append(f"Design: T={T_MAIN}, n={N_BID}, lognormal(0,1) truncated to [{VL}, {VH}], reps={REPS}")
lines.append(f"Mass of the lognormal inside [{VL},{VH}]: {MASS:.4f} (left out {1-MASS:.4f})")
lines.append(f"Bid support: [{B_LOW:.4f}, {B_HIGH:.4f}]")
lines.append(f"E[V] = {np.trapezoid(VGRID * f_true(VGRID), VGRID):.4f}; "
             f"g(b_low) = n/(n-1) f(v_low) = {N_BID/(N_BID-1)*f_true(VL):.4f}")

results = {}
for c, seed in ((1.0, SEED_MAIN), (2.978, SEED_MAIN)):
    rng = np.random.default_rng(seed)
    XI = np.empty((REPS, bgrid.size))
    FH = np.empty((REPS, vgrid.size))
    hgs, hfs, los, his, lovs, hivs, trimfrac = [], [], [], [], [], [], []
    t0 = time.perf_counter()
    for r in range(REPS):
        vals = draw_values(rng, T_MAIN * N_BID)
        bids = beta(vals)
        est = gpv(bids, N_BID, c=c, bgrid=bgrid, vgrid=vgrid)
        XI[r], FH[r] = est["xi"], est["f"]
        hgs.append(est["hg"]); hfs.append(est["hf"]); los.append(est["lo"]); his.append(est["hi"])
        lovs.append(est["lo_v"]); hivs.append(est["hi_v"]); trimfrac.append(1 - est["keep"].mean())
    secs = (time.perf_counter() - t0) / REPS
    lo_b, hi_b = np.mean(los), np.mean(his)
    lo_v, hi_v = np.mean(lovs), np.mean(hivs)
    inb = (bgrid >= lo_b) & (bgrid <= hi_b)
    inv = (vgrid >= lo_v) & (vgrid <= hi_v)
    xi_mean, f_mean = XI.mean(0), FH.mean(0)
    xi_lo, xi_hi = np.percentile(XI, [5, 95], axis=0)
    f_lo, f_hi = np.percentile(FH, [5, 95], axis=0)
    xt, ft = xi_true(bgrid), f_true(vgrid)
    cover_xi = np.mean((xt[inb] >= xi_lo[inb]) & (xt[inb] <= xi_hi[inb]))
    cover_f = np.mean((ft[inv] >= f_lo[inv]) & (ft[inv] <= f_hi[inv]))
    bias_xi = np.max(np.abs(xi_mean[inb] - xt[inb]))
    bias_f = np.max(np.abs(f_mean[inv] - ft[inv]))
    sup_f = np.mean(np.max(np.abs(FH[:, inv] - ft[inv]), axis=1))
    sup_xi = np.mean(np.max(np.abs(XI[:, inb] - xt[inb]), axis=1))
    results[c] = dict(XI=XI, FH=FH, xi_mean=xi_mean, f_mean=f_mean, xi_lo=xi_lo, xi_hi=xi_hi,
                      f_lo=f_lo, f_hi=f_hi, lo_b=lo_b, hi_b=hi_b, lo_v=lo_v, hi_v=hi_v)
    lines.append("")
    lines.append(f"--- bandwidth constant c = {c} (seed {seed}) ---")
    lines.append(f"mean h_g = {np.mean(hgs):.4f}, mean h_f = {np.mean(hfs):.4f}, "
                 f"mean share of bids trimmed = {np.mean(trimfrac):.4f}")
    lines.append(f"mean trimming bounds on bids: [{lo_b:.4f}, {hi_b:.4f}]; "
                 f"inner value region (GPV fn 9): [{lo_v:.4f}, {hi_v:.4f}]")
    lines.append(f"max |mean xi_hat - xi| on trimmed bid range: {bias_xi:.4f}")
    lines.append(f"max |mean f_hat - f| on inner value region: {bias_f:.4f}")
    lines.append(f"share of grid points with truth inside 90% band: xi {cover_xi:.3f}, f {cover_f:.3f}")
    lines.append(f"mean sup-norm error: xi {sup_xi:.4f}, f {sup_f:.4f}")
    lines.append(f"seconds per replication (Python/numpy): {secs:.4f}")

# ----------------------------------------------------------------- sample-size experiment
lines.append("")
lines.append("--- sample-size experiment (c = 1, 200 reps each, seed 15402) ---")
lines.append("fixed inner sets: bids [0.25, 1.35], values [0.40, 1.80]")
lines.append("T, mean sup|xi_hat - xi|, mean sup|f_hat - f| (two-step), "
             "mean sup|f_tilde - f| (true values, same kernel and rule)")
rng = np.random.default_rng(SEED_SIZE)
b_in = np.linspace(0.25, 1.35, 300)
v_in = np.linspace(0.40, 1.80, 300)
size_rows = []
for T in (100, 200, 400, 800, 1600):
    e_xi, e_f, e_ft = [], [], []
    for r in range(200):
        vals = draw_values(rng, T * N_BID)
        bids = beta(vals)
        est = gpv(bids, N_BID, c=1.0, bgrid=b_in, vgrid=v_in)
        e_xi.append(np.max(np.abs(est["xi"] - xi_true(b_in))))
        e_f.append(np.max(np.abs(est["f"] - f_true(v_in))))
        hf0 = 1.06 * vals.std(ddof=1) * vals.size ** (-0.2)
        ft_ = kde(vals, v_in, hf0, vals.size)
        e_ft.append(np.max(np.abs(ft_ - f_true(v_in))))
    row = (T, np.mean(e_xi), np.mean(e_f), np.mean(e_ft))
    size_rows.append(row)
    lines.append(f"{T:5d}  {row[1]:.4f}  {row[2]:.4f}  {row[3]:.4f}  ratio f/f_tilde = {row[2]/row[3]:.2f}")
Ts = np.log([r[0] for r in size_rows])
for k, name in ((1, "xi"), (2, "f two-step"), (3, "f infeasible")):
    slope = np.polyfit(Ts, np.log([r[k] for r in size_rows]), 1)[0]
    lines.append(f"log-log slope of mean sup error in T, {name}: {slope:.3f}")

# ----------------------------------------------------------------- single sample for trimming figure
rng = np.random.default_rng(SEED_ONE)
vals1 = draw_values(rng, T_MAIN * N_BID)
bids1 = beta(vals1)
bfine = np.linspace(B_LOW, B_HIGH, 600)
one = gpv(bids1, N_BID, c=1.0, bgrid=bfine, vgrid=vgrid, trim=True)
g_at_top = one["g_grid"][-1] / g_bid_true(B_HIGH)
lines.append("")
lines.append("--- single sample (seed 15403) ---")
lines.append(f"h_g = {one['hg']:.4f}; B_min = {bids1.min():.4f}, B_max = {bids1.max():.4f}")
lines.append(f"g_hat(b_high)/g(b_high) = {g_at_top:.3f}")
top = bids1 > one["hi"]
lines.append(f"bids above B_max - h_g: {top.sum()}; mean (V_hat - V) among them: "
             f"{np.mean(one['V'][top] - vals1[top]):.4f}; max: {np.max(one['V'][top] - vals1[top]):.4f}")
kept = one["keep"]
lines.append(f"kept bids: mean |V_hat - V| = {np.mean(np.abs(one['V'][kept] - vals1[kept])):.4f}")

OUT_TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ----------------------------------------------------------------- figure: Monte Carlo
R = results[1.0]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
ax = axes[0]
ax.fill_between(bgrid, R["xi_lo"], R["xi_hi"], color=BLUE, alpha=0.18, linewidth=0)
ax.plot(bgrid, xi_true(bgrid), color=GRAY, lw=2.0)
ax.plot(bgrid, R["xi_mean"], color=BLUE, lw=1.4, ls="--")
for x in (R["lo_b"], R["hi_b"]):
    ax.axvline(x, color=GRAY, lw=0.7, ls=":")
ax.set_xlim(B_LOW, B_HIGH)
ax.set_ylim(0, 4.2)
ax.set_xlabel(r"bid $b$")
ax.set_ylabel(r"value $\xi(b)$")
ax.annotate(r"true $\xi=\beta^{-1}$", xy=(1.0, xi_true(1.0)), xytext=(0.25, 2.6), color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
k = np.argmin(np.abs(bgrid - 1.45))
ax.annotate(r"mean of $\hat\xi$", xy=(bgrid[k], R["xi_mean"][k]), xytext=(0.95, 3.7), color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
kb = np.argmin(np.abs(bgrid - 1.57))
ax.annotate("90% band", xy=(bgrid[kb], 0.5 * (R["xi_lo"][kb] + R["xi_hi"][kb])), xytext=(1.0, 0.6),
            color=BLUE, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
ax.set_title("(a) Inverse bid function", fontsize=10)

ax = axes[1]
ax.fill_between(vgrid, R["f_lo"], R["f_hi"], color=BLUE, alpha=0.18, linewidth=0)
ax.plot(vgrid, f_true(vgrid), color=GRAY, lw=2.0)
ax.plot(vgrid, R["f_mean"], color=BLUE, lw=1.4, ls="--")
for x in (R["lo_v"], R["hi_v"]):
    ax.axvline(x, color=GRAY, lw=0.7, ls=":")
ax.set_xlim(0, VH)
ax.set_ylim(0, 1.15)
ax.set_xlabel(r"value $v$")
ax.set_ylabel(r"density $f(v)$")
ax.annotate(r"true $f$", xy=(1.3, f_true(1.3)), xytext=(1.55, 0.75), color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
k = np.argmin(np.abs(vgrid - 0.8))
ax.annotate(r"mean of $\hat f$", xy=(vgrid[k], R["f_mean"][k]), xytext=(1.0, 1.02), color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
ax.set_title("(b) Density of values", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_MC)
fig.savefig(HERE / "gpv_mc_preview.png", dpi=130)

# ----------------------------------------------------------------- figure: trimming
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
ax = axes[0]
ax.axvspan(B_LOW, one["lo"], color=ORANGE, alpha=0.12, linewidth=0)
ax.axvspan(one["hi"], B_HIGH, color=ORANGE, alpha=0.12, linewidth=0)
ax.plot(bfine, g_bid_true(bfine), color=GRAY, lw=2.0)
ax.plot(bfine, one["g_grid"], color=BLUE, lw=1.4, ls="--")
ax.set_xlim(B_LOW, B_HIGH)
ax.set_ylim(0, 1.45)
ax.set_xlabel(r"bid $b$")
ax.set_ylabel("bid density")
k = np.argmin(np.abs(bfine - 1.0))
ax.annotate(r"true $g$", xy=(bfine[k], g_bid_true(bfine[k])), xytext=(1.05, 1.15), color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
k = np.argmin(np.abs(bfine - 0.62))
ax.annotate(r"kernel estimate $\hat g$", xy=(bfine[k], one["g_grid"][k]), xytext=(0.35, 0.25),
            color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
ax.text(B_HIGH - 0.01, 1.40, "trimmed", color=ORANGE, fontsize=8.5, ha="right", va="top")
ax.text(B_LOW + 0.01, 1.40, "trimmed", color=ORANGE, fontsize=8.5, ha="left", va="top")
ax.set_title(r"(a) Boundary bias of $\hat g$", fontsize=10)

ax = axes[1]
ax.axvspan(B_LOW, one["lo"], color=ORANGE, alpha=0.12, linewidth=0)
ax.axvspan(one["hi"], B_HIGH, color=ORANGE, alpha=0.12, linewidth=0)
ax.scatter(bids1[kept], one["V"][kept], s=4, color=BLUE, alpha=0.5, linewidths=0)
ax.scatter(bids1[~kept], one["V"][~kept], s=9, color=ORANGE, alpha=0.9, linewidths=0)
ax.plot(bfine, xi_true(bfine), color=GRAY, lw=1.6)
ax.plot(bfine, one["xi"], color=BLUE, lw=1.0, ls="--")
ax.set_xlim(B_LOW, B_HIGH)
ax.set_ylim(0, 4.0)
ax.set_xlabel(r"bid $b$")
ax.set_ylabel(r"pseudo-value $\hat V$")
ax.annotate(r"true $\xi(b)$", xy=(0.9, xi_true(0.9)), xytext=(0.2, 2.4), color=GRAY,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
k = np.argmin(np.abs(bfine - 1.6))
ax.annotate(r"untrimmed $\hat\xi(b)$", xy=(bfine[k], min(one["xi"][k], 3.9)), xytext=(0.7, 3.5),
            color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
ax.text(0.95, 0.35, "orange dots: trimmed bids", color=ORANGE, fontsize=8.5)
ax.set_title("(b) Pseudo-values before trimming", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_TRIM)
fig.savefig(HERE / "gpv_trim_preview.png", dpi=130)
print("saved", OUT_MC, OUT_TRIM)
