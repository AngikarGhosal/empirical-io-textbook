"""What happens when GPV is applied to bids with unobserved auction heterogeneity (Section 15.7.1).

Design (our simulation):
    symmetric IPV first-price auction, n = 4 bidders, T = 500 auctions per sample;
    values V = omega * A with A ~ U[1, 2] and ln(omega) ~ N(0, 0.3^2), omega common knowledge
    among bidders and unobserved by the econometrician.
    Equilibrium bid at omega (multiplicative separability):  beta(v; omega) = omega * beta0(v / omega),
    beta0(a) = a - int_1^a (t-1)^3 dt / (a-1)^3 = (3a + 1) / 4, so B = omega (3A + 1) / 4.
    True inverse bid function at omega = 1: v = (4b - 1) / 3 on [1, 7/4].

The econometrician has no covariates, pools all bids, and applies GPV:
    G-hat = empirical CDF, g-hat = triweight kernel density,
    h = 2.978 * 1.06 * sd(b) * (nT)^(-1/5), trimming within h of the extreme bids,
    pseudo-value  v-hat = b + G-hat(b) / ((n - 1) g-hat(b)),
    value density by a triweight kernel on the pseudo-values with the same rule of thumb, using the
    robust scale min(sd, IQR / 1.349) because pseudo-values have a long upper tail.
We report the median and the 5-95 percent band across R = 200 samples.

Outputs: figures/ch15/uh_bias.pdf and figures/ch15/src/uh_bias_numbers.txt
Seed: 15701.
"""
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_FIG = HERE.parent / "uh_bias.pdf"
OUT_TXT = HERE / "uh_bias_numbers.txt"

SEED = 15701
N_BID = 4
T = 500
SIG = 0.3
R = 200

rng = np.random.default_rng(SEED)


def triweight(u):
    return np.where(np.abs(u) <= 1.0, 35.0 / 32.0 * (1.0 - u**2) ** 3, 0.0)


def kde(x_eval, data, h):
    u = (x_eval[:, None] - data[None, :]) / h
    return triweight(u).mean(axis=1) / h


def simulate():
    omega = np.exp(SIG * rng.standard_normal(T))
    A = rng.uniform(1.0, 2.0, size=(T, N_BID))
    V = omega[:, None] * A
    B = omega[:, None] * (3.0 * A + 1.0) / 4.0
    return V.ravel(), B.ravel()


b_grid = np.linspace(0.75, 2.6, 300)
v_grid = np.linspace(0.4, 3.6, 300)
# wide grid for moments: covers the full support of V and the upper tail of the GPV density
v_mom = np.linspace(0.0, 10.0, 2001)

xi_hat = np.full((R, b_grid.size), np.nan)
f_hat = np.zeros((R, v_grid.size))
f_hat_mom = np.zeros((R, v_mom.size))
iqr_hat = np.zeros(R)
iqr_true_sample = np.zeros(R)
markdown_hat = np.zeros(R)
markdown_true_sample = np.zeros(R)

for r in range(R):
    V, B = simulate()
    nb = B.size
    h = 2.978 * 1.06 * B.std(ddof=1) * nb ** (-0.2)
    Bs = np.sort(B)
    G_at_B = np.searchsorted(Bs, B, side="right") / nb
    g_at_B = kde(B, B, h)
    keep = (B >= Bs[0] + h) & (B <= Bs[-1] - h)
    vhat = B[keep] + G_at_B[keep] / ((N_BID - 1) * g_at_B[keep])
    markdown_hat[r] = np.mean((vhat - B[keep]) / vhat)
    markdown_true_sample[r] = np.mean((V - B) / V)
    # inverse bid function on the grid (inside the trimmed range)
    inside = (b_grid >= Bs[0] + h) & (b_grid <= Bs[-1] - h)
    Gg = np.searchsorted(Bs, b_grid[inside], side="right") / nb
    gg = kde(b_grid[inside], B, h)
    xi_hat[r, inside] = b_grid[inside] + Gg / ((N_BID - 1) * gg)
    # value density from pseudo-values (GPV second step, same bandwidth rule)
    # robust scale: pseudo-values have a long upper tail where g-hat is small
    q75, q25 = np.percentile(vhat, [75, 25])
    scale = min(vhat.std(ddof=1), (q75 - q25) / 1.349)
    hf = 2.978 * 1.06 * scale * vhat.size ** (-0.2)
    f_hat[r] = kde(v_grid, vhat, hf)
    f_hat_mom[r] = kde(v_mom, vhat, hf)
    iqr_hat[r] = q75 - q25
    iqr_true_sample[r] = np.subtract(*np.percentile(V, [75, 25]))

# true marginal density of V = omega * A (integrate over omega numerically)
z = np.linspace(-5, 5, 4001)
w = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
om = np.exp(SIG * z)


def f_true_on(grid):
    dens = ((grid[:, None] >= om[None, :]) & (grid[:, None] <= 2 * om[None, :])) / om[None, :]
    return np.trapezoid(dens * w[None, :], z, axis=1)


f_true = f_true_on(v_grid)
f_true_mom = f_true_on(v_mom)

true_markdown = (1.0 - np.log(2.0)) / 4.0

xi_med = np.nanmedian(xi_hat, axis=0)
xi_lo = np.nanpercentile(xi_hat, 5, axis=0)
xi_hi = np.nanpercentile(xi_hat, 95, axis=0)
coverage = np.mean(np.isfinite(xi_hat), axis=0)
ok = coverage >= 0.9

f_med = np.median(f_hat, axis=0)
f_lo = np.percentile(f_hat, 5, axis=0)
f_hi = np.percentile(f_hat, 95, axis=0)

# slope of the estimated inverse bid function over the central bid range
cent = ok & (b_grid >= 1.1) & (b_grid <= 1.6)
slope_hat = np.polyfit(b_grid[cent], xi_med[cent], 1)[0]

# variance of values: true marginal variance vs variance implied by pseudo-values,
# both computed on the wide grid v_mom so that no tail mass is lost
EV_exact = 1.5 * np.exp(SIG**2 / 2)
VarV_exact = (7.0 / 3.0) * np.exp(2 * SIG**2) - EV_exact**2
mass_true = np.trapezoid(f_true_mom, v_mom)
EV = np.trapezoid(v_mom * f_true_mom, v_mom)
VarV = np.trapezoid((v_mom - EV) ** 2 * f_true_mom, v_mom)
f_med_mom = np.median(f_hat_mom, axis=0)
mass_hat = np.trapezoid(f_med_mom, v_mom)
EVh = np.trapezoid(v_mom * f_med_mom, v_mom) / mass_hat
VarVh = np.trapezoid((v_mom - EVh) ** 2 * f_med_mom, v_mom) / mass_hat
# per-replication variance of the GPV density, then the median across replications
var_rep = np.empty(R)
for r in range(R):
    m_r = np.trapezoid(f_hat_mom[r], v_mom)
    e_r = np.trapezoid(v_mom * f_hat_mom[r], v_mom) / m_r
    var_rep[r] = np.trapezoid((v_mom - e_r) ** 2 * f_hat_mom[r], v_mom) / m_r
# the same moments over the plotted range only (truncated, unnormalized), for comparison
EV_plot = np.trapezoid(v_grid * f_true, v_grid)
VarV_plot = np.trapezoid((v_grid - EV_plot) ** 2 * f_true, v_grid)
EVh_plot = np.trapezoid(v_grid * f_med, v_grid)
VarVh_plot = np.trapezoid((v_grid - EVh_plot) ** 2 * f_med, v_grid)

lines = [
    f"seed = {SEED}, n = {N_BID}, T = {T}, sigma_lnomega = {SIG}, replications = {R}",
    f"true average markdown E[(V-B)/V] = (1 - ln 2)/4 = {true_markdown:.4f}",
    f"sample average true markdown (mean over replications) = {markdown_true_sample.mean():.4f}",
    f"GPV pooled estimate of average markdown: median = {np.median(markdown_hat):.4f},"
    f" 5th pct = {np.percentile(markdown_hat, 5):.4f}, 95th pct = {np.percentile(markdown_hat, 95):.4f}",
    f"slope of true inverse bid function at omega = 1: 4/3 = {4/3:.4f}",
    f"slope of median GPV pooled inverse bid function on b in [1.1, 1.6]: {slope_hat:.4f}",
    f"exact mean and variance of V: 1.5 exp(0.045) = {EV_exact:.4f}, (7/3) exp(0.18) - mean^2 = {VarV_exact:.4f}",
    f"true marginal mean and variance of V on [0, 10] (mass {mass_true:.4f}): {EV:.4f}, {VarV:.4f}",
    f"mean and variance from median GPV density on [0, 10] (mass {mass_hat:.4f}): {EVh:.4f}, {VarVh:.4f}",
    f"variance of GPV density per replication: median = {np.median(var_rep):.4f},"
    f" 5th pct = {np.percentile(var_rep, 5):.4f}, 95th pct = {np.percentile(var_rep, 95):.4f}",
    f"variance of V and of the median GPV density over the plotted range [0.4, 3.6]:"
    f" {VarV_plot:.4f}, {VarVh_plot:.4f}",
    f"interquartile range of pseudo-values: median = {np.median(iqr_hat):.4f},"
    f" 5th pct = {np.percentile(iqr_hat, 5):.4f}, 95th pct = {np.percentile(iqr_hat, 95):.4f}",
    f"interquartile range of true values V (median over samples) = {np.median(iqr_true_sample):.4f};"
    f" interquartile range of A = 0.5",
]
OUT_TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ------------------------------------------------------------------ figure
C_TRUE = "#1f3b73"
C_EST = "#c4512b"
C_ALT = "#7a7a7a"

# The figure is drawn at its printed size (included at width=\textwidth = 6.2 in), so the font sizes
# below are the sizes on the page.
FINAL_W, FINAL_H = 6.2, 3.0
FS_TICK, FS_LABEL, FS_TITLE, FS_NOTE = 8.0, 8.5, 9.0, 8.0
plt.rcParams.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.labelsize": FS_LABEL})

fig, axes = plt.subplots(1, 2, figsize=(FINAL_W, FINAL_H))

ax = axes[0]
bb = np.linspace(1.0, 1.75, 50)
ax.plot(bb, (4 * bb - 1) / 3, color=C_TRUE, lw=1.4, label=r"true, $\omega=1$")
for om_, ls in ((np.exp(-SIG), ":"), (np.exp(SIG), "--")):
    bb2 = np.linspace(om_, 1.75 * om_, 50)
    ax.plot(bb2, (4 * bb2 - om_) / 3, color=C_TRUE, lw=0.9, ls=ls)
ax.fill_between(b_grid[ok], xi_lo[ok], xi_hi[ok], color=C_EST, alpha=0.18, lw=0)
ax.plot(b_grid[ok], xi_med[ok], color=C_EST, lw=1.4, label="GPV on pooled bids")
ax.text(0.74, 0.66, r"true, $\omega=e^{-0.3}$", color=C_TRUE, fontsize=FS_NOTE)
ax.text(1.95, 1.80, r"true, $\omega=e^{0.3}$", color=C_TRUE, fontsize=FS_NOTE)
ax.set_xlabel(r"bid $b$")
ax.set_ylabel(r"implied value $\xi(b)$")
ax.set_title("(a) Inverse bid functions", fontsize=FS_TITLE)
ax.legend(frameon=False, fontsize=FS_NOTE, loc="upper left")
ax.set_xlim(0.7, 2.5)
ax.set_ylim(0.6, 3.6)

ax = axes[1]
ax.plot(v_grid, f_true, color=C_TRUE, lw=1.4, label=r"true density of $V=\omega A$")
ax.fill_between(v_grid, f_lo, f_hi, color=C_EST, alpha=0.18, lw=0)
ax.plot(v_grid, f_med, color=C_EST, lw=1.4, label="GPV on pooled bids")
ax.plot([1, 1, 2, 2], [0, 1, 1, 0], color=C_ALT, lw=0.9, ls="--", label=r"density of $A$ ($\omega=1$)")
ax.set_xlabel(r"value $v$")
ax.set_ylabel("density")
ax.set_title("(b) Value densities", fontsize=FS_TITLE)
ax.legend(frameon=False, fontsize=FS_NOTE, loc="upper right")
ax.set_xlim(0.4, 3.6)
ax.set_ylim(0, 1.45)

for a in axes:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_FIG)
print("saved", OUT_FIG)
