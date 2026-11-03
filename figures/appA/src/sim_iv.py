"""
Simulations for Appendix A, Part 1 (identification, least squares, instrumental variables).

Produces (paths relative to Textbook/):
    figures/appA/simultaneity.pdf     fig:A-simultaneity
    figures/appA/weakiv.pdf           fig:A-weakiv
    figures/appA/src/sim_iv_numbers.json   every number quoted in the text
    figures/appA/src/*_preview.png    previews for checking

Design 1 (simultaneous equations, Section A.1.2 and A.3):
    demand  q = a_d - b p + u_d,          u_d ~ N(0, s_d^2)
    supply  q = a_s + c p + delta w + u_s, u_s ~ N(0, s_s^2), w ~ N(0, 1)
    with a_d = 10, b = 1, a_s = 2, c = 1, delta = 1, s_d = 0.6, s_s = 0.5.
    The cost shifter w is excluded from demand and serves as the instrument.
    Panel (a): one sample of n = 200 equilibria. Panel (b): 2,000 replications.

Design 2 (weak instruments, Section A.3.5):
    y = beta x + e,  x = Z pi + v,  (e, v) bivariate normal, unit variances, corr rho,
    beta = 0, n = 500, L = 3 instruments, Z drawn once (fixed design) and pi scaled so that
    the concentration parameter mu^2 = pi'Z'Z pi / sigma_v^2 takes the target value exactly.
    Panel (a): densities of 2SLS for mu^2 in {1, 10, 100}, rho = 0.5.
    Panel (b): rejection rates of nominal 5% Wald (t) and Anderson-Rubin tests of beta = 0.

Design 3 (forbidden regression, Section A.3.3): computed on one sample of 10^6 draws.
    z ~ N(0,1), eta, eps ~ N(0,1), v = s(z) eta, x = z + v, e = 0.8 eta + 0.6 eps,
    y = 1 + x + 0.5 x^2 + e. Heteroskedastic first stage s(z)^2 = 0.25 + z^2;
    homoskedastic comparison s(z)^2 = 1.25 (same mean variance).
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
OUT = {"seed": SEED}

# ------------------------------------------------------------------ style (book palette)
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7b5cd6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral"})


def ols(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b


def tsls(y, X, Z):
    """2SLS coefficients: (X'PzX)^{-1} X'Pz y."""
    Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return np.linalg.lstsq(Xh, y, rcond=None)[0]


# ================================================================== Design 1
A_D, B, A_S, C, DELTA, S_D, S_S = 10.0, 1.0, 2.0, 1.0, 1.0, 0.6, 0.5
N1, R1 = 200, 2000


def draw_market(n):
    w = rng.standard_normal(n)
    ud = S_D * rng.standard_normal(n)
    us = S_S * rng.standard_normal(n)
    p = (A_D - A_S - DELTA * w + ud - us) / (B + C)
    q = A_D - B * p + ud
    return p, q, w


sig_sup2 = DELTA ** 2 + S_S ** 2          # variance of the composite supply shock
plim_ols = (C * S_D ** 2 - B * sig_sup2) / (S_D ** 2 + sig_sup2)
OUT["design1"] = dict(a_d=A_D, b=B, a_s=A_S, c=C, delta=DELTA, s_d=S_D, s_s=S_S, n=N1, R=R1,
                      plim_ols=plim_ols, var_supply_shock=sig_sup2)

p1, q1, w1 = draw_market(N1)
one = np.ones(N1)
b_ols1 = ols(q1, np.column_stack([one, p1]))
b_iv1 = tsls(q1, np.column_stack([one, p1]), np.column_stack([one, w1]))
OUT["design1"]["sample_ols"] = b_ols1.tolist()
OUT["design1"]["sample_iv"] = b_iv1.tolist()

ols_draws, iv_draws, fs_F = np.empty(R1), np.empty(R1), np.empty(R1)
for r in range(R1):
    p, q, w = draw_market(N1)
    X = np.column_stack([one, p])
    Z = np.column_stack([one, w])
    ols_draws[r] = ols(q, X)[1]
    iv_draws[r] = tsls(q, X, Z)[1]
    g = ols(p, Z)
    res = p - Z @ g
    s2 = res @ res / (N1 - 2)
    wc = w - w.mean()
    fs_F[r] = g[1] ** 2 / (s2 / (wc @ wc))
OUT["design1"]["mc"] = dict(ols_mean=float(ols_draws.mean()), ols_sd=float(ols_draws.std(ddof=1)),
                            iv_mean=float(iv_draws.mean()), iv_sd=float(iv_draws.std(ddof=1)),
                            iv_median=float(np.median(iv_draws)), fsF_mean=float(fs_F.mean()))

# figure
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
ax = axes[0]
ax.scatter(p1, q1, s=7, color=BLUE, alpha=0.45, linewidths=0)
pg = np.linspace(p1.min() - 0.3, p1.max() + 0.3, 50)
ax.plot(pg, A_D - B * pg, color=INK, lw=1.4, label="Demand")
ax.plot(pg, A_S + C * pg, color=INK, lw=1.0, ls=(0, (4, 2)), label="Supply ($w=0$)")
ax.plot(pg, b_ols1[0] + b_ols1[1] * pg, color=ORANGE, lw=1.6, label="OLS fit")
ax.plot(pg, b_iv1[0] + b_iv1[1] * pg, color=AQUA, lw=1.6, ls=(0, (1.2, 1.2)), label="IV fit")
ax.set_xlabel("Price $p$")
ax.set_ylabel("Quantity $q$")
ax.set_title("(a) One sample of 200 equilibria", loc="left")
ax.set_ylim(1.4, 8.6)
ax.legend(frameon=False, loc="lower right", ncol=2, handlelength=2.0, columnspacing=1.0)
ax.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)

ax = axes[1]
grid = np.linspace(-1.8, 0.2, 400)
k_ols = stats.gaussian_kde(ols_draws)(grid)
k_iv = stats.gaussian_kde(iv_draws)(grid)
ax.fill_between(grid, k_ols, color=ORANGE, alpha=0.18, lw=0)
ax.plot(grid, k_ols, color=ORANGE, lw=1.5)
ax.fill_between(grid, k_iv, color=AQUA, alpha=0.18, lw=0)
ax.plot(grid, k_iv, color=AQUA, lw=1.5)
ax.axvline(-B, color=INK, lw=1.0)
ax.axvline(plim_ols, color=INK2, lw=0.8, ls=(0, (3, 2)))
ymax = max(k_ols.max(), k_iv.max())
ax.text(-B - 0.03, 1.02 * ymax, "true slope $-b$", ha="right", va="bottom", fontsize=7, color=INK)
ax.text(plim_ols + 0.03, 1.02 * ymax, "OLS limit", ha="left", va="bottom", fontsize=7, color=INK2)
ax.text(np.median(iv_draws), 0.55 * k_iv.max(), "2SLS", ha="center", fontsize=7.5, color=INK,
        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
ax.text(np.median(ols_draws), 0.55 * k_ols.max(), "OLS", ha="center", fontsize=7.5, color=INK,
        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
ax.set_ylim(0, 1.15 * ymax)
ax.set_xlabel("Estimated slope of $q$ on $p$")
ax.set_ylabel("Density")
ax.set_title("(b) Sampling distributions, 2,000 samples", loc="left")
ax.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIGDIR, "simultaneity.pdf"))
fig.savefig(os.path.join(HERE, "simultaneity_preview.png"), dpi=150)
plt.close(fig)

# ================================================================== Design 2
N2, L2, R2 = 500, 3, 10000
Zfix = rng.standard_normal((N2, L2))
Qz, _ = np.linalg.qr(Zfix)            # orthonormal basis of col(Z)
ZtZ = Zfix.T @ Zfix
fcrit_ar = stats.f.ppf(0.95, L2, N2 - L2)
zcrit = stats.norm.ppf(0.975)


def pi_for(mu2):
    d = np.ones(L2)
    scale = np.sqrt(mu2 / (d @ ZtZ @ d))
    return scale * d


def simulate_weak(mu2, rho, R=R2, beta=0.0):
    pi = pi_for(mu2)
    zp = Zfix @ pi
    e = rng.standard_normal((N2, R))
    v = rho * e + np.sqrt(1 - rho ** 2) * rng.standard_normal((N2, R))
    x = zp[:, None] + v
    y = beta * x + e
    Qx, Qy = Qz.T @ x, Qz.T @ y
    xPx = (Qx * Qx).sum(0)
    xPy = (Qx * Qy).sum(0)
    xx = (x * x).sum(0)
    xy = (x * y).sum(0)
    yy = (y * y).sum(0)
    b2 = xPy / xPx
    bols = xy / xx
    ssr = yy - 2 * b2 * xy + b2 ** 2 * xx
    se = np.sqrt(ssr / (N2 - 1) / xPx)
    wald_rej = np.abs((b2 - beta) / se) > zcrit
    y0 = y - beta * x
    Qy0 = Qz.T @ y0
    yPy0 = (Qy0 * Qy0).sum(0)
    yy0 = (y0 * y0).sum(0)
    ar = (yPy0 / L2) / ((yy0 - yPy0) / (N2 - L2))
    ar_rej = ar > fcrit_ar
    fs = (xPx / L2) / ((xx - xPx) / (N2 - L2))
    return dict(b2=b2, bols=bols, wald=wald_rej.mean(), ar=ar_rej.mean(), F=fs.mean())


RHO_A = 0.5
dens = {}
OUT["design2"] = dict(n=N2, L=L2, R=R2, rho_panel_a=RHO_A, ar_fcrit=float(fcrit_ar))
for mu2 in (1.0, 10.0, 100.0):
    s = simulate_weak(mu2, RHO_A)
    dens[mu2] = s["b2"]
    approx_bias = RHO_A * L2 / (mu2 + L2)          # ratio-of-expectations approximation
    nagar_bias = RHO_A * (L2 - 2) / mu2             # Nagar (1959) second-order approximation
    OUT["design2"]["mu2_%g" % mu2] = dict(
        mean_2sls=float(s["b2"].mean()), median_2sls=float(np.median(s["b2"])),
        approx_bias=float(approx_bias), nagar_bias=float(nagar_bias),
        mean_ols=float(s["bols"].mean()), mean_F=float(s["F"]),
        theory_EF=1 + mu2 / L2, wald=float(s["wald"]), ar=float(s["ar"]))

grid_ratio = np.array([0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 50, 100])
rej = {"wald_05": [], "wald_95": [], "ar_95": [], "F_95": []}
for g in grid_ratio:
    s5 = simulate_weak(g * L2, 0.5)
    s9 = simulate_weak(g * L2, 0.95)
    rej["wald_05"].append(float(s5["wald"]))
    rej["wald_95"].append(float(s9["wald"]))
    rej["ar_95"].append(float(s9["ar"]))
    rej["F_95"].append(float(s9["F"]))
OUT["design2"]["grid_mu2_over_L"] = grid_ratio.tolist()
OUT["design2"]["rejection"] = rej
OUT["design2"]["mc_se_at_5pct"] = float(np.sqrt(0.05 * 0.95 / R2))

fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
ax = axes[0]
xg = np.linspace(-1.0, 1.2, 500)
cols = {1.0: ORANGE, 10.0: PURPLE, 100.0: BLUE}
styles = {1.0: "-", 10.0: (0, (4, 1.5)), 100.0: (0, (1.2, 1.2))}
for mu2 in (1.0, 10.0, 100.0):
    d = dens[mu2]
    d = d[np.abs(d) < 5]
    k = stats.gaussian_kde(d)(xg)
    ax.plot(xg, k, color=cols[mu2], lw=1.6, ls=styles[mu2], label=r"$\mu^2 = %g$" % mu2)
ax.axvline(0.0, color=INK, lw=1.0)
ax.axvline(RHO_A, color=INK2, lw=0.8, ls=(0, (3, 2)))
ax.text(-0.03, ax.get_ylim()[1] * 0.97, r"$\beta = 0$", ha="right", va="top", fontsize=7.5)
ax.text(RHO_A + 0.02, ax.get_ylim()[1] * 0.97, r"OLS limit, $\pi = 0$", ha="left", va="top",
        fontsize=7, color=INK2)
ax.set_xlabel(r"2SLS estimate $\hat\beta$")
ax.set_ylabel("Density")
ax.set_title(r"(a) 2SLS with $L = 3$, $\rho = 0.5$", loc="left")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.55))
ax.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)

ax = axes[1]
ax.plot(grid_ratio, rej["wald_95"], color=ORANGE, lw=1.6, marker="o", ms=3.5,
        label=r"Wald, $\rho = 0.95$")
ax.plot(grid_ratio, rej["wald_05"], color=PURPLE, lw=1.6, marker="s", ms=3.2, ls=(0, (4, 1.5)),
        label=r"Wald, $\rho = 0.5$")
ax.plot(grid_ratio, rej["ar_95"], color=AQUA, lw=1.6, marker="^", ms=3.5, ls=(0, (1.2, 1.2)),
        label=r"Anderson-Rubin, $\rho = 0.95$")
ax.axhline(0.05, color=INK, lw=0.8)
ax.set_xscale("log")
ax.set_xticks([0.1, 1, 10, 100])
ax.set_xticklabels(["0.1", "1", "10", "100"])
ax.set_xlabel(r"$\frac{\mu^2}{L}$ (log scale)")
ax.set_ylabel("Rejection rate of a true null")
ax.set_title("(b) Size of nominal 5% tests", loc="left")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0, max(rej["wald_95"]) * 1.12)
ax.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIGDIR, "weakiv.pdf"))
fig.savefig(os.path.join(HERE, "weakiv_preview.png"), dpi=150)
plt.close(fig)

# ================================================================== Design 3
N3 = 1_000_000
z = rng.standard_normal(N3)
eta = rng.standard_normal(N3)
eps = rng.standard_normal(N3)
e3 = 0.8 * eta + 0.6 * eps
OUT["design3"] = {"n": N3}
for name, s2 in (("hetero", 0.25 + z ** 2), ("homo", np.full(N3, 1.25))):
    v = np.sqrt(s2) * eta
    x = z + v
    y = 1 + x + 0.5 * x ** 2 + e3
    one3 = np.ones(N3)
    Z1 = np.column_stack([one3, z])
    g = ols(x, Z1)
    xh = Z1 @ g
    vh = x - xh
    forb = ols(y, np.column_stack([one3, xh, xh ** 2]))
    iv = tsls(y, np.column_stack([one3, x, x ** 2]), np.column_stack([one3, z, z ** 2]))
    olsq = ols(y, np.column_stack([one3, x, x ** 2]))
    cf = ols(y, np.column_stack([one3, x, x ** 2, vh]))
    OUT["design3"][name] = dict(forbidden=forb.tolist(), iv_z_z2=iv.tolist(), ols=olsq.tolist(),
                                control_function=cf.tolist())

with open(os.path.join(HERE, "sim_iv_numbers.json"), "w") as fh:
    json.dump(OUT, fh, indent=2)
print(json.dumps(OUT, indent=2))
