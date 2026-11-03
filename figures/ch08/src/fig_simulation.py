"""Figure fig:08-simerror and tables tab:08-simdelta, tab:08-integration, tab:08-importance
(Chapter 8, section 8.4).

One market with five inside products and an outside good. Random coefficients on the constant
(sd 1.0), on a characteristic x (sd 1.5) and on price (sd 0.5), nu ~ N(0, I_3), mean price
coefficient alpha = 2 (so alpha_i = 2 - 0.5 nu_ip). Observed shares
s = (0.10, 0.05, 0.03, 0.015, 0.005), s_0 = 0.80. The mean utilities delta are obtained by the
contraction under a 40-point-per-dimension Gauss-Hermite product rule (64,000 nodes); these delta
define the true model. True shares are checked against 10^6 pseudo-random draws.

1. Sampling distribution of the smooth simulator s^ns_j and of ln s^ns_j for n_s in {20, 100, 1000}
   (2,000 replications each, fresh pseudo-random draws), and the mean bias of ln s^ns_j over a grid of
   n_s (20,000 replications per n_s) against the second-order approximation
   -Var(s_ij) / (2 n_s s_j^2).
2. Error in the inverted mean utility: for each replication we solve s^ns(delta) = s by the
   contraction and record delta_hat - delta (500 replications per n_s).
3. Integration rules: relative RMSE of s_j for pseudo-Monte Carlo, scrambled Halton draws and the
   Gauss-Hermite product rule with 125, 1000 and 8000 nodes.
4. Importance sampling (BLP 1995, eq. 6.13): accept a draw with probability
   f_bar(nu, theta') = sum_j s_ij(theta') and weight it by s_bar(theta') / f_bar(nu, theta').
   Relative RMSE for n_s = 100 accepted draws, with theta' = theta and with a preliminary estimate
   that understates every sigma by 20 percent.

Output: figures/ch08/fig_simulation.pdf, src/simulation_numbers.txt
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rclogit_p1 import (individual_shares, shares, contraction, gauss_hermite_nodes,
                        halton_normal)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "fig_simulation.pdf"
PNG = HERE / "fig_simulation_preview.png"
TXT = HERE / "simulation_numbers.txt"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

lines = []


def out(s=""):
    print(s)
    lines.append(s)


rng = np.random.default_rng(20261022)

# ----------------------------------------------------------------------------- market
J = 5
x = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
p = np.array([2.0, 1.6, 1.2, 1.0, 0.8])
s_obs = np.array([0.10, 0.05, 0.03, 0.015, 0.005])
xr = np.column_stack([np.ones(J), x, p])
sig = np.array([1.0, 1.5, 0.5])
K = 3
nodes_q, w_q = gauss_hermite_nodes(K, 40)
delta, it = contraction(s_obs, xr, sig, nodes_q, w_q)
s_true, s0_true = shares(delta, xr, sig, nodes_q, w_q)
sij_q, _ = individual_shares(delta, xr, sig, nodes_q)
var_sij = w_q @ (sij_q ** 2) - s_true ** 2
cv = np.sqrt(var_sij) / s_true
out("Market: shares " + ", ".join(f"{v:.3f}" for v in s_obs) + f"; outside {1 - s_obs.sum():.3f}")
out(f"  delta = {np.round(delta, 4).tolist()} (contraction {it} iterations)")
big = rng.standard_normal((1_000_000, K))
s_mc = np.zeros(J)
for chunk in np.array_split(big, 20):
    s_mc += shares(delta, xr, sig, chunk)[0] * chunk.shape[0]
s_mc /= big.shape[0]
out("  shares from 10^6 pseudo-random draws: " + ", ".join(f"{v:.5f}" for v in s_mc))
out("  sd(s_ij)/s_j (coefficient of variation of the integrand): "
    + ", ".join(f"{v:.3f}" for v in cv))

# ----------------------------------------------------------------------------- 1. distributions
NS_SHOW = [20, 100, 1000]
R_SHOW = 2000
R_INV = 500
draws_log = {}
out()
out(f"1. Smooth simulator, {R_SHOW} replications")
for ns in NS_SHOW:
    sims = np.array([shares(delta, xr, sig, rng.standard_normal((ns, K)))[0]
                     for _ in range(R_SHOW)])
    lg = np.log(sims) - np.log(s_true)
    draws_log[ns] = lg
    out(f"  n_s = {ns:5d}: mean(s^ns/s - 1) = "
        + ", ".join(f"{v:+.4f}" for v in (sims / s_true - 1).mean(axis=0))
        + " | sd(s^ns/s) = " + ", ".join(f"{v:.3f}" for v in (sims / s_true).std(axis=0)))
    out(f"               mean(ln s^ns - ln s) = " + ", ".join(f"{v:+.4f}" for v in lg.mean(axis=0))
        + " | approx -CV^2/(2 n_s) = " + ", ".join(f"{-c**2/(2*ns):+.4f}" for c in cv))

NS_GRID = [10, 20, 50, 100, 200, 500, 1000]
R_GRID = 20000
bias_grid = np.zeros((len(NS_GRID), J))
bias_se = np.zeros((len(NS_GRID), J))
out()
out(f"   Mean bias of ln s^ns over n_s grid ({R_GRID} replications)")
for a, ns in enumerate(NS_GRID):
    acc = []
    for _ in range(R_GRID // 500):
        nu = rng.standard_normal((500, ns, K))
        mu = np.einsum("rnk,jk->rnj", nu, xr * sig)
        v = delta[None, None, :] + mu
        m = np.maximum(0.0, v.max(axis=2, keepdims=True))
        ev = np.exp(v - m)
        den = np.exp(-m) + ev.sum(axis=2, keepdims=True)
        sim = (ev / den).mean(axis=1)
        acc.append(np.log(sim) - np.log(s_true))
    acc = np.vstack(acc)
    bias_grid[a] = acc.mean(axis=0)
    bias_se[a] = acc.std(axis=0) / np.sqrt(acc.shape[0])
    out(f"  n_s = {ns:5d}: bias " + ", ".join(f"{v:+.5f}" for v in bias_grid[a])
        + " | se " + ", ".join(f"{v:.5f}" for v in bias_se[a])
        + " | approx " + ", ".join(f"{-c**2/(2*ns):+.5f}" for c in cv))

# ----------------------------------------------------------------------------- 2. inverted delta
out()
out("2. Error in the inverted mean utility, delta_hat - delta (500 replications)")
for ns in NS_SHOW:
    errs = []
    for _ in range(R_INV):
        nu = rng.standard_normal((ns, K))
        dh, _ = contraction(s_true, xr, sig, nu, tol=1e-12)
        errs.append(dh - delta)
    errs = np.array(errs)
    out(f"  n_s = {ns:5d}: mean " + ", ".join(f"{v:+.4f}" for v in errs.mean(axis=0))
        + " | sd " + ", ".join(f"{v:.4f}" for v in errs.std(axis=0)))

# ----------------------------------------------------------------------------- 3. integration rules
out()
out("3. Integration rules: relative RMSE of s_j in percent (200 replications for random rules)")
R_INT = 200
for q in (5, 10, 20):
    n = q ** K
    nq, wq = gauss_hermite_nodes(K, q)
    s_gh = shares(delta, xr, sig, nq, wq)[0]
    e_gh = 100 * np.abs(s_gh / s_true - 1)
    e_mc = np.zeros((R_INT, J))
    e_ha = np.zeros((R_INT, J))
    for r in range(R_INT):
        e_mc[r] = shares(delta, xr, sig, rng.standard_normal((n, K)))[0] / s_true - 1
        e_ha[r] = shares(delta, xr, sig, halton_normal(n, K, seed=1000 + r))[0] / s_true - 1
    rm_mc = 100 * np.sqrt((e_mc ** 2).mean(axis=0))
    rm_ha = 100 * np.sqrt((e_ha ** 2).mean(axis=0))
    out(f"  n = {n:5d}: pseudo-MC " + ", ".join(f"{v:.3f}" for v in rm_mc))
    out(f"            Halton    " + ", ".join(f"{v:.3f}" for v in rm_ha))
    out(f"            GH (q={q:2d}) " + ", ".join(f"{v:.1e}" for v in e_gh))

# ----------------------------------------------------------------------------- 4. importance sampling
out()
out("4. Importance sampling, n_s = 100 accepted draws, relative RMSE of s_j in percent "
    "(1000 replications)")
NS_IS = 100
R_IS = 1000


def is_estimator(delta_p, sig_p, sbar_p, ns):
    """BLP (6.13) with the factor 1/ns: accept nu with prob f_bar(nu, theta')."""
    acc = []
    need = ns
    while need > 0:
        nu = rng.standard_normal((4 * need * int(1 / sbar_p + 1), K))
        sij_p, _ = individual_shares(delta_p, xr, sig_p, nu)
        fbar = sij_p.sum(axis=1)
        keep = rng.uniform(size=nu.shape[0]) < fbar
        nu_k, fb_k = nu[keep][:need], fbar[keep][:need]
        acc.append((nu_k, fb_k))
        need -= nu_k.shape[0]
    nu_a = np.vstack([a[0] for a in acc])
    fb_a = np.concatenate([a[1] for a in acc])
    sij, _ = individual_shares(delta, xr, sig, nu_a)
    wts = sbar_p / fb_a
    return (wts[:, None] * sij).mean(axis=0)


rm_simple = np.zeros(J)
e_s = np.zeros((R_IS, J))
for r in range(R_IS):
    e_s[r] = shares(delta, xr, sig, rng.standard_normal((NS_IS, K)))[0] / s_true - 1
rm_simple = 100 * np.sqrt((e_s ** 2).mean(axis=0))
out("  simple smooth simulator        : " + ", ".join(f"{v:.2f}" for v in rm_simple))
for label, scale in (("theta' = theta", 1.0), ("sigma' = 0.8 sigma", 0.8)):
    sig_p = sig * scale
    delta_p, _ = contraction(s_true, xr, sig_p, nodes_q, w_q)
    sbar_p = shares(delta_p, xr, sig_p, nodes_q, w_q)[0].sum()
    e_i = np.zeros((R_IS, J))
    for r in range(R_IS):
        e_i[r] = is_estimator(delta_p, sig_p, sbar_p, NS_IS) / s_true - 1
    rm_is = 100 * np.sqrt((e_i ** 2).mean(axis=0))
    mean_is = 100 * e_i.mean(axis=0)
    out(f"  importance sampling, {label:18s}: " + ", ".join(f"{v:.2f}" for v in rm_is)
        + " | mean error (%) " + ", ".join(f"{v:+.2f}" for v in mean_is))
    out(f"    acceptance rate s_bar(theta') = {sbar_p:.4f}")

TXT.write_text("\n".join(lines) + "\n")

# ----------------------------------------------------------------------------- figure
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral",
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
})
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"wspace": 0.32})
ax = axes[0]
cols = {20: BLUE, 100: ORANGE, 1000: AQUA}
jsmall = J - 1
bins = np.linspace(-2.0, 1.5, 50)
for ns in NS_SHOW:
    d = draws_log[ns][:, jsmall]
    ax.hist(d, bins=bins, density=True, histtype="step", lw=1.6, color=cols[ns],
            label=rf"$n_s = {ns}$")
    ax.axvline(d.mean(), color=cols[ns], lw=1.0, ls=(0, (3, 2)))
ax.axvline(0.0, color=INK, lw=0.8)
ax.set_xlabel(r"$\ln s_j^{n_s} - \ln s_j$ for the product with $s_j = 0.005$")
ax.set_ylabel("Density")
ax.set_xlim(-2.0, 1.5)
ax.legend(frameon=False, loc="upper left")
ax.set_title("(a) Sampling distribution of the log share", loc="left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.5)

ax = axes[1]
ns_arr = np.array(NS_GRID, float)
fine = np.geomspace(8, 1300, 100)
# Labels sit off the lines: the s_j = 0.10 label ends left of x = 150 below its (falling) line,
# the s_j = 0.005 label starts at x = 200 above its line.
for j, colr, lab, xpos, ypos, ha, va in ((0, BLUE, r"$s_j = 0.10$", 150, 0.5, "right", "top"),
                                         (jsmall, ORANGE, r"$s_j = 0.005$", 200, 1.5, "left", "bottom")):
    ax.plot(ns_arr, -bias_grid[:, j], "o", ms=4.5, color=colr, mec="white", mew=0.6, zorder=3)
    ax.plot(fine, cv[j] ** 2 / (2 * fine), color=colr, lw=1.3)
    ax.text(xpos, ypos * cv[j] ** 2 / (2 * xpos), lab, color=INK, fontsize=8, ha=ha, va=va)
ax.plot([], [], "o", ms=4.5, color=INK2, mec="white", label="Monte Carlo mean (20,000 replications)")
ax.plot([], [], color=INK2, lw=1.3, label="Second-order approximation")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Number of draws $n_s$")
ax.set_ylabel(r"Downward bias $\ln s_j - \mathrm{E}[\ln s_j^{n_s}]$")
ax.set_title("(b) Bias of the log share", loc="left")
ax.legend(frameon=False, loc="upper right", fontsize=7.2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, which="major", color=GRID, lw=0.5)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(PNG, dpi=150, bbox_inches="tight")
print("saved", OUT)
