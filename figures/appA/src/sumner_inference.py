"""
Inference for the cigarette tax pass-through regression under dependence: Appendix A, Part 3
(Sections A.8.5 and A.9).

Data: ProblemSet_1/data_sumner.csv, a reconstructed state-year panel (47 states plus DC,
1954-1978, 1,200 observations): weighted-average retail price per pack (cents), combined state
plus federal excise tax (cents per pack), national CPI-U. Prices and taxes come from The Tax
Burden on Tobacco, historical compilation (1983), Table 13; CPI from BLS.

Produces (paths relative to Textbook/):
    figures/appA/delta_boot.pdf                 fig:A-delta-boot
    figures/appA/src/sumner_inference.json      every number quoted in the text and in tab:A-sumner
    figures/appA/src/delta_boot_preview.png     preview for checking

Specifications (rows of tab:A-sumner):
    (1) p_sy = mu_s + tau_y + rho t_sy + e_sy, OLS (two-way fixed effects)
    (2) p_sy = a + rho t_sy + c CPI_y + e_sy, pooled OLS
    (3) same regressors as (2), two-way random effects (variance components) GLS, Swamy-Arora type
        variance estimates, as in Fuller and Battese (1974).
Standard errors for rho: conventional; HC1; CR1 by state (48 clusters); CR1 by year (25);
two-way clustered (Cameron, Gelbach and Miller 2011); Driscoll-Kraay with Bartlett weights and
2 lags. For row (3) the robust variants are applied to the GLS-transformed regression.
Wild cluster bootstrap-t by state (null rho = 1 imposed, Rademacher weights, B = 9,999).
Pairs cluster bootstrap by state for row (1), B = 9,999: rho*, its CR1 s.e., and the implied
elasticity eps = rho / (rho - 1). Jackknife over states for the BCa acceleration.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))       # ECON_257
DATA = os.path.join(ROOT, "ProblemSet_1", "data_sumner.csv")
SEED = 20261022
B_WILD = 9999
B_PAIRS = 9999
DK_LAGS = 2

d = pd.read_csv(DATA).sort_values(["state_abbr", "year"]).reset_index(drop=True)
states = d.state_abbr.unique()
years = np.sort(d.year.unique())
S, Y = len(states), len(years)
n = len(d)
assert n == S * Y == 1200
sid = pd.factorize(d.state_abbr)[0]
yid = pd.factorize(d.year)[0]
p = d.price_cents.to_numpy(float)
t = d.tax_cents_combined.to_numpy(float)
cpi = d.cpi_u_annual.to_numpy(float)


def within(v):
    """Two-way within transformation for the balanced panel (rows sorted by state, then year)."""
    M = v.reshape(S, Y)
    return (M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()).ravel()


def variances(h, e, K, s_ids=sid, y_ids=yid):
    """Variance estimates of the coefficient b = h @ y with residuals e and K regressors."""
    u = h * e
    s2 = e @ e / (n - K)
    out = {"conv": s2 * (h @ h), "hc1": n / (n - K) * np.sum(u ** 2)}
    us = np.bincount(s_ids, u)
    uy = np.bincount(y_ids, u)
    Cs, Cy = len(us), len(uy)
    out["state"] = Cs / (Cs - 1) * (n - 1) / (n - K) * np.sum(us ** 2)
    out["year"] = Cy / (Cy - 1) * (n - 1) / (n - K) * np.sum(uy ** 2)
    out["twoway"] = out["state"] + out["year"] - out["hc1"]
    # Driscoll-Kraay: Newey-West on the year-level sums of scores
    U = uy
    dk = np.sum(U ** 2)
    for j in range(1, DK_LAGS + 1):
        dk += 2 * (1 - j / (DK_LAGS + 1)) * np.sum(U[j:] * U[:-j])
    out["dk"] = dk
    return out


def pvals(b, V, null=1.0):
    """Two-sided p-values for H0: coefficient = null, with reference distributions by variant."""
    df = {"conv": None, "hc1": None, "state": S - 1, "year": Y - 1, "twoway": min(S, Y) - 1, "dk": Y - 1}
    out = {}
    for k, v in V.items():
        tstat = (b - null) / np.sqrt(v)
        out[k] = float(2 * stats.norm.sf(abs(tstat))) if df[k] is None else float(2 * stats.t.sf(abs(tstat), df[k]))
    return out


res = {"seed": SEED, "n": n, "states": S, "years": Y, "B_wild": B_WILD, "B_pairs": B_PAIRS, "dk_lags": DK_LAGS}

# ------------------------------------------------------------------ row 1: two-way FE OLS
xw, yw = within(t), within(p)
h1 = xw / (xw @ xw)
rho1 = h1 @ yw
e1 = yw - rho1 * xw
K1 = 1 + 1 + (S - 1) + (Y - 1)
V1 = variances(h1, e1, K1)
res["row1"] = {"rho": rho1, "se": {k: float(np.sqrt(v)) for k, v in V1.items()}, "p": pvals(rho1, V1), "K": K1}

# ------------------------------------------------------------------ row 2: pooled OLS with CPI
X2 = np.column_stack([np.ones(n), t, cpi])
H2 = np.linalg.solve(X2.T @ X2, X2.T)
b2 = H2 @ p
e2 = p - X2 @ b2
V2 = variances(H2[1], e2, 3)
res["row2"] = {"rho": b2[1], "se": {k: float(np.sqrt(v)) for k, v in V2.items()}, "p": pvals(b2[1], V2), "K": 3}


# ------------------------------------------------------------------ row 3: two-way RE GLS with CPI
def two_way_re():
    def m(v, ids):
        return (np.bincount(ids, v) / np.bincount(ids))[ids]

    bw = (xw @ yw) / (xw @ xw)
    ew = yw - bw * xw
    s2e = (ew @ ew) / ((S - 1) * (Y - 1) - 1)
    ps, ts = np.bincount(sid, p) / Y, np.bincount(sid, t) / Y
    xs = np.column_stack([np.ones(S), ts])
    es = ps - xs @ np.linalg.lstsq(xs, ps, rcond=None)[0]
    s21 = Y * (es @ es) / (S - 2)
    py, ty, cy = np.bincount(yid, p) / S, np.bincount(yid, t) / S, np.bincount(yid, cpi) / S
    xt = np.column_stack([np.ones(Y), ty, cy])
    et = py - xt @ np.linalg.lstsq(xt, py, rcond=None)[0]
    s22 = S * (et @ et) / (Y - 3)
    s2a, s2b = max((s21 - s2e) / Y, 0.0), max((s22 - s2e) / S, 0.0)
    l1 = 1 - np.sqrt(s2e / (s2e + Y * s2a))
    l2 = 1 - np.sqrt(s2e / (s2e + S * s2b))
    l3 = l1 + l2 - 1 + np.sqrt(s2e / (s2e + Y * s2a + S * s2b))

    def tr(v):
        return v - l1 * m(v, sid) - l2 * m(v, yid) + l3 * v.mean()

    Xs = np.column_stack([tr(c) for c in (np.ones(n), t, cpi)])
    ys = tr(p)
    H = np.linalg.solve(Xs.T @ Xs, Xs.T)
    b = H @ ys
    e = ys - Xs @ b
    V = variances(H[1], e, 3)
    V["conv"] = s2e * np.linalg.inv(Xs.T @ Xs)[1, 1]        # GLS variance with the estimated s2e
    return b[1], V, dict(s2e=s2e, s2a=s2a, s2b=s2b, l1=l1, l2=l2, l3=l3)


rho3, V3, vc = two_way_re()
res["row3"] = {"rho": rho3, "se": {k: float(np.sqrt(v)) for k, v in V3.items()}, "p": pvals(rho3, V3),
               "vc": vc}

# ------------------------------------------------------------------ dependence diagnostics
E1 = e1.reshape(S, Y)
XW = xw.reshape(S, Y)
res["ar1_resid_within"] = float(np.sum(E1[:, 1:] * E1[:, :-1]) / np.sum(E1[:, :-1] ** 2))
res["ar1_tax_within"] = float(np.sum(XW[:, 1:] * XW[:, :-1]) / np.sum(XW[:, :-1] ** 2))
E2 = e2.reshape(S, Y)
res["ar1_resid_pooled"] = float(np.sum(E2[:, 1:] * E2[:, :-1]) / np.sum(E2[:, :-1] ** 2))
res["ratio_state_to_conv_row1"] = res["row1"]["se"]["state"] / res["row1"]["se"]["conv"]

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------ wild cluster bootstrap by state
def wild_row1(B, ids, nclus):
    yr = within(p - t)                      # restricted residuals (null rho = 1)
    yfit = p - yr
    t0 = (rho1 - 1) / np.sqrt(V1["state" if ids is sid else "year"])
    count, done = 0, 0
    kcr = nclus / (nclus - 1) * (n - 1) / (n - K1)
    while done < B:
        b = min(1000, B - done)
        W = rng.choice([-1.0, 1.0], size=(nclus, b))[ids]
        Ys = yfit[:, None] + yr[:, None] * W
        Yw = (Ys.reshape(S, Y, b) - Ys.reshape(S, Y, b).mean(1, keepdims=True)
              - Ys.reshape(S, Y, b).mean(0, keepdims=True) + Ys.reshape(S, Y, b).mean((0, 1), keepdims=True)).reshape(n, b)
        rs = h1 @ Yw
        es = Yw - xw[:, None] * rs
        us = h1[:, None] * es
        cs = np.zeros((nclus, b))
        np.add.at(cs, ids, us)
        se = np.sqrt(kcr * np.sum(cs ** 2, axis=0))
        count += np.sum(np.abs((rs - 1) / se) >= abs(t0))
        done += b
    return float(count / B)


def wild_row2(B, ids, nclus):
    Xr = np.column_stack([np.ones(n), cpi])
    yr_full = p - t
    er = yr_full - Xr @ np.linalg.lstsq(Xr, yr_full, rcond=None)[0]
    yfit = p - er
    h = H2[1]
    t0 = (b2[1] - 1) / np.sqrt(V2["state" if ids is sid else "year"])
    kcr = nclus / (nclus - 1) * (n - 1) / (n - 3)
    count, done = 0, 0
    while done < B:
        b = min(1000, B - done)
        W = rng.choice([-1.0, 1.0], size=(nclus, b))[ids]
        Ys = yfit[:, None] + er[:, None] * W
        Bs = H2 @ Ys
        es = Ys - X2 @ Bs
        cs = np.zeros((nclus, b))
        np.add.at(cs, ids, h[:, None] * es)
        se = np.sqrt(kcr * np.sum(cs ** 2, axis=0))
        count += np.sum(np.abs((Bs[1] - 1) / se) >= abs(t0))
        done += b
    return float(count / B)


res["row1"]["wild_state"] = wild_row1(B_WILD, sid, S)
res["row1"]["wild_year"] = wild_row1(B_WILD, yid, Y)
res["row2"]["wild_state"] = wild_row2(B_WILD, sid, S)
res["row2"]["wild_year"] = wild_row2(B_WILD, yid, Y)

# ------------------------------------------------------------------ pairs cluster bootstrap by state (row 1)
P = p.reshape(S, Y)
Tm = t.reshape(S, Y)
rho_star, se_star = np.empty(B_PAIRS), np.empty(B_PAIRS)
kcr1 = S / (S - 1) * (n - 1) / (n - K1)
done = 0
while done < B_PAIRS:
    b = min(1000, B_PAIRS - done)
    idx = rng.integers(0, S, size=(b, S))
    Pb, Tb = P[idx], Tm[idx]                                      # (b, S, Y)

    def wb(A):
        return A - A.mean(2, keepdims=True) - A.mean(1, keepdims=True) + A.mean((1, 2), keepdims=True)

    xb, yb = wb(Tb), wb(Pb)
    sxx = np.sum(xb ** 2, axis=(1, 2))
    rb = np.sum(xb * yb, axis=(1, 2)) / sxx
    eb = yb - rb[:, None, None] * xb
    cs = np.sum(xb * eb, axis=2) / sxx[:, None]                    # (b, S) cluster sums of h*e
    rho_star[done:done + b] = rb
    se_star[done:done + b] = np.sqrt(kcr1 * np.sum(cs ** 2, axis=1))
    done += b

se_cl = res["row1"]["se"]["state"]
tcrit = stats.t.ppf(0.975, S - 1)
eps_hat = rho1 / (rho1 - 1)
eps_star = rho_star / (rho_star - 1)
q = lambda a, pr: float(np.quantile(a, pr))
boot = {
    "se_boot_rho": float(rho_star.std(ddof=1)),
    "mean_rho_star": float(rho_star.mean()),
    "share_rho_star_le_1": float(np.mean(rho_star <= 1)),
    "pct_rho": [q(rho_star, 0.025), q(rho_star, 0.975)],
    "basic_rho": [2 * rho1 - q(rho_star, 0.975), 2 * rho1 - q(rho_star, 0.025)],
}
tstar = (rho_star - rho1) / se_star
boot["pct_t_rho"] = [rho1 - q(tstar, 0.975) * se_cl, rho1 - q(tstar, 0.025) * se_cl]
boot["tstar_quantiles"] = [q(tstar, 0.025), q(tstar, 0.975)]
# BCa: bias correction and jackknife acceleration over states
z0 = stats.norm.ppf(np.mean(rho_star < rho1))
jack = np.empty(S)
for s in range(S):
    keep = np.arange(S) != s
    A, Bm = Tm[keep], P[keep]
    xa = A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    ya = Bm - Bm.mean(1, keepdims=True) - Bm.mean(0, keepdims=True) + Bm.mean()
    jack[s] = np.sum(xa * ya) / np.sum(xa ** 2)
dj = jack.mean() - jack
acc = np.sum(dj ** 3) / (6 * np.sum(dj ** 2) ** 1.5)
za = stats.norm.ppf([0.025, 0.975])
alphas = stats.norm.cdf(z0 + (z0 + za) / (1 - acc * (z0 + za)))
boot["bca_rho"] = [q(rho_star, alphas[0]), q(rho_star, alphas[1])]
boot["bca_z0"], boot["bca_acc"], boot["bca_alphas"] = float(z0), float(acc), alphas.tolist()
boot["jackknife_se_rho"] = float(np.sqrt((S - 1) / S * np.sum(dj ** 2)))
# implied elasticity
se_d_conv = res["row1"]["se"]["conv"] / (rho1 - 1) ** 2
se_d_cl = se_cl / (rho1 - 1) ** 2
ci_rho_cl = [rho1 - tcrit * se_cl, rho1 + tcrit * se_cl]
ci_rho_conv = [rho1 - 1.96 * res["row1"]["se"]["conv"], rho1 + 1.96 * res["row1"]["se"]["conv"]]
g = lambda r: r / (r - 1)
eps = {
    "eps_hat": eps_hat, "sumner_style": 1 / (1 - rho1),
    "se_delta_conv": se_d_conv, "se_delta_cluster": se_d_cl,
    "delta_ci_conv": [eps_hat - 1.96 * se_d_conv, eps_hat + 1.96 * se_d_conv],
    "delta_ci_cluster": [eps_hat - tcrit * se_d_cl, eps_hat + tcrit * se_d_cl],
    "ci_rho_cluster": ci_rho_cl, "ci_rho_conv": ci_rho_conv,
    "mapped_ci_conv": [g(ci_rho_conv[1]), g(ci_rho_conv[0])],
    "mapped_ci_cluster": [g(ci_rho_cl[1]), g(ci_rho_cl[0])],
    "pct_ci_eps": [q(eps_star, 0.025), q(eps_star, 0.975)],
    "pct_ci_eps_mapped_from_rho": [g(boot["pct_rho"][1]), g(boot["pct_rho"][0])],
    "basic_ci_eps": [2 * eps_hat - q(eps_star, 0.975), 2 * eps_hat - q(eps_star, 0.025)],
    "bca_ci_eps_mapped": [g(boot["bca_rho"][1]), g(boot["bca_rho"][0])],
    "pct_t_ci_eps_mapped": [g(boot["pct_t_rho"][1]), g(boot["pct_t_rho"][0])],
    "basic_ci_eps_mapped": [g(boot["basic_rho"][1]), g(boot["basic_rho"][0])],
    "mcse_wild_p_row1": float(np.sqrt(res["row1"]["wild_state"] * (1 - res["row1"]["wild_state"]) / B_WILD)),
    "median_eps_star": q(eps_star, 0.5),
    "share_eps_star_gt_60": float(np.mean((eps_star > 60) | (eps_star < 0))),
    "sd_eps_star": float(eps_star.std(ddof=1)),
    "iqr_eps_star": [q(eps_star, 0.25), q(eps_star, 0.75)],
}
res["boot"] = boot
res["eps"] = eps
res["tcrit_47"] = float(tcrit)
# t-statistic for Sumner's published numbers
res["sumner_t"] = (1.074 - 1) / 0.013
res["sumner_delta_se"] = 0.013 / (0.074 ** 2)
res["sumner_eps"] = 1.074 / 0.074
# Dafny (2009) worked example: exp(beta) - 1 and its delta-method s.e.
res["dafny"] = {"beta": 0.376, "se": 0.132, "pct": float(np.exp(0.376) - 1),
                "se_pct": float(np.exp(0.376) * 0.132),
                "ci_mapped": [float(np.exp(0.376 - 1.96 * 0.132) - 1), float(np.exp(0.376 + 1.96 * 0.132) - 1)]}
# bootstrap of the maximum: probability that the maximum is redrawn
res["boot_max"] = {str(m): float(1 - (1 - 1 / m) ** m) for m in (10, 100, 1000)}


def to_float(o):
    if isinstance(o, dict):
        return {k: to_float(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [to_float(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


res = to_float(res)
with open(os.path.join(HERE, "sumner_inference.json"), "w") as fh:
    json.dump(res, fh, indent=1)

for r in ("row1", "row2", "row3"):
    print(r, round(res[r]["rho"], 4), {k: round(v, 4) for k, v in res[r]["se"].items()})
    print("   p", {k: round(v, 4) for k, v in res[r]["p"].items()},
          "wild_state", res[r].get("wild_state"), "wild_year", res[r].get("wild_year"))
print("diag", res["ar1_resid_within"], res["ar1_tax_within"], res["ar1_resid_pooled"])
print("boot", json.dumps(res["boot"], indent=0))
print("eps", json.dumps(res["eps"], indent=0))
print("dafny", res["dafny"], "bootmax", res["boot_max"])

# ------------------------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7b5cd6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral"})
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.75))
ax = axes[0]
bins = np.linspace(0.95, 1.20, 60)
ax.hist(rho_star, bins=bins, density=True, color=BLUE, alpha=0.35, edgecolor="white", lw=0.3,
        label="state bootstrap")
gr = np.linspace(0.95, 1.20, 400)
ax.plot(gr, stats.norm.pdf(gr, rho1, se_cl), color=INK, lw=1.2, label="normal, clustered s.e.")
ax.axvline(1.0, color=ORANGE, lw=1.0, ls="--")
ax.annotate(r"$\rho = 1$", (1.0, ax.get_ylim()[1] * 0.92), xytext=(3, 0), textcoords="offset points",
            color=INK, fontsize=7.5)
ax.set_xlabel(r"pass-through $\hat\rho^{*}$")
ax.set_ylabel("density")
ax.set_title(r"(a) Pass-through", loc="left")
ax.legend(frameon=False, loc="upper right")

ax = axes[1]
lo, hi = 0, 60
inside = eps_star[(eps_star > lo) & (eps_star < hi)]
bins = np.linspace(lo, hi, 61)
ax.hist(inside, bins=bins, density=False, weights=np.full(inside.size, 1.0 / (B_PAIRS * (bins[1] - bins[0]))),
        color=BLUE, alpha=0.35, edgecolor="white", lw=0.3, label="state bootstrap")
ge = np.linspace(lo, hi, 600)
ax.plot(ge, stats.norm.pdf(ge, eps_hat, se_d_cl), color=INK, lw=1.2, label="delta method")
ymax = ax.get_ylim()[1]
lines = [(eps["delta_ci_cluster"], INK2, "-", "delta interval"),
         (eps["pct_ci_eps"], ORANGE, "--", "percentile interval"),
         (eps["mapped_ci_cluster"], PURPLE, ":", "endpoint-mapped interval")]
for k, (ci, col, ls, lab) in enumerate(lines):
    yy = ymax * (1.02 + 0.07 * k)
    a, bnd = max(ci[0], lo), min(ci[1], hi)
    ax.plot([a, bnd], [yy, yy], color=col, lw=1.6, ls=ls, label=lab)
    for xv in ci:
        if lo <= xv <= hi:
            ax.plot([xv, xv], [yy - ymax * 0.02, yy + ymax * 0.02], color=col, lw=1.2)
    if ci[1] > hi:
        ax.annotate("", xy=(hi + 1.5, yy), xytext=(hi - 0.5, yy),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0))
ax.set_ylim(0, ymax * 1.3)
ax.set_xlim(lo, hi + 2)
ax.set_xlabel(r"implied elasticity $\frac{\hat\rho^{*}}{\hat\rho^{*} - 1}$")
ax.set_ylabel("density")
ax.set_title("(b) Implied elasticity", loc="left")
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.45))
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "delta_boot.pdf"))
fig.savefig(os.path.join(HERE, "delta_boot_preview.png"), dpi=160)
