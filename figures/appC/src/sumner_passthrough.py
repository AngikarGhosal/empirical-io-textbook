"""
Appendix C, Problem C.2: what cigarette tax pass-through reveals about conduct.

Data: figures/appC/data/cigarette_panel.csv, a reconstructed state-year panel (47 states plus
the District of Columbia, 1954-1978, 1,200 observations). Columns: state_abbr, year,
price_cents (weighted-average retail price per pack, nominal cents), tax_cents_combined
(state plus federal excise tax, nominal cents per pack; federal tax 8 cents throughout),
cpi_u_annual (national CPI-U, 1982-84 = 100). Source of prices and taxes: The Tax Burden on
Tobacco, historical volume (1983), Table 13; CPI-U from the BLS.

Everything is computed with numpy (statsmodels is used only for an optional cross-check).
  M0  p_sy = mu_s + tau_y + rho t_sy + e_sy               (nominal; conventional and clustered s.e.)
  R2  p_sy = a + rho t_sy + g CPI_y + e_sy                 (Sumner's row 2)
  R3  same regressors, two-way random effects GLS          (Sumner's row 3)
  M1  M0 in real terms (deflated by the CPI-U)
  M2  P_sy = mu_s + tau_y + b1 (T - Tbar) + b2 (T - Tbar)^2 + e_sy
Cluster-robust variance by state with the small-sample factor G/(G-1) (N-1)/(N-K) and
t critical values with G - 1 = 47 degrees of freedom.

Outputs (relative to Textbook/):
  figures/appC/rho_T.pdf
  figures/appC/src/sumner_passthrough_results.json
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
DATA = os.path.join(FIGDIR, "data", "cigarette_panel.csv")

d = pd.read_csv(DATA).sort_values(["state_abbr", "year"]).reset_index(drop=True)
S, Y, N = d.state_abbr.nunique(), d.year.nunique(), len(d)
assert N == S * Y == 1200, "panel must be balanced"
sid = pd.factorize(d.state_abbr)[0]
yid = pd.factorize(d.year)[0]
p = d.price_cents.to_numpy(float)
t = d.tax_cents_combined.to_numpy(float)
cpi = d.cpi_u_annual.to_numpy(float)
P = 100.0 * p / cpi                    # real price, 1982-84 cents
T = 100.0 * t / cpi                    # real tax
share = t / p

G = S
TCRIT = stats.t.ppf(0.975, G - 1)
DS = np.eye(S)[sid]                    # state dummies (all S)
DY = np.eye(Y)[yid][:, 1:]             # year dummies, first year dropped


def ols(y, X, w=None):
    """OLS (or WLS with weights w). Returns coefficients, residuals and (X'WX)^{-1}."""
    if w is None:
        w = np.ones_like(y)
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    beta = XtX_inv @ Xw.T @ yw
    e = y - X @ beta
    return beta, e, XtX_inv, Xw, e * sw


def vcov(Xw, ew, XtX_inv, cluster=None):
    n, k = Xw.shape
    if cluster is None:
        s2 = (ew @ ew) / (n - k)
        return s2 * XtX_inv
    meat = np.zeros((k, k))
    for g in np.unique(cluster):
        sg = Xw[cluster == g].T @ ew[cluster == g]
        meat += np.outer(sg, sg)
    ng = len(np.unique(cluster))
    fac = ng / (ng - 1.0) * (n - 1.0) / (n - k)
    return fac * XtX_inv @ meat @ XtX_inv


def fe_regression(y, regs, cluster=True, w=None):
    """y on regs plus state and year dummies. Coefficients of regs come first."""
    X = np.column_stack([regs, DS, DY])
    beta, e, XtX_inv, Xw, ew = ols(y, X, w)
    Vc = vcov(Xw, ew, XtX_inv, sid if cluster else None)
    Vo = vcov(Xw, ew, XtX_inv, None)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - (e @ e) / tss
    k = regs.shape[1]
    return dict(b=beta[:k], Vc=Vc[:k, :k], Vo=Vo[:k, :k], e=e, r2=r2, K=X.shape[1])


res = {"n": N, "states": S, "years": Y, "tcrit_47": TCRIT}

# ------------------------------------------------------------------ summary statistics
def desc(v):
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()),
                max=float(v.max()))


res["stats"] = {"price": desc(p), "tax": desc(t), "real_price": desc(P), "real_tax": desc(T),
                "tax_share": desc(share)}
for yr in (1954, 1978):
    sel = d.year.to_numpy() == yr
    res[f"price_mean_{yr}"] = float(p[sel].mean())
    res[f"tax_mean_{yr}"] = float(t[sel].mean())
    res[f"share_mean_{yr}"] = float(share[sel].mean())
    res[f"share_of_means_{yr}"] = float(t[sel].mean() / p[sel].mean())
    res[f"statetax_median_{yr}"] = float(np.median(t[sel] - 8))
    res[f"real_price_mean_{yr}"] = float(P[sel].mean())
    res[f"real_tax_mean_{yr}"] = float(T[sel].mean())
    res[f"cpi_{yr}"] = float(cpi[sel][0])
dtax = d.groupby("state_abbr").tax_cents_combined.diff().dropna().to_numpy()
res["n_transitions"] = int(dtax.size)
res["n_tax_changes"] = int((dtax != 0).sum())
res["n_tax_decreases"] = int((dtax < 0).sum())
res["mean_tax_change"] = float(dtax[dtax != 0].mean())
res["states_never_change"] = int((d.groupby("state_abbr").tax_cents_combined.nunique() == 1).sum())

# ------------------------------------------------------------------ M0: Sumner row 1
m0 = fe_regression(p, t[:, None])
rho = m0["b"][0]
se_o, se_c = np.sqrt(m0["Vo"][0, 0]), np.sqrt(m0["Vc"][0, 0])
res["M0"] = dict(rho=rho, se_conv=se_o, se_cl=se_c, r2=m0["r2"], K=m0["K"],
                 t_conv=(rho - 1) / se_o, t_cl=(rho - 1) / se_c,
                 p_conv=2 * stats.t.sf(abs(rho - 1) / se_o, N - m0["K"]),
                 p_cl=2 * stats.t.sf(abs(rho - 1) / se_c, G - 1),
                 ci_lo=rho - TCRIT * se_c, ci_hi=rho + TCRIT * se_c,
                 ci_conv_lo=rho - stats.t.ppf(0.975, N - m0["K"]) * se_o,
                 ci_conv_hi=rho + stats.t.ppf(0.975, N - m0["K"]) * se_o)

# residual serial correlation within states (pooled AR(1) of the M0 residuals)
e0 = m0["e"].reshape(S, Y)
res["M0"]["resid_ar1"] = float(np.sum(e0[:, 1:] * e0[:, :-1]) / np.sum(e0[:, :-1] ** 2))
res["M0"]["se_ratio"] = se_c / se_o
# serial correlation of the two-way demeaned tax (the identifying variation) and the
# textbook AR(1) approximation to the variance inflation, (1 + r_x r_e) / (1 - r_x r_e)
tw = t - (np.bincount(sid, t) / Y)[sid] - (np.bincount(yid, t) / S)[yid] + t.mean()
tw2 = tw.reshape(S, Y)
r_x = float(np.sum(tw2[:, 1:] * tw2[:, :-1]) / np.sum(tw2[:, :-1] ** 2))
r_e = res["M0"]["resid_ar1"]
res["M0"]["tax_within_ar1"] = r_x
res["M0"]["approx_se_inflation"] = float(np.sqrt((1 + r_x * r_e) / (1 - r_x * r_e)))
res["M0"]["sd_within_tax"] = float(tw.std(ddof=0))
# power of the clustered test of rho = 1 (two-sided, 5 percent, t with G - 1 df)
T80 = stats.t.ppf(0.80, G - 1)
res["M0"]["mde80_cl"] = float((TCRIT + T80) * se_c)
res["M0"]["mde80_conv"] = float((TCRIT + T80) * se_o)
ncp = (rho - 1) / se_c
res["M0"]["power_at_estimate_cl"] = float(stats.t.sf(TCRIT - ncp, G - 1) + stats.t.cdf(-TCRIT - ncp, G - 1))
# markup in cents implied by the Lerner index 1 - 1/rho at mean prices
res["M0"]["markup_cents_mean_price"] = float((1 - 1 / rho) * p.mean())
res["M0"]["markup_cents_1978"] = float((1 - 1 / rho) * p[d.year.to_numpy() == 1978].mean())

# ------------------------------------------------------------------ Sumner rows 2 and 3
X2 = np.column_stack([np.ones(N), t, cpi])
b2_, e2_, XtX2, Xw2, ew2 = ols(p, X2)
V2 = vcov(Xw2, ew2, XtX2)
res["R2"] = dict(rho=b2_[1], se=np.sqrt(V2[1, 1]), r2=1 - (e2_ @ e2_) / np.sum((p - p.mean()) ** 2),
                 g_cpi=b2_[2])
res["R2"]["t_one"] = (res["R2"]["rho"] - 1) / res["R2"]["se"]
res["R2"]["p_one"] = 2 * stats.t.sf(abs(res["R2"]["t_one"]), N - 3)


def two_way_re():
    """Two-way random effects GLS (variance components) for the balanced panel.

    sigma_e^2 from the two-way within regression; state and year components from the
    between-state and between-year regressions (Swamy-Arora type). Regressors: 1, t, CPI.
    """
    def mean_by(v, ids):
        return (np.bincount(ids, v) / np.bincount(ids))[ids]

    def within(v):
        return v - mean_by(v, sid) - mean_by(v, yid) + v.mean()

    yw, xw = within(p), within(t)
    bw = (xw @ yw) / (xw @ xw)
    ew = yw - bw * xw
    s2e = (ew @ ew) / ((S - 1) * (Y - 1) - 1)
    # between states: state means of p on (1, state mean of t); CPI is constant across states
    ps, ts = np.bincount(sid, p) / Y, np.bincount(sid, t) / Y
    Xs = np.column_stack([np.ones(S), ts])
    es = ps - Xs @ np.linalg.lstsq(Xs, ps, rcond=None)[0]
    s21 = Y * (es @ es) / (S - 2)
    # between years: year means of p on (1, year mean of t, CPI)
    py, ty, cy = (np.bincount(yid, v) / S for v in (p, t, cpi))
    Xy = np.column_stack([np.ones(Y), ty, cy])
    ey = py - Xy @ np.linalg.lstsq(Xy, py, rcond=None)[0]
    s22 = S * (ey @ ey) / (Y - 3)
    s2a, s2b = max((s21 - s2e) / Y, 0.0), max((s22 - s2e) / S, 0.0)
    l1 = 1 - np.sqrt(s2e / (s2e + Y * s2a))
    l2 = 1 - np.sqrt(s2e / (s2e + S * s2b))
    l3 = l1 + l2 - 1 + np.sqrt(s2e / (s2e + Y * s2a + S * s2b))

    def tr(v):
        return v - l1 * mean_by(v, sid) - l2 * mean_by(v, yid) + l3 * v.mean()

    Xs_ = np.column_stack([tr(col) for col in X2.T])
    b = np.linalg.lstsq(Xs_, tr(p), rcond=None)[0]
    V = s2e * np.linalg.inv(Xs_.T @ Xs_)
    return dict(rho=b[1], se=np.sqrt(V[1, 1]), s2e=s2e, s2_state=s2a, s2_year=s2b)


res["R3"] = two_way_re()
res["R3"]["t_one"] = (res["R3"]["rho"] - 1) / res["R3"]["se"]
res["R3"]["p_one"] = 2 * stats.norm.sf(abs(res["R3"]["t_one"]))

# Sumner's Table 1 transformations, using his printed estimates
sumner_rows = [("row1", 1.074, 0.013), ("row2", 1.029, 0.011), ("row3", 1.069, 0.013)]
res["sumner_table1"] = {
    k: dict(theta=th, se=se, t_one=(th - 1) / se, inv=1 / (1 - th), exact=th / (1 - th),
            lerner=1 - 1 / th)
    for k, th, se in sumner_rows}

# ------------------------------------------------------------------ M1: real terms
m1 = fe_regression(P, T[:, None])
r1, s1 = m1["b"][0], np.sqrt(m1["Vc"][0, 0])
res["M1"] = dict(rho=r1, se_cl=s1, se_conv=np.sqrt(m1["Vo"][0, 0]),
                 p_cl=2 * stats.t.sf(abs(r1 - 1) / s1, G - 1),
                 ci_lo=r1 - TCRIT * s1, ci_hi=r1 + TCRIT * s1)

# decomposition of M0 versus M1: M1 equals WLS of the nominal equation with state effects
# proportional to the price level, mu_s * CPI_y / 100, and weights (100 / CPI_y)^2.
Xa = np.column_stack([t, DS * (cpi / 100.0)[:, None], DY])
ba, ea, XtXa, Xwa, ewa = ols(p, Xa)
Va = vcov(Xwa, ewa, XtXa, sid)
bw_, ew_, XtXw, Xww, eww = ols(p, Xa, w=(100.0 / cpi) ** 2)
res["decomp"] = dict(rho_scaled_fe_ols=ba[0], se_cl=np.sqrt(Va[0, 0]), rho_scaled_fe_wls=bw_[0])
res["decomp"]["p_cl"] = 2 * stats.t.sf(abs(ba[0] - 1) / np.sqrt(Va[0, 0]), G - 1)
res["decomp"]["ci_lo"] = ba[0] - TCRIT * np.sqrt(Va[0, 0])
res["decomp"]["ci_hi"] = ba[0] + TCRIT * np.sqrt(Va[0, 0])
res["decomp"]["weight_ratio_1954_1978"] = float((cpi.max() / cpi.min()) ** 2)
# conventional 95 percent intervals for Sumner's rows 2 and 3
res["R2"]["ci_lo"] = res["R2"]["rho"] - stats.t.ppf(0.975, N - 3) * res["R2"]["se"]
res["R2"]["ci_hi"] = res["R2"]["rho"] + stats.t.ppf(0.975, N - 3) * res["R2"]["se"]
res["R3"]["ci_lo"] = res["R3"]["rho"] - stats.norm.ppf(0.975) * res["R3"]["se"]
res["R3"]["ci_hi"] = res["R3"]["rho"] + stats.norm.ppf(0.975) * res["R3"]["se"]
assert abs(bw_[0] - r1) < 1e-8, "WLS identity with M1 failed"

# ------------------------------------------------------------------ M2: quadratic in real tax
Tbar = T.mean()
Tc = T - Tbar
m2 = fe_regression(P, np.column_stack([Tc, Tc ** 2]))
b1, b2 = m2["b"]
V = m2["Vc"]
res["M2"] = dict(b1=b1, b2=b2, se_b1=np.sqrt(V[0, 0]), se_b2=np.sqrt(V[1, 1]),
                 p_b2=2 * stats.t.sf(abs(b2) / np.sqrt(V[1, 1]), G - 1), Tbar=Tbar)
for q in (10, 50, 90):
    dev = np.percentile(Tc, q)
    g = np.array([1.0, 2 * dev])
    val, sev = b1 + 2 * b2 * dev, float(np.sqrt(g @ V @ g))
    res["M2"][f"T_p{q}"] = float(Tbar + dev)
    res["M2"][f"rho_p{q}"] = float(val)
    res["M2"][f"se_p{q}"] = sev
    res["M2"][f"p_one_p{q}"] = float(2 * stats.t.sf(abs(val - 1) / sev, G - 1))
    res["M2"][f"ci_lo_p{q}"] = float(val - TCRIT * sev)
    res["M2"][f"ci_hi_p{q}"] = float(val + TCRIT * sev)
spread = 2 * (np.percentile(Tc, 90) - np.percentile(Tc, 10))
diff, diff_se = spread * b2, spread * np.sqrt(V[1, 1])
res["M2"].update(diff=diff, diff_se=diff_se, diff_lo=diff - TCRIT * diff_se,
                 diff_hi=diff + TCRIT * diff_se, mde80_diff=float((TCRIT + T80) * diff_se),
                 p10_p90_gap=float(spread / 2))

# ------------------------------------------------------------------ identified set, bounds
kap, klo, khi = 1 - 1 / rho, 1 - 1 / res["M0"]["ci_lo"], 1 - 1 / res["M0"]["ci_hi"]
res["kappa"] = dict(kappa=kap, lo=klo, hi=khi, chi_monopoly=1 + kap, chi_cournot6=1 + 6 * kap,
                    theta_ce04=0.4 * kap, theta_ce04_lo=0.4 * klo, theta_ce04_hi=0.4 * khi,
                    theta_ce03=0.3 * kap, theta_ce08=0.8 * kap,
                    N_cournot_equiv_04=1 / (0.4 * kap), sumner_eta_exact=-rho / (rho - 1))
res["lerner_bound"] = dict(share_mean=share.mean(), share_min=share.min(), share_max=share.max(),
                           theta_max_mean=0.4 * (1 - share.mean()),
                           theta_max_minshare=0.4 * (1 - share.min()),
                           theta_max_maxshare=0.4 * (1 - share.max()),
                           theta_max_mean_eps03=0.3 * (1 - share.mean()),
                           theta_max_mean_eps08=0.8 * (1 - share.mean()),
                           eps_needed_monopoly_mean=1 / (1 - share.mean()),
                           eps_needed_monopoly_minshare=1 / (1 - share.min()))
# M1 conventional p-value
res["M1"]["p_conv"] = 2 * stats.t.sf(abs(r1 - 1) / res["M1"]["se_conv"], N - m1["K"])

# ------------------------------------------------------------------ optional cross-check
try:
    import statsmodels.formula.api as smf
    fit = smf.ols("price_cents ~ tax_cents_combined + C(state_abbr) + C(year)", d)
    r_o = fit.fit()
    r_c = fit.fit(cov_type="cluster", cov_kwds={"groups": sid}, use_t=True)
    assert abs(r_o.params["tax_cents_combined"] - rho) < 1e-9
    assert abs(r_o.bse["tax_cents_combined"] - se_o) < 1e-9
    assert abs(r_c.bse["tax_cents_combined"] - se_c) < 1e-9
    res["statsmodels_check"] = "passed"
except ImportError:
    res["statsmodels_check"] = "statsmodels not installed"


def tofloat(o):
    if isinstance(o, dict):
        return {k: tofloat(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


res = tofloat(res)
with open(os.path.join(HERE, "sumner_passthrough_results.json"), "w") as fh:
    json.dump(res, fh, indent=2)


def show(prefix, o):
    for k, v in o.items():
        if isinstance(v, dict):
            show(prefix + k + ".", v)
        else:
            print(f"{prefix + k:40s} {v}")


show("", res)

# ------------------------------------------------------------------ figure: rho(T)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, BAND, ORANGE, BAR = "#2a78d6", "#b7d3f6", "#eb6834", "#9ec5f4"
plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})
grid = np.linspace(T.min(), T.max(), 300)
gdev = grid - Tbar
rho_g = b1 + 2 * b2 * gdev
se_g = np.sqrt(V[0, 0] + 4 * gdev * V[0, 1] + 4 * gdev ** 2 * V[1, 1])
fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.6, 3.6), sharex=True,
                              gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.08))
ax.fill_between(grid, rho_g - TCRIT * se_g, rho_g + TCRIT * se_g, color=BAND, lw=0,
                label="95 percent confidence band")
ax.plot(grid, rho_g, color=BLUE, lw=2, label=r"$\hat\rho(T) = \hat b_1 + 2\hat b_2 (T - \bar T)$")
ax.axhline(r1, color=BLUE, lw=1, ls="--", label=r"constant pass-through, real terms ($\hat\rho = %.3f$)" % r1)
ax.axhline(1.0, color=ORANGE, lw=1.2, label=r"competition with flat cost ($\rho = 1$)")
for q in (10, 90):
    ax.plot(res["M2"][f"T_p{q}"], res["M2"][f"rho_p{q}"], "o", color=BLUE, ms=5, mec="white",
            mew=1, zorder=5)
    ax.annotate(f"{q}th percentile", (res["M2"][f"T_p{q}"], res["M2"][f"rho_p{q}"]),
                xytext=(4, -12 if q == 10 else 6), textcoords="offset points", fontsize=8, color=INK)
ax.set_ylabel(r"pass-through $\rho$")
ax.set_ylim(0.85, 1.62)
ax.grid(True, axis="y", color=GRID, lw=0.5)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
axh.hist(T, bins=30, color=BAR, edgecolor="white", lw=0.6)
axh.set_ylabel("state-years")
axh.set_xlabel(r"real excise tax $T$ (cents per pack, 1982–84 prices)")
for s in ("top", "right"):
    axh.spines[s].set_visible(False)
fig.align_ylabels([ax, axh])
fig.savefig(os.path.join(FIGDIR, "rho_T.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "rho_T_preview.png"), dpi=150, bbox_inches="tight")
print("wrote", os.path.join(FIGDIR, "rho_T.pdf"))
