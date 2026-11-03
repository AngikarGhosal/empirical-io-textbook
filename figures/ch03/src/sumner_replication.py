"""Replication and extension of the cigarette tax pass-through regressions.

Data: data_sumner.csv, a reconstructed state-year panel (47 states plus DC,
1954-1978, 1,200 observations) with the weighted-average retail price per pack,
the combined state plus federal excise tax (cents per pack) and the CPI-U.
Source of prices and taxes: The Tax Burden on Tobacco, historical compilation
(1983), Table 13. CPI-U: BLS.

Outputs
  ../sumner_identified_set.pdf   identified set of (curvature, conduct) pairs
  sumner_results.json            every number quoted in the text
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
d = pd.read_csv(HERE / "data_sumner.csv").sort_values(["state_abbr", "year"]).reset_index(drop=True)
N_STATES = d.state_abbr.nunique()
N_YEARS = d.year.nunique()
d["P"] = 100 * d.price_cents / d.cpi_u_annual          # real price, 1982-84 cents
d["T"] = 100 * d.tax_cents_combined / d.cpi_u_annual   # real tax
d["Tc"] = d["T"] - d["T"].mean()
FE = " + C(state_abbr) + C(year)"
CL = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.state_abbr)[0]}, use_t=True)

# ---------------------------------------------------------------- Table 1 rows
# Row 1: state and year dummies, OLS (conventional and state-clustered s.e.)
r1 = smf.ols("price_cents ~ tax_cents_combined" + FE, d).fit()
r1c = smf.ols("price_cents ~ tax_cents_combined" + FE, d).fit(**CL)
# Row 2: national CPI in place of the year dummies, no state dummies, OLS
r2 = smf.ols("price_cents ~ tax_cents_combined + cpi_u_annual", d).fit()


# Row 3: CPI, two-way variance components (random state and year effects), GLS
def two_way_re(data):
    """Swamy-Arora type two-way random-effects GLS for a balanced panel."""
    y = data.price_cents.to_numpy(float)
    tax = data.tax_cents_combined.to_numpy(float)
    cpi = data.cpi_u_annual.to_numpy(float)
    n, t = N_STATES, N_YEARS

    def m(v, key):
        return pd.Series(v).groupby(data[key].to_numpy()).transform("mean").to_numpy()

    def within(v):
        return v - m(v, "state_abbr") - m(v, "year") + v.mean()

    yw, xw = within(y), within(tax)
    bw = (xw @ yw) / (xw @ xw)
    ew = yw - bw * xw
    s2e = (ew @ ew) / ((n - 1) * (t - 1) - 1)
    g = data.groupby("state_abbr")
    xs = np.column_stack([np.ones(n), g.tax_cents_combined.mean()])
    es = g.price_cents.mean().to_numpy() - xs @ np.linalg.lstsq(xs, g.price_cents.mean().to_numpy(), rcond=None)[0]
    s21 = t * (es @ es) / (n - 2)
    g = data.groupby("year")
    xt = np.column_stack([np.ones(t), g.tax_cents_combined.mean(), g.cpi_u_annual.mean()])
    et = g.price_cents.mean().to_numpy() - xt @ np.linalg.lstsq(xt, g.price_cents.mean().to_numpy(), rcond=None)[0]
    s22 = n * (et @ et) / (t - 3)
    s2a, s2b = max((s21 - s2e) / t, 0.0), max((s22 - s2e) / n, 0.0)
    l1 = 1 - np.sqrt(s2e / (s2e + t * s2a))
    l2 = 1 - np.sqrt(s2e / (s2e + n * s2b))
    l3 = l1 + l2 - 1 + np.sqrt(s2e / (s2e + t * s2a + n * s2b))

    def tr(v):
        return v - l1 * m(v, "state_abbr") - l2 * m(v, "year") + l3 * v.mean()

    X = np.column_stack([np.ones_like(y), tax, cpi])
    Xs = np.column_stack([tr(X[:, j]) for j in range(X.shape[1])])
    b = np.linalg.lstsq(Xs, tr(y), rcond=None)[0]
    V = s2e * np.linalg.inv(Xs.T @ Xs)
    return b[1], np.sqrt(V[1, 1]), dict(s2e=s2e, s2a=s2a, s2b=s2b)


rho3, se3, vc = two_way_re(d)

# ---------------------------------------------------------------- extensions
m1 = smf.ols("P ~ T" + FE, d).fit(**CL)                 # real terms
m2 = smf.ols("P ~ Tc + I(Tc**2)" + FE, d).fit(**CL)     # quadratic in real tax

rho, se_cl = r1c.params["tax_cents_combined"], r1c.bse["tax_cents_combined"]
lo, hi = r1c.conf_int().loc["tax_cents_combined"]
b1, b2 = m2.params["Tc"], m2.params["I(Tc ** 2)"]
V = m2.cov_params().loc[["Tc", "I(Tc ** 2)"], ["Tc", "I(Tc ** 2)"]].to_numpy()
pct = {}
for q in (10, 90):
    dev = np.percentile(d["Tc"], q)
    g = np.array([1.0, 2 * dev])
    pct[q] = (b1 + 2 * b2 * dev, float(np.sqrt(g @ V @ g)))
spread = 2 * (np.percentile(d["Tc"], 90) - np.percentile(d["Tc"], 10))
tcrit = 2.0117  # t with 47 degrees of freedom, 0.975 quantile
diff, diff_se = spread * b2, spread * m2.bse["I(Tc ** 2)"]
kap, kap_lo, kap_hi = 1 - 1 / rho, 1 - 1 / lo, 1 - 1 / hi
share = d.tax_cents_combined / d.price_cents

# variation in taxes
dd = d.copy()
dd["dtax"] = dd.groupby("state_abbr").tax_cents_combined.diff()
changes = dd.dtax.dropna()

res = {
    "n": int(r1.nobs),
    "row1_rho": r1.params["tax_cents_combined"], "row1_se": r1.bse["tax_cents_combined"], "row1_r2": r1.rsquared,
    "row1_se_cluster": se_cl, "row1_ci_lo": lo, "row1_ci_hi": hi,
    "row1_p_comp_cluster": float(r1c.t_test("tax_cents_combined = 1").pvalue),
    "row2_rho": r2.params["tax_cents_combined"], "row2_se": r2.bse["tax_cents_combined"], "row2_r2": r2.rsquared,
    "row3_rho": rho3, "row3_se": se3, **{f"vc_{k}": v for k, v in vc.items()},
    "real_rho": m1.params["T"], "real_se": m1.bse["T"], "real_p": float(m1.t_test("T = 1").pvalue),
    "quad_b1": b1, "quad_b2": b2, "quad_se_b2": m2.bse["I(Tc ** 2)"], "quad_p_b2": m2.pvalues["I(Tc ** 2)"],
    "rho_p10": pct[10][0], "se_p10": pct[10][1], "rho_p90": pct[90][0], "se_p90": pct[90][1],
    "diff_lo": diff - tcrit * diff_se, "diff_hi": diff + tcrit * diff_se,
    "kappa": kap, "kappa_lo": kap_lo, "kappa_hi": kap_hi,
    "chi_monopoly": 1 + kap, "chi_cournot6": 1 + 6 * kap,
    "theta_ce04": 0.4 * kap, "theta_ce04_lo": 0.4 * kap_lo, "theta_ce04_hi": 0.4 * kap_hi,
    "theta_ce03": 0.3 * kap, "theta_ce08": 0.8 * kap,
    "tax_share_mean": share.mean(), "tax_share_min": share.min(), "tax_share_max": share.max(),
    "theta_max04_mean": 0.4 * (1 - share.mean()), "theta_max04_minshare": 0.4 * (1 - share.min()),
    "price_mean": d.price_cents.mean(), "price_sd": d.price_cents.std(),
    "tax_mean": d.tax_cents_combined.mean(), "tax_sd": d.tax_cents_combined.std(),
    "price_1954": d.loc[d.year == 1954, "price_cents"].mean(),
    "price_1978": d.loc[d.year == 1978, "price_cents"].mean(),
    "statetax_median_1954": (d.loc[d.year == 1954, "tax_cents_combined"] - 8).median(),
    "statetax_median_1978": (d.loc[d.year == 1978, "tax_cents_combined"] - 8).median(),
    "n_transitions": int(changes.size), "n_changes": int((changes != 0).sum()),
    "n_decreases": int((changes < 0).sum()), "mean_change": changes[changes != 0].mean(),
    "max_change": changes.max(),
}
res = {k: float(v) for k, v in res.items()}
for k, v in res.items():
    print(f"{k:22s} {v:10.4f}")
(HERE / "sumner_results.json").write_text(json.dumps(res, indent=2))

# ---------------------------------------------------------------- figure
BLUE, BAND, ORANGE, INK, MUTED = "#2a78d6", "#a9c9ef", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "mathtext.fontset": "cm",
                     "font.family": "serif", "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"]})
chi = np.linspace(1.0005, 4.0, 600)
fig, ax = plt.subplots(figsize=(5.6, 2.9))
ax.fill_between(chi, np.minimum(kap_lo / (chi - 1), 1.0), np.minimum(kap_hi / (chi - 1), 1.0),
                color=BAND, alpha=0.7, lw=0, label="95 percent confidence band")
line = np.where(kap / (chi - 1) <= 1.0, kap / (chi - 1), np.nan)
ax.plot(chi, line, color=BLUE, lw=2, label=r"$\theta(\chi - 1) = \hat\kappa$")
ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
pts = [
    (1 + kap, 1.0, "monopoly", (9, -14)),
    (1 + 6 * kap, 1 / 6, "symmetric Cournot, $N = 6$", (8, 2)),
    (3.5, 0.4 * kap, r"constant elasticity, $\varepsilon = 0.4$", (-75, -16)),
]
for x, y, lab, off in pts:
    ax.plot(x, y, "o", color=ORANGE, ms=6, mec="white", mew=1.2, zorder=5)
    ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=8.5, color=INK)
ax.set_yscale("log")
ax.set_ylim(0.002, 1.8)
ax.set_xlim(1, 4)
ax.set_xlabel(r"curvature of inverse demand $\chi = -\frac{Q P^{\prime\prime}(Q)}{P^{\prime}(Q)}$")
ax.set_ylabel(r"conduct $\theta$")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "sumner_identified_set.pdf")
fig.savefig(HERE / "sumner_identified_set_preview.png", dpi=150)
