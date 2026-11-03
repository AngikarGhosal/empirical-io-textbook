"""Markups over time in the U.S. automobile market (Appendix C.8).

Estimates a plain logit, a nested logit with vehicle-type nests (car,
SUV, truck, van) and two logits in log price (one coefficient, or one
per five-year period) by 2SLS on the product data of the public
replication package of Grieco, Murry and Yurukoglu (2024), and the
period model also by LIML. Computes multi-product Bertrand markups with
ownership by parent company, sales-weighted Lerner indices by year and
the exact decomposition of the change in the mean log Lerner index.
The nesting parameter is also matched to the second-choice survey
counts in the package. Writes autos_results.json, autos_macros.tex
(every number used in the text) and figures/appC/autos_lerner.pdf.
Values reported by GMY in their manuscript are kept in GMY_REPORTED for
reference only; they are not plotted.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io as sio  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

# The input data come from the replication package of Grieco, Murry and Yurukoglu (2024), which is not
# redistributed here. Download it and point GMY_DATA at its data/ folder.
GMY_DATA = os.environ.get("GMY_DATA", "GMY_Replication/data/").rstrip("/") + "/"
BOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = BOOK + "/src"
FIG_DIR = BOOK
FIG_NAME = "autos_lerner.pdf"
SEED = 20261020
N_DRAWS = 2000
SC_YEARS = (1991, 1999, 2005, 2015)
FIRST, LAST = 1980, 2018

# Values reported by GMY, transcribed from the July 5, 2023 author
# manuscript (Table II, Section 5.4 text, Table VII):
# https://charliemurry.github.io/working_papers/CarMarkupsJuly2023.pdf
GMY_REPORTED = {
    "sw_mean_lerner": {"1980": 0.42, "2018": 0.22},
    "median_lerner": {"1980": 0.325, "2018": 0.185},
    "logprice_sw_mean_lerner": {"1980": 0.32, "2018": 0.27},
    "table2_iv_price_coef": -1.696,
    "table2_iv_price_se": 0.598,
    "table2_ols_price_coef": -0.334,
    "table2_ols_price_se": 0.042,
    "table2_first_stage_rxr_coef": 0.411,
    "table2_first_stage_rxr_se": 0.110,
    "table2_first_stage_F": 14.086,
    "table2_iv_mean_own_elasticity": -6.107,
    "table2_pass_through": 0.117,
}


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def zscore(x):
    """Standardize as MATLAB zscore (sample standard deviation)."""
    return (x - x.mean()) / x.std(ddof=1)


def load_data():
    """Build shares, real prices, nests and counts as in GMY."""
    d = pd.read_csv(GMY_DATA + "data_blp.csv")
    d["msize"] = d["number_households"] / 2.5
    d["s"] = d["sales"] / d["msize"]
    d["s0"] = 1.0 - d.groupby("year")["s"].transform("sum")
    d["p"] = d["msrp"] / 10.0  # 2015 dollars, units of $10,000
    d["y"] = np.log(d["s"]) - np.log(d["s0"])
    d["rxr"] = d["lag_pl_con"]
    d["nest"] = np.select(
        [d["suv"] == 1, d["truck"] == 1, d["van"] == 1],
        ["suv", "truck", "van"],
        "car",
    )
    grp = d.groupby(["year", "nest"])["s"]
    d["sg"] = grp.transform("sum")
    d["sjg"] = d["s"] / d["sg"]
    d["lsjg"] = np.log(d["sjg"])
    d["n_nest"] = grp.transform("size")
    own = d.groupby(["year", "nest", "parent"])["s"].transform("size")
    d["n_rival"] = d["n_nest"] - own
    assert (d["n_rival"] > 0).all(), "a nest-year has a single firm"
    d["trend"] = d["year"] - FIRST
    return d


def characteristics(d):
    """GMY linear characteristics ($X in codes/linearDemand.do)."""
    return pd.DataFrame(
        {
            "height": zscore(d["log_height"]),
            "footprint": zscore(d["log_footprint"]),
            "hp": zscore(d["log_hp"]),
            "mpg": zscore(d["log_mpg"]),
            "curbweight": zscore(d["log_curbweight"]),
            "numtrims": zscore(d["log_number_trims"]),
            "releaseYear": d["releaseYear"],
            "yearsSinceDesign": d["yearsSinceDesign"].clip(3, 12),
            "sport": d["sport"],
            "EV": np.maximum(d["PHEV"], d["EV"]),
            "truck": d["truck"],
            "suv": d["suv"],
            "van": d["van"],
        }
    )


def controls(d, X, trends=0):
    """Constant, characteristics, year and make dummies, nest trends."""
    yfe = pd.get_dummies(d["year"], drop_first=True).astype(float)
    mfe = pd.get_dummies(d["make2"], drop_first=True).astype(float)
    cols = [np.ones(len(d)), X.values, yfe.values, mfe.values]
    t = d["trend"].values.astype(float)
    types = d[["suv", "truck", "van"]].values
    for k in range(1, trends + 1):
        cols.append(types * (t**k)[:, None])
    return np.column_stack(cols)


def nest_instruments(d, X):
    """Candidate excluded instruments for the within-nest share."""
    z = {
        "ln_n_nest": np.log(d["n_nest"].values),
        "ln_n_rival": np.log(d["n_rival"].values),
    }
    keys = [d["year"], d["nest"]]
    for c in ("hp", "footprint", "curbweight", "mpg"):
        v = pd.Series(X[c].values, index=d.index)
        tot = v.groupby(keys).transform("sum")
        own = v.groupby(keys + [d["parent"]]).transform("sum")
        z["rival_sum_" + c] = (tot - own).values
    for c in ("hp", "footprint"):
        out = np.zeros(len(d))
        for idx in d.groupby(["year", "nest"]).indices.values():
            v = X[c].values[idx]
            par = d["parent"].values[idx]
            close = np.abs(v[:, None] - v[None, :]) < 1.0
            out[idx] = (close & (par[:, None] != par[None, :])).sum(1)
        z["diff_" + c] = out
    return z


# ----------------------------------------------------------------------
# Linear IV machinery
# ----------------------------------------------------------------------
def cluster_vcov(bread, scores, groups, n, k, small):
    """Cluster-robust sandwich; small applies Stata's regress factor."""
    codes, uniq = pd.factorize(np.asarray(groups))
    ng = len(uniq)
    sums = np.zeros((ng, scores.shape[1]))
    np.add.at(sums, codes, scores)
    V = bread @ (sums.T @ sums) @ bread
    if small:
        V = V * ng / (ng - 1) * (n - 1) / (n - k)
    return V, ng


def as_2d(a):
    """Return a two-dimensional column array."""
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def tsls(y, endog, inst, W, groups, small=False):
    """Two-stage least squares with cluster-robust standard errors."""
    endog, inst = as_2d(endog), as_2d(inst)
    Xf = np.column_stack([endog, W])
    Zf = np.column_stack([inst, W])
    Xh = Zf @ np.linalg.lstsq(Zf, Xf, rcond=None)[0]
    b = np.linalg.lstsq(Xh, y, rcond=None)[0]
    u = y - Xf @ b
    bread = np.linalg.pinv(Xh.T @ Xh)
    k = np.linalg.matrix_rank(Xf)
    V, ng = cluster_vcov(bread, Xh * u[:, None], groups, len(y), k, small)
    ke = endog.shape[1]
    Vk = V[:ke, :ke]
    return {"b": b[:ke], "V": Vk, "se": np.sqrt(np.diag(Vk)), "u": u, "G": ng}


def ols(y, Xf, groups, small=True):
    """OLS with cluster-robust standard errors."""
    b = np.linalg.lstsq(Xf, y, rcond=None)[0]
    u = y - Xf @ b
    bread = np.linalg.pinv(Xf.T @ Xf)
    k = np.linalg.matrix_rank(Xf)
    V, ng = cluster_vcov(bread, Xf * u[:, None], groups, len(y), k, small)
    return {"b": b, "V": V, "se": np.sqrt(np.diag(V)), "u": u, "G": ng}


def wald_f(r, q):
    """Cluster-robust F statistic for the first q coefficients."""
    bz, Vz = r["b"][:q], r["V"][:q, :q]
    return float(bz @ np.linalg.solve(Vz, bz) / q)


def first_stage(endog, inst, W, groups):
    """First-stage coefficients and cluster-robust F on instruments."""
    inst = as_2d(inst)
    q = inst.shape[1]
    r = ols(endog, np.column_stack([inst, W]), groups, small=True)
    return {
        "coef": r["b"][:q],
        "se": r["se"][:q],
        "F": wald_f(r, q),
        "fitted": endog - r["u"],
    }


def sw_fstats(endog, inst, W, groups):
    """Sanderson-Windmeijer conditional first-stage F statistics."""
    endog, inst = as_2d(endog), as_2d(inst)
    ke, q = endog.shape[1], inst.shape[1]
    out = []
    for k in range(ke):
        others = np.delete(endog, k, axis=1)
        e = tsls(endog[:, k], others, inst, W, groups)["u"]
        r = ols(e, np.column_stack([inst, W]), groups, small=True)
        bz, Vz = r["b"][:q], r["V"][:q, :q]
        out.append(float(bz @ np.linalg.solve(Vz, bz) / (q - ke + 1)))
    return out


def partial_out(A, W):
    """Residuals of A after least-squares projection on W."""
    return A - W @ np.linalg.lstsq(W, A, rcond=None)[0]


def liml(y, endog, inst, W):
    """Limited-information maximum likelihood point estimates."""
    yt = partial_out(y, W)
    Yt = partial_out(as_2d(endog), W)
    Zt = partial_out(as_2d(inst), W)
    Y0 = np.column_stack([yt, Yt])
    PY0 = Zt @ np.linalg.lstsq(Zt, Y0, rcond=None)[0]
    A = (Y0 - PY0).T @ Y0
    k = np.min(np.real(np.linalg.eigvals(np.linalg.solve(A, Y0.T @ Y0))))
    Yk = (1.0 - k) * Yt + k * PY0[:, 1:]
    return np.linalg.solve(Yk.T @ Yt, Yk.T @ yt), float(k)


def period_index(d):
    """GMY five-year periods: 1980-84, ..., 2010-14, 2015-18."""
    return np.clip((d["year"].values - FIRST) // 5, 0, 7)


# ----------------------------------------------------------------------
# Second-choice moment for the nested logit
# ----------------------------------------------------------------------
def removal_same_type(s, sjg, sg, nest, sigma):
    """Nested-logit probability that an inside second choice shares
    the first choice's nest, computed by removing the first choice."""
    R = 1.0 - sg * (1.0 - (1.0 - sjg) ** (1.0 - sigma))
    same = nest[:, None] == nest[None, :]
    lift = np.where(same, (1.0 - sjg[:, None]) ** (-sigma), 1.0)
    new = s[None, :] * lift / R[:, None]
    div = (new - s[None, :]) / s[:, None]
    np.fill_diagonal(div, 0.0)
    return (div * same).sum(1) / div.sum(1)


def load_counts():
    """GMY second-choice counts (rows: first choice, cols: second)."""
    m = sio.loadmat(GMY_DATA + "secondChoiceOut.mat")
    return {yr: m["secondChoiceN_%d" % yr].astype(float) for yr in SC_YEARS}


def load_boot_counts():
    """GMY's 500 bootstrap samples of the second-choice counts."""
    m = sio.loadmat(GMY_DATA + "secondChoice_bootstrap.mat")
    nb = m["scboot%d" % SC_YEARS[0]].shape[1]
    return [
        {yr: m["scboot%d" % yr][0, b].astype(float) for yr in SC_YEARS}
        for b in range(nb)
    ]


def year_cache(d):
    """Shares, nest shares and nests by year (data order kept)."""
    out = {}
    for yr in SC_YEARS:
        sub = d[d["year"] == yr]
        out[yr] = (
            sub["s"].values,
            sub["sjg"].values,
            sub["sg"].values,
            sub["nest"].values,
        )
    return out


def sc_data_moment(counts, cache):
    """Observed share of second choices in the first choice's nest."""
    num = den = 0.0
    for yr, N in counts.items():
        nest = cache[yr][3]
        r, c = N.shape
        num += (N * (nest[:r, None] == nest[None, :c])).sum()
        den += N.sum()
    return num / den


def sc_model_moment(sigma, counts, cache):
    """Model share of second choices in the first choice's nest."""
    num = den = 0.0
    for yr, N in counts.items():
        s, sjg, sg, nest = cache[yr]
        pr = removal_same_type(s, sjg, sg, nest, sigma)[: N.shape[0]]
        nj = N.sum(1)
        num += (nj * pr).sum()
        den += nj.sum()
    return num / den


def fit_sigma(counts, cache):
    """Nesting parameter that matches the same-nest second choices."""
    target = sc_data_moment(counts, cache)
    sig = brentq(
        lambda x: sc_model_moment(x, counts, cache) - target,
        0.0,
        0.995,
        xtol=1e-12,
    )
    return sig, target


# ----------------------------------------------------------------------
# Demand derivatives and Bertrand markups
# ----------------------------------------------------------------------
def share_jacobian(model, s, p, sjg, nest, alpha, sigma):
    """Matrix D with D[j, k] equal to the derivative of s_j in p_k."""
    if model == "logprice":
        D = alpha * np.outer(s, s) / p[None, :]
        np.fill_diagonal(D, -alpha * s * (1.0 - s) / p)
        return D
    sig = sigma if model == "nested" else 0.0
    D = alpha * np.outer(s, s)
    if sig > 0.0:
        same = nest[:, None] == nest[None, :]
        D = D + same * (alpha * sig / (1.0 - sig)) * np.outer(s, sjg)
    own = 1.0 / (1.0 - sig) - sig / (1.0 - sig) * sjg - s
    np.fill_diagonal(D, -alpha * s * own)
    return D


def bertrand_markups(D, owner, s):
    """Solve s + (Omega * D') m = 0 for the dollar markups m."""
    O = owner[:, None] == owner[None, :]
    return np.linalg.solve(O * D.T, -s)


def alpha_in(alpha, yr):
    """Price coefficient for a year (scalar or year-keyed dict)."""
    return alpha[int(yr)] if isinstance(alpha, dict) else alpha


def model_markups(d, model, alpha, sigma=0.0, years=None):
    """Markups (units of $10,000) and own-price elasticities."""
    m = np.full(len(d), np.nan)
    el = np.full(len(d), np.nan)
    yrs = sorted(d["year"].unique()) if years is None else years
    for yr in yrs:
        idx = np.where(d["year"].values == yr)[0]
        s, p = d["s"].values[idx], d["p"].values[idx]
        D = share_jacobian(
            model,
            s,
            p,
            d["sjg"].values[idx],
            d["nest"].values[idx],
            alpha_in(alpha, yr),
            sigma,
        )
        m[idx] = bertrand_markups(D, d["parent"].values[idx], s)
        el[idx] = np.diag(D) * p / s
    return m, el


def yearly_summary(d, m, el, alpha):
    """Sales-weighted Lerner, dollar markup and price by year."""
    rows = []
    for yr, idx in sorted(d.groupby("year").indices.items()):
        s, p, mk = d["s"].values[idx], d["p"].values[idx], m[idx]
        w = s / s.sum()
        L = mk / p
        a = alpha_in(alpha, yr)
        rows.append(
            {
                "year": int(yr),
                "alpha": float(a),
                "sw_mean_lerner": float(w @ L),
                "median_lerner": float(np.median(L)),
                "sw_mean_markup_usd": float(1e4 * (w @ mk)),
                "sw_mean_price_usd": float(1e4 * (w @ p)),
                "sw_mean_log_price": float(w @ np.log(p)),
                "sw_mean_log_alpha_markup": float(w @ np.log(a * mk)),
                "sw_mean_log_lerner": float(w @ np.log(L)),
                "n_lerner_at_least_one": int((L >= 1.0).sum()),
                "mean_own_elasticity": float(el[idx].mean()),
            }
        )
    return rows


def trend_stats(rows, log_price=False):
    """Level, change, slope and exact log decomposition of the trend."""
    yrs = np.array([r["year"] for r in rows], dtype=float)
    L = np.array([r["sw_mean_lerner"] for r in rows])
    a, b = rows[0], rows[-1]
    # Identity: mean log Lerner = mean log(alpha m) - log alpha
    # - mean log p, all sales-weighted within a year.
    dlogl = b["sw_mean_log_lerner"] - a["sw_mean_log_lerner"]
    dlogp = b["sw_mean_log_price"] - a["sw_mean_log_price"]
    dlogm = b["sw_mean_log_alpha_markup"] - a["sw_mean_log_alpha_markup"]
    dloga = float(np.log(b["alpha"]) - np.log(a["alpha"]))
    # Dollar price sensitivity: alpha / p with log price, alpha if linear.
    dlogs = dloga - dlogp if log_price else dloga
    return {
        "lerner_first": a["sw_mean_lerner"],
        "lerner_last": b["sw_mean_lerner"],
        "lerner_change": b["sw_mean_lerner"] - a["sw_mean_lerner"],
        "lerner_ratio_last_first": b["sw_mean_lerner"] / a["sw_mean_lerner"],
        "lerner_pct_change": 100.0
        * (b["sw_mean_lerner"] / a["sw_mean_lerner"] - 1.0),
        "lerner_avg_1980_1982": float(L[:3].mean()),
        "lerner_avg_2016_2018": float(L[-3:].mean()),
        "slope_per_decade": float(10.0 * np.polyfit(yrs, L, 1)[0]),
        "median_first": a["median_lerner"],
        "median_last": b["median_lerner"],
        "markup_usd_first": a["sw_mean_markup_usd"],
        "markup_usd_last": b["sw_mean_markup_usd"],
        "markup_usd_pct_change": 100.0
        * (b["sw_mean_markup_usd"] / a["sw_mean_markup_usd"] - 1.0),
        "dlog_lerner": dlogl,
        "dlog_price": dlogp,
        "dlog_alpha_markup": dlogm,
        "dlog_alpha": dloga,
        "dlog_dollar_price_sensitivity": dlogs,
        "identity_residual": dlogl - (dlogm - dloga - dlogp),
        "price_share_of_log_decline": (
            dlogp / (-dlogl) if dlogl < 0 else None
        ),
        "alpha_share_of_log_decline": (
            dloga / (-dlogl) if dlogl < 0 else None
        ),
    }


def alpha_scaled_lerner(d, model, sigma, years=(FIRST, LAST)):
    """Sales-weighted mean of alpha times the Lerner index by year."""
    m, _ = model_markups(d, model, 1.0, sigma, list(years))
    out = []
    for yr in years:
        idx = np.where(d["year"].values == yr)[0]
        w = d["s"].values[idx] / d["s"].values[idx].sum()
        out.append(float(w @ (m[idx] / d["p"].values[idx])))
    return np.array(out)


def interval_summary(draws):
    """Percentile 95 percent intervals for Lerner levels and change."""
    draws = np.asarray(draws)
    out = {}
    names = ("lerner_first", "lerner_last", "lerner_change", "ratio")
    cols = [
        draws[:, 0],
        draws[:, 1],
        draws[:, 1] - draws[:, 0],
        draws[:, 1] / draws[:, 0],
    ]
    for n, c in zip(names, cols):
        out[n] = [float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5))]
    return out


# ----------------------------------------------------------------------
# Self-checks
# ----------------------------------------------------------------------
def nl_shares(delta, nest, sigma):
    """Nested-logit market shares given mean utilities."""
    a = np.exp(delta / (1.0 - sigma))
    groups = np.unique(nest)
    tot = {g: a[nest == g].sum() for g in groups}
    dj = np.array([tot[g] for g in nest])
    den = 1.0 + sum(v ** (1.0 - sigma) for v in tot.values())
    return a * dj ** (-sigma) / den


def self_checks(d, alpha_l, alpha_n, sigma):
    """Verify closed forms against brute force on one year."""
    sub = d[d["year"] == 2015]
    s, sjg, sg = sub["s"].values, sub["sjg"].values, sub["sg"].values
    nest, p = sub["nest"].values, sub["p"].values
    s0 = 1.0 - s.sum()
    delta = np.log(s) - np.log(s0) - sigma * np.log(sjg)
    err_inv = np.abs(nl_shares(delta, nest, sigma) - s).max()
    closed = removal_same_type(s, sjg, sg, nest, sigma)
    err_rm = 0.0
    for j in (0, 50, 150, 250):
        keep = np.arange(len(s)) != j
        new = nl_shares(delta[keep], nest[keep], sigma)
        div = (new - s[keep]) / s[j]
        same = nest[keep] == nest[j]
        err_rm = max(err_rm, abs(div[same].sum() / div.sum() - closed[j]))
    D = share_jacobian("nested", s, p, sjg, nest, alpha_n, sigma)
    h = 1e-6
    err_fd = 0.0
    for k in (0, 50, 150, 250):
        e = np.zeros(len(s))
        e[k] = h
        up = nl_shares(delta - alpha_n * e, nest, sigma)
        dn = nl_shares(delta + alpha_n * e, nest, sigma)
        err_fd = max(err_fd, np.abs((up - dn) / (2 * h) - D[:, k]).max())
    owner = sub["parent"].values
    m = bertrand_markups(
        share_jacobian("logit", s, p, sjg, nest, alpha_l, 0.0), owner, s
    )
    sf = pd.Series(s).groupby(owner).transform("sum").values
    err_logit = np.abs(m - 1.0 / (alpha_l * (1.0 - sf))).max()
    return {
        "nested_share_inversion_max_abs_error": float(err_inv),
        "removal_formula_vs_brute_force_max_abs_error": float(err_rm),
        "nested_jacobian_vs_finite_difference_max_abs_error": float(err_fd),
        "logit_markup_vs_closed_form_max_abs_error": float(err_logit),
    }


# ----------------------------------------------------------------------
# Figure and output
# ----------------------------------------------------------------------
def make_figure(series, path):
    """Plot the sales-weighted mean Lerner index by year for each model."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.edgecolor": "#8a8984",
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    fig.subplots_adjust(left=0.1, right=0.66, bottom=0.15, top=0.97)
    styles = [
        ("logit", "Plain logit", "#2a78d6", "-"),
        ("nested_2sls", "Instrumented nested logit", "#eb6834", "--"),
        ("nested_survey", "Survey nested logit", "#1baf7a", "-."),
        ("logprice", "Log-price logit", "#b07d00", ":"),
        (
            "logprice_period",
            "Period log-price logit",
            "#c4457a",
            (0, (6, 1.5, 1, 1.5, 1, 1.5)),
        ),
        (
            "logprice_period_liml",
            "Period log-price logit, LIML",
            "#6b6a66",
            (0, (1, 1.2)),
        ),
    ]
    yrs = series["years"]
    lines = []
    for key, lab, col, ls in styles:
        lines += ax.plot(
            yrs,
            series[key],
            color=col,
            ls=ls,
            lw=1.5,
            label=lab,
            solid_capstyle="round",
        )
    ax.set_ylabel(r"Lerner index $\frac{p - mc}{p}$", fontsize=11)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_xlim(FIRST - 1, LAST + 1)
    ax.set_ylim(0.0, 0.5)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(True, color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        handles=lines,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8.5,
        handlelength=2.4,
        handletextpad=0.5,
        labelspacing=0.5,
        borderaxespad=0.0,
    )
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "autos_lerner_preview.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def to_py(obj):
    """Convert numpy containers to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_py(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def data_summary(d):
    """Sample facts used in the write-up."""
    out = {
        "n_obs": len(d),
        "n_years": int(d["year"].nunique()),
        "n_parents": int(d["parent"].nunique()),
        "n_make_clusters": int(d["make2"].nunique()),
        "n_models": int(d["model"].nunique()),
        "min_products_per_year": int(d.groupby("year").size().min()),
        "max_products_per_year": int(d.groupby("year").size().max()),
    }
    for yr in (FIRST, LAST):
        sub = d[d["year"] == yr]
        w = sub["s"] / sub["s"].sum()
        firm = sub.groupby("parent")["s"].transform("sum")
        firm_nest = sub.groupby(["parent", "nest"])["sjg"].transform("sum")
        out[str(yr)] = {
            "sw_mean_firm_share": float(w @ firm),
            "sw_mean_firm_within_nest_share": float(w @ firm_nest),
            "n_products": len(sub),
            "market_size": float(sub["msize"].iloc[0]),
            "outside_share": float(sub["s0"].iloc[0]),
            "sw_mean_price_usd": float(1e4 * (w @ sub["p"])),
            "nest_sales_share": sub.groupby("nest")["s"]
            .sum()
            .div(sub["s"].sum())
            .to_dict(),
            "nest_n_products": sub.groupby("nest").size().to_dict(),
            "mean_within_nest_share": float(sub["sjg"].mean()),
            "sw_mean_within_nest_share": float(w @ sub["sjg"]),
        }
    return out


def fmt(x, nd):
    """Fixed-point string with nd decimals."""
    return "%.*f" % (nd, x)


def usd(x):
    """Whole number with thousands separators."""
    return "{:,.0f}".format(x)


def latex_macros(res):
    """LaTeX macros (prefix gx) for every number the write-up cites."""
    g, gd, mk = res["gmy_reported"], res["gmy_derived"], res["markups"]
    sc, lpp, n2 = (
        res["second_choice"],
        res["logprice_period"],
        res["nested_2sls"],
    )
    dat, lg, ns = res["data"], res["logit"], res["nested_survey"]
    m = {
        "Nobs": usd(dat["n_obs"]),
        "Nresp": usd(sc["n_respondents"]),
        "Nboot": "%d" % sc["n_boot"],
        "Ngrid": "%d" % res["nested_2sls_sensitivity_counts"]["n_specs"],
        "Ninst": "%d" % lpp["n_excluded_instruments"],
        "GmyLFirst": fmt(g["sw_mean_lerner"]["1980"], 2),
        "GmyLLast": fmt(g["sw_mean_lerner"]["2018"], 2),
        "GmyRatio": fmt(gd["sw_ratio_2018_1980"], 2),
        "GmyLogLFirst": fmt(g["logprice_sw_mean_lerner"]["1980"], 2),
        "GmyLogLLast": fmt(g["logprice_sw_mean_lerner"]["2018"], 2),
        "GmyLogRatio": fmt(gd["logprice_ratio_2018_1980"], 2),
        "PriceFirst": usd(dat["1980"]["sw_mean_price_usd"]),
        "PriceLast": usd(dat["2018"]["sw_mean_price_usd"]),
        "PriceRatio": fmt(dat["sw_mean_price_ratio_first_last"], 3),
        "DlogPrice": fmt(mk["logit"]["trend"]["dlog_price"], 2),
        "FirmNestShareFirst": fmt(
            dat["1980"]["sw_mean_firm_within_nest_share"], 3
        ),
        "FirmNestShareLast": fmt(
            dat["2018"]["sw_mean_firm_within_nest_share"], 3
        ),
        "LogitCoef": fmt(lg["price_coef"], 3),
        "LogitSe": fmt(lg["se_make_cluster"], 3),
        "LogitF": fmt(lg["first_stage_F"], 1),
        "LogitCiLo": fmt(res["uncertainty"]["logit"]["lerner_change"][0], 3),
        "LogitCiHi": fmt(res["uncertainty"]["logit"]["lerner_change"][1], 3),
        "LogitPriceShare": fmt(
            100 * mk["logit"]["trend"]["price_share_of_log_decline"], 0
        ),
        "NlCoef": fmt(n2["make_cluster"]["price"]["coef"], 3),
        "NlCoefSe": fmt(n2["make_cluster"]["price"]["se"], 3),
        "NlSigma": fmt(n2["make_cluster"]["ln_within_nest_share"]["coef"], 3),
        "NlSigmaSe": fmt(n2["make_cluster"]["ln_within_nest_share"]["se"], 3),
        "NlSwFp": fmt(n2["sw_F_price_lsjg"][0], 1),
        "NlSwFs": fmt(n2["sw_F_price_lsjg"][1], 1),
        "GridSigmaMin": fmt(res["nested_2sls_sigma_range"][0], 2),
        "GridSigmaMax": fmt(res["nested_2sls_sigma_range"][1], 2),
        "ScObsPct": fmt(100 * sc["observed_same_nest_share"], 1),
        "ScLogitPct": fmt(100 * sc["logit_implied_same_nest_share"], 1),
        "ScSigma": fmt(sc["sigma_hat"], 3),
        "ScSigmaSe": fmt(sc["sigma_boot_se"], 3),
        "ScSigmaYrMin": fmt(sc["sigma_by_year_range"][0], 2),
        "ScSigmaYrMax": fmt(sc["sigma_by_year_range"][1], 2),
        "SvCoef": fmt(ns["price_coef"], 3),
        "SvCoefSe": fmt(ns["se_two_step"], 3),
        "SvMkDrop": fmt(
            -mk["nested_survey"]["trend"]["markup_usd_pct_change"], 1
        ),
        "LpCoef": fmt(res["logprice_logit"]["log_price_coef"], 3),
        "LpCoefSe": fmt(res["logprice_logit"]["se_make_cluster"], 3),
        "LpMkPct": fmt(mk["logprice"]["trend"]["markup_usd_pct_change"], 0),
        "PerCoefFirst": fmt(-lpp["alpha_2sls"][0], 3),
        "PerCoefLast": fmt(-lpp["alpha_2sls"][-1], 3),
        "PerAlphaFirst": fmt(lpp["alpha_2sls"][0], 2),
        "PerAlphaLast": fmt(lpp["alpha_2sls"][-1], 2),
        "PerDlogAlpha": fmt(lpp["dlog_alpha_2sls"], 2),
        "PerSensDrop": fmt(
            -mk["logprice_period"]["trend"]["dlog_dollar_price_sensitivity"],
            2,
        ),
        "LimlCoefFirst": fmt(-lpp["alpha_liml"][0], 3),
        "LimlCoefLast": fmt(-lpp["alpha_liml"][-1], 3),
        "LimlDlogAlpha": fmt(lpp["dlog_alpha_liml"], 2),
        "LimlUpFirst": fmt(
            100 * lpp["liml_over_2sls_alpha_first_last"][0] - 100, 0
        ),
        "LimlUpLast": fmt(
            100 * lpp["liml_over_2sls_alpha_first_last"][1] - 100, 0
        ),
        "RxrMinSwF": fmt(min(lpp["sw_F_rxr_only"]), 1),
        "MkPctMin": fmt(
            res["summary"]["markup_usd_pct_change_range_excl_logprice"][0], 1
        ),
        "MkPctMax": fmt(
            res["summary"]["markup_usd_pct_change_range_excl_logprice"][1], 1
        ),
    }
    rows = (
        ("Logit", "logit"),
        ("Nl", "nested_2sls"),
        ("Sv", "nested_survey"),
        ("Lp", "logprice"),
        ("Per", "logprice_period"),
        ("Liml", "logprice_period_liml"),
    )
    for pre, key in rows:
        t = mk[key]["trend"]
        m[pre + "LFirst"] = fmt(t["lerner_first"], 3)
        m[pre + "LLast"] = fmt(t["lerner_last"], 3)
        m[pre + "Ratio"] = fmt(t["lerner_ratio_last_first"], 3)
        m[pre + "MkFirst"] = usd(t["markup_usd_first"])
        m[pre + "MkLast"] = usd(t["markup_usd_last"])
        m[pre + "LDropPct"] = fmt(-t["lerner_pct_change"], 0)
    lines = ["% Generated by code/extension_gmy.py. Do not edit by hand."]
    for k in sorted(m):
        lines.append("\\newcommand{\\gx%s}{%s}" % (k, m[k]))
    return "\n".join(lines) + "\n"


def rounding_bounds(vals, half=0.005):
    """Range of the 2018 to 1980 ratio given two-decimal rounding."""
    a, b = vals["1980"], vals["2018"]
    return [(b - half) / (a + half), (b + half) / (a - half)]


def iv_block(r, endog_names):
    """Package 2SLS coefficients and standard errors."""
    return {
        n: {"coef": float(r["b"][i]), "se": float(r["se"][i])}
        for i, n in enumerate(endog_names)
    }


def main():
    """Run all estimates, markups, checks and write outputs."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load_data()
    X = characteristics(d)
    W0, W1, W2 = (controls(d, X, k) for k in (0, 1, 2))
    y, p, rxr = d["y"].values, d["p"].values, d["rxr"].values
    lsjg, mk = d["lsjg"].values, d["make2"].values
    res = {"gmy_reported": GMY_REPORTED, "data": data_summary(d)}
    g = GMY_REPORTED
    res["gmy_derived"] = {
        "sw_ratio_2018_1980": g["sw_mean_lerner"]["2018"]
        / g["sw_mean_lerner"]["1980"],
        "sw_change": g["sw_mean_lerner"]["2018"] - g["sw_mean_lerner"]["1980"],
        "median_ratio_2018_1980": g["median_lerner"]["2018"]
        / g["median_lerner"]["1980"],
        "logprice_ratio_2018_1980": g["logprice_sw_mean_lerner"]["2018"]
        / g["logprice_sw_mean_lerner"]["1980"],
        "logprice_change": g["logprice_sw_mean_lerner"]["2018"]
        - g["logprice_sw_mean_lerner"]["1980"],
        "sw_ratio_rounding_bounds": rounding_bounds(g["sw_mean_lerner"]),
        "logprice_ratio_rounding_bounds": rounding_bounds(
            g["logprice_sw_mean_lerner"]
        ),
    }
    pr = [res["data"][str(yr)]["sw_mean_price_usd"] for yr in (FIRST, LAST)]
    res["data"]["sw_mean_price_ratio_first_last"] = pr[0] / pr[1]

    # (a) Plain logit, GMY Table II specification.
    lg = tsls(y, p, rxr, W0, mk)
    lg_small = tsls(y, p, rxr, W0, mk, small=True)
    lg_model = tsls(y, p, rxr, W0, d["model"].values)
    lg_year = tsls(y, p, rxr, W0, d["year"].values)
    lg_trend = tsls(y, p, rxr, W1, mk)
    ols_lg = ols(y, np.column_stack([p, W0]), mk, small=True)
    fs = first_stage(p, rxr, W0, mk)
    alpha_l = -float(lg["b"][0])
    res["logit"] = {
        "price_coef": float(lg["b"][0]),
        "se_make_cluster": float(lg["se"][0]),
        "se_make_cluster_small_sample": float(lg_small["se"][0]),
        "se_model_cluster": float(lg_model["se"][0]),
        "se_year_cluster": float(lg_year["se"][0]),
        "n_clusters": {
            "make": lg["G"],
            "model": lg_model["G"],
            "year": lg_year["G"],
        },
        "ols_price_coef": float(ols_lg["b"][0]),
        "ols_price_se": float(ols_lg["se"][0]),
        "first_stage_rxr_coef": float(fs["coef"][0]),
        "first_stage_rxr_se": float(fs["se"][0]),
        "first_stage_F": fs["F"],
        "pass_through_eyex": float(
            np.mean(fs["coef"][0] * rxr / fs["fitted"])
        ),
        "mean_own_elasticity": float(np.mean(lg["b"][0] * (1 - d.s) * p)),
        "price_coef_with_nest_trends": float(lg_trend["b"][0]),
        "se_with_nest_trends": float(lg_trend["se"][0]),
    }

    # (c) Logit in log price (diagnostic for the price-level channel).
    lp = tsls(y, np.log(p), rxr, W0, mk)
    fs_lp = first_stage(np.log(p), rxr, W0, mk)
    alpha_lp = -float(lp["b"][0])
    res["logprice_logit"] = {
        "log_price_coef": float(lp["b"][0]),
        "se_make_cluster": float(lp["se"][0]),
        "first_stage_rxr_coef": float(fs_lp["coef"][0]),
        "first_stage_F": fs_lp["F"],
    }

    # (d) Log price with a separate coefficient for each five-year
    # period, instrumented as GMY describe for their period models:
    # rxr by period plus assembly-country dummies by period.
    per = period_index(d)
    P = (per[:, None] == np.arange(8)[None, :]).astype(float)
    lp_per = P * np.log(p)[:, None]
    z_rxr = P * rxr[:, None]
    cdum = pd.get_dummies(d["country1"], drop_first=True).astype(float)
    z_cty = np.column_stack(
        [z_rxr] + [cdum.values * P[:, [k]] for k in range(8)]
    )
    z_cty = z_cty[:, z_cty.std(0) > 0]
    n_inst = int(
        np.linalg.matrix_rank(np.column_stack([z_cty, W1]))
        - np.linalg.matrix_rank(W1)
    )
    lpp = tsls(y, lp_per, z_cty, W1, mk)
    lpp_rxr = tsls(y, lp_per, z_rxr, W1, mk)
    b_liml, k_liml = liml(y, lp_per, z_cty, W1)
    ols_per = ols(y, np.column_stack([lp_per, W1]), mk)
    alpha_per = -lpp["b"]
    alpha_year = {int(yr): float(alpha_per[k]) for yr, k in zip(d.year, per)}
    alpha_year_liml = {
        int(yr): float(-b_liml[k]) for yr, k in zip(d.year, per)
    }
    res["logprice_period"] = {
        "periods": [
            "1980-84",
            "1985-89",
            "1990-94",
            "1995-99",
            "2000-04",
            "2005-09",
            "2010-14",
            "2015-18",
        ],
        "n_excluded_instruments": n_inst,
        "n_assembly_countries": int(d["country1"].nunique()),
        "alpha_2sls": alpha_per,
        "se_2sls_make_cluster": lpp["se"],
        "alpha_liml": -b_liml,
        "liml_kappa": k_liml,
        "alpha_ols": -ols_per["b"][:8],
        "alpha_2sls_rxr_only": -lpp_rxr["b"],
        "se_2sls_rxr_only": lpp_rxr["se"],
        "sw_F_rxr_only": sw_fstats(lp_per, z_rxr, W1, mk),
        "dlog_alpha_2sls": float(np.log(alpha_per[7] / alpha_per[0])),
        "dlog_alpha_liml": float(np.log(b_liml[7] / b_liml[0])),
        "dlog_alpha_ols": float(np.log(ols_per["b"][7] / ols_per["b"][0])),
        "liml_over_2sls_alpha_first_last": [
            float(b_liml[0] / lpp["b"][0]),
            float(b_liml[7] / lpp["b"][7]),
        ],
    }

    # (b) Nested logit by 2SLS: main specification and sensitivity.
    Z = nest_instruments(d, X)
    endog = np.column_stack([p, lsjg])
    zmain = np.column_stack([rxr, Z["ln_n_nest"]])
    nl = tsls(y, endog, zmain, W1, mk)
    nl_model = tsls(y, endog, zmain, W1, d["model"].values)
    nl_year = tsls(y, endog, zmain, W1, d["year"].values)
    fs_p = first_stage(p, zmain, W1, mk)
    fs_l = first_stage(lsjg, zmain, W1, mk)
    alpha_n2, sigma_n2 = -float(nl["b"][0]), float(nl["b"][1])
    res["nested_2sls"] = {
        "spec": "instruments rxr and ln(products in nest-year); GMY "
        "controls plus nest-specific linear trends",
        "make_cluster": iv_block(nl, ["price", "ln_within_nest_share"]),
        "model_cluster_se": nl_model["se"].tolist(),
        "year_cluster_se": nl_year["se"].tolist(),
        "cov_price_sigma": float(nl["V"][0, 1]),
        "first_stage_price": {
            "coef": fs_p["coef"],
            "se": fs_p["se"],
            "F": fs_p["F"],
        },
        "first_stage_lsjg": {
            "coef": fs_l["coef"],
            "se": fs_l["se"],
            "F": fs_l["F"],
        },
        "sw_F_price_lsjg": sw_fstats(endog, zmain, W1, mk),
    }
    sets = {
        "count": ["ln_n_nest"],
        "rival_count": ["ln_n_rival"],
        "rival_chars": [
            "rival_sum_hp",
            "rival_sum_footprint",
            "rival_sum_curbweight",
            "rival_sum_mpg",
        ],
        "differentiation": ["diff_hp", "diff_footprint"],
    }
    grid = []
    for tname, Wt in (("none", W0), ("linear", W1), ("quadratic", W2)):
        for sname, cols in sets.items():
            zz = np.column_stack([rxr] + [Z[c] for c in cols])
            r = tsls(y, endog, zz, Wt, mk)
            grid.append(
                {
                    "nest_trends": tname,
                    "instruments": sname,
                    "price_coef": float(r["b"][0]),
                    "price_se": float(r["se"][0]),
                    "sigma": float(r["b"][1]),
                    "sigma_se": float(r["se"][1]),
                    "sw_F": sw_fstats(endog, zz, Wt, mk),
                }
            )
    res["nested_2sls_sensitivity"] = grid
    sig_vals = [r["sigma"] for r in grid]
    res["nested_2sls_sigma_range"] = [min(sig_vals), max(sig_vals)]
    res["nested_2sls_sensitivity_counts"] = {
        "n_specs": len(grid),
        "n_sigma_outside_unit_interval": sum(
            not (0.0 <= r["sigma"] < 1.0) for r in grid
        ),
        "n_positive_price_coef": sum(r["price_coef"] > 0 for r in grid),
        "n_min_sw_F_below_10": sum(min(r["sw_F"]) < 10 for r in grid),
    }

    # Nesting parameter from GMY's second-choice counts.
    counts, cache = load_counts(), year_cache(d)
    sigma_sc, target = fit_sigma(counts, cache)
    by_year = {}
    for yr in SC_YEARS:
        sg_y, tg_y = fit_sigma({yr: counts[yr]}, cache)
        by_year[str(yr)] = {
            "sigma": sg_y,
            "observed_same_nest": tg_y,
            "logit_implied_same_nest": sc_model_moment(
                0.0, {yr: counts[yr]}, cache
            ),
        }
    boot = np.array([fit_sigma(bc, cache)[0] for bc in load_boot_counts()])
    res["second_choice"] = {
        "n_respondents": float(sum(N.sum() for N in counts.values())),
        "observed_same_nest_share": target,
        "logit_implied_same_nest_share": sc_model_moment(0.0, counts, cache),
        "nested_2sls_implied_same_nest_share": sc_model_moment(
            float(np.clip(sigma_n2, 0.0, 0.99)), counts, cache
        ),
        "sigma_hat": sigma_sc,
        "sigma_boot_se": float(boot.std(ddof=1)),
        "sigma_boot_ci95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "n_boot": len(boot),
        "by_year": by_year,
        "sigma_by_year_range": [
            min(v["sigma"] for v in by_year.values()),
            max(v["sigma"] for v in by_year.values()),
        ],
    }

    # Price coefficient given the survey sigma (two-step 2SLS).
    b_y = tsls(y, p, rxr, W0, mk)["b"][0]
    b_l = tsls(lsjg, p, rxr, W0, mk)["b"][0]
    nls = tsls(y - sigma_sc * lsjg, p, rxr, W0, mk)
    alpha_ns = -float(nls["b"][0])
    var_cond = float(nls["V"][0, 0])
    se_two = np.sqrt(var_cond + b_l**2 * boot.var(ddof=1))
    res["nested_survey"] = {
        "price_coef": float(nls["b"][0]),
        "se_conditional_on_sigma": float(np.sqrt(var_cond)),
        "se_two_step": float(se_two),
        "dprice_coef_dsigma": float(-b_l),
        "sigma": sigma_sc,
        "first_stage_F": fs["F"],
        "check_linear_in_sigma": float(
            abs((b_y - sigma_sc * b_l) - nls["b"][0])
        ),
    }

    # Markups under every model.
    models = {
        "logit": ("logit", alpha_l, 0.0),
        "nested_2sls": ("nested", alpha_n2, sigma_n2),
        "nested_survey": ("nested", alpha_ns, sigma_sc),
        "logprice": ("logprice", alpha_lp, 0.0),
        "logprice_period": ("logprice", alpha_year, 0.0),
        "logprice_period_liml": ("logprice", alpha_year_liml, 0.0),
    }
    series = {"years": sorted(int(v) for v in d["year"].unique())}
    res["markups"] = {}
    for key, (kind, a, sgm) in models.items():
        m, el = model_markups(d, kind, a, sgm)
        rows = yearly_summary(d, m, el, a)
        series[key] = [r["sw_mean_lerner"] for r in rows]
        res["markups"][key] = {
            "alpha": a,
            "sigma": sgm,
            "trend": trend_stats(rows, kind == "logprice"),
            "mean_own_elasticity_all": float(np.mean(el)),
            "n_lerner_at_least_one_all": int(
                sum(r["n_lerner_at_least_one"] for r in rows)
            ),
            "by_year": rows,
        }
    pct = [
        v["trend"]["markup_usd_pct_change"]
        for k, v in res["markups"].items()
        if k != "logprice"
    ]
    res["summary"] = {
        "markup_usd_pct_change_range_excl_logprice": [min(pct), max(pct)]
    }

    # Sampling uncertainty of the 1980 and 2018 Lerner indices.
    unc = {}
    a_l = alpha_scaled_lerner(d, "logit", 0.0)
    dr = rng.normal(alpha_l, lg["se"][0], N_DRAWS)
    dr = dr[dr > 0]
    unc["logit"] = interval_summary(np.outer(1.0 / dr, a_l))
    unc["logit"]["share_draws_kept"] = len(dr) / N_DRAWS
    a_p = alpha_scaled_lerner(d, "logprice", 0.0)
    dr = rng.normal(alpha_lp, lp["se"][0], N_DRAWS)
    dr = dr[dr > 0]
    unc["logprice"] = interval_summary(np.outer(1.0 / dr, a_p))
    unc["logprice"]["share_draws_kept"] = len(dr) / N_DRAWS
    dr = rng.multivariate_normal(alpha_per, lpp["V"], N_DRAWS)[:, [0, 7]]
    dr = dr[(dr > 0).all(1)]
    unc["logprice_period"] = interval_summary(a_p[None, :] / dr)
    unc["logprice_period"]["share_draws_kept"] = len(dr) / N_DRAWS
    out = []
    for _ in range(N_DRAWS):
        sb = boot[rng.integers(len(boot))]
        ab = -(b_y - sb * b_l) + rng.normal(0.0, np.sqrt(var_cond))
        if ab > 0:
            out.append(alpha_scaled_lerner(d, "nested", sb) / ab)
    unc["nested_survey"] = interval_summary(out)
    unc["nested_survey"]["share_draws_kept"] = len(out) / N_DRAWS
    V2 = np.array(
        [[nl["V"][0, 0], -nl["V"][0, 1]], [-nl["V"][0, 1], nl["V"][1, 1]]]
    )
    draws = rng.multivariate_normal([alpha_n2, sigma_n2], V2, N_DRAWS)
    out = []
    for ab, sb in draws:
        if ab > 0 and 0.0 <= sb < 1.0:
            out.append(alpha_scaled_lerner(d, "nested", sb) / ab)
    unc["nested_2sls"] = interval_summary(out)
    unc["nested_2sls"]["share_draws_kept"] = len(out) / N_DRAWS
    res["uncertainty"] = unc
    res["uncertainty_note"] = (
        "Parametric draws, seed %d, %d draws; logit and log-price draw "
        "alpha from its 2SLS normal; survey nested logit draws sigma "
        "from GMY bootstrap counts and alpha given sigma; 2SLS nested "
        "logit draws (alpha, sigma) jointly and keeps 0<=sigma<1."
        % (SEED, N_DRAWS)
    )

    res["self_checks"] = self_checks(d, alpha_l, alpha_ns, sigma_sc)
    res["figure"] = "figures/appC/" + FIG_NAME
    make_figure(series, os.path.join(FIG_DIR, FIG_NAME))
    res = to_py(res)
    with open(os.path.join(OUT_DIR, "autos_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT_DIR, "autos_macros.tex"), "w") as f:
        f.write(latex_macros(res))
    print(
        json.dumps(
            to_py(
                {
                    k: res[k]
                    for k in (
                        "logit",
                        "logprice_logit",
                        "logprice_period",
                        "nested_2sls",
                        "second_choice",
                        "nested_survey",
                        "uncertainty",
                        "self_checks",
                    )
                }
            ),
            indent=1,
        )
    )
    for key in models:
        print(key, json.dumps(to_py(res["markups"][key]["trend"])))


if __name__ == "__main__":
    main()
