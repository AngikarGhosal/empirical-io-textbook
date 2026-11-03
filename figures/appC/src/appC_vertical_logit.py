"""Appendix C, Sections C.5 and C.6: vertical and logit demand on the simulated market.

Reads figures/appC/data/simulated_market.csv. Estimation uses only the observable
columns (market_id, product_id, owner, X0, X1, X2, Z, M, price, share_realized).
The true objects (mc_true, xi, omega, delta_true) are read only to print benchmarks.

C.5 (vertical model, exponential tastes, lambda = 1):
  inversion y_jt, clustered OLS of y on x, lambda = 2 scale check, supply under
  marginal-cost pricing and multi-product Nash pricing (tridiagonal Jacobian),
  complex-step derivative check, market-0 elasticity matrix and price ladder,
  single-product markup bound.
C.6 (logit):
  Berry inversion, instruments and their strength, OLS and 2SLS with clustered
  standard errors, Hansen J, milk-only 2SLS, product-effect test, market-0
  elasticities, cost recovery, market bootstrap (5,000 resamples, seed 2031),
  stacked-moment sandwich for the two-step estimator, joint GMM with a just- and an
  over-identified supply block, gap term r_jt from the random coefficients
  inversion (1,000 draws per market, seed 2030), its second-order approximation,
  and the within-market and full derivatives of the approximation's price component.

Writes ../ladder.pdf, ../gap.pdf, previews in this folder, and
appC_vertical_logit_output.json. Prints every number used in the text.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import optimize, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
DATA = FIG / "data" / "simulated_market.csv"
OBS = ["market_id", "product_id", "owner", "X0", "X1", "X2", "Z", "M", "price",
       "share_realized"]
TRUTH = ["mc_true", "xi", "omega"]
X_COLS = ["X0", "X1", "X2"]
W_COLS = X_COLS + ["Z"]
IV = ["Z", "x1_own", "x1_rival", "x2_own", "x2_rival", "n_own", "n_rival"]
SIGMA = np.diag([0.5, 0.2, 0.3, 0.5])
LAM = 1.0
N_BOOT, SEED_BOOT = 5000, 2031
N_DRAWS, SEED_EST = 1000, 2030
TOL = 1e-10
OUT = {}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 9, "axes.edgecolor": MUTED})


def show(key, val):
    """Print a number and store it for the JSON summary."""
    OUT[key] = val
    print(f"{key}: {val}")


# ----------------------------------------------------------------- regressions
def small_sample(n, k, g):
    return g / (g - 1) * (n - 1) / (n - k)


def cluster_meat(scores, groups):
    sc = pd.DataFrame(scores).groupby(groups).sum().to_numpy()
    return sc.T @ sc, sc.shape[0]


def ols(y, x, groups):
    """OLS with market-clustered SEs, factor G/(G-1)(N-1)/(N-K)."""
    bread = np.linalg.inv(x.T @ x)
    b = bread @ x.T @ y
    e = y - x @ b
    meat, g = cluster_meat(x * e[:, None], groups)
    v = small_sample(*x.shape, g) * bread @ meat @ bread
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return b, np.sqrt(np.diag(v)), v, e, r2


def tsls(y, x, z, groups):
    """2SLS with market-clustered SEs using the projected regressors."""
    xh = z @ np.linalg.lstsq(z, x, rcond=None)[0]
    bread = np.linalg.inv(xh.T @ xh)
    b = bread @ xh.T @ y
    e = y - x @ b
    meat, g = cluster_meat(xh * e[:, None], groups)
    v = small_sample(*x.shape, g) * bread @ meat @ bread
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return b, np.sqrt(np.diag(v)), v, e, r2


def wald_f(b, v, cols):
    bb, vv = b[cols], v[np.ix_(cols, cols)]
    return float(bb @ np.linalg.solve(vv, bb) / len(cols))


def hansen_j(y, x, z, groups):
    """Two-step GMM: W from 2SLS residuals (clustered, uncentered); J at step 2."""
    n = len(y)

    def est(w):
        a = x.T @ z @ w @ z.T @ x
        return np.linalg.solve(a, x.T @ z @ w @ z.T @ y)

    b1 = est(np.linalg.inv(z.T @ z / n))
    meat, _ = cluster_meat(z * (y - x @ b1)[:, None], groups)
    w2 = np.linalg.inv(meat / n)
    b2 = est(w2)
    gbar = z.T @ (y - x @ b2) / n
    j = float(n * gbar @ w2 @ gbar)
    df = z.shape[1] - x.shape[1]
    return j, df, float(stats.chi2.sf(j, df)), b2


# --------------------------------------------------------------- vertical model
def vertical_delta(p, s, lam):
    """y_jt: qualities (times lambda) from prices and shares, one market."""
    order = np.argsort(p)
    tail = np.cumsum(s[order][::-1])[::-1]          # bar S_(k)
    steps = np.diff(p[order], prepend=0.0)          # p_(k) - p_(k-1)
    delta = np.empty_like(p)
    delta[order] = np.cumsum(steps * -np.log1p(-tail) / lam)
    return delta


def vertical_shares(delta, p, lam):
    """Shares given qualities and prices (complex-safe)."""
    order = np.argsort(p.real)
    cut = np.diff(delta[order], prepend=0.0) / np.diff(p[order], prepend=0.0)
    if np.any(np.diff(cut.real) >= 0):
        raise ValueError("dominated product")
    bigf = 1.0 - np.exp(-lam * cut)
    s = np.empty_like(p)
    s[order] = bigf - np.append(bigf[1:], 0.0)
    return s, 1.0 - bigf[0]


def vertical_jacobian(p, s, lam):
    """D_jk = ds_j/dp_k (tridiagonal on the price ladder) and ds_0/dp_k."""
    n = len(p)
    order = np.argsort(p)
    tail = np.cumsum(s[order][::-1])[::-1]
    cut = -np.log1p(-tail) / lam
    g = lam * (1.0 - tail) * cut / np.diff(p[order], prepend=0.0)
    ds = np.diag(-g - np.append(g[1:], 0.0))
    ds += np.diag(g[1:], 1) + np.diag(g[1:], -1)
    jac = np.empty((n, n))
    jac[np.ix_(order, order)] = ds
    d0 = np.zeros(n)
    d0[order[0]] = g[0]
    return jac, d0, order, tail, cut, g


def ownership(owner):
    return (owner[:, None] == owner[None, :]).astype(float)


def implied_mc(p, s, jac, own):
    return p + np.linalg.solve(own * jac.T, s)


def elasticity_matrix(jac, d0, p, s, s0):
    e = jac * p[None, :] / s[:, None]
    return np.vstack([e, d0 * p / s0])


# ------------------------------------------------------------------ RC shares
def choice_probs(delta, xtilde, nu, sigma):
    v = delta[None, :] + nu @ sigma.T @ xtilde.T
    m = np.maximum(v.max(axis=1, keepdims=True), 0.0)
    ev = np.exp(v - m)
    return ev / (np.exp(-m) + ev.sum(axis=1, keepdims=True))


def invert_shares(s_obs, xtilde, nu, sigma, delta0):
    delta, log_s = delta0.copy(), np.log(s_obs)
    for it in range(1, 100_001):
        new = delta + log_s - np.log(choice_probs(delta, xtilde, nu, sigma).mean(0))
        if np.max(np.abs(new - delta)) < TOL:
            return new, it
        delta = new
    raise RuntimeError("contraction failed")


# ======================================================================== main
def main():
    raw = pd.read_csv(DATA)
    missing = [c for c in OBS if c not in raw.columns]
    if missing:
        raise KeyError(f"missing columns {missing}")
    df = raw[OBS + [c for c in TRUTH if c in raw.columns]].copy()
    df = df.sort_values(["market_id", "product_id"]).reset_index(drop=True)
    assert (df["share_realized"] > 0).all()
    mid = df["market_id"].to_numpy()
    markets = np.unique(mid)
    idx = [np.flatnonzero(mid == m) for m in markets]
    groups = mid
    n = len(df)
    p_all = df["price"].to_numpy()
    s_all = df["share_realized"].to_numpy()
    owner = df["owner"].to_numpy()
    x = df[X_COLS].to_numpy()
    w = df[W_COLS].to_numpy()
    s0_t = 1.0 - df.groupby("market_id")["share_realized"].transform("sum").to_numpy()
    show("n_obs", n)
    show("n_markets", len(markets))
    show("outside_share_min", float(s0_t.min()))
    show("outside_share_max", float(s0_t.max()))
    show("outside_share_mean_markets", float(np.mean([s0_t[r[0]] for r in idx])))

    # ================================================================ C.5
    print("\n===== C.5 vertical model =====")
    y_v = np.empty(n)
    y_v2 = np.empty(n)
    mc_nash = np.empty(n)
    mc_nash2 = np.empty(n)
    own_e = np.empty(n)
    mk_single = np.empty(n)
    bound_single = np.empty(n)
    bound_lower = np.empty(n)
    cs_err = 0.0
    roundtrip = 0.0
    colsum = 0.0
    max_deriv = 0.0
    max_cut = 0.0
    for rows in idx:
        p, s = p_all[rows], s_all[rows]
        y_v[rows] = vertical_delta(p, s, LAM)
        y_v2[rows] = vertical_delta(p, s, 2 * LAM)
        jac, d0, order, tail, cut, g = vertical_jacobian(p, s, LAM)
        max_cut = max(max_cut, cut.max())
        own = ownership(owner[rows])
        mc_nash[rows] = implied_mc(p, s, jac, own)
        jac2, _, _, _, _, _ = vertical_jacobian(p, s, 2 * LAM)
        mc_nash2[rows] = implied_mc(p, s, jac2, own)
        own_e[rows] = np.diag(jac) * p / s
        # single-product markups and the bound m <= s dp / (f nu)
        mk_single[rows] = -s / np.diag(jac)
        b = np.empty(len(p))
        g_up = np.append(g[1:], 0.0)
        b[order] = s[order] / np.maximum(g, g_up)
        bound_single[rows] = b
        bl = np.empty(len(p))
        bl[order] = np.diff(p[order], prepend=0.0) / (1.0 - tail)
        bound_lower[rows] = bl
        # derivative check by complex step
        s_fit, _ = vertical_shares(y_v[rows], p, LAM)
        roundtrip = max(roundtrip, np.max(np.abs(s_fit - s)))
        colsum = max(colsum, np.max(np.abs(jac.sum(0) + d0)))
        max_deriv = max(max_deriv, np.abs(jac).max(), np.abs(d0).max())
        for k in range(len(p)):
            h = np.zeros(len(p), dtype=complex)
            h[k] = 1e-20j
            sc, s0c = vertical_shares(y_v[rows].astype(complex), p + h, LAM)
            num = np.append(sc.imag, s0c.imag) / 1e-20
            cs_err = max(cs_err, np.max(np.abs(num - np.append(jac[:, k], d0[k]))))
    show("v_max_cutoff", float(max_cut))
    show("v_bound_minus_ln_s0_max", float((-np.log(s0_t)).max()))
    show("v_complex_step_max_abs_error", float(cs_err))
    show("v_max_abs_share_derivative", float(max_deriv))
    show("v_share_roundtrip_error", float(roundtrip))
    show("v_max_abs_column_sum", float(colsum))

    bv, sev, _, _, r2v = ols(y_v, x, groups)
    show("v_demand_coef", bv.tolist())
    show("v_demand_se", sev.tolist())
    show("v_demand_r2", float(r2v))
    bv2, _, _, _, r2v2 = ols(y_v2, x, groups)
    show("v_lambda2_coef_ratio", (bv2 / bv).tolist())
    show("v_lambda2_r2", float(r2v2))
    show("v_lambda2_max_mc_diff", float(np.max(np.abs(mc_nash2 - mc_nash))))
    y0 = y_v[idx[0]]
    show("v_quality_m0", dict(zip(df["product_id"].iloc[idx[0]].astype(int).astype(str), y0.round(4).tolist())))
    within_range = [y_v[r].max() - y_v[r].min() for r in idx]
    show("v_within_market_quality_range_mean", float(np.mean(within_range)))
    show("v_y_mean", float(y_v.mean()))
    show("v_y_sd", float(y_v.std(ddof=1)))

    bm, sem, _, _, r2m = ols(p_all, w, groups)
    show("v_supply_mcp_coef", bm.tolist())
    show("v_supply_mcp_se", sem.tolist())
    show("v_supply_mcp_r2", float(r2m))
    bn, sen, _, _, r2n = ols(mc_nash, w, groups)
    show("v_supply_nash_coef", bn.tolist())
    show("v_supply_nash_se", sen.tolist())
    show("v_supply_nash_r2", float(r2n))
    mk_n = p_all - mc_nash
    show("v_nash_markup_mean", float(mk_n.mean()))
    show("v_nash_markup_median", float(np.median(mk_n)))
    show("v_nash_markup_min", float(mk_n.min()))
    show("v_nash_markup_max", float(mk_n.max()))
    show("v_nash_mc_mean", float(mc_nash.mean()))
    show("v_nash_mc_min", float(mc_nash.min()))
    show("v_nash_n_negative_mc", int((mc_nash < 0).sum()))
    show("v_single_markup_mean", float(mk_single.mean()))
    show("v_single_bound_holds", bool(np.all(mk_single <= bound_single + 1e-12)))
    show("v_single_lowergap_bound_holds", bool(np.all(mk_single <= bound_lower + 1e-12)))
    show("v_single_bound_median", float(np.median(bound_single)))
    show("v_single_markup_median", float(np.median(mk_single)))
    same = np.concatenate([owner[r][np.argsort(p_all[r])][1:] == owner[r][np.argsort(p_all[r])][:-1]
                           for r in idx])
    show("v_adjacent_same_owner_pairs", int(same.sum()))
    show("v_markets_with_same_owner_neighbors", int(sum(
        np.any(owner[r][np.argsort(p_all[r])][1:] == owner[r][np.argsort(p_all[r])][:-1])
        for r in idx)))
    show("v_share_obs_nash_eq_single", float(np.mean(np.abs(mk_n - mk_single) < 1e-12)))
    gaps = np.concatenate([np.diff(np.sort(p_all[r])) for r in idx])
    show("v_gap_n", int(gaps.size))
    show("v_gap_median", float(np.median(gaps)))
    show("v_gap_below_0.01", int((gaps < 0.01).sum()))
    show("v_gap_below_0.001", int((gaps < 0.001).sum()))
    show("v_gap_min", float(gaps.min()))
    lowest_gap = np.concatenate([np.diff(np.sort(p_all[r]), prepend=0.0) for r in idx])
    show("v_cheapest_price_mean", float(np.mean([np.sort(p_all[r])[0] for r in idx])))
    show("v_own_elast_all_mean", float(own_e.mean()))
    show("v_own_elast_all_median", float(np.median(own_e)))
    show("v_own_elast_all_min", float(own_e.min()))
    show("v_own_elast_all_max", float(own_e.max()))
    if "mc_true" in df:
        true_mk = p_all - df["mc_true"].to_numpy()
        show("true_markup_mean", float(true_mk.mean()))
        show("true_markup_min", float(true_mk.min()))
        show("true_markup_max", float(true_mk.max()))
        show("true_mc_mean", float(df["mc_true"].mean()))
        show("corr_vnash_markup_true_markup", float(np.corrcoef(mk_n, true_mk)[0, 1]))
        show("corr_vnash_mc_true_mc", float(np.corrcoef(mc_nash, df["mc_true"])[0, 1]))
        show("corr_price_true_mc", float(np.corrcoef(p_all, df["mc_true"])[0, 1]))

    # market 0 detail
    r0 = idx[0]
    p0, s0v = p_all[r0], s_all[r0]
    prod0 = df["product_id"].iloc[r0].astype(int).to_numpy()
    jac0, d00, order0, tail0, cut0, g0 = vertical_jacobian(p0, s0v, LAM)
    e0 = elasticity_matrix(jac0, d00, p0, s0v, 1.0 - s0v.sum())
    show("m0_products", prod0.tolist())
    show("m0_outside_share", float(1.0 - s0v.sum()))
    show("v_elast_m0", np.round(e0, 2).tolist())
    ladder = []
    for k, j in enumerate(order0):
        ladder.append({
            "rank": k + 1, "product": int(prod0[j]), "owner": int(owner[r0][j]),
            "price": round(float(p0[j]), 4),
            "gap": round(float(p0[j] - (p0[order0[k - 1]] if k else 0.0)), 4),
            "share": round(float(s0v[j]), 4), "Sbar": round(float(tail0[k]), 4),
            "cutoff": round(float(cut0[k]), 4), "quality": round(float(y_v[r0][j]), 4),
            "own_elast": round(float(e0[j, j]), 2),
            "nash_markup": round(float(mk_n[r0][j]), 4),
            "single_markup": round(float(mk_single[r0][j]), 4),
            "single_bound": round(float(bound_single[r0][j]), 4),
            "lower_gap_bound": round(float(bound_lower[r0][j]), 4),
            "true_markup": round(float(true_mk[r0][j]), 4) if "mc_true" in df else None,
        })
    show("v_ladder_m0", ladder)
    show("v_outside_elast_m0", float(e0[-1, order0[0]]))

    # ladder figure
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.9), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.6]})
    ax = axes[0]
    ps = p0[order0]
    ax.hlines(0, ps.min() - 0.03, ps.max() + 0.03, color=MUTED, lw=0.8)
    for k, j in enumerate(order0):
        ax.plot(p0[j], 0, "o", ms=8, color=BLUE, mec="white", mew=1.2, zorder=3)
        ax.annotate(f"{prod0[j]}", (p0[j], 0), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
        if k:
            mid_p = 0.5 * (p0[j] + p0[order0[k - 1]])
            gap = p0[j] - p0[order0[k - 1]]
            ax.annotate(f"{gap:.3f}", (mid_p, 0), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=7, color=MUTED)
    ax.set_ylim(-0.9, 0.9)
    ax.set_yticks([])
    ax.set_title("(a) price ladder in market 0: product labels above, price gaps below",
                 loc="left", fontsize=9, color=INK)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax = axes[1]
    for j in order0:
        ax.vlines(p0[j], e0[j, j], 0, color=ORANGE, lw=2)
        ax.plot(p0[j], e0[j, j], "o", ms=7, color=ORANGE, mec="white", mew=1.0, zorder=3)
        off = (-30, -3) if prod0[j] == 4 else (5, -3)
        ax.annotate(f"${e0[j, j]:.1f}$", (p0[j], e0[j, j]), xytext=off,
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("own-price elasticity", color=MUTED)
    ax.set_xlabel(r"price $p_{j0}$", color=MUTED)
    ax.set_ylim(-400, 20)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("(b) own-price elasticity implied by the vertical model",
                 loc="left", fontsize=9, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "ladder.pdf")
    fig.savefig(HERE / "ladder_preview.png", dpi=130)
    plt.close(fig)

    # ================================================================ C.6
    print("\n===== C.6 logit =====")
    firm = df.groupby(["market_id", "owner"])
    mkt = df.groupby("market_id")
    for col in ("X1", "X2"):
        fs = firm[col].transform("sum")
        df[col.lower() + "_own"] = fs - df[col]
        df[col.lower() + "_rival"] = mkt[col].transform("sum") - fs
    fn = firm["X0"].transform("size")
    df["n_own"] = fn - 1
    df["n_rival"] = mkt["X0"].transform("size") - fn
    zmat = df[X_COLS + IV].to_numpy(float)
    show("iv_rank_full", int(np.linalg.matrix_rank(zmat)) == zmat.shape[1])
    vif = {}
    for j, name in enumerate(X_COLS + IV):
        if name not in IV:
            continue
        rest = np.delete(zmat, j, axis=1)
        fit = rest @ np.linalg.lstsq(rest, zmat[:, j], rcond=None)[0]
        vif[name] = float(np.sum((zmat[:, j] - zmat[:, j].mean()) ** 2)
                          / np.sum((zmat[:, j] - fit) ** 2))
    show("iv_vif", vif)
    show("corr_x1own_nown", float(df["x1_own"].corr(df["n_own"])))
    show("corr_x1rival_nrival", float(df["x1_rival"].corr(df["n_rival"])))
    show("iv_describe", df[IV].describe().loc[["mean", "std", "min", "max"]].round(3).to_dict())

    y_l = np.log(s_all) - np.log(s0_t)
    xp = np.column_stack([x, p_all])
    b_ols, se_ols, _, _, r2_ols = ols(y_l, xp, groups)
    b_iv, se_iv, v_iv, e_iv, r2_iv = tsls(y_l, xp, zmat, groups)
    alpha = float(-b_iv[3])
    show("l_ols_coef", b_ols.tolist())
    show("l_ols_se", se_ols.tolist())
    show("l_ols_r2", float(r2_ols))
    show("l_iv_coef", b_iv.tolist())
    show("l_iv_se", se_iv.tolist())
    show("l_iv_r2", float(r2_iv))
    show("l_alpha", alpha)
    show("l_t_alpha_vs_3", float((alpha - 3.0) / se_iv[3]))
    # first stage
    b_fs, se_fs, v_fs, e_fs, r2_fs = ols(p_all, zmat, groups)
    show("fs_coef", dict(zip(X_COLS + IV, b_fs.round(4).tolist())))
    show("fs_se", dict(zip(X_COLS + IV, se_fs.round(4).tolist())))
    show("fs_r2", float(r2_fs))
    ex = list(range(3, 3 + len(IV)))
    show("fs_cluster_F_all", wald_f(b_fs, v_fs, ex))
    show("fs_cluster_F_nonZ", wald_f(b_fs, v_fs, ex[1:]))
    show("fs_cluster_F_nonZ_p", float(stats.f.sf(wald_f(b_fs, v_fs, ex[1:]), 6, 99)))
    _, _, _, e_r, _ = ols(p_all, x, groups)
    ssr_u, ssr_r = e_fs @ e_fs, e_r @ e_r
    f_homo = (ssr_r - ssr_u) / len(IV) / (ssr_u / (n - zmat.shape[1]))
    show("fs_homo_F", float(f_homo))
    show("fs_partial_r2", float((ssr_r - ssr_u) / ssr_r))
    # first stage with Z only
    b_fz, _, v_fz, _, _ = ols(p_all, w, groups)
    show("fs_cluster_F_Zonly", wald_f(b_fz, v_fz, [3]))
    # Hansen J and two-step GMM
    j, jdf, jp, b_gmm = hansen_j(y_l, xp, zmat, groups)
    show("hansen_J", j)
    show("hansen_df", jdf)
    show("hansen_p", jp)
    show("gmm2_alpha", float(-b_gmm[3]))
    # milk-only 2SLS
    b_z, se_z, _, _, r2_z = tsls(y_l, xp, w, groups)
    show("l_ivZ_coef", b_z.tolist())
    show("l_ivZ_se", se_z.tolist())
    show("l_ivZ_r2", float(r2_z))
    # check the hand-coded clustered 2SLS against a packaged routine
    try:
        from linearmodels.iv import IV2SLS
        diffs = []
        for zz, bh, sh in ((zmat, b_iv, se_iv), (w, b_z, se_z)):
            res = IV2SLS(y_l, x, p_all[:, None], zz[:, 3:]).fit(
                cov_type="clustered", clusters=groups, debiased=True)
            diffs += [np.max(np.abs(res.params.to_numpy() - bh)),
                      np.max(np.abs(res.std_errors.to_numpy() - sh))]
        show("linearmodels_max_abs_diff_coef_se", float(max(diffs)))
    except ImportError:
        show("linearmodels_max_abs_diff_coef_se", None)
    # product-effect test on 2SLS residuals
    dummies = pd.get_dummies(df["product_id"]).to_numpy(float)
    fit_w = w @ np.linalg.lstsq(w, e_iv, rcond=None)[0]
    fit_d = dummies @ np.linalg.lstsq(dummies, e_iv, rcond=None)[0]
    ssr_w, ssr_d = np.sum((e_iv - fit_w) ** 2), np.sum((e_iv - fit_d) ** 2)
    df1 = dummies.shape[1] - np.linalg.matrix_rank(w)
    df2 = n - dummies.shape[1]
    fpe = (ssr_w - ssr_d) / df1 / (ssr_d / df2)
    show("product_effect_F", float(fpe))
    show("product_effect_df", [int(df1), int(df2)])
    show("product_effect_p", float(stats.f.sf(fpe, df1, df2)))
    # implied own elasticities
    own_l = -alpha * p_all * (1 - s_all)
    show("l_own_elast_all_mean", float(own_l.mean()))
    show("l_own_elast_all_min", float(own_l.min()))
    show("l_own_elast_all_max", float(own_l.max()))
    show("l_ols_n_inelastic", int(np.sum(-b_ols[3] * p_all * (1 - s_all) < 1)))
    # market 0 elasticities
    jac_l = -alpha * (np.diag(s0v) - np.outer(s0v, s0v))
    d0_l = alpha * (1 - s0v.sum()) * s0v
    el0 = elasticity_matrix(jac_l, d0_l, p0, s0v, 1 - s0v.sum())
    show("l_elast_m0", np.round(el0, 2).tolist())
    show("l_elast_m0_raw_col_entries", (alpha * p0 * s0v).round(4).tolist())
    show("l_diversion_outside_m0", ((1 - s0v.sum()) / (1 - s0v)).round(4).tolist())
    show("v_diversion_outside_m0", float(d00[order0[0]] / -jac0[order0[0], order0[0]]))

    # cost recovery
    sf = df.groupby(["market_id", "owner"])["share_realized"].transform("sum").to_numpy()
    kfac = 1.0 / (1.0 - sf)
    mc_l = p_all - kfac / alpha
    # check equality with the matrix formula in every market
    chk = 0.0
    for rows in idx:
        jl = -alpha * (np.diag(s_all[rows]) - np.outer(s_all[rows], s_all[rows]))
        chk = max(chk, np.max(np.abs(implied_mc(p_all[rows], s_all[rows], jl,
                                                ownership(owner[rows])) - mc_l[rows])))
    show("l_mc_closed_form_vs_matrix", float(chk))
    mk_l = p_all - mc_l
    show("l_markup_mean", float(mk_l.mean()))
    show("l_markup_min", float(mk_l.min()))
    show("l_markup_max", float(mk_l.max()))
    show("l_markup_m0", dict(zip([str(v) for v in prod0], mk_l[r0].round(4).tolist())))
    show("l_Sf_max", float(sf.max()))
    show("l_n_negative_mc", int((mc_l < 0).sum()))
    if "mc_true" in df:
        show("corr_logit_markup_true", float(np.corrcoef(mk_l, true_mk)[0, 1]))
        show("corr_logit_mc_true_mc", float(np.corrcoef(mc_l, df["mc_true"])[0, 1]))
    if "mc_true" in df and "omega" in df:
        # gamma_hat - gamma = OLS(omega) + OLS(true markup - model markup)
        proj = np.linalg.solve(w.T @ w, w.T)
        show("decomp_sampling_part", (proj @ df["omega"].to_numpy()).tolist())
        show("decomp_markup_part_mcp", (proj @ true_mk).tolist())
        show("decomp_markup_part_vnash", (proj @ (true_mk - mk_n)).tolist())
        show("decomp_markup_part_logit", (proj @ (true_mk - mk_l)).tolist())
        show("true_markup_by_organic", [float(true_mk[x[:, 2] == 0].mean()),
                                        float(true_mk[x[:, 2] == 1].mean())])
        show("logit_markup_by_organic", [float(mk_l[x[:, 2] == 0].mean()),
                                         float(mk_l[x[:, 2] == 1].mean())])
    b_s, se_s, _, e_s, r2_s = ols(mc_l, w, groups)
    show("l_supply_coef", b_s.tolist())
    show("l_supply_se_naive", se_s.tolist())
    show("l_supply_r2", float(r2_s))
    # market bootstrap (same draw order as the original code)
    rng = np.random.default_rng(SEED_BOOT)
    draws = np.empty((N_BOOT, 8))
    for bb in range(N_BOOT):
        pick = rng.integers(0, len(idx), size=len(idx))
        rows = np.concatenate([idx[m] for m in pick])
        xh = zmat[rows] @ np.linalg.lstsq(zmat[rows], xp[rows], rcond=None)[0]
        coef = np.linalg.lstsq(xh, y_l[rows], rcond=None)[0]
        mc_b = p_all - kfac / (-coef[3])
        gam = np.linalg.lstsq(w[rows], mc_b[rows], rcond=None)[0]
        draws[bb] = np.append(coef, gam)
    draws[:, 3] = -draws[:, 3]
    bse = draws.std(axis=0, ddof=1)
    show("boot_se_demand", bse[:4].tolist())
    show("boot_se_supply", bse[4:].tolist())
    lo, hi = np.percentile(draws[:, 4:], [2.5, 97.5], axis=0)
    show("boot_ci_supply_lo", lo.tolist())
    show("boot_ci_supply_hi", hi.tolist())
    dev2 = (draws - draws.mean(0)) ** 2
    show("boot_se_mc_error", (dev2.std(0, ddof=1) / (2 * bse * np.sqrt(N_BOOT))).tolist())
    # stacked-moment sandwich for the sequential estimator
    xh = zmat @ np.linalg.lstsq(zmat, xp, rcond=None)[0]
    a_mat = np.linalg.inv(xh.T @ xh)                     # (Xh'Xh)^-1
    if_d = (xh * e_iv[:, None]) @ a_mat.T                # influence of theta_d (per obs)
    if_alpha = -if_d[:, 3]                               # alpha = -coef
    ww_inv = np.linalg.inv(w.T @ w)
    dgam_dalpha = ww_inv @ (w.T @ kfac) / alpha ** 2     # d gamma_hat / d alpha
    if_s = (w * e_s[:, None]) @ ww_inv.T + np.outer(if_alpha, dgam_dalpha)
    meat, g_cl = cluster_meat(if_s, groups)
    v_stack = small_sample(n, 4, g_cl) * meat
    show("stacked_se_supply", np.sqrt(np.diag(v_stack)).tolist())
    show("dgamma0_dalpha", float(dgam_dalpha[0]))
    show("delta_method_markup_se", float(mk_l.mean() / alpha * se_iv[3]))

    # supply side alone (Berry 1994, eq. 31, multi-product version):
    # p = x g1 + g2 w + (1/alpha) k + omega, k = 1/(1 - S_f) instrumented by
    # the six competition instruments
    xk = np.column_stack([w, kfac])
    b_so, se_so, _, _, _ = tsls(p_all, xk, zmat, groups)
    show("supply_only_coef", b_so.tolist())
    show("supply_only_se", se_so.tolist())
    show("supply_only_alpha", float(1.0 / b_so[4]))
    show("supply_only_alpha_se_delta", float(se_so[4] / b_so[4] ** 2))
    b_kf, _, v_kf, _, _ = ols(kfac, zmat, groups)
    show("supply_only_fs_F_k", wald_f(b_kf, v_kf, list(range(4, zmat.shape[1]))))
    show("k_min_max_mean", [float(kfac.min()), float(kfac.max()), float(kfac.mean())])

    # joint GMM
    def joint_gmm(zs):
        zd = zmat
        nd, ns_ = zd.shape[1], zs.shape[1]
        bmat = np.zeros((nd + ns_, 7))
        bmat[:nd, :3] = zd.T @ x / n
        bmat[nd:, 3:] = zs.T @ w / n

        def bvec(a):
            return np.concatenate([zd.T @ (y_l + a * p_all), zs.T @ (p_all - kfac / a)]) / n

        def conc(a, wm):
            bv_ = bvec(a)
            phi = np.linalg.solve(bmat.T @ wm @ bmat, bmat.T @ wm @ bv_)
            gbar = bv_ - bmat @ phi
            return float(gbar @ wm @ gbar), phi

        def solve(wm):
            grid = np.linspace(0.8, 8.0, 361)
            q = [conc(a, wm)[0] for a in grid]
            a0 = grid[int(np.argmin(q))]
            res = optimize.minimize_scalar(lambda a: conc(a, wm)[0],
                                           bounds=(max(0.5, a0 - 0.1), a0 + 0.1),
                                           method="bounded", options={"xatol": 1e-10})
            return res.x, conc(res.x, wm)[1]

        def resid(a, phi):
            xi_ = y_l + a * p_all - x @ phi[:3]
            om_ = p_all - kfac / a - w @ phi[3:]
            return xi_, om_

        w1 = np.zeros((nd + ns_, nd + ns_))
        w1[:nd, :nd] = np.linalg.inv(zd.T @ zd / n)
        w1[nd:, nd:] = np.linalg.inv(zs.T @ zs / n)
        a1, phi1 = solve(w1)
        xi1, om1 = resid(a1, phi1)
        sc = np.column_stack([zd * xi1[:, None], zs * om1[:, None]])
        meat_, _ = cluster_meat(sc, groups)
        w2 = np.linalg.inv(meat_ / n)
        a2, phi2 = solve(w2)
        xi2, om2 = resid(a2, phi2)
        gbar = np.concatenate([zd.T @ xi2, zs.T @ om2]) / n
        jstat = float(n * gbar @ w2 @ gbar)
        # Jacobian wrt (beta, alpha, gamma)
        gmat = np.zeros((nd + ns_, 8))
        gmat[:nd, :3] = -zd.T @ x / n
        gmat[:nd, 3] = zd.T @ p_all / n
        gmat[nd:, 3] = zs.T @ (kfac / a2 ** 2) / n
        gmat[nd:, 4:] = -zs.T @ w / n
        vcov = np.linalg.inv(gmat.T @ w2 @ gmat) / n
        dfj = nd + ns_ - 8
        return {"alpha_step1": float(a1), "beta_step1": phi1[:3].tolist(),
                "gamma_step1": phi1[3:].tolist(),
                "alpha": float(a2), "beta": phi2[:3].tolist(), "gamma": phi2[3:].tolist(),
                "se": np.sqrt(np.diag(vcov)).tolist(), "J": jstat, "J_df": dfj,
                "J_p": float(stats.chi2.sf(jstat, dfj)) if dfj > 0 else None}

    ja = joint_gmm(w)
    show("joint_gmm_just_identified", ja)
    zs_b = df[W_COLS + IV[1:]].to_numpy(float)
    jb = joint_gmm(zs_b)
    show("joint_gmm_overidentified", jb)

    # ----------------------------------------------------- gap term r_jt
    print("\n===== C.6(e) gap term =====")
    rng = np.random.default_rng(SEED_EST)
    nu = rng.standard_normal((len(idx), N_DRAWS, 4))
    xt_all = np.column_stack([x, p_all])
    delta = np.empty(n)
    iters = []
    for t, rows in enumerate(idx):
        delta[rows], it = invert_shares(s_all[rows], xt_all[rows], nu[t], SIGMA, y_l[rows])
        iters.append(it)
    show("rc_iterations_min_mean_max", [int(min(iters)), float(np.mean(iters)), int(max(iters))])
    r_gap = y_l - delta
    show("r_all_positive", bool(np.all(r_gap > 0)))
    show("r_min", float(r_gap.min()))
    show("r_max", float(r_gap.max()))
    show("r_mean", float(r_gap.mean()))
    show("r_corr_price", float(np.corrcoef(r_gap, p_all)[0, 1]))
    show("r_m0", dict(zip([str(v) for v in prod0], r_gap[r0].round(4).tolist())))
    b_rc, se_rc, _, _, _ = tsls(delta, xp, zmat, groups)
    b_r, _, _, _, _ = tsls(r_gap, xp, zmat, groups)
    show("rc_iv_coef", b_rc.tolist())
    show("rc_iv_se", se_rc.tolist())
    show("rc_alpha", float(-b_rc[3]))
    show("b_r_price", float(b_r[3]))
    show("identity_gap", float(abs((-b_rc[3]) - b_r[3] - alpha)))
    show("b_r_full", b_r.tolist())
    b_r_ols, _, _, _, _ = ols(r_gap, xp, groups)
    show("b_r_ols_price", float(b_r_ols[3]))
    show("ols_rc_alpha", float(-ols(delta, xp, groups)[0][3]))
    # second-order approximation r ~ 1/2 x Sigma^2 x' - x Sigma^2 xbar'
    sig2 = np.diag(SIGMA) ** 2
    r_approx = np.empty(n)
    for rows in idx:
        xt = xt_all[rows]
        xbar = s_all[rows] @ xt
        r_approx[rows] = 0.5 * (xt ** 2) @ sig2 - xt @ (sig2 * xbar)
    show("r_approx_corr", float(np.corrcoef(r_gap, r_approx)[0, 1]))
    show("r_approx_mean", float(r_approx.mean()))
    show("r_approx_m0", dict(zip([str(v) for v in prod0], r_approx[r0].round(4).tolist())))
    b_ra, _, _, _, _ = tsls(r_approx, xp, zmat, groups)
    show("b_r_approx_price", float(b_ra[3]))
    # price-only component of the approximation and its 2SLS slope
    r_price = np.empty(n)
    for rows in idx:
        pbar = s_all[rows] @ p_all[rows]
        r_price[rows] = sig2[3] * (0.5 * p_all[rows] ** 2 - p_all[rows] * pbar)
    b_rp, _, _, _, _ = tsls(r_price, xp, zmat, groups)
    show("b_r_price_component", float(b_rp[3]))
    show("sigma_p2_times_mean_price", float(sig2[3] * p_all.mean()))
    # share-weighted inside price pbar_t, the within-market slope of the price
    # component, sigma_p^2 (p - pbar), and its full derivative in p_jt when pbar_t
    # moves with p_jt through logit share derivatives at the RC price coefficient:
    # sigma_p^2 [(p - pbar)(1 + alpha p s) - p s]
    pbar_obs = np.empty(n)
    for rows in idx:
        pbar_obs[rows] = s_all[rows] @ p_all[rows]
    gap_p = p_all - pbar_obs
    a_rc = float(-b_rc[3])
    full_der = gap_p * (1 + a_rc * p_all * s_all) - p_all * s_all
    show("pbar_min_mean_max", [float(pbar_obs.min()), float(pbar_obs.mean()),
                               float(pbar_obs.max())])
    show("p_minus_pbar_min", float(gap_p.min()))
    show("full_derivative_price_component_min_over_sigma_p2", float(full_der.min()))
    show("sigma_p2_times_mean_p_minus_pbar", float(sig2[3] * gap_p.mean()))
    show("price_mean", float(p_all.mean()))
    show("price_min", float(p_all.min()))
    show("price_max", float(p_all.max()))
    # a perfect-fit check of the RC model: alpha recovered without the gap
    if "xi" in df:
        show("corr_xi_hat_rc_true", float(np.corrcoef(
            delta - xp @ b_rc, df["xi"].to_numpy())[0, 1]))

    # true own elasticities in market 0 (firm draws, seed 2026, true parameters)
    if "xi" in df:
        nu_f = np.random.default_rng(2026).standard_normal((len(idx), 100, 4))[0]
        beta_t, alpha_t = np.array([-1.0, 0.2, 0.15]), 3.0
        d_true = x[r0] @ beta_t - alpha_t * p0 + df["xi"].to_numpy()[r0]
        sij = choice_probs(d_true, xt_all[r0], nu_f, SIGMA)
        a_i = alpha_t - nu_f @ SIGMA[3, :]
        wgt = a_i[:, None] * sij
        jac_t = wgt.T @ sij / len(nu_f) - np.diag(wgt.mean(0))
        s_t = sij.mean(0)
        show("true_own_elast_m0", dict(zip([str(v) for v in prod0],
                                           (np.diag(jac_t) * p0 / s_t).round(3).tolist())))

    # gap figure
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    others = np.ones(n, bool)
    others[r0] = False
    ax.scatter(p_all[others], r_gap[others], s=10, color=BLUE, alpha=0.45, lw=0,
               label="all other product-markets (647)")
    ax.scatter(p_all[r0], r_gap[r0], s=46, color=ORANGE, edgecolor="white", lw=1.0,
               zorder=3, label="market 0")
    for j in range(len(r0)):
        ax.annotate(str(prod0[j]), (p0[j], r_gap[r0][j]), xytext=(5, -3),
                    textcoords="offset points", fontsize=7, color=INK)
    order = np.argsort(p_all)
    xs = np.linspace(p_all.min(), p_all.max(), 50)
    cf = np.polyfit(p_all, r_gap, 1)
    ax.plot(xs, np.polyval(cf, xs), color=MUTED, lw=1.0, ls="--",
            label="least-squares line")
    ax.set_xlabel(r"price $p_{jt}$", color=MUTED)
    ax.set_ylabel(r"gap term $r_{jt}$", color=MUTED)
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "gap.pdf")
    fig.savefig(HERE / "gap_preview.png", dpi=130)
    plt.close(fig)
    show("gap_fig_ols_slope", float(cf[0]))

    (HERE / "appC_vertical_logit_output.json").write_text(
        json.dumps(OUT, indent=1, default=float))


if __name__ == "__main__":
    main()
