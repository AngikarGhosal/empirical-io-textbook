"""Appendix C.7: random coefficients estimation with Sigma known, and a
comparison of the vertical, logit and random coefficients models.

Reads the simulated market of C.4 (figures/appC/data/simulated_market.csv)
and the firm draws (figures/appC/data/firm_draws.npz). Only the
observables (market, product, owner, X0-X2, Z, M, price, realized share)
enter estimation. True objects (delta_true, xi, mc_true, firm draws) are
used only as benchmarks.

Writes
  figures/appC/data/est_draws.npz      estimation draws (seed 2030)
  figures/appC/src/appC_rc_results.json every number used in C.7
  figures/appC/own_elas.pdf            own-elasticity comparison figure
and prints every number.

All regressions are coded by hand with numpy: 2SLS, market-clustered
sandwich with the factor G/(G-1) (N-1)/(N-K), cluster-robust first-stage
F, two-step clustered GMM and Hansen's J, market bootstrap.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
APPC = HERE.parent
DATA = APPC / "data"

X_COLS = ["X0", "X1", "X2"]
W_COLS = X_COLS + ["Z"]
IV_CANDIDATES = ["Z", "x1_own", "x1_rival", "x2_own", "x2_rival", "n_own", "n_rival"]

TRUE_BETA = np.array([-1.0, 0.2, 0.15])
TRUE_ALPHA = 3.0
TRUE_GAMMA = np.array([0.2, 0.25, 0.3, 0.5])
SIGMA = np.diag([0.5, 0.2, 0.3, 0.5])  # known in C.7

N_DRAWS_EST = 1000
SEED_EST = 2030
N_DRAWS_CHECK = 100_000
SEED_CHECK = 2032
SEED_BOOT = 2031
N_BOOT = 5000
TOL = 1e-10
MAX_ITER = 100_000


# ---------------------------------------------------------------------
# Demand primitives
# ---------------------------------------------------------------------
def choice_probs(delta, xtilde, nu, sigma):
    """Logit probabilities s_ij for every draw (rows) and product (cols).

    Utility delta_j + xtilde_j Sigma nu_i, outside good 0. Subtracts
    max(0, max_k V_ik) from every utility of draw i before exponentiating.
    """
    v = delta[None, :] + nu @ sigma.T @ xtilde.T
    m = np.maximum(v.max(axis=1, keepdims=True), 0.0)
    ev = np.exp(v - m)
    return ev / (np.exp(-m) + ev.sum(axis=1, keepdims=True))


def shares_delta(delta, xtilde, nu, sigma):
    return choice_probs(delta, xtilde, nu, sigma).mean(axis=0)


def invert_shares(s_obs, xtilde, nu, sigma, delta0, tol=TOL):
    """BLP contraction delta <- delta + ln s_obs - ln s(delta)."""
    delta, log_s = delta0.copy(), np.log(s_obs)
    for it in range(1, MAX_ITER + 1):
        new = delta + log_s - np.log(shares_delta(delta, xtilde, nu, sigma))
        if np.max(np.abs(new - delta)) < tol:
            return new, it
        delta = new
    raise RuntimeError("contraction did not converge")


def rc_parts(delta, xtilde, nu, sigma):
    """Pieces of the RC Jacobian that are linear in alpha.

    D(alpha) = alpha * A + B with
    A_jk = mean_i[-s_ij (1{j=k} - s_ik)],
    B_jk = mean_i[(Sigma nu_i)_4 s_ij (1{j=k} - s_ik)].
    d0(alpha) = alpha * a0 + b0, d0_k = ds_0/dp_k.
    """
    sij = choice_probs(delta, xtilde, nu, sigma)
    si0 = 1.0 - sij.sum(axis=1)
    sp = nu @ sigma[3, :]
    n = len(nu)
    A = sij.T @ sij / n - np.diag(sij.mean(axis=0))
    B = -((sp[:, None] * sij).T @ sij / n - np.diag((sp[:, None] * sij).mean(axis=0)))
    a0 = (sij * si0[:, None]).mean(axis=0)
    b0 = -((sp[:, None] * sij) * si0[:, None]).mean(axis=0)
    return A, B, a0, b0, sij.mean(axis=0), si0.mean()


def rc_jacobian(delta, xtilde, nu, sigma, alpha):
    A, B, a0, b0, s, s0 = rc_parts(delta, xtilde, nu, sigma)
    return alpha * A + B, alpha * a0 + b0, s, s0


def elasticity_matrix(jac, d0, p, s, s0):
    e = jac * p[None, :] / s[:, None]
    return np.vstack([e, d0 * p / s0])


def ownership(owner):
    return (owner[:, None] == owner[None, :]).astype(float)


def implied_mc(p, s, jac, own):
    """Nash FOC s + (Omega .* D')(p - mc) = 0 solved for mc."""
    return p + np.linalg.solve(own * jac.T, s)


# ---------------------------------------------------------------------
# Vertical model (for the comparison only)
# ---------------------------------------------------------------------
def vertical_delta(p, s, lam=1.0):
    order = np.argsort(p)
    tail = np.cumsum(s[order][::-1])[::-1]
    steps = np.diff(p[order], prepend=0.0)
    delta = np.empty_like(p)
    delta[order] = np.cumsum(steps * -np.log1p(-tail) / lam)
    return delta


def vertical_jacobian(p, s, lam=1.0):
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
    return jac, d0


# ---------------------------------------------------------------------
# Linear estimators
# ---------------------------------------------------------------------
def cluster_sums(m, groups):
    codes, uniq = pd.factorize(np.asarray(groups))
    out = np.zeros((len(uniq), m.shape[1]))
    np.add.at(out, codes, m)
    return out


def tsls(y, x, z, groups):
    """2SLS coefficients and clustered SEs (G/(G-1)(N-1)/(N-K))."""
    xhat = z @ np.linalg.lstsq(z, x, rcond=None)[0]
    bread = np.linalg.inv(xhat.T @ xhat)
    b = bread @ xhat.T @ y
    e = y - x @ b
    sc = cluster_sums(xhat * e[:, None], groups)
    n, k = x.shape
    g = sc.shape[0]
    c = g / (g - 1) * (n - 1) / (n - k)
    V = c * bread @ sc.T @ sc @ bread
    return {"b": b, "se": np.sqrt(np.diag(V)), "V": V, "e": e}


def ols(y, x, groups):
    return tsls(y, x, x, groups)


def first_stage_f(price, xexog, inst, groups):
    """Cluster-robust Wald statistic of the excluded instruments over q."""
    r = ols(price, np.column_stack([inst, xexog]), groups)
    q = inst.shape[1]
    b, V = r["b"][:q], r["V"][:q, :q]
    return float(b @ np.linalg.solve(V, b) / q), r


def hansen_j(y, x, z, groups):
    """Two-step GMM with clustered weight, as in linearmodels IVGMM."""
    n = len(y)
    w0 = np.linalg.inv(z.T @ z / n)

    def est(w):
        a = x.T @ z @ w @ z.T @ x
        return np.linalg.solve(a, x.T @ z @ w @ z.T @ y)

    b1 = est(w0)
    e1 = y - x @ b1
    sc = cluster_sums(z * e1[:, None], groups)
    s = sc.T @ sc / n
    w = np.linalg.inv(s)
    b2 = est(w)
    gbar = (z * (y - x @ b2)[:, None]).mean(axis=0)
    j = float(n * gbar @ w @ gbar)
    df = z.shape[1] - x.shape[1]
    return {"J": j, "df": df, "p": float(stats.chi2.sf(j, df)), "b2": b2}


def blp_instruments(df):
    firm = df.groupby(["market_id", "owner"])
    mkt = df.groupby("market_id")
    out = pd.DataFrame({"Z": df["Z"]}, index=df.index)
    for col in ("X1", "X2"):
        firm_sum = firm[col].transform("sum")
        out[col.lower() + "_own"] = firm_sum - df[col]
        out[col.lower() + "_rival"] = mkt[col].transform("sum") - firm_sum
    firm_n = firm["X0"].transform("size")
    out["n_own"] = firm_n - 1
    out["n_rival"] = mkt["X0"].transform("size") - firm_n
    return out[IV_CANDIDATES]


def drop_collinear(df, base, cands):
    kept = []
    for c in cands:
        mat = df[base + kept + [c]].to_numpy()
        if np.linalg.matrix_rank(mat) == mat.shape[1]:
            kept.append(c)
    return kept


def jacobian_rowsums(delta, xtilde, nu, sigma):
    """Row sums of d f / d delta for f = delta + ln s_obs - ln s(delta).

    Row j equals 1 - mean_i[s_ij s_i0] / s_j (BLP 1995, p. 888).
    """
    sij = choice_probs(delta, xtilde, nu, sigma)
    s = sij.mean(axis=0)
    si0 = 1.0 - sij.sum(axis=1)
    return 1.0 - (sij * si0[:, None]).mean(axis=0) / s


def contraction_anatomy(s_obs, xtilde, nu, sigma, delta0, delta_star):
    """Step sizes, row sums along the path and an a priori iteration bound."""
    delta, log_s = delta0.copy(), np.log(s_obs)
    steps, kappas = [], []
    for _ in range(MAX_ITER):
        kappas.append(float(jacobian_rowsums(delta, xtilde, nu, sigma).max()))
        new = delta + log_s - np.log(shares_delta(delta, xtilde, nu, sigma))
        steps.append(float(np.max(np.abs(new - delta))))
        delta = new
        if steps[-1] < TOL:
            break
    kappa = max(kappas)
    # ||delta^{n} - delta^{n-1}|| <= kappa^{n-1} ||delta^1 - delta^0||,
    # so the stopping rule is met by the first n with kappa^{n-1} step_1 < TOL.
    bound = int(np.floor(np.log(TOL / steps[0]) / np.log(kappa))) + 2
    ratios = [steps[k + 1] / steps[k] for k in range(len(steps) - 1)]
    # Buyers' outside-good probability and relative simulation error.
    sij = choice_probs(delta_star, xtilde, nu, sigma)
    si0 = 1.0 - sij.sum(axis=1)
    s = sij.mean(axis=0)
    return {
        "iterations": len(steps),
        "initial_error_sup": float(np.max(np.abs(delta0 - delta_star))),
        "first_step_sup": steps[0],
        "steps": steps,
        "step_ratios": ratios,
        "rowsum_max_along_path": kappa,
        "rowsum_at_solution": jacobian_rowsums(delta_star, xtilde, nu, sigma),
        "iteration_bound": bound,
        "buyers_outside_prob": (sij * si0[:, None]).mean(axis=0) / s,
        "outside_prob_mean": float(si0.mean()),
        "rel_sim_se_share": sij.std(axis=0, ddof=1) / np.sqrt(len(nu)) / s,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA / "simulated_market.csv")
    df = df.sort_values(["market_id", "product_id"]).reset_index(drop=True)
    assert (df["share_realized"] > 0).all()
    mid = df["market_id"].to_numpy()
    markets = np.unique(mid)
    idx = [np.flatnonzero(mid == m) for m in markets]
    groups = mid
    inst = blp_instruments(df)
    df = pd.concat([df, inst.drop(columns="Z")], axis=1)
    instr = drop_collinear(df, X_COLS, IV_CANDIDATES)

    p_all = df["price"].to_numpy()
    s_all = df["share_realized"].to_numpy()
    owner = df["owner"].to_numpy()
    X = df[X_COLS].to_numpy()
    XP = np.column_stack([X, p_all])
    Zm = df[X_COLS + instr].to_numpy()
    Wc = df[W_COLS].to_numpy()
    s0_all = 1.0 - df.groupby("market_id")["share_realized"].transform("sum").to_numpy()
    y_logit = np.log(s_all) - np.log(s0_all)
    out = {"n_obs": len(df), "n_markets": len(markets), "instruments": instr}

    # ---------------- logit (benchmark for comparisons) ----------------
    lg_ols = ols(y_logit, XP, groups)
    lg_iv = tsls(y_logit, XP, Zm, groups)
    alpha_l = -lg_iv["b"][3]
    fsF, fs = first_stage_f(p_all, X, df[instr].to_numpy(), groups)
    out["logit"] = {
        "ols": {"b": lg_ols["b"], "se": lg_ols["se"]},
        "iv": {"b": lg_iv["b"], "se": lg_iv["se"]},
        "alpha_2sls": alpha_l,
        "first_stage_F": fsF,
        "first_stage_milk": [fs["b"][0], fs["se"][0]],
        "hansen": hansen_j(y_logit, XP, Zm, groups),
    }

    # ---------------- RC: draws and contraction ----------------
    rng = np.random.default_rng(SEED_EST)
    nu = rng.standard_normal((len(idx), N_DRAWS_EST, 4))
    np.savez(DATA / "est_draws.npz", nu=nu, market_id=markets)
    delta = np.empty(len(df))
    iters, fit_err = [], []
    for t, rows in enumerate(idx):
        delta[rows], it = invert_shares(s_all[rows], XP[rows], nu[t], SIGMA, y_logit[rows])
        s_fit = shares_delta(delta[rows], XP[rows], nu[t], SIGMA)
        iters.append(it)
        fit_err.append(np.max(np.abs(np.log(s_fit / s_all[rows]))))
    iters = np.array(iters)
    r_term = y_logit - delta
    out["contraction"] = {
        "tol": TOL,
        "iter_min": int(iters.min()),
        "iter_max": int(iters.max()),
        "iter_mean": float(iters.mean()),
        "iter_m0": int(iters[0]),
        "max_abs_log_share_gap": float(max(fit_err)),
        "share_obs_delta_below_logit_start": float((delta < y_logit).mean()),
        "n_obs_delta_below_logit_start": int((delta < y_logit).sum()),
        "r_min": float(r_term.min()),
        "r_max": float(r_term.max()),
        "r_mean": float(r_term.mean()),
    }

    # Contraction modulus check: largest row sum of d Phi / d delta.
    max_rowsum = 0.0
    for t, rows in enumerate(idx):
        sij = choice_probs(delta[rows], XP[rows], nu[t], SIGMA)
        s = sij.mean(axis=0)
        si0 = 1.0 - sij.sum(axis=1)
        rowsum = 1.0 - (sij * si0[:, None]).mean(axis=0) / s
        max_rowsum = max(max_rowsum, rowsum.max())
    out["contraction"]["max_rowsum_jacobian"] = float(max_rowsum)

    # ---------------- convergence anatomy in market 0 ----------------
    out["contraction_m0"] = contraction_anatomy(
        s_all[idx[0]], XP[idx[0]], nu[0], SIGMA, y_logit[idx[0]], delta[idx[0]])

    # ---------------- market 0 table and gap decomposition ----------------
    r0 = idx[0]
    M0 = float(df["M"].iloc[r0[0]])
    rng_c = np.random.default_rng(SEED_CHECK)
    nu_big = rng_c.standard_normal((N_DRAWS_CHECK, 4))
    d_exact, _ = invert_shares(s_all[r0], XP[r0], nu_big, SIGMA, delta[r0])
    true_d = df["delta_true"].to_numpy()[r0]
    buyers = np.rint(s_all[r0] * M0).astype(int)
    out["m0"] = {
        "product_id": df["product_id"].to_numpy()[r0],
        "owner": owner[r0],
        "price": p_all[r0],
        "share": s_all[r0],
        "buyers": buyers,
        "logit_start": y_logit[r0],
        "delta": delta[r0],
        "delta_true": true_d,
        "delta_100k": d_exact,
        "simulation_part": delta[r0] - d_exact,
        "sampling_part": d_exact - true_d,
        "approx_sd_log_share": np.sqrt((1.0 - s_all[r0]) / buyers),
        "M0": M0,
        "outside_share": 1.0 - s_all[r0].sum(),
        "r_term": r_term[r0],
    }

    # ---------------- RC linear parameters ----------------
    rc_ols = ols(delta, XP, groups)
    rc_iv = tsls(delta, XP, Zm, groups)
    alpha_rc = -rc_iv["b"][3]
    gap = tsls(r_term, XP, Zm, groups)
    z_only = tsls(delta, XP, df[X_COLS + ["Z"]].to_numpy(), groups)
    out["rc"] = {
        "ols": {"b": rc_ols["b"], "se": rc_ols["se"]},
        "iv": {"b": rc_iv["b"], "se": rc_iv["se"]},
        "alpha_2sls": alpha_rc,
        "hansen": hansen_j(delta, XP, Zm, groups),
        "gap_price_coef": float(gap["b"][3]),
        "alpha_rc_minus_logit": float(alpha_rc - alpha_l),
        "identity_error": float(abs(-gap["b"][3] - (alpha_l - alpha_rc))),
        "corr_r_price": float(np.corrcoef(r_term, p_all)[0, 1]),
        "t_vs_true": float((alpha_rc - TRUE_ALPHA) / rc_iv["se"][3]),
        "min_alpha_i": float((alpha_rc - nu.reshape(-1, 4) @ SIGMA[3, :]).min()),
        "z_only": z_only["b"],
        "z_only_se": z_only["se"],
    }

    # ---------------- elasticities, market 0 ----------------
    p0, s0v = p_all[r0], s_all[r0]
    jac_rc, d0_rc, s_sim, s0_sim = rc_jacobian(delta[r0], XP[r0], nu[0], SIGMA, alpha_rc)
    e_rc = elasticity_matrix(jac_rc, d0_rc, p0, s_sim, s0_sim)
    # True model: true parameters, true delta, the 100 firm draws.
    nu_firm = np.load(DATA / "firm_draws.npz")["nu"]
    jac_tr, d0_tr, s_tr, s0_tr = rc_jacobian(true_d, XP[r0], nu_firm[0], SIGMA, TRUE_ALPHA)
    e_tr = elasticity_matrix(jac_tr, d0_tr, p0, s_tr, s0_tr)
    # Logit.
    jac_l = -alpha_l * (np.diag(s0v) - np.outer(s0v, s0v))
    d0_l = alpha_l * (1.0 - s0v.sum()) * s0v
    e_l = elasticity_matrix(jac_l, d0_l, p0, s0v, 1.0 - s0v.sum())
    # Vertical.
    jac_v, d0_v = vertical_jacobian(p0, s0v)
    e_v = elasticity_matrix(jac_v, d0_v, p0, s0v, 1.0 - s0v.sum())

    def div0(jac, d0):
        return d0 / (-np.diag(jac))

    def nnz_cross(e):
        m = e[:-1].copy()
        np.fill_diagonal(m, 0.0)
        return [int((np.abs(row) > 0).sum()) for row in m]

    out["elasticities_m0"] = {
        "rc_est": e_rc,
        "true": e_tr,
        "logit": e_l,
        "vertical": e_v,
        "own": {
            "rc_est": np.diag(e_rc[:-1]),
            "true": np.diag(e_tr[:-1]),
            "logit": np.diag(e_l[:-1]),
            "vertical": np.diag(e_v[:-1]),
        },
        "diversion_outside": {
            "rc_est": div0(jac_rc, d0_rc),
            "true": div0(jac_tr, d0_tr),
            "logit": div0(jac_l, d0_l),
            "vertical": div0(jac_v, d0_v),
        },
        "nonzero_cross_per_row": {
            "rc_est": nnz_cross(e_rc),
            "true": nnz_cross(e_tr),
            "logit": nnz_cross(e_l),
            "vertical": nnz_cross(e_v),
        },
        "max_abs_gap_rc_est_vs_true": float(np.max(np.abs(e_rc - e_tr))),
        "ratio_vertical_logit_of_means": float(np.diag(e_v[:-1]).mean() / np.diag(e_l[:-1]).mean()),
    }
    # Within-column spread of cross elasticities (zero under logit).
    spread = {}
    for key, e in (("rc_est", e_rc), ("true", e_tr), ("logit", e_l)):
        col = e[:-1].copy()
        np.fill_diagonal(col, np.nan)
        spread[key] = float(np.nanmax(np.nanmax(col, axis=0) / np.nanmin(col, axis=0)))
    out["elasticities_m0"]["max_ratio_within_column"] = spread

    # Own elasticities over all observations.
    own_all = {"rc": [], "logit": []}
    for t, rows in enumerate(idx):
        jac, _, s_sim_t, _ = rc_jacobian(delta[rows], XP[rows], nu[t], SIGMA, alpha_rc)
        own_all["rc"].extend(np.diag(jac) * p_all[rows] / s_sim_t)
    own_all["logit"] = -alpha_l * p_all * (1.0 - s_all)
    out["own_all"] = {k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for k, v in own_all.items()}

    # ---------------- implied marginal costs and supply ----------------
    parts = []
    for t, rows in enumerate(idx):
        parts.append(rc_parts(delta[rows], XP[rows], nu[t], SIGMA))

    def rc_mc(alpha):
        mc = np.empty(len(df))
        for t, rows in enumerate(idx):
            A, B, _, _, s_sim_t, _ = parts[t]
            mc[rows] = implied_mc(p_all[rows], s_all[rows], alpha * A + B, ownership(owner[rows]))
        return mc

    def logit_mc(alpha):
        mc = np.empty(len(df))
        for rows in idx:
            s = s_all[rows]
            jac = -alpha * (np.diag(s) - np.outer(s, s))
            mc[rows] = implied_mc(p_all[rows], s, jac, ownership(owner[rows]))
        return mc

    mc_v = np.empty(len(df))
    for rows in idx:
        jac, _ = vertical_jacobian(p_all[rows], s_all[rows])
        mc_v[rows] = implied_mc(p_all[rows], s_all[rows], jac, ownership(owner[rows]))

    mc_rc = rc_mc(alpha_rc)
    mc_l = logit_mc(alpha_l)
    sup = {}
    for key, mc in (("vertical", mc_v), ("logit", mc_l), ("rc", mc_rc), ("mcp", p_all)):
        r = ols(mc, Wc, groups)
        mk = p_all - mc
        sup[key] = {
            "b": r["b"],
            "se_clustered": r["se"],
            "markup_mean": float(mk.mean()),
            "markup_min": float(mk.min()),
            "markup_max": float(mk.max()),
            "lerner_mean": float((mk / p_all).mean()),
            "n_negative_mc": int((mc < 0).sum()),
        }
    mk_true = p_all - df["mc_true"].to_numpy()
    sup["true"] = {
        "b": TRUE_GAMMA,
        "markup_mean": float(mk_true.mean()),
        "markup_min": float(mk_true.min()),
        "markup_max": float(mk_true.max()),
        "lerner_mean": float((mk_true / p_all).mean()),
    }
    # Market bootstrap for the logit and RC cost equations (alpha re-estimated).
    rng_b = np.random.default_rng(SEED_BOOT)
    draws = {"logit": np.empty((N_BOOT, 4)), "rc": np.empty((N_BOOT, 4))}
    for b in range(N_BOOT):
        pick = rng_b.integers(0, len(idx), size=len(idx))
        rows = np.concatenate([idx[m] for m in pick])

        def coef(yv):
            xhat = Zm[rows] @ np.linalg.lstsq(Zm[rows], XP[rows], rcond=None)[0]
            return np.linalg.lstsq(xhat, yv[rows], rcond=None)[0]

        a_l = -coef(y_logit)[3]
        a_r = -coef(delta)[3]
        draws["logit"][b] = np.linalg.lstsq(Wc[rows], logit_mc(a_l)[rows], rcond=None)[0]
        draws["rc"][b] = np.linalg.lstsq(Wc[rows], rc_mc(a_r)[rows], rcond=None)[0]
    for key in ("logit", "rc"):
        sup[key]["se_bootstrap"] = draws[key].std(axis=0, ddof=1)
    sup["bootstrap"] = {"replications": N_BOOT, "seed": SEED_BOOT}
    # Correlation of RC implied markups with the true markups.
    sup["rc"]["corr_markup_true"] = float(np.corrcoef(p_all - mc_rc, mk_true)[0, 1])
    sup["logit"]["corr_markup_true"] = float(np.corrcoef(p_all - mc_l, mk_true)[0, 1])
    sup["rc"]["mean_abs_markup_error"] = float(np.mean(np.abs((p_all - mc_rc) - mk_true)))
    sup["logit"]["mean_abs_markup_error"] = float(np.mean(np.abs((p_all - mc_l) - mk_true)))
    sup["vertical"]["corr_markup_true"] = float(np.corrcoef(p_all - mc_v, mk_true)[0, 1])
    sup["vertical"]["mean_abs_markup_error"] = float(np.mean(np.abs((p_all - mc_v) - mk_true)))
    out["supply"] = sup

    # ---------------- figure ----------------
    plot_own(out, APPC / "own_elas.pdf")

    res = to_py(out)
    (HERE / "appC_rc_results.json").write_text(json.dumps(res, indent=1))
    report(res)


def plot_own(out, path):
    own = out["elasticities_m0"]["own"]
    pid = out["m0"]["product_id"]
    labels = ["product %d" % k for k in pid]
    y = np.arange(len(labels))
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    series = [
        ("logit", "Logit, 2SLS", "#2a78d6", "o", -0.22),
        ("rc_est", "RC logit, 2SLS", "#eb6834", "s", 0.0),
        ("true", "RC logit, true", "#1b8a5a", "D", 0.22),
    ]
    for key, name, col, mk, dy in series:
        axes[0].scatter(own[key], y + dy, s=26, marker=mk, color=col, edgecolor="#0b0b0b",
                        linewidth=0.4, label=name, zorder=3)
    axes[1].scatter(own["vertical"], y, s=26, marker="o", color="#c48a00", edgecolor="#0b0b0b",
                    linewidth=0.4, label="Vertical", zorder=3)
    axes[0].set_xlim(-4.0, -2.3)
    axes[1].set_xlim(-380, 0)
    titles = ["(a) Logit and random coefficients", "(b) Vertical model"]
    for ax, ttl in zip(axes, titles):
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.set_xlabel(r"own-price elasticity $\varepsilon_{jj}$", fontsize=9)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    fig.savefig(HERE / "own_elas_preview.png", dpi=150)
    plt.close(fig)


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def report(r):
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print("instruments", r["instruments"], "N", r["n_obs"])
    lg = r["logit"]
    print("LOGIT ols b", np.round(lg["ols"]["b"], 3), "se", np.round(lg["ols"]["se"], 3))
    print("LOGIT iv  b", np.round(lg["iv"]["b"], 3), "se", np.round(lg["iv"]["se"], 3))
    print("first-stage F %.2f milk %.3f (%.3f)" % (lg["first_stage_F"], *lg["first_stage_milk"]))
    print("logit Hansen J %.2f df %d p %.3f" % (lg["hansen"]["J"], lg["hansen"]["df"], lg["hansen"]["p"]))
    c = r["contraction"]
    print("CONTRACTION", json.dumps(c))
    a = r["contraction_m0"]
    print("CONTRACTION M0 iterations %d bound %d kappa_path %.4f init err %.4f first step %.4f" % (
        a["iterations"], a["iteration_bound"], a["rowsum_max_along_path"],
        a["initial_error_sup"], a["first_step_sup"]))
    print(" step ratios", np.round(a["step_ratios"], 4))
    print(" rowsum at solution", np.round(a["rowsum_at_solution"], 4))
    print(" buyers' outside prob", np.round(a["buyers_outside_prob"], 4), "mean outside prob %.4f" % a["outside_prob_mean"])
    print(" rel sim se of share", np.round(a["rel_sim_se_share"], 4))
    m0 = r["m0"]
    print("MARKET 0  M0 %.0f outside share %.4f" % (m0["M0"], m0["outside_share"]))
    print(" prod own   price  share buyers  start   delta   true   d100k   sim    samp   sdlog  r")
    for k in range(len(m0["product_id"])):
        print(" %3d %3d  %.4f %.4f %4d  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f" % (
            m0["product_id"][k], m0["owner"][k], m0["price"][k], m0["share"][k], m0["buyers"][k],
            m0["logit_start"][k], m0["delta"][k], m0["delta_true"][k], m0["delta_100k"][k],
            m0["simulation_part"][k], m0["sampling_part"][k], m0["approx_sd_log_share"][k],
            m0["r_term"][k]))
    rc = r["rc"]
    print("RC ols b", np.round(rc["ols"]["b"], 3), "se", np.round(rc["ols"]["se"], 3))
    print("RC iv  b", np.round(rc["iv"]["b"], 3), "se", np.round(rc["iv"]["se"], 3))
    print("RC Hansen J %.2f p %.3f; gap coef %.4f; alpha_rc - alpha_l %.4f; identity err %.1e; corr(r,p) %.3f; t %.2f; min alpha_i %.3f" % (
        rc["hansen"]["J"], rc["hansen"]["p"], rc["gap_price_coef"], rc["alpha_rc_minus_logit"],
        rc["identity_error"], rc["corr_r_price"], rc["t_vs_true"], rc["min_alpha_i"]))
    print("RC z-only b", np.round(rc["z_only"], 3), "se", np.round(rc["z_only_se"], 3))
    el = r["elasticities_m0"]
    for key in ("rc_est", "true", "logit", "vertical"):
        print("ELAST", key)
        print(np.array(el[key]))
    for key in ("rc_est", "true", "logit", "vertical"):
        print("own", key, np.round(el["own"][key], 3))
        print("div0", key, np.round(100 * np.array(el["diversion_outside"][key]), 1))
        print("nnz cross per row", key, el["nonzero_cross_per_row"][key])
    print("max |e_rc - e_true|", el["max_abs_gap_rc_est_vs_true"], "vertical/logit means", el["ratio_vertical_logit_of_means"])
    print("max within-column ratio", el["max_ratio_within_column"])
    print("own all obs", r["own_all"])
    for key, v in r["supply"].items():
        print("SUPPLY", key, json.dumps(v))


if __name__ == "__main__":
    main()
