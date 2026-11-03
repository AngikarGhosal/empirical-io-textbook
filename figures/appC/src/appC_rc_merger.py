"""Appendix C.7(f): a merger in market 0 under three demand models.

The firms that own products {1, 4, 6, 7} (owner id 1, firm B) and
{2, 3, 10} (owner id 2, firm C) merge. In market 0 the merging products
are 4 and 6 (B) and 2, 3 and 10 (C). We predict the post-merger
equilibrium under
  (i)   the logit estimated by 2SLS (C.6), costs recovered from its FOCs;
  (ii)  the random coefficients model with Sigma known and (beta, alpha)
        estimated by 2SLS (C.7), with the 1,000 estimation draws and
        costs recovered from its FOCs;
  (iii) the true model: true parameters, true xi, the 100 firm draws
        and true marginal costs.
Prices are solved by the logit markup iteration (i) and the
Morrow-Skerlos iteration (ii)-(iii). Reads appC_rc_results.json (run
appC_rc.py first), est_draws.npz, firm_draws.npz, simulated_market.csv.
Writes appC_rc_merger_results.json and prints every number.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from appC_rc import SIGMA, choice_probs, ownership, vertical_jacobian

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TOL = 1e-12
MAX_ITER = 100_000
MERGE = (1, 2)  # owner ids of firms B and C


def merged_owner(owner):
    out = owner.copy()
    out[np.isin(out, MERGE)] = MERGE[0]
    return out


# ---------------------------------------------------------------- logit
def logit_shares(dbase, p, p0, alpha):
    v = dbase - alpha * (p - p0)
    ev = np.exp(v)
    return ev / (1.0 + ev.sum())


def logit_solve(dbase, p0, alpha, mc, owner, start):
    p = start.copy()
    for it in range(1, MAX_ITER + 1):
        s = logit_shares(dbase, p, p0, alpha)
        sf = pd.Series(s).groupby(owner).transform("sum").to_numpy()
        new = mc + 1.0 / (alpha * (1.0 - sf))
        if np.max(np.abs(new - p)) < TOL:
            return new, it
        p = new
    raise RuntimeError("logit iteration did not converge")


def logit_jac(s, alpha):
    return -alpha * (np.diag(s) - np.outer(s, s))


def logit_cs(dbase, p, p0, alpha):
    return np.log1p(np.exp(dbase - alpha * (p - p0)).sum()) / alpha


# ---------------------------------------------------------------- RC
def rc_moments(dbase, p, p0, x, nu, alpha):
    """Shares, Lambda (vector), Gamma and s_ij at prices p.

    Mean utility moves with price as delta_j(p) = dbase_j - alpha (p_j - p0_j);
    the random part uses xtilde = (x, p).
    """
    xt = np.column_stack([x, p])
    sij = choice_probs(dbase - alpha * (p - p0), xt, nu, SIGMA)
    ai = alpha - nu @ SIGMA[3, :]
    w = ai[:, None] * sij
    return sij.mean(axis=0), w.mean(axis=0), w.T @ sij / len(nu), sij, ai


def rc_solve(dbase, p0, x, nu, alpha, mc, owner, start):
    own = ownership(owner)
    p = start.copy()
    for it in range(1, MAX_ITER + 1):
        s, lam, gam, _, _ = rc_moments(dbase, p, p0, x, nu, alpha)
        new = mc + (s + (own * gam.T) @ (p - mc)) / lam
        if np.max(np.abs(new - p)) < TOL:
            return new, it
        p = new
    raise RuntimeError("Morrow-Skerlos iteration did not converge")


def rc_foc(dbase, p, p0, x, nu, alpha, mc, owner):
    s, lam, gam, _, _ = rc_moments(dbase, p, p0, x, nu, alpha)
    jac = gam - np.diag(lam)
    return s + (ownership(owner) * jac.T) @ (p - mc)


def rc_cs(dbase, p, p0, x, nu, alpha):
    """Mean over draws of the log-sum divided by alpha_i (per consumer)."""
    xt = np.column_stack([x, p])
    v = dbase - alpha * (p - p0) + (nu @ SIGMA.T @ xt.T)
    ai = alpha - nu @ SIGMA[3, :]
    m = np.maximum(v.max(axis=1), 0.0)
    ls = m + np.log(np.exp(-m) + np.exp(v - m[:, None]).sum(axis=1))
    return float(np.mean(ls / ai))


def diversion(jac):
    """D[j, k] = -(ds_k/dp_j) / (ds_j/dp_j)."""
    return -jac.T / np.diag(jac)[:, None]


def main():
    res = json.loads((HERE / "appC_rc_results.json").read_text())
    df = pd.read_csv(DATA / "simulated_market.csv")
    df = df.sort_values(["market_id", "product_id"]).reset_index(drop=True)
    g = df[df["market_id"] == 0]
    pid = g["product_id"].to_numpy()
    owner = g["owner"].to_numpy()
    owner_post = merged_owner(owner)
    x = g[["X0", "X1", "X2"]].to_numpy()
    p0 = g["price"].to_numpy()
    s_obs = g["share_realized"].to_numpy()
    M0 = float(g["M"].iloc[0])
    merging = np.isin(owner, MERGE)

    alpha_l = res["logit"]["alpha_2sls"]
    alpha_rc = res["rc"]["alpha_2sls"]
    delta_rc = np.array(res["m0"]["delta"])
    nu_est = np.load(DATA / "est_draws.npz")["nu"][0]
    nu_firm = np.load(DATA / "firm_draws.npz")["nu"][0]
    delta_true = g["delta_true"].to_numpy()
    mc_true = g["mc_true"].to_numpy()

    out = {"product_id": pid, "owner": owner, "price_pre": p0, "M0": M0}

    # (i) logit
    d_l = np.log(s_obs) - np.log(1.0 - s_obs.sum())
    s_l0 = logit_shares(d_l, p0, p0, alpha_l)
    mc_l = p0 + np.linalg.solve(ownership(owner) * logit_jac(s_l0, alpha_l).T, s_l0)
    pre_l, _ = logit_solve(d_l, p0, alpha_l, mc_l, owner, p0 + 0.3)
    post_l, it_l = logit_solve(d_l, p0, alpha_l, mc_l, owner_post, p0)
    s_l1 = logit_shares(d_l, post_l, p0, alpha_l)

    # (ii) RC estimated
    s_r0, lam, gam, _, _ = rc_moments(delta_rc, p0, p0, x, nu_est, alpha_rc)
    jac_r0 = gam - np.diag(lam)
    mc_r = p0 + np.linalg.solve(ownership(owner) * jac_r0.T, s_r0)
    pre_r, _ = rc_solve(delta_rc, p0, x, nu_est, alpha_rc, mc_r, owner, p0 + 0.3)
    post_r, it_r = rc_solve(delta_rc, p0, x, nu_est, alpha_rc, mc_r, owner_post, p0)
    s_r1 = rc_moments(delta_rc, post_r, p0, x, nu_est, alpha_rc)[0]

    # (iii) true model
    s_t0, lam_t, gam_t, _, _ = rc_moments(delta_true, p0, p0, x, nu_firm, 3.0)
    jac_t0 = gam_t - np.diag(lam_t)
    foc_pre_t = np.max(np.abs(rc_foc(delta_true, p0, p0, x, nu_firm, 3.0, mc_true, owner)))
    pre_t, _ = rc_solve(delta_true, p0, x, nu_firm, 3.0, mc_true, owner, p0 + 0.3)
    post_t, it_t = rc_solve(delta_true, p0, x, nu_firm, 3.0, mc_true, owner_post, p0)
    s_t1 = rc_moments(delta_true, post_t, p0, x, nu_firm, 3.0)[0]

    # Restarts from random prices for the post-merger games.
    rng = np.random.default_rng(2033)
    dev = {"logit": 0.0, "rc": 0.0, "true": 0.0}
    for _ in range(20):
        st = mc_true + rng.uniform(0.0, 2.0, size=len(p0))
        dev["logit"] = max(dev["logit"], np.max(np.abs(
            logit_solve(d_l, p0, alpha_l, mc_l, owner_post, st)[0] - post_l)))
        dev["rc"] = max(dev["rc"], np.max(np.abs(
            rc_solve(delta_rc, p0, x, nu_est, alpha_rc, mc_r, owner_post, st)[0] - post_r)))
        dev["true"] = max(dev["true"], np.max(np.abs(
            rc_solve(delta_true, p0, x, nu_firm, 3.0, mc_true, owner_post, st)[0] - post_t)))

    def profits(p, mc, s, own_ids):
        pi = (p - mc) * s
        return {int(k): float(pi[own_ids == k].sum()) for k in np.unique(own_ids)}

    cs = {
        "logit": [logit_cs(d_l, p0, p0, alpha_l), logit_cs(d_l, post_l, p0, alpha_l)],
        "rc": [rc_cs(delta_rc, p0, p0, x, nu_est, alpha_rc), rc_cs(delta_rc, post_r, p0, x, nu_est, alpha_rc)],
        "true": [rc_cs(delta_true, p0, p0, x, nu_firm, 3.0), rc_cs(delta_true, post_t, p0, x, nu_firm, 3.0)],
    }
    blocks = {
        "logit": (post_l, s_l0, s_l1, mc_l, logit_jac(s_l0, alpha_l), it_l, pre_l),
        "rc": (post_r, s_r0, s_r1, mc_r, jac_r0, it_r, pre_r),
        "true": (post_t, s_t0, s_t1, mc_true, jac_t0, it_t, pre_t),
    }
    for key, (post, s_pre, s_post, mc, jac, it, pre) in blocks.items():
        dp = post - p0
        D = diversion(jac)
        # Diversion from each merging product to the partner firm's products.
        dpart = np.zeros(len(p0))
        for j in range(len(p0)):
            if merging[j]:
                partner = merging & (owner != owner[j])
                dpart[j] = D[j, partner].sum()
        # First-order upward pricing pressure: sum over partner products of
        # D_jk (p_k - mc_k), and GUPPI = UPP / p_j.
        upp = np.zeros(len(p0))
        for j in range(len(p0)):
            if merging[j]:
                partner = merging & (owner != owner[j])
                upp[j] = (D[j, partner] * (p0[partner] - mc[partner])).sum()
        out[key] = {
            "mc": mc,
            "markup_pre": p0 - mc,
            "price_post": post,
            "dp": dp,
            "dp_pct": 100.0 * dp / p0,
            "share_pre": s_pre,
            "share_post": s_post,
            "outside_pre": 1.0 - s_pre.sum(),
            "outside_post": 1.0 - s_post.sum(),
            "mean_dp_merging": float(dp[merging].mean()),
            "mean_dp_rivals": float(dp[~merging].mean()),
            "sw_dp_merging": float((s_pre[merging] * dp[merging]).sum() / s_pre[merging].sum()),
            "cs_pre": cs[key][0],
            "cs_post": cs[key][1],
            "dcs_per_consumer": cs[key][1] - cs[key][0],
            "dcs_total": M0 * (cs[key][1] - cs[key][0]),
            "profit_pre": profits(p0, mc, s_pre, owner),
            "profit_post": profits(post, mc, s_post, owner),
            "diversion_to_partner": dpart,
            "upp": upp,
            "guppi": upp / p0,
            "iterations": it,
            "pre_merger_resolve_max_dev": float(np.max(np.abs(pre - p0))),
            "restart_max_dev": float(dev[key]),
        }
        pp, po = out[key]["profit_pre"], out[key]["profit_post"]
        out[key]["merged_profit_pre_total"] = M0 * (pp.get(1, 0) + pp.get(2, 0))
        out[key]["merged_profit_post_total"] = M0 * (po.get(1, 0) + po.get(2, 0))
        out[key]["rival_profit_change_total"] = M0 * sum(po[k] - pp[k] for k in pp if k not in MERGE)
    out["true"]["foc_pre_residual"] = float(foc_pre_t)
    for key in ("logit", "rc", "true"):
        o = out[key]
        o["markup_post"] = o["price_post"] - o["mc"]
        o["dprofit_merged_total"] = o["merged_profit_post_total"] - o["merged_profit_pre_total"]
        o["dwelfare_total"] = o["dcs_total"] + o["dprofit_merged_total"] + o["rival_profit_change_total"]
    # Vertical model: diversion from each merging product to the partner firm.
    jac_v, d0_v = vertical_jacobian(p0, s_obs)
    Dv = diversion(jac_v)
    dv = np.zeros(len(p0))
    for j in range(len(p0)):
        if merging[j]:
            partner = merging & (owner != owner[j])
            dv[j] = Dv[j, partner].sum()
    out["vertical_diversion_to_partner"] = dv
    out["vertical_diversion_to_outside"] = d0_v / (-np.diag(jac_v))
    out["price_ladder"] = pid[np.argsort(p0)]

    res_out = to_py(out)
    (HERE / "appC_rc_merger_results.json").write_text(json.dumps(res_out, indent=1))
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print("products", pid, "owners", owner, "merging", merging.astype(int))
    print("pre prices", p0)
    for key in ("logit", "rc", "true"):
        o = res_out[key]
        print("==", key, "iterations", o["iterations"], "pre-resolve dev %.1e restart dev %.1e" % (
            o["pre_merger_resolve_max_dev"], o["restart_max_dev"]))
        print(" mc", np.array(o["mc"]))
        print(" markup pre", np.array(o["markup_pre"]))
        print(" post price", np.array(o["price_post"]))
        print(" dp", np.array(o["dp"]))
        print(" dp pct", np.array(o["dp_pct"]))
        print(" share pre", np.array(o["share_pre"]))
        print(" share post", np.array(o["share_post"]))
        print(" outside %.4f -> %.4f" % (o["outside_pre"], o["outside_post"]))
        print(" mean dp merging %.4f rivals %.4f sw merging %.4f" % (
            o["mean_dp_merging"], o["mean_dp_rivals"], o["sw_dp_merging"]))
        print(" CS per consumer %.5f -> %.5f d %.6f total %.2f" % (
            o["cs_pre"], o["cs_post"], o["dcs_per_consumer"], o["dcs_total"]))
        print(" merged profit total %.2f -> %.2f; rival profit change %.2f" % (
            o["merged_profit_pre_total"], o["merged_profit_post_total"], o["rival_profit_change_total"]))
        print(" diversion to partner", np.array(o["diversion_to_partner"]))
        print(" UPP", np.array(o["upp"]), "GUPPI", np.array(o["guppi"]))
    print("true pre FOC residual", res_out["true"]["foc_pre_residual"])
    for key in ("logit", "rc", "true"):
        o = res_out[key]
        print(key, "markup post", np.round(o["markup_post"], 4), "dprofit merged %.2f dW %.2f" % (
            o["dprofit_merged_total"], o["dwelfare_total"]))
    print("vertical diversion to partner", np.round(res_out["vertical_diversion_to_partner"], 4))
    print("vertical diversion to outside", np.round(res_out["vertical_diversion_to_outside"], 4))
    print("price ladder", res_out["price_ladder"])


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


if __name__ == "__main__":
    main()
