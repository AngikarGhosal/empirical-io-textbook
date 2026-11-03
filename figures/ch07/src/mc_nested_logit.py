"""Monte Carlo study of the nested-logit regression (Berry 1994, eq. 28).

Design. Each sample has T = 500 independent markets. Every market has two
inside nests, g = 1, 2, and an outside good in its own nest. The number of
products in nest g of market t, J_gt, is drawn uniformly from {1, ..., 5}
(main design) or fixed at 3 (fixed-size design). Each product is sold by its
own single-product firm. Utility is
    u_ij = delta_j + zeta_ig + (1 - sigma) eps_ij,
    delta_j = beta0 + betax * x_j - alpha * p_j + xi_j,
with u_i0 = eps_i0. Marginal cost is
    c_j = exp(gamma0 + gammax * x_j + gammaw * w_j + sigma_c * xi_j + sigma_om * om_j).
x, xi, w, om are independent standard normal. True values: beta0 = 2,
betax = 1, alpha = 1, sigma = 0.5, gamma0 = 0.5, gammax = 0.25, gammaw = 0.25,
sigma_c = 0.1, sigma_om = 0.25. Prices are the single-product Bertrand-Nash
equilibrium with nested-logit demand,
    p_j = c_j + (1 - sigma) / (alpha * [1 - sigma s_{j|g} - (1 - sigma) s_j]),
solved market by market by simultaneous best-response iteration, with each
best response found exactly by bisection on its first-order condition.
Shares are exact.

Estimators of ln s_j - ln s_0 = beta0 + betax x_j - alpha p_j + sigma ln s_{j|g} + xi_j:
  OLS        all regressors treated as exogenous;
  IV-p       price instrumented, ln s_{j|g} treated as exogenous;
  IV-both    price and ln s_{j|g} instrumented (the correct estimator);
  Logit-IV   sigma restricted to zero, price instrumented (misspecified).
Excluded instruments: own cost shifter w_j; number of products in the own
nest J_gt; sum of x over the other products in the own nest; sum of x over
the products in the other nest; sum of w over the other products in the own
nest. IV-p and Logit-IV use the same excluded instruments.

Outputs (paths relative to Textbook/):
    figures/ch07/mc_nl_table.tex           LaTeX rows for tab:07-mc-nl
    figures/ch07/src/mc_nl_numbers.txt     every number quoted in the text
Only numpy is used. Seeds are fixed.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)

TRUE = dict(beta0=2.0, betax=1.0, alpha=1.0, sigma=0.5)
COST = dict(gamma0=0.5, gammax=0.25, gammaw=0.25, sigma_c=0.1, sigma_om=0.25)
T = 500          # markets per sample
R = 500          # samples per design
JMAX = 5         # maximum products per nest
G = 2            # inside nests


def draw_market_data(rng, fixed_size):
    """Draw characteristics for T markets with G nests of up to JMAX slots."""
    nslot = G * JMAX
    if fixed_size:
        jg = np.full((T, G), 3)
    else:
        jg = rng.integers(1, JMAX + 1, size=(T, G))
    slot_nest = np.repeat(np.arange(G), JMAX)            # nest of each slot
    slot_pos = np.tile(np.arange(JMAX), G)               # position within nest
    active = slot_pos[None, :] < jg[:, slot_nest]        # (T, nslot)
    x = rng.standard_normal((T, nslot))
    xi = rng.standard_normal((T, nslot))
    w = rng.standard_normal((T, nslot))
    om = rng.standard_normal((T, nslot))
    return jg, slot_nest, active, x, xi, w, om


def shares(delta, active, slot_nest, sigma):
    """Nested-logit shares: product, within-nest, outside."""
    e = np.where(active, np.exp(delta / (1.0 - sigma)), 0.0)
    D = np.stack([e[:, slot_nest == g].sum(axis=1) for g in range(G)], axis=1)
    incl = np.where(D > 0, D ** (1.0 - sigma), 0.0)       # D_g^{1-sigma}
    denom = 1.0 + incl.sum(axis=1, keepdims=True)
    s_g = incl / denom
    Dj = D[:, slot_nest]
    s_jg = np.where(active, e / np.where(Dj > 0, Dj, 1.0), 0.0)
    s_j = s_jg * s_g[:, slot_nest]
    s_0 = 1.0 / denom[:, 0]
    return s_j, s_jg, s_0


def best_response(p, quality, mc, active, slot_nest, nbis=60, hi=30.0):
    """Exact single-product best responses to rivals' current prices.

    For product j in nest g, given rivals' prices, the FOC
        p_j - c_j = (1 - sigma) / (alpha [1 - sigma s_{j|g} - (1 - sigma) s_j])
    has a unique root: the left side rises in p_j and the right side falls.
    We find it by vectorized bisection on [c_j, c_j + hi].
    """
    a_, sg = TRUE["alpha"], TRUE["sigma"]
    e = np.where(active, np.exp((quality - a_ * p) / (1.0 - sg)), 0.0)
    D = np.stack([e[:, slot_nest == g].sum(axis=1) for g in range(G)], axis=1)
    incl = D ** (1.0 - sg)
    R_in = D[:, slot_nest] - e                                   # rivals in own nest
    other = 1.0 + incl.sum(axis=1, keepdims=True) - incl[:, slot_nest]

    def foc(pt):
        et = np.exp((quality - a_ * pt) / (1.0 - sg))
        Dt = R_in + et
        s_jg = et / Dt
        inc_t = Dt ** (1.0 - sg)
        s_j = s_jg * inc_t / (other + inc_t)
        return pt - mc - (1.0 - sg) / (a_ * (1.0 - sg * s_jg - (1.0 - sg) * s_j))

    lo = mc.copy()
    up = mc + hi
    if np.any(np.where(active, foc(up), 1.0) <= 0):
        raise RuntimeError("bisection bracket too small")
    for _ in range(nbis):
        mid = 0.5 * (lo + up)
        f = foc(mid)
        lo = np.where(f < 0, mid, lo)
        up = np.where(f < 0, up, mid)
    return 0.5 * (lo + up)


def equilibrium(x, xi, w, om, active, slot_nest, tol=1e-10, maxit=500):
    """Single-product Bertrand-Nash prices with nested-logit demand.

    Simultaneous (Jacobi) best-response iteration with exact best responses.
    """
    a_, sg = TRUE["alpha"], TRUE["sigma"]
    mc = np.exp(COST["gamma0"] + COST["gammax"] * x + COST["gammaw"] * w
                + COST["sigma_c"] * xi + COST["sigma_om"] * om)
    quality = TRUE["beta0"] + TRUE["betax"] * x + xi
    p = mc + (1.0 - sg) / a_
    for it in range(maxit):
        p_new = best_response(p, quality, mc, active, slot_nest)
        gap = np.max(np.abs(np.where(active, p_new - p, 0.0)))
        p = p_new
        if gap < tol:
            break
    else:
        raise RuntimeError("equilibrium did not converge")
    s_j, s_jg, s_0 = shares(quality - a_ * p, active, slot_nest, sg)
    m = (1.0 - sg) / (a_ * (1.0 - sg * s_jg - (1.0 - sg) * s_j))
    foc = np.max(np.abs(np.where(active, p - mc - m, 0.0)))
    return p, s_j, s_jg, s_0, mc, it + 1, foc


def tsls(y, X, Z):
    """Two-stage least squares coefficients."""
    ZZ = Z.T @ Z
    PzX = Z @ np.linalg.solve(ZZ, Z.T @ X)
    return np.linalg.solve(PzX.T @ X, PzX.T @ y)


def first_stage_F(v, Zexog, Zexcl):
    """F statistic for the excluded instruments in the first stage of v."""
    Zfull = np.column_stack([Zexog, Zexcl])
    n, k_full = Zfull.shape
    q = Zexcl.shape[1]
    b_r = np.linalg.lstsq(Zexog, v, rcond=None)[0]
    b_u = np.linalg.lstsq(Zfull, v, rcond=None)[0]
    ssr_r = np.sum((v - Zexog @ b_r) ** 2)
    ssr_u = np.sum((v - Zfull @ b_u) ** 2)
    return ((ssr_r - ssr_u) / q) / (ssr_u / (n - k_full))


def one_sample(rng, fixed_size):
    jg, slot_nest, active, x, xi, w, om = draw_market_data(rng, fixed_size)
    p, s_j, s_jg, s_0, mc, nit, foc = equilibrium(x, xi, w, om, active, slot_nest)
    # instruments built market by market
    xa = np.where(active, x, 0.0)
    wa = np.where(active, w, 0.0)
    xsum_nest = np.stack([xa[:, slot_nest == g].sum(axis=1) for g in range(G)], axis=1)
    wsum_nest = np.stack([wa[:, slot_nest == g].sum(axis=1) for g in range(G)], axis=1)
    own_nest_x = xsum_nest[:, slot_nest] - xa             # other products, own nest
    other_nest_x = xsum_nest.sum(axis=1, keepdims=True) - xsum_nest[:, slot_nest]
    own_nest_w = wsum_nest[:, slot_nest] - wa
    n_nest = jg[:, slot_nest].astype(float)
    sel = active.ravel()
    flat = lambda a: a.ravel()[sel]
    s0_rep = np.repeat(s_0[:, None], G * JMAX, axis=1)
    y = np.log(flat(s_j)) - np.log(flat(s0_rep))
    lnsjg = np.log(flat(s_jg))
    const = np.ones(sel.sum())
    X_nl = np.column_stack([const, flat(x), -flat(p), lnsjg])   # coefficients: beta0, betax, alpha, sigma
    X_lg = np.column_stack([const, flat(x), -flat(p)])
    excl_cols = [flat(w), flat(own_nest_x), flat(other_nest_x), flat(own_nest_w)]
    if not fixed_size:
        excl_cols.insert(1, flat(n_nest))
    excl = np.column_stack(excl_cols)
    Zexog = np.column_stack([const, flat(x)])
    Z_both = np.column_stack([Zexog, excl])
    Z_ponly = np.column_stack([Zexog, lnsjg, excl])
    res = {}
    res["OLS"] = np.linalg.lstsq(X_nl, y, rcond=None)[0]
    res["IV-p"] = tsls(y, X_nl, Z_ponly)
    res["IV-both"] = tsls(y, X_nl, Z_both)
    b_lg = tsls(y, X_lg, Z_both)
    res["Logit-IV"] = np.append(b_lg, 0.0)
    # implied mean own-price elasticity (positive number)
    pj, sj, sjg = flat(p), flat(s_j), flat(s_jg)
    def own_el(b):
        a_, sg = b[2], b[3]
        return np.mean(a_ * pj * (1.0 / (1.0 - sg) - sg / (1.0 - sg) * sjg - sj))
    el = {k: own_el(v) for k, v in res.items()}
    el["True"] = own_el(np.array([TRUE["beta0"], TRUE["betax"], TRUE["alpha"], TRUE["sigma"]]))
    F_p = first_stage_F(flat(p), Zexog, excl)
    F_s = first_stage_F(lnsjg, Zexog, excl)
    diag = dict(nit=nit, foc=foc, mean_s0=s_0.mean(), mean_p=pj.mean(),
                mean_mc=flat(mc).mean(), mean_sjg=sjg.mean(), n=sel.sum(),
                F_p=F_p, F_s=F_s, mean_sj=sj.mean())
    return res, el, diag


def run(fixed_size, seed):
    rng = np.random.default_rng(seed)
    est = {k: [] for k in ["OLS", "IV-p", "IV-both", "Logit-IV"]}
    els = {k: [] for k in ["OLS", "IV-p", "IV-both", "Logit-IV", "True"]}
    diags = []
    for r in range(R):
        res, el, diag = one_sample(rng, fixed_size)
        for k in est:
            est[k].append(res[k])
        for k in els:
            els[k].append(el[k])
        diags.append(diag)
    est = {k: np.array(v) for k, v in est.items()}
    els = {k: np.array(v) for k, v in els.items()}
    return est, els, diags


def fmt(m, s, nd=3):
    return f"{m:.{nd}f}", f"({s:.{nd}f})"


def main():
    lines = []
    out_main = run(fixed_size=False, seed=20261022)
    out_fixed = run(fixed_size=True, seed=20261023)
    names = ["beta0", "betax", "alpha", "sigma"]
    for label, (est, els, diags) in [("Main design: J_gt uniform on {1,...,5}", out_main),
                                     ("Fixed-size design: J_gt = 3", out_fixed)]:
        lines.append(f"=== {label}; R = {R} samples of T = {T} markets ===")
        for k, b in est.items():
            parts = [f"{nm}={b[:, i].mean():.3f} ({b[:, i].std(ddof=1):.3f})"
                     for i, nm in enumerate(names)]
            lines.append(f"{k:9s} " + "  ".join(parts))
        for k, e in els.items():
            lines.append(f"own elasticity {k:9s} mean {np.nanmean(e):.3f} sd {np.nanstd(e, ddof=1):.3f}")
        sig = est["IV-both"][:, 3]
        lines.append(f"IV-both sigma outside [0,1): {np.sum((sig < 0) | (sig >= 1))} of {R}")
        sig_o = est["OLS"][:, 3]
        lines.append(f"OLS sigma >= 1: {np.sum(sig_o >= 1)} of {R}")
        for key in ["nit", "foc", "mean_s0", "mean_p", "mean_mc", "mean_sjg", "mean_sj", "n", "F_p", "F_s"]:
            v = np.array([d[key] for d in diags], dtype=float)
            lines.append(f"{key}: mean {v.mean():.4g}, median {np.median(v):.4g}, min {v.min():.4g}, max {v.max():.4g}")
        lines.append("")
    txt = "\n".join(lines)
    with open(os.path.join(HERE, "mc_nl_numbers.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)

    # LaTeX table body: rows = parameters, columns = estimators (main design)
    # plus IV-both under the fixed-size design.
    est_m, els_m, _ = out_main
    est_f, els_f, _ = out_fixed
    labels = [r"$\beta_0$", r"$\beta_x$", r"$\alpha$", r"$\sigma$"]
    truth = [TRUE["beta0"], TRUE["betax"], TRUE["alpha"], TRUE["sigma"]]
    cols = [est_m["OLS"], est_m["IV-p"], est_m["IV-both"], est_m["Logit-IV"], est_f["IV-both"]]
    rows = []
    for i, lab in enumerate(labels):
        means, sds = [], []
        for j, c in enumerate(cols):
            if j == 3 and i == 3:
                means.append(r"0 (imposed)")
                sds.append("")
                continue
            m_, s_ = fmt(c[:, i].mean(), c[:, i].std(ddof=1))
            means.append(m_)
            sds.append(s_)
        rows.append(f"{lab} & {truth[i]:.1f} & " + " & ".join(means) + r" \\")
        rows.append(" & & " + " & ".join(sds) + r" \\")
    el_cols = [els_m["OLS"], els_m["IV-p"], els_m["IV-both"], els_m["Logit-IV"], els_f["IV-both"]]
    el_true_m = els_m["True"].mean()
    el_true_f = els_f["True"].mean()
    means = [f"{np.mean(e):.2f}" for e in el_cols]
    sds = [f"({np.std(e, ddof=1):.2f})" for e in el_cols]
    rows.append(r"\addlinespace")
    rows.append(r"Mean own elasticity & " + f"{el_true_m:.2f}" + " & " + " & ".join(means) + r" \\")
    rows.append(" & & " + " & ".join(sds) + r" \\")
    with open(os.path.join(OUT, "mc_nl_table.tex"), "w") as f:
        f.write("% Generated by figures/ch07/src/mc_nested_logit.py; do not edit by hand.\n")
        f.write(f"% True mean own elasticity: main design {el_true_m:.3f}, fixed-size design {el_true_f:.3f}\n")
        f.write("\n".join(rows) + "\n")
    print("table written")


if __name__ == "__main__":
    main()
