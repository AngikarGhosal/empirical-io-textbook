"""Monte Carlo replication of Berry (1994, Section 8 and Table 1).

Design (Berry 1994, p. 258): each sample has 500 independent duopoly markets.
Each firm sells one product. Consumer utility is
    u_ij = beta0 + betax * x_j + sigma_d * xi_j - alpha * p_j + eps_ij,
with an outside good u_i0 = eps_i0 and type-I extreme value eps.
Marginal cost is
    c_j = exp(gamma0 + gammax * x_j + sigma_c * xi_j + gammaw * w_j + sigma_w * omega_j).
x, xi, w, omega are independent standard normal. Parameters: beta0 = 5,
betax = 2, alpha = 1, gamma0 = 1, gammax = 0.5, gammaw = sigma_w = sigma_c = 0.25.
Prices are the single-product Bertrand-Nash equilibrium, solved market by
market with Lambert-W best responses. Shares are exact (no sampling error).

Estimators: OLS of ln s_j - ln s_0 on (1, x_j, p_j); 2SLS with the excluded
instruments (w_j, x_rival). We also report 2SLS with each excluded instrument
alone and an extension in which the number of products per market grows.

Outputs (paths relative to Textbook/):
    figures/ch07/mc_logit_ols_iv.pdf   sampling distributions of alpha-hat
    figures/ch07/mc_logit_table.tex    LaTeX rows for tab:07-mc-logit
    figures/ch07/src/mc_logit_numbers.txt   every number quoted in the text
Only numpy, scipy and matplotlib are used. Seeds are fixed.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)

PAR = dict(beta0=5.0, betax=2.0, alpha=1.0, gamma0=1.0, gammax=0.5,
           gammaw=0.25, sigma_w=0.25, sigma_c=0.25)

# Berry (1994), Table 1, empirical means and (standard errors).
BERRY = {
    1: {"OLS": {"beta0": (3.46, 0.158), "betax": (1.41, 0.058), "alpha": (0.726, 0.029)},
        "IV": {"beta0": (4.98, 0.226), "betax": (1.99, 0.091), "alpha": (0.995, 0.039)}},
    3: {"OLS": {"beta0": (0.378, 0.415), "betax": (0.325, 0.127), "alpha": (0.181, 0.076)},
        "IV": {"beta0": (4.89, 0.738), "betax": (1.95, 0.272), "alpha": (0.979, 0.128)}},
}


def equilibrium(x, xi, w, om, sigma_d, tol=1e-13, maxit=5000):
    """Single-product logit Bertrand-Nash equilibrium in every market.

    Arrays have shape (T, J). Returns prices, shares, outside shares, costs.
    Best response: m_j = (1/alpha) * [1 + W(exp(v_j - 1) / D_{-j})], where
    v_j = a_j - alpha * mc_j and D_{-j} = 1 + sum_{k != j} exp(delta_k).
    """
    a_ = PAR["alpha"]
    mc = np.exp(PAR["gamma0"] + PAR["gammax"] * x + PAR["sigma_c"] * xi
                + PAR["gammaw"] * w + PAR["sigma_w"] * om)
    quality = PAR["beta0"] + PAR["betax"] * x + sigma_d * xi
    v = quality - a_ * mc
    m = np.full_like(mc, 1.0 / a_)
    for it in range(maxit):
        e = np.exp(v - a_ * m)
        tot = 1.0 + e.sum(axis=1, keepdims=True)
        dminus = tot - e
        z = np.exp(v - 1.0 - np.log(dminus))
        m_new = (1.0 + lambertw(z).real) / a_
        if np.max(np.abs(m_new - m)) < tol:
            m = m_new
            break
        m = m_new
    else:
        raise RuntimeError("equilibrium did not converge")
    e = np.exp(v - a_ * m)
    tot = 1.0 + e.sum(axis=1, keepdims=True)
    s = e / tot
    s0 = (1.0 / tot).ravel()
    # FOC check: m = 1 / (alpha (1 - s))
    foc = np.max(np.abs(m - 1.0 / (a_ * (1.0 - s))))
    return mc + m, s, s0, mc, it + 1, foc


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def tsls(y, X, Z):
    PZ_X = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return np.linalg.solve(PZ_X.T @ X, PZ_X.T @ y)


def first_stage_F(p, Zexog, Zexcl):
    """Homoskedastic F statistic for the excluded instruments."""
    Zfull = np.column_stack([Zexog, Zexcl])
    r_u = p - Zfull @ ols(p, Zfull)
    r_r = p - Zexog @ ols(p, Zexog)
    q = Zexcl.shape[1] if Zexcl.ndim > 1 else 1
    n, k = Zfull.shape
    return ((r_r @ r_r - r_u @ r_u) / q) / ((r_u @ r_u) / (n - k))


def one_sample(rng, sigma_d, T=500, J=2):
    x = rng.standard_normal((T, J))
    xi = rng.standard_normal((T, J))
    w = rng.standard_normal((T, J))
    om = rng.standard_normal((T, J))
    p, s, s0, mc, nit, foc = equilibrium(x, xi, w, om, sigma_d)
    y = (np.log(s) - np.log(s0)[:, None]).ravel()
    xr = (x.sum(axis=1, keepdims=True) - x).ravel()  # rivals' characteristics
    xv, pv, wv, sv = x.ravel(), p.ravel(), w.ravel(), s.ravel()
    one = np.ones_like(xv)
    X = np.column_stack([one, xv, pv])
    Zexog = np.column_stack([one, xv])
    res = {}
    b = ols(y, X)
    res["OLS"] = np.array([b[0], b[1], -b[2]])
    b = tsls(y, X, np.column_stack([Zexog, wv, xr]))
    res["IV"] = np.array([b[0], b[1], -b[2]])
    b = tsls(y, X, np.column_stack([Zexog, wv]))
    res["IV_w"] = np.array([b[0], b[1], -b[2]])
    b = tsls(y, X, np.column_stack([Zexog, xr]))
    res["IV_xr"] = np.array([b[0], b[1], -b[2]])
    res["F_both"] = first_stage_F(pv, Zexog, np.column_stack([wv, xr]))
    res["F_w"] = first_stage_F(pv, Zexog, wv[:, None])
    res["F_xr"] = first_stage_F(pv, Zexog, xr[:, None])
    # implied own elasticities alpha_hat * p * (1 - s) under OLS
    el_ols = res["OLS"][2] * pv * (1.0 - sv)
    res["share_inelastic_ols"] = np.mean(el_ols < 1.0)
    el_iv = res["IV"][2] * pv * (1.0 - sv)
    res["share_inelastic_iv"] = np.mean(el_iv < 1.0)
    res["mean_s0"] = s0.mean()
    res["min_s"] = s.min()
    res["mean_p"] = pv.mean()
    res["mean_mc"] = mc.mean()
    res["mean_markup"] = (pv - mc.ravel()).mean()
    res["corr_p_xi"] = None
    res["nit"] = nit
    res["foc"] = foc
    return res


def run(sigma_d, R, seed):
    rng = np.random.default_rng(seed)
    return [one_sample(rng, sigma_d) for _ in range(R)]


def summarize(results, key):
    arr = np.array([r[key] for r in results])
    return arr.mean(axis=0), arr.std(axis=0, ddof=1)


def many_products(Jlist, R=200, N=1000, sigma_d=1.0, seed=7, gammax=0.0):
    """2SLS with the rival-characteristics instrument only, as J grows.

    By default x is removed from marginal cost (gammax = 0), so that the
    rivals' characteristics shift rivals' appeal one for one and the
    instrument is strong in duopoly.
    """
    saved = PAR["gammax"]
    PAR["gammax"] = gammax
    rng = np.random.default_rng(seed)
    out = []
    for J in Jlist:
        T = N // J
        a_iv, a_w, a_o, Fs, s0s, mk = [], [], [], [], [], []
        for _ in range(R):
            x = rng.standard_normal((T, J))
            xi = rng.standard_normal((T, J))
            w = rng.standard_normal((T, J))
            om = rng.standard_normal((T, J))
            p, s, s0, mc, _, _ = equilibrium(x, xi, w, om, sigma_d)
            y = (np.log(s) - np.log(s0)[:, None]).ravel()
            xr = (x.sum(axis=1, keepdims=True) - x).ravel()
            xv, pv, wv = x.ravel(), p.ravel(), w.ravel()
            one = np.ones_like(xv)
            X = np.column_stack([one, xv, pv])
            Zexog = np.column_stack([one, xv])
            a_iv.append(-tsls(y, X, np.column_stack([Zexog, xr]))[2])
            a_w.append(-tsls(y, X, np.column_stack([Zexog, wv]))[2])
            a_o.append(-ols(y, X)[2])
            Fs.append(first_stage_F(pv, Zexog, xr[:, None]))
            s0s.append(s0.mean())
            mk.append((pv - mc.ravel()).std())
        a_iv, a_w = np.array(a_iv), np.array(a_w)
        out.append(dict(J=J, T=T, medF=np.median(Fs), med_a=np.median(a_iv),
                        q25=np.percentile(a_iv, 25), q75=np.percentile(a_iv, 75),
                        med_a_w=np.median(a_w), med_ols=np.median(a_o), s0=np.mean(s0s), sd_markup=np.mean(mk)))
    PAR["gammax"] = saved
    return out


def fmt(v, d=3):
    return f"{v:.{d}f}"


def main():
    names = ["beta0", "betax", "alpha"]
    lines = []
    tab = {}
    R_table, R_fig = 100, 1000
    for sd, seed in [(1, 1994), (3, 2581)]:
        res = run(sd, R_table, seed)
        tab[sd] = res
        lines.append(f"=== sigma_d = {sd}, R = {R_table} samples of 500 duopoly markets ===")
        for est in ["OLS", "IV", "IV_w", "IV_xr"]:
            m, s = summarize(res, est)
            lines.append(f"{est:6s} " + "  ".join(f"{n}={m[i]:.3f} ({s[i]:.3f})" for i, n in enumerate(names)))
        for k in ["F_both", "F_w", "F_xr"]:
            arr = np.array([r[k] for r in res])
            lines.append(f"{k}: mean {arr.mean():.1f}, median {np.median(arr):.1f}, min {arr.min():.1f}, max {arr.max():.1f}")
        a_ols = np.array([r["OLS"][2] for r in res])
        a_iv = np.array([r["IV"][2] for r in res])
        inel = np.array([r["share_inelastic_ols"] for r in res])
        inel_iv = np.array([r["share_inelastic_iv"] for r in res])
        lines.append(f"OLS alpha<0 in {np.sum(a_ols < 0)} of {R_table} samples")
        lines.append(f"OLS: mean share of products with inelastic demand {inel.mean():.3f}; samples with any inelastic {np.sum(inel > 0)}")
        lines.append(f"IV: mean share of products with inelastic demand {inel_iv.mean():.4f}; samples with any inelastic {np.sum(inel_iv > 0)}")
        for k in ["mean_s0", "min_s", "mean_p", "mean_mc", "mean_markup", "nit", "foc"]:
            arr = np.array([r[k] for r in res])
            lines.append(f"{k}: mean {arr.mean():.4g}, min {arr.min():.4g}, max {arr.max():.4g}")
        lines.append("")

    # LaTeX rows: parameter, true value, Berry OLS, ours OLS, Berry IV, ours IV.
    true = {"beta0": 5, "betax": 2, "alpha": 1}
    sym = {"beta0": r"$\beta_0$", "betax": r"$\beta_x$", "alpha": r"$\alpha$"}
    tex = []
    for sd in [1, 3]:
        tex.append(r"\multicolumn{6}{l}{\textit{Panel " + ("A" if sd == 1 else "B")
                   + r": $\sigma_d = " + str(sd) + r"$}} \\")
        mo, so = summarize(tab[sd], "OLS")
        mi, si = summarize(tab[sd], "IV")
        for i, n in enumerate(names):
            bo, bi = BERRY[sd]["OLS"][n], BERRY[sd]["IV"][n]
            tex.append(f"{sym[n]} & {true[n]} & {bo[0]:.3f} & {mo[i]:.3f} & {bi[0]:.3f} & {mi[i]:.3f} \\\\")
            tex.append(f" & & ({bo[1]:.3f}) & ({so[i]:.3f}) & ({bi[1]:.3f}) & ({si[i]:.3f}) \\\\")
        if sd == 1:
            tex.append(r"\addlinespace")
    with open(os.path.join(OUT, "mc_logit_table.tex"), "w") as f:
        f.write("% Generated by figures/ch07/src/mc_logit_iv.py\n")
        f.write("\n".join(tex) + "\n")

    # Figure: 1,000 samples per design.
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    figstats = {}
    for ax, (sd, seed) in zip(axes, [(1, 11), (3, 33)]):
        res = run(sd, R_fig, seed)
        a_ols = np.array([r["OLS"][2] for r in res])
        a_iv = np.array([r["IV"][2] for r in res])
        figstats[sd] = (a_ols.mean(), a_ols.std(ddof=1), a_iv.mean(), a_iv.std(ddof=1),
                        np.sum(a_ols < 0), np.mean(np.abs(a_iv - 1) < np.abs(a_ols - 1)))
        lo = min(a_ols.min(), a_iv.min())
        hi = max(a_ols.max(), a_iv.max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(a_ols, bins=bins, color="#eb6834", alpha=0.85, label="OLS",
                edgecolor="white", linewidth=0.4)
        ax.hist(a_iv, bins=bins, color="#2a78d6", alpha=0.85, label="2SLS",
                edgecolor="white", linewidth=0.4)
        ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.2)
        ax.text(1.0, ax.get_ylim()[1] * 0.93, r" true $\alpha = 1$", fontsize=8,
                color="#333333", ha="left", va="top")
        ax.set_title(r"$\sigma_d = %d$" % sd, fontsize=10)
        ax.set_xlabel(r"Estimated price coefficient $\hat\alpha$", fontsize=9)
        ax.tick_params(labelsize=8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Number of samples", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mc_logit_ols_iv.pdf"))
    fig.savefig(os.path.join(HERE, "mc_logit_ols_iv_preview.png"), dpi=130)
    lines.append(f"=== Figure: R = {R_fig} samples per design ===")
    for sd, v in figstats.items():
        lines.append(f"sigma_d={sd}: OLS mean {v[0]:.3f} sd {v[1]:.3f}; IV mean {v[2]:.3f} sd {v[3]:.3f}; "
                     f"OLS<0 in {v[4]} samples; IV closer to truth in {v[5]:.3f} of samples")
    lines.append("")

    # Extension: weak BLP-type instruments as the number of products grows.
    lines.append("=== Extension: rival-characteristics instrument only, gammax = 0, N = 1000 obs, sigma_d = 1, R = 200 ===")
    lines.append("J  T  median_F  median_alpha_IV(xr)  IQR  median_alpha_IV(w)  median_alpha_OLS  mean_s0  mean_sd_markup")
    for r in many_products([2, 5, 10, 25, 50]):
        lines.append(f"{r['J']:3d} {r['T']:4d} {r['medF']:8.1f} {r['med_a']:8.3f} "
                     f"[{r['q25']:.3f}, {r['q75']:.3f}] {r['med_a_w']:8.3f} {r['med_ols']:8.3f} {r['s0']:.4f} {r['sd_markup']:.4f}")
    lines.append("")

    # Why the rival's characteristic is a weak instrument in this design:
    # x raises marginal cost as well as utility, so net mean utility barely moves.
    rng = np.random.default_rng(1)
    T, J = 20000, 2
    x, xi, w, om = [rng.standard_normal((T, J)) for _ in range(4)]
    p, s, s0, mc, _, _ = equilibrium(x, xi, w, om, 1.0)
    net = (PAR["beta0"] + PAR["betax"] * x + 1.0 * xi - PAR["alpha"] * p).ravel()
    Xc = np.column_stack([np.ones(T * J), x.ravel()])
    lines.append("=== Large-sample projections on x (sigma_d = 1, 20,000 markets) ===")
    lines.append(f"slope of p on x: {ols(p.ravel(), Xc)[1]:.3f}; slope of mc on x: {ols(mc.ravel(), Xc)[1]:.3f}; "
                 f"slope of net mean utility (beta0 + betax x + xi - alpha p) on x: {ols(net, Xc)[1]:.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "mc_logit_numbers.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
