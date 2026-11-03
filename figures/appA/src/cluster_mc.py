"""
Monte Carlo of inference with clustered errors for Appendix A, Part 3 (Section A.8.2).

Produces (paths relative to Textbook/):
    figures/appA/cluster_mc.pdf                fig:A-cluster-mc
    figures/appA/src/cluster_mc_numbers.json   every number quoted in the text
    figures/appA/src/cluster_mc_preview.png    preview for checking

Design:
    y_ic = b0 + b1 x_ic + e_ic,  c = 1..C clusters, i = 1..n_c observations per cluster (n_c = 20),
    x_ic = sqrt(rho_x) a_c + sqrt(1 - rho_x) u_ic,   e_ic = sqrt(rho_e) g_c + sqrt(1 - rho_e) v_ic,
    a_c, u_ic, g_c, v_ic i.i.d. N(0, 1), rho_x = rho_e = 0.5, b0 = 0, b1 = 0 (the null is true).
    C in {5, 10, 20, 50, 100}; 1,000 Monte Carlo replications per C.
Tests of H0: b1 = 0 at nominal level 0.05:
    (1) conventional OLS standard error, normal quantiles
    (2) heteroskedasticity-robust HC1 standard error, normal quantiles
    (3) cluster-robust CR1 standard error, normal quantiles
    (4) CR1 standard error, t(C - 1) quantiles
    (5) wild cluster bootstrap-t: restricted residuals (null imposed), Rademacher weights drawn by
        cluster, symmetric p-value. When 2^C <= B the p-value is the exact conditional bootstrap
        p-value: all 2^C weight vectors are enumerated, each with probability 2^{-C}. Otherwise
        it is simulated from B = 399 random weight vectors.
    (6) wild cluster bootstrap-t as in (5), with the six-point weights of Webb (values
        -sqrt(3/2), -1, -sqrt(1/2), sqrt(1/2), 1, sqrt(3/2), each with probability 1/6) and
        B = 399 random weight vectors drawn from a separate random stream.
CR1: V = C/(C-1) (n-1)/(n-K) (X'X)^{-1} [sum_c X_c' e_c e_c' X_c] (X'X)^{-1}, K = 2.

Ties. The weight vectors w and -w give the same |t*|, and the all-ones vector reproduces the
data, so |t*| = |t| holds exactly for some draws. Floating-point roundoff can push such a tie
just below |t|. The p-values therefore count |t*| >= |t| - TIE_TOL max(1, |t|), with
TIE_TOL = 1e-10, far above roundoff (about 1e-13) and far below any genuine gap.
For C = 5 the json also records, as a diagnostic, the rejection rate of the simulated p-value
from the B random Rademacher draws, with ties counted ("wild_sim_tol") and with the raw
comparison |t*| >= |t| ("wild_sim_raw").
"""
import os
import json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
SEED = 20261022
RHO_X, RHO_E, NC = 0.5, 0.5, 20
C_LIST = [5, 10, 20, 50, 100]
REPS = 1000
B = 399
ALPHA = 0.05
TIE_TOL = 1e-10
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def boot_t(W, cl, y, X, H, h, G, kcr):
    """Bootstrap t statistics of the slope for a (C, B) matrix of cluster weights."""
    etil = y - y.mean()                     # restricted residuals (regression on a constant)
    ystar = y.mean() + etil[:, None] * W[cl]
    bstar = H @ ystar                       # (2, B)
    estar = ystar - X @ bstar
    sestar = np.sqrt(kcr * np.sum((G @ (h[:, None] * estar)) ** 2, axis=0))
    return bstar[1] / sestar


def pval(tstar, t, tol=True):
    """Symmetric bootstrap p-value, with numerical ties counted when tol is True."""
    cut = abs(t) - TIE_TOL * max(1.0, abs(t)) if tol else abs(t)
    return float(np.mean(np.abs(tstar) >= cut))


def all_signs(C):
    """All 2^C Rademacher weight vectors as a (C, 2^C) matrix."""
    k = np.arange(2 ** C)
    return (((k[None, :] >> np.arange(C)[:, None]) & 1) * 2.0 - 1.0)


def one_rep(rng, rng_webb, C):
    n = C * NC
    cl = np.repeat(np.arange(C), NC)
    x = np.sqrt(RHO_X) * rng.standard_normal(C)[cl] + np.sqrt(1 - RHO_X) * rng.standard_normal(n)
    e = np.sqrt(RHO_E) * rng.standard_normal(C)[cl] + np.sqrt(1 - RHO_E) * rng.standard_normal(n)
    y = e                                   # b0 = b1 = 0
    X = np.column_stack([np.ones(n), x])
    XtXi = np.linalg.inv(X.T @ X)
    H = XtXi @ X.T                          # (2, n): rows give the coefficient weights
    h = H[1]                                # slope weights: b1_hat = h @ y
    G = np.zeros((C, n))
    G[cl, np.arange(n)] = 1.0
    K = 2
    kcr = C / (C - 1) * (n - 1) / (n - K)

    b = H @ y
    ehat = y - X @ b
    s2 = ehat @ ehat / (n - K)
    se_conv = np.sqrt(s2 * XtXi[1, 1])
    se_hc1 = np.sqrt(n / (n - K) * np.sum((h * ehat) ** 2))
    se_cr1 = np.sqrt(kcr * np.sum((G @ (h * ehat)) ** 2))
    t_conv, t_hc1, t_cr1 = b[1] / se_conv, b[1] / se_hc1, b[1] / se_cr1
    z = stats.norm.ppf(1 - ALPHA / 2)
    tc = stats.t.ppf(1 - ALPHA / 2, C - 1)

    # wild cluster bootstrap-t with the null imposed, Rademacher weights
    args = (cl, y, X, H, h, G, kcr)
    W = rng.choice([-1.0, 1.0], size=(C, B))              # (C, B) Rademacher by cluster
    tstar_sim = boot_t(W, *args)
    p_sim_tol = pval(tstar_sim, t_cr1)
    p_sim_raw = pval(tstar_sim, t_cr1, tol=False)
    if 2 ** C <= B:
        tstar_all = boot_t(all_signs(C), *args)
        p_wild = pval(tstar_all, t_cr1)
        n_distinct = int(np.unique(np.round(np.abs(tstar_all), 8)).size)
    else:
        p_wild = p_sim_tol
        n_distinct = None
    # wild cluster bootstrap-t with the six-point weights, separate random stream
    Wwebb = rng_webb.choice(WEBB, size=(C, B))
    p_webb = pval(boot_t(Wwebb, *args), t_cr1)

    return {"conv": abs(t_conv) > z, "hc1": abs(t_hc1) > z, "cr1_z": abs(t_cr1) > z,
            "cr1_t": abs(t_cr1) > tc, "wild": p_wild <= ALPHA, "webb": p_webb <= ALPHA,
            "wild_sim_tol": p_sim_tol <= ALPHA, "wild_sim_raw": p_sim_raw <= ALPHA,
            "p_wild": p_wild, "n_distinct": n_distinct,
            "b1": b[1], "se_conv": se_conv, "se_cr1": se_cr1}


def main():
    out = {"seed": SEED, "rho_x": RHO_X, "rho_e": RHO_E, "n_c": NC, "reps": REPS, "B": B,
           "tie_tol": TIE_TOL, "webb_weights": [float(v) for v in WEBB],
           "moulton_factor_theory": 1 + (NC - 1) * RHO_X * RHO_E, "results": {}}
    for C in C_LIST:
        rng = np.random.default_rng([SEED, C])
        rng_webb = np.random.default_rng([SEED, C, 6])
        res = [one_rep(rng, rng_webb, C) for _ in range(REPS)]
        rr = {k: float(np.mean([r[k] for r in res])) for k in ("conv", "hc1", "cr1_z", "cr1_t", "wild", "webb")}
        rr["wild_method"] = "exact enumeration of 2^C weight vectors" if 2 ** C <= B else "B random draws"
        if 2 ** C <= B:
            pw = np.array([r["p_wild"] for r in res])
            rr["wild_exact_min_p"] = float(pw.min())
            rr["wild_exact_p_floor"] = 2.0 / 2 ** C
            rr["wild_max_distinct_abs_t"] = int(max(r["n_distinct"] for r in res))
            rr["wild_sim_tol"] = float(np.mean([r["wild_sim_tol"] for r in res]))
            rr["wild_sim_raw"] = float(np.mean([r["wild_sim_raw"] for r in res]))
        b1 = np.array([r["b1"] for r in res])
        rr["sd_b1"] = float(b1.std(ddof=1))
        rr["mean_se_conv"] = float(np.mean([r["se_conv"] for r in res]))
        rr["mean_se_cr1"] = float(np.mean([r["se_cr1"] for r in res]))
        rr["var_ratio_true_to_conv"] = float(b1.var(ddof=1) / np.mean([r["se_conv"] ** 2 for r in res]))
        out["results"][str(C)] = rr
        print(C, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rr.items()})
    out["mcse_at_005"] = float(np.sqrt(ALPHA * (1 - ALPHA) / REPS))
    with open(os.path.join(HERE, "cluster_mc_numbers.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out)


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, PURPLE, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#7b5cd6", "#d55181"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "mathtext.fontset": "stix", "font.family": "STIXGeneral"})
    Cs = np.array(C_LIST)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    specs = [("conv", "conventional s.e.", INK2, "D", ":"),
             ("hc1", "HC1 s.e.", PURPLE, "v", ":"),
             ("cr1_z", "CR1 s.e., normal quantiles", BLUE, "o", "-"),
             ("cr1_t", r"CR1 s.e., $t_{C-1}$ quantiles", ORANGE, "s", "--"),
             ("wild", "wild cluster bootstrap-$t$, Rademacher", AQUA, "^", "-."),
             ("webb", "wild cluster bootstrap-$t$, six-point", MAGENTA, "P", (0, (5, 1.5, 1, 1.5, 1, 1.5)))]
    for key, lab, col, mk, ls in specs:
        vals = [out["results"][str(C)][key] for C in C_LIST]
        ax.plot(Cs, vals, color=col, marker=mk, ms=4.5, lw=1.4, ls=ls, label=lab)
    band = 1.96 * out["mcse_at_005"]
    ax.axhspan(ALPHA - band, ALPHA + band, color=GRID, alpha=0.8, lw=0, zorder=0)
    ax.axhline(ALPHA, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks(Cs)
    ax.set_xticklabels([str(C) for C in C_LIST])
    ax.minorticks_off()
    ax.set_ylim(0, 0.8)
    ax.set_xlabel(r"number of clusters $C$ (20 observations per cluster)")
    ax.set_ylabel("rejection rate of a nominal 5% test")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, loc="upper center", ncol=2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "cluster_mc.pdf"))
    fig.savefig(os.path.join(HERE, "cluster_mc_preview.png"), dpi=160)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--figure-only":
        with open(os.path.join(HERE, "cluster_mc_numbers.json")) as fh:
            make_figure(json.load(fh))
    else:
        main()
