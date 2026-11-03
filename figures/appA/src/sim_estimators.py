"""
Monte Carlo comparison of simulation estimators for Appendix A, Part 3 (Section A.7).

Produces (paths relative to Textbook/):
    figures/appA/simest.pdf                 fig:A-simest
    figures/appA/src/simest_numbers.json    every number quoted in the text
    figures/appA/src/simest_preview.png     preview for checking

Design (a panel binary logit with a normal random coefficient):
    u_it = b0 + beta_i x_it + eps_it,  beta_i = b1 + sigma nu_i,  nu_i ~ N(0, 1),
    eps_it i.i.d. logistic, y_it = 1{u_it > 0},  i = 1..N individuals, t = 1..T occasions.
    True values: b0 = -0.5, b1 = 1.0, sigma = 1.0. x_it ~ N(0.5, 1). N = 1,000, T = 5.
    The random coefficient is constant over the T occasions of an individual, so the likelihood
    of individual i is f_i = int prod_t Lambda(v_it(nu))^{y_it} (1 - Lambda(v_it(nu)))^{1 - y_it} phi(nu) dnu.

Estimators (draws nu_ir, r = 1..n_s, independent across i, held fixed during each optimization):
    SML     maximize sum_i ln f~_i, f~_i = (1/n_s) sum_r L_ir
    BC-SML  maximize sum_i [ln f~_i + 0.5 V~_i / f~_i^2], V~_i = sum_r (L_ir - f~_i)^2 / (n_s (n_s - 1))
            (first-order bias correction, Gourieroux and Monfort 1996)
    MSM     just-identified method of simulated moments with three moments per individual:
            m1 = sum_t (y_it - P~_it), m2 = sum_t (y_it - P~_it) x_it,
            m3 = sum_{t<s} (y_it y_is - P~_its) x_it x_is,
            with P~_it = (1/n_s) sum_r Lambda_itr and P~_its = (1/n_s) sum_r Lambda_itr Lambda_isr.
            Each simulated moment is linear in an unbiased simulator.
    Benchmarks with exact integrals (40-node Gauss-Hermite quadrature): ML and exact-moment MM.
    n_s in {5, 10, 25, 50, 200}; R = 300 Monte Carlo replications; fresh data and draws each replication.
"""
import os
import json
import time
import numpy as np
from scipy.optimize import minimize, least_squares
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
SEED = 20261022
B0, B1, SIG = -0.5, 1.0, 1.0
THETA0 = np.array([B0, B1, SIG])
N, T = 1000, 5
NS_LIST = [5, 10, 25, 50, 200]
R = 300
GH_NODES = 40


def draw_data(rng):
    x = 0.5 + rng.standard_normal((N, T))
    nu = rng.standard_normal(N)
    v = B0 + (B1 + SIG * nu)[:, None] * x
    y = (rng.random((N, T)) < 1.0 / (1.0 + np.exp(-v))).astype(float)
    return y, x


def lam(v):
    return 1.0 / (1.0 + np.exp(-v))


# ----------------------------------------------------------------------------- likelihood
def indiv_lik(theta, y, x, nodes):
    """L_ir and its gradient. nodes: (N, R) array of draws. Returns L (N,R), dL (3,N,R)."""
    b0, b1, s = theta
    v = b0 + (b1 + s * nodes)[:, :, None] * x[:, None, :]          # (N, R, T)
    p = lam(v)
    yy = y[:, None, :]
    ll = np.where(yy > 0.5, np.log(np.clip(p, 1e-300, None)), np.log(np.clip(1 - p, 1e-300, None)))
    L = np.exp(ll.sum(axis=2))                                       # (N, R)
    resid = yy - p                                                   # d ln L / d v
    d0 = resid.sum(axis=2)
    d1 = (resid * x[:, None, :]).sum(axis=2)
    d2 = d1 * nodes
    dL = np.stack([L * d0, L * d1, L * d2])                          # (3, N, R)
    return L, dL


def neg_sml(theta, y, x, nodes, weights, correct):
    L, dL = indiv_lik(theta, y, x, nodes)
    f = L @ weights                                                  # (N,)
    df = dL @ weights                                                # (3, N)
    f = np.clip(f, 1e-300, None)
    obj = np.log(f).sum()
    grad = (df / f).sum(axis=1)
    if correct:
        ns = nodes.shape[1]
        dev = L - f[:, None]
        V = (dev ** 2).sum(axis=1) / (ns * (ns - 1))
        dV = 2.0 * (dev[None] * (dL - df[:, :, None])).sum(axis=2) / (ns * (ns - 1))
        corr = 0.5 * V / f ** 2
        dcorr = 0.5 * (dV / f ** 2 - 2.0 * V * df / f ** 3)
        obj += corr.sum()
        grad += dcorr.sum(axis=1)
    return -obj / N, -grad / N


def fit_sml(y, x, nodes, weights, correct, start):
    res = minimize(neg_sml, start, args=(y, x, nodes, weights, correct), jac=True,
                   method="BFGS", options={"gtol": 1e-7, "maxiter": 500})
    th = res.x.copy()
    th[2] = abs(th[2])
    return th, res.success


# ----------------------------------------------------------------------------- moments
def moments(theta, y, x, nodes, weights):
    """Average moment vector (3,) and Jacobian (3,3)."""
    b0, b1, s = theta
    v = b0 + (b1 + s * nodes)[:, :, None] * x[:, None, :]          # (N, R, T)
    p = lam(v)
    dp = p * (1 - p)
    xx = x[:, None, :]
    # simulated P_it and derivatives
    P = np.einsum("nrt,r->nt", p, weights)
    dPv = [np.einsum("nrt,r->nt", dp, weights),
           np.einsum("nrt,r->nt", dp * xx, weights),
           np.einsum("nrt,r->nt", dp * xx * nodes[:, :, None], weights)]
    # pairwise term: sum_{t<s} p_t p_s x_t x_s = 0.5[(sum p x)^2 - sum p^2 x^2], per draw
    a = (p * xx).sum(axis=2)
    bq = (p ** 2 * xx ** 2).sum(axis=2)
    pair = 0.5 * (a ** 2 - bq)                                       # (N, R)
    Ppair = pair @ weights
    dv = [np.ones_like(p), np.broadcast_to(xx, p.shape), xx * nodes[:, :, None]]
    dPpair = []
    for k in range(3):
        da = (dp * dv[k] * xx).sum(axis=2)
        dbq = (2 * p * dp * dv[k] * xx ** 2).sum(axis=2)
        dPpair.append((a * da - 0.5 * dbq) @ weights)
    ya = (y * x).sum(axis=1)
    ypair = 0.5 * (ya ** 2 - (y * x ** 2).sum(axis=1))
    g = np.array([(y - P).sum(axis=1).mean(),
                  ((y - P) * x).sum(axis=1).mean(),
                  (ypair - Ppair).mean()])
    J = np.zeros((3, 3))
    for k in range(3):
        J[0, k] = -dPv[k].sum(axis=1).mean()
        J[1, k] = -(dPv[k] * x).sum(axis=1).mean()
        J[2, k] = -dPpair[k].mean()
    return g, J


def fit_msm(y, x, nodes, weights, start):
    res = least_squares(lambda th: moments(th, y, x, nodes, weights)[0], start,
                        jac=lambda th: moments(th, y, x, nodes, weights)[1],
                        method="lm", xtol=1e-10, ftol=1e-12)
    th = res.x.copy()
    th[2] = abs(th[2])
    return th, res.success


# ----------------------------------------------------------------------------- one replication
def one_rep(r):
    rng = np.random.default_rng([SEED, r])
    y, x = draw_data(rng)
    out = {}
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(GH_NODES)
    gh_w = gh_w / gh_w.sum()
    nodes_q = np.broadcast_to(gh_x, (N, GH_NODES))
    start = THETA0.copy()
    out["ML"], _ = fit_sml(y, x, nodes_q, gh_w, False, start)
    out["MM"], _ = fit_msm(y, x, nodes_q, gh_w, start)
    for ns in NS_LIST:
        nodes = rng.standard_normal((N, ns))
        w = np.full(ns, 1.0 / ns)
        out[f"SML_{ns}"], _ = fit_sml(y, x, nodes, w, False, start)
        out[f"BCSML_{ns}"], _ = fit_sml(y, x, nodes, w, True, start)
        out[f"MSM_{ns}"], _ = fit_msm(y, x, nodes, w, start)
    return {k: v.tolist() for k, v in out.items()}


def summarize(results):
    keys = results[0].keys()
    S = {}
    for k in keys:
        a = np.array([res[k] for res in results])            # (R, 3)
        err = a - THETA0
        S[k] = {"mean": a.mean(0).tolist(), "bias": err.mean(0).tolist(),
                "sd": a.std(0, ddof=1).tolist(), "rmse": np.sqrt((err ** 2).mean(0)).tolist(),
                "mcse_bias": (a.std(0, ddof=1) / np.sqrt(len(a))).tolist()}
    return S


def make_figure(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7b5cd6"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "mathtext.fontset": "stix", "font.family": "STIXGeneral"})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    ns = np.array(NS_LIST)
    specs = [("SML", "SML", BLUE, "o", "-"), ("BCSML", "bias-corrected SML", ORANGE, "s", "--"),
             ("MSM", "MSM", AQUA, "^", "-.")]
    for j, (stat, ylabel, title) in enumerate([("bias", r"mean bias of $\hat\sigma$", "(a) Bias"),
                                                ("rmse", r"RMSE of $\hat\sigma$", "(b) Root mean squared error")]):
        ax = axes[j]
        for key, lab, col, mk, ls in specs:
            vals = np.array([S[f"{key}_{n}"][stat][2] for n in NS_LIST])
            ax.plot(ns, vals, color=col, marker=mk, ms=4.5, lw=1.4, ls=ls, label=lab)
            if stat == "bias":
                se = np.array([S[f"{key}_{n}"]["mcse_bias"][2] for n in NS_LIST])
                ax.fill_between(ns, vals - 1.96 * se, vals + 1.96 * se, color=col, alpha=0.12, lw=0)
        ax.axhline(S["ML"][stat][2], color=PURPLE, lw=1.0, ls=":", label="ML, exact integral")
        ax.axhline(S["MM"][stat][2], color=INK2, lw=1.0, ls=(0, (1, 2)), label="MM, exact moments")
        if stat == "bias":
            ax.axhline(0.0, color=GRID, lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in NS_LIST])
        ax.minorticks_off()
        ax.set_xlabel(r"simulation draws per individual, $n_s$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "simest.pdf"))
    fig.savefig(os.path.join(HERE, "simest_preview.png"), dpi=160)


if __name__ == "__main__":
    import sys
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else R
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(one_rep, range(reps)))
    S = summarize(results)
    S["_meta"] = {"seed": SEED, "N": N, "T": T, "R": reps, "theta0": THETA0.tolist(),
                  "ns_list": NS_LIST, "gh_nodes": GH_NODES, "seconds": time.time() - t0}
    # share of y = 1 and an illustrative simulator variance ratio for the text
    rng = np.random.default_rng([SEED, 999999])
    y, x = draw_data(rng)
    S["_meta"]["mean_y"] = float(y.mean())
    with open(os.path.join(HERE, "simest_numbers.json"), "w") as fh:
        json.dump(S, fh, indent=1)
    for k in ["ML", "MM"] + [f"{e}_{n}" for n in NS_LIST for e in ("SML", "BCSML", "MSM")]:
        b, s, rm = S[k]["bias"][2], S[k]["sd"][2], S[k]["rmse"][2]
        print(f"{k:10s} sigma: bias {b:+.4f} sd {s:.4f} rmse {rm:.4f} | b1 bias {S[k]['bias'][1]:+.4f}")
    print("seconds", time.time() - t0)
    make_figure(S)
