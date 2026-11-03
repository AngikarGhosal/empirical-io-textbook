"""
Staggered difference in differences for Appendix A, Part 5 (Section A.12.6).

Produces (paths relative to Textbook/):
    figures/appA/staggered.pdf              fig:A-staggered
    figures/appA/src/staggered_numbers.json every number quoted in the text
    figures/appA/src/staggered_preview.png  preview for checking

Design (balanced panel, parallel trends and no anticipation hold exactly):
    Units i = 1..N, periods t = 1..T with T = 20.
    Adoption cohorts E_i in {5, 10, 15} and a never-treated group (E_i = inf).
    Group sizes: N_E = 100 per adoption cohort, N_U = 100 never treated (N = 400).
    Untreated outcome y_it(0) = alpha_i + lambda_t + eps_it with
        alpha_i ~ N(0, 1), lambda_t = 0.05 t + N(0, 0.2^2) (redrawn each replication),
        eps_it ~ N(0, 0.5^2) i.i.d.
    Effect of treatment k = t - E_i >= 0 periods after adoption:
        tau_e(k) = gamma_e (k + 1), gamma_5 = 0.10, gamma_10 = 0.06, gamma_15 = 0.03.
    Effects therefore grow with time since adoption and are larger for earlier cohorts.

Estimators (computed in each of R = 500 replications):
    (1) Static TWFE: y_it = mu_i + lambda_t + beta D_it + e_it.
    (2) Goodman-Bacon decomposition of (1) into 2x2 comparisons (checked to sum to beta-hat).
    (3) Dynamic TWFE event study: dummies for k = -5..5 except k = -1, with binned end
        points k <= -6 and k >= 6 (not plotted); never-treated units have all dummies zero.
    (4) Callaway-Sant'Anna group-time ATTs with never-treated controls and base period E - 1,
        aggregated by cohort shares at each k:  ATT(e, e + k) = (ybar_{e, e+k} - ybar_{e, e-1})
        - (ybar_{U, e+k} - ybar_{U, e-1}). In this balanced design without covariates the
        Sun-Abraham interaction-weighted estimator with never-treated controls coincides with it
        (checked numerically on the first replication).
    (5) Imputation estimator (Borusyak, Jaravel and Spiess): fit mu_i + lambda_t on untreated
        observations (D_it = 0) by alternating projections, impute y_it(0) for treated
        observations, and average y_it - yhat_it(0) by relative time k.
Truth at relative time k: phi_k = sum_e w_e(k) tau_e(k), w_e(k) = cohort shares among cohorts
observed at k (all three cohorts are observed for k in [-4, 5]).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
SEED = 20261022
rng = np.random.default_rng(SEED)
OUT = {"seed": SEED}

# ------------------------------------------------------------------ style (book palette)
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7b5cd6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral"})

# ------------------------------------------------------------------ design
T = 20
COHORTS = [5, 10, 15]
GAMMA = {5: 0.10, 10: 0.06, 15: 0.03}
N_PER = 100          # units per adoption cohort
N_U = 100            # never-treated units
R = 500              # Monte Carlo replications
KMIN, KMAX = -5, 5   # plotted event window
SIG_EPS, SIG_LAM, TREND = 0.5, 0.2, 0.05

E = np.concatenate([np.full(N_PER, e, dtype=float) for e in COHORTS] + [np.full(N_U, np.inf)])
N = E.size
tt = np.arange(1, T + 1)
Emat = np.repeat(E[:, None], T, axis=1)
Tmat = np.repeat(tt[None, :], N, axis=0)
Kmat = Tmat - Emat                      # relative time (-inf for never treated)
Dmat = (Kmat >= 0).astype(float)
treated_cohort = np.isfinite(E)


def tau(e, k):
    k = np.asarray(k, dtype=float)
    return np.where(k >= 0, GAMMA[int(e)] * (k + 1.0), 0.0)


TAU = np.zeros((N, T))
for e in COHORTS:
    m = E == e
    TAU[m] = tau(e, Kmat[m])

# truth: event-time average with cohort shares among cohorts observed at k
def truth_phi(k):
    obs = [e for e in COHORTS if 1 <= e + k <= T]
    return float(np.mean([tau(e, k) for e in obs]))


KS = list(range(KMIN, KMAX + 1))
PHI = {k: truth_phi(k) for k in KS}
ATT_TRUE = float(TAU[Dmat == 1].mean())   # average effect over all treated unit-periods
OUT["truth_phi"] = {str(k): round(v, 4) for k, v in PHI.items()}
OUT["att_true_all_treated_cells"] = round(ATT_TRUE, 4)


# ------------------------------------------------------------------ helpers
def twoway_demean(M):
    """Double-demean an N x T (or N x T x p) array in a balanced panel (FWL for unit + time FE)."""
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean(axis=(0, 1), keepdims=True)


def twfe(Y, Xlist):
    """OLS of Y on regressors X (each N x T) with unit and time fixed effects (balanced panel)."""
    y = twoway_demean(Y).ravel()
    X = np.column_stack([twoway_demean(X).ravel() for X in Xlist])
    return np.linalg.lstsq(X, y, rcond=None)[0]


# event-study regressors: k=-5..5 except -1, plus bins k<=-6 and k>=6
ES_KEYS = [k for k in KS if k != -1]
ES_REGS = [(Kmat == k).astype(float) for k in ES_KEYS]
ES_REGS.append((Kmat <= KMIN - 1).astype(float))
ES_REGS.append((Kmat >= KMAX + 1).astype(float))


def cs_event(Y):
    """Callaway-Sant'Anna with never-treated controls, base period e-1, aggregated by cohort shares."""
    ybar = {e: Y[E == e].mean(axis=0) for e in COHORTS}
    yU = Y[~treated_cohort].mean(axis=0)
    est = {}
    for k in KS:
        vals, wts = [], []
        for e in COHORTS:
            t = e + k
            if 1 <= t <= T and k != -1:
                b = e - 1
                att = (ybar[e][t - 1] - ybar[e][b - 1]) - (yU[t - 1] - yU[b - 1])
                vals.append(att)
                wts.append(np.sum(E == e))
        est[k] = float(np.average(vals, weights=wts)) if vals else 0.0
    return est


def sa_event(Y):
    """Sun-Abraham: saturated cohort x relative-time TWFE regression, never-treated controls."""
    regs, keys = [], []
    for e in COHORTS:
        for k in range(1 - e, T - e + 1):
            if k == -1:
                continue
            regs.append(((E[:, None] == e) & (Kmat == k)).astype(float))
            keys.append((e, k))
    b = twfe(Y, regs)
    delta = dict(zip(keys, b))
    est = {}
    for k in KS:
        if k == -1:
            est[k] = 0.0
            continue
        vals = [delta[(e, k)] for e in COHORTS if (e, k) in delta]
        est[k] = float(np.mean(vals))
    return est


def impute_event(Y, tol=1e-11, maxit=5000):
    """Imputation estimator: unit and time effects from D=0 cells by alternating projections."""
    W = (Dmat == 0)
    mu = np.zeros(N)
    lam = np.zeros(T)
    cnt_i = W.sum(axis=1)
    cnt_t = W.sum(axis=0)
    for _ in range(maxit):
        mu_new = np.where(W, Y - lam[None, :], 0.0).sum(axis=1) / cnt_i
        lam_new = np.where(W, Y - mu_new[:, None], 0.0).sum(axis=0) / cnt_t
        diff = max(np.abs(mu_new - mu).max(), np.abs(lam_new - lam).max())
        mu, lam = mu_new, lam_new
        if diff < tol:
            break
    resid = Y - mu[:, None] - lam[None, :]
    est = {}
    for k in KS:
        if k < 0:
            est[k] = np.nan
            continue
        m = (Kmat == k)
        est[k] = float(resid[m].mean())
    return est


def bacon(Y):
    """Goodman-Bacon decomposition of the static TWFE coefficient (balanced panel)."""
    groups = COHORTS
    nshare = {e: np.mean(E == e) for e in groups}
    nU = np.mean(~treated_cohort)
    Dbar = {e: np.mean(tt >= e) for e in groups}
    VD = np.mean(twoway_demean(Dmat) ** 2)
    ybar = {e: Y[E == e].mean(axis=0) for e in groups}
    yU = Y[~treated_cohort].mean(axis=0)
    comps = []

    def dd(ya, yb, pre, post):
        return (ya[post].mean() - ya[pre].mean()) - (yb[post].mean() - yb[pre].mean())

    for e in groups:
        pre, post = tt < e, tt >= e
        b = dd(ybar[e], yU, pre, post)
        s = nshare[e] * nU * Dbar[e] * (1 - Dbar[e]) / VD
        comps.append(("treated vs never", e, None, b, s))
    for i, k in enumerate(groups):
        for l in groups[i + 1:]:
            # k adopts earlier than l: Dbar_k > Dbar_l
            win = tt < l
            pre, post = (tt < k)[win], ((tt >= k) & (tt < l))[win]
            b_early = dd(ybar[k][win], ybar[l][win], pre, post)
            s_early = nshare[k] * nshare[l] * (Dbar[k] - Dbar[l]) * (1 - Dbar[k]) / VD
            comps.append(("earlier vs later (not yet treated)", k, l, b_early, s_early))
            win = tt >= k
            pre, post = ((tt >= k) & (tt < l))[win], (tt >= l)[win]
            b_late = dd(ybar[l][win], ybar[k][win], pre, post)
            s_late = nshare[k] * nshare[l] * Dbar[l] * (Dbar[k] - Dbar[l]) / VD
            comps.append(("later vs earlier (already treated)", l, k, b_late, s_late))
    return comps


# ------------------------------------------------------------------ Monte Carlo
res = {"twfe_static": [], "es_twfe": {k: [] for k in KS}, "cs": {k: [] for k in KS},
       "imp": {k: [] for k in KS}}
bacon_first = None
for r in range(R):
    alpha = rng.standard_normal(N)
    lam = TREND * tt + SIG_LAM * rng.standard_normal(T)
    Y = alpha[:, None] + lam[None, :] + TAU + SIG_EPS * rng.standard_normal((N, T))
    b_static = twfe(Y, [Dmat])[0]
    res["twfe_static"].append(b_static)
    b_es = twfe(Y, ES_REGS)
    es = dict(zip(ES_KEYS, b_es[:len(ES_KEYS)]))
    es[-1] = 0.0
    cs = cs_event(Y)
    cs[-1] = 0.0
    imp = impute_event(Y)
    for k in KS:
        res["es_twfe"][k].append(es[k])
        res["cs"][k].append(cs[k])
        res["imp"][k].append(imp[k])
    if r == 0:
        comps = bacon(Y)
        total = sum(c[3] * c[4] for c in comps)
        wsum = sum(c[4] for c in comps)
        sa = sa_event(Y)
        OUT["check_bacon_sum_equals_twfe"] = {"twfe": round(b_static, 6), "weighted_sum": round(total, 6),
                                              "weights_sum": round(wsum, 6)}
        OUT["check_sa_equals_cs_maxabsdiff"] = float(max(abs(sa[k] - cs[k]) for k in KS))
        bacon_first = comps

# ------------------------------------------------------------------ Bacon decomposition in expectation
# Use the noise-free outcome (alpha = lambda = eps = 0) so that each 2x2 equals its expectation.
comps_exp = bacon(TAU.copy())
beta_exp = twfe(TAU.copy(), [Dmat])[0]
OUT["twfe_static_expectation"] = round(float(beta_exp), 4)
OUT["twfe_static_mc_mean"] = round(float(np.mean(res["twfe_static"])), 4)
OUT["twfe_static_mc_sd"] = round(float(np.std(res["twfe_static"], ddof=1)), 4)
OUT["bacon_expectation"] = [
    {"type": c[0], "treated": c[1], "control": (c[2] if c[2] is not None else "never"),
     "estimate": round(float(c[3]), 4), "weight": round(float(c[4]), 4)} for c in comps_exp]
by_type = {}
for c in comps_exp:
    d = by_type.setdefault(c[0], {"weight": 0.0, "wsum": 0.0})
    d["weight"] += c[4]
    d["wsum"] += c[3] * c[4]
OUT["bacon_by_type"] = {k: {"weight": round(v["weight"], 4),
                            "avg_estimate": round(v["wsum"] / v["weight"], 4)} for k, v in by_type.items()}

summ = {}
for name in ["es_twfe", "cs", "imp"]:
    summ[name] = {}
    for k in KS:
        a = np.array(res[name][k], dtype=float)
        if np.all(np.isnan(a)):
            continue
        summ[name][str(k)] = {"mean": round(float(np.nanmean(a)), 4),
                              "p05": round(float(np.nanpercentile(a, 5)), 4),
                              "p95": round(float(np.nanpercentile(a, 95)), 4)}
OUT["event_study"] = summ
OUT["design"] = {"T": T, "cohorts": COHORTS, "gamma": GAMMA, "N_per_cohort": N_PER, "N_never": N_U,
                 "R": R, "sigma_eps": SIG_EPS, "sigma_lambda": SIG_LAM, "trend": TREND}

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.75), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
ks = np.array(KS)
ax.axhline(0, color=INK2, lw=0.6)
ax.axvline(-0.5, color=GRID, lw=0.8, ls="--")
ax.plot(ks, [PHI[k] for k in KS], color=INK, lw=1.4, marker="o", ms=3.5, label=r"true $\phi_k$", zorder=5)


def band(name, color, marker, dx, label):
    kk = [k for k in KS if str(k) in summ[name]]
    m = np.array([summ[name][str(k)]["mean"] for k in kk])
    lo = np.array([summ[name][str(k)]["p05"] for k in kk])
    hi = np.array([summ[name][str(k)]["p95"] for k in kk])
    x = np.array(kk) + dx
    ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt=marker, color=color, ms=4, lw=0.9, capsize=1.5,
                label=label, zorder=4)


band("es_twfe", ORANGE, "s", -0.18, "TWFE event study")
band("cs", BLUE, "D", 0.0, "Callaway-Sant'Anna")
band("imp", AQUA, "^", 0.18, "imputation")
ax.set_xticks(ks)
ax.set_xlabel(r"periods relative to adoption, $k$")
ax.set_ylabel("effect on $y$")
ax.set_title("(a) Event-study estimates")
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", color=GRID, lw=0.5)

ax = axes[1]
styles = {"treated vs never": (BLUE, "o"), "earlier vs later (not yet treated)": (AQUA, "^"),
          "later vs earlier (already treated)": (ORANGE, "s")}
labels = {"treated vs never": "treated vs never treated",
          "earlier vs later (not yet treated)": "earlier vs not-yet-treated",
          "later vs earlier (already treated)": "later vs already-treated"}
for typ, (col, mk) in styles.items():
    pts = [(c[4], c[3]) for c in comps_exp if c[0] == typ]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, s=22, label=labels[typ],
               edgecolor="white", linewidth=0.6, zorder=4)
ax.axhline(beta_exp, color=INK2, lw=0.9, ls="--")
ax.axhline(ATT_TRUE, color=INK, lw=0.9)
ax.axhline(0, color=GRID, lw=0.6)
xmax = max(c[4] for c in comps_exp) * 1.6
ax.text(xmax * 0.98, beta_exp - 0.02, r"TWFE $\hat\beta$", ha="right", va="top", fontsize=7, color=INK2)
ax.text(xmax * 0.98, ATT_TRUE + 0.02, "ATT", ha="right", va="bottom", fontsize=7, color=INK)
ax.set_xlim(0, xmax)
ax.set_xlabel("weight in TWFE coefficient")
ax.set_ylabel(r"$2\times 2$ DiD estimate")
ax.set_title("(b) Goodman-Bacon decomposition")
ax.legend(frameon=False, loc="lower right", fontsize=6.5, borderaxespad=0.2)
ax.grid(axis="y", color=GRID, lw=0.5)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIGDIR, "staggered.pdf"))
fig.savefig(os.path.join(HERE, "staggered_preview.png"), dpi=170)

with open(os.path.join(HERE, "staggered_numbers.json"), "w") as fh:
    json.dump(OUT, fh, indent=2)
print(json.dumps(OUT, indent=2))
