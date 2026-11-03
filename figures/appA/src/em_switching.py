"""
Simulated two-regime supply relation for Appendix A, Part 2 (Section A.6.6, fig:A-em).

Produces (paths relative to Textbook/):
    figures/appA/em_switching.pdf            fig:A-em
    figures/appA/src/em_switching_numbers.json   every number quoted in the text
    figures/appA/src/em_switching_preview.png    preview for checking

Design (weekly data in the spirit of the Joint Executive Committee railroad cartel):
    demand  log Q_t = a0 + a1 log p_t + a2 L_t + U1_t
    supply  log p_t = b0 + b1 log Q_t + b2 S_t + b3 I_t + U2_t
    (U1_t, U2_t) i.i.d. N(0, Sigma), Sigma = diag(s1^2, s2^2) in the data generating process.
    L_t = 1 in weeks 13-42 of each 52-week year (lake navigation), S_t = 1 from week 150 on (entry).
    Regime I_t (1 = collusion, 0 = price war) is a two-state Markov chain with
    Pr(I_{t+1}=1 | I_t=1) = 0.975 and Pr(I_{t+1}=1 | I_t=0) = 0.067.
    T = 328 weeks.

Estimators:
    (1) 2SLS of each equation with the regime known.
    (2) Mixture maximum likelihood with an i.i.d. Bernoulli(lambda) regime, computed by the EM
        algorithm. The M step maximizes the exact expected complete-data log-likelihood
        sum_t [w_t ln h(y_t | I=1) + (1 - w_t) ln h(y_t | I=0)] (weighted FIML with Jacobian |det B|).
        Three starting classifications. A variant that plugs w_t into the known-regime likelihood
        (as in the verbal description of the algorithm in Porter 1983) is also run.
    (3) Markov-switching maximum likelihood (Hamilton filter), with logit transition probabilities
        Pr(I_{t+1}=1 | I_t = k) = exp(g_k) / (1 + exp(g_k)), started at the EM estimates;
        filtered and smoothed (Kim) regime probabilities.
"""
import os
import json
import numpy as np
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
SEED = 20261022
rng = np.random.default_rng(SEED)
OUT = {"seed": SEED}

BLUE, ORANGE, INK, INK2, GRID, SHADE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e6e3", "#d9d8d4"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral"})

# ------------------------------------------------------------------ data generating process
T = 328
TRUE = dict(a0=9.09, a1=-0.80, a2=-0.43, b0=-2.70, b1=0.09, b2=-0.20, b3=0.545,
            s1=0.40, s2=0.16, r12=0.0, p11=0.975, p01=0.067)
OUT["true"] = TRUE
week = np.arange(T)
L = (((week % 52) >= 13) & ((week % 52) < 43)).astype(float)
S = (week >= 150).astype(float)
I = np.empty(T, dtype=int)
I[0] = 1
for t in range(1, T):
    p = TRUE["p11"] if I[t - 1] == 1 else TRUE["p01"]
    I[t] = int(rng.random() < p)
U = np.column_stack([TRUE["s1"] * rng.standard_normal(T), TRUE["s2"] * rng.standard_normal(T)])
# solve the structural system B y = Gamma X + Delta I + U for y = (log Q, log p)
B0 = np.array([[1.0, -TRUE["a1"]], [-TRUE["b1"], 1.0]])
X = np.column_stack([np.ones(T), L, S])
G0 = np.array([[TRUE["a0"], TRUE["a2"], 0.0], [TRUE["b0"], 0.0, TRUE["b2"]]])
D0 = np.array([0.0, TRUE["b3"]])
Y = np.linalg.solve(B0, (X @ G0.T + np.outer(I, D0) + U).T).T
logQ, logp = Y[:, 0], Y[:, 1]
OUT["n_war_weeks"] = int((I == 0).sum())
OUT["n_wars"] = int(((I[1:] == 0) & (I[:-1] == 1)).sum() + (I[0] == 0))
OUT["share_collusive"] = float(I.mean())


# ------------------------------------------------------------------ 2SLS with known regime
def tsls(y, Xr, Z):
    PZ = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    b = np.linalg.solve(Xr.T @ PZ @ Xr, Xr.T @ PZ @ y)
    e = y - Xr @ b
    s2 = e @ e / len(y)
    V = s2 * np.linalg.inv(Xr.T @ PZ @ Xr)
    return b, np.sqrt(np.diag(V)), e


Zall = np.column_stack([np.ones(T), L, S, I])
bd, sed, _ = tsls(logQ, np.column_stack([np.ones(T), logp, L]), Zall)
bs, ses, es = tsls(logp, np.column_stack([np.ones(T), logQ, S, I]), Zall)
OUT["tsls_known"] = dict(a0=bd[0], a1=bd[1], a2=bd[2], b0=bs[0], b1=bs[1], b2=bs[2], b3=bs[3],
                         se_a1=sed[1], se_b3=ses[3])

# ------------------------------------------------------------------ likelihood pieces
# parameter vector: a0 a1 a2 b0 b1 b2 b3 log s1 log s2 atanh r12
NAMES = ["a0", "a1", "a2", "b0", "b1", "b2", "b3", "ls1", "ls2", "zr"]


def unpack(th):
    a0, a1, a2, b0, b1, b2, b3, ls1, ls2, zr = th
    s1, s2, r = np.exp(ls1), np.exp(ls2), np.tanh(zr)
    Sig = np.array([[s1 * s1, r * s1 * s2], [r * s1 * s2, s2 * s2]])
    return a0, a1, a2, b0, b1, b2, b3, Sig


def resid0(th):
    """e_t = B y_t - Gamma X_t (regime-free structural residuals) and Delta."""
    a0, a1, a2, b0, b1, b2, b3, Sig = unpack(th)
    e1 = logQ - a0 - a1 * logp - a2 * L
    e2 = logp - b0 - b1 * logQ - b2 * S
    return np.column_stack([e1, e2]), np.array([0.0, b3]), Sig, 1.0 - a1 * b1


def logh(th):
    """ln h(y_t | I_t = 1) and ln h(y_t | I_t = 0), t = 1..T."""
    e, D, Sig, detB = resid0(th)
    Si = np.linalg.inv(Sig)
    const = -np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(Sig)) + np.log(abs(detB))
    q0 = np.einsum("ti,ij,tj->t", e, Si, e)
    e1 = e - D
    q1 = np.einsum("ti,ij,tj->t", e1, Si, e1)
    return const - 0.5 * q1, const - 0.5 * q0


def mix_loglik(th, lam):
    l1, l0 = logh(th)
    m = np.maximum(l1, l0)
    return np.sum(m + np.log(lam * np.exp(l1 - m) + (1 - lam) * np.exp(l0 - m)))


def estep(th, lam):
    l1, l0 = logh(th)
    d = np.clip(l0 - l1, -700, 700)
    return lam / (lam + (1 - lam) * np.exp(d))


def mstep(th, w, exact=True):
    """Maximize the expected complete-data log-likelihood given posterior weights w."""
    if exact:
        def negQ(x):
            l1, l0 = logh(x)
            return -np.sum(w * l1 + (1 - w) * l0)
    else:  # plug w_t into the known-regime likelihood in place of I_t
        def negQ(x):
            e, D, Sig, detB = resid0(x)
            Si = np.linalg.inv(Sig)
            ew = e - np.outer(w, D)
            q = np.einsum("ti,ij,tj->t", ew, Si, ew)
            const = -np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(Sig)) + np.log(abs(detB))
            return -np.sum(const - 0.5 * q)
    res = optimize.minimize(negQ, th, method="BFGS", options={"gtol": 1e-9, "maxiter": 2000})
    return res.x


def run_em(w0, th0, exact=True, maxit=500, tol=1e-9):
    w = w0.copy()
    th = th0.copy()
    lam = w.mean()
    path, corr_path = [], []
    th = mstep(th, w, exact)
    for it in range(maxit):
        ll = mix_loglik(th, lam)
        path.append(ll)
        w_new = estep(th, lam)
        corr_path.append(np.corrcoef(w, w_new)[0, 1] if w.std() > 0 else np.nan)
        lam = w_new.mean()
        th = mstep(th, w_new, exact)
        w = w_new
        if it > 2 and abs(path[-1] - path[-2]) < tol:
            break
    path.append(mix_loglik(th, lam))
    return th, lam, w, np.array(path), np.array(corr_path)


# starting values: regime-free 2SLS of each equation (the true regime is never used by the estimators)
Zfree = np.column_stack([np.ones(T), L, S])
bd0, _, _ = tsls(logQ, np.column_stack([np.ones(T), logp, L]), Zfree)
b_noI, _, e_noI = tsls(logp, np.column_stack([np.ones(T), logQ, S]), Zfree)
th_start = np.array([bd0[0], bd0[1], bd0[2], b_noI[0], b_noI[1], b_noI[2], 0.2, np.log(0.4), np.log(0.25), 0.0])
OUT["tsls_no_regime"] = dict(a1=float(bd0[1]), b1=float(b_noI[1]))
# the supply residual ignoring the regime is used to build the starting classifications
starts = {
    "median split of supply residual": (e_noI > np.median(e_noI)).astype(float),
    "lowest quartile = war": (e_noI > np.quantile(e_noI, 0.25)).astype(float),
    "random classification": rng.random(T),
}
em_runs = {}
for name, w0 in starts.items():
    th_hat, lam_hat, w_hat, path, cpath = run_em(w0, th_start, exact=True)
    em_runs[name] = dict(th=th_hat, lam=lam_hat, w=w_hat, path=path, corr=cpath)

best = max(em_runs, key=lambda k: em_runs[k]["path"][-1])
th_em, lam_em, w_em = em_runs[best]["th"], em_runs[best]["lam"], em_runs[best]["w"]
for k, r in em_runs.items():
    d = np.diff(r["path"])
    OUT.setdefault("em_starts", {})[k] = dict(
        iterations=int(len(r["path"]) - 1), final_loglik=float(r["path"][-1]),
        min_increment=float(d.min()), b3=float(r["th"][6]), lam=float(r["lam"]),
        first_iter_corr_above_0999=int(np.argmax(r["corr"] > 0.999)) if np.any(r["corr"] > 0.999) else None)


# observed-information standard errors at the EM solution (numerical Hessian of the mixture likelihood)
def num_hessian(f, x, h=1e-4):
    k = len(x)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h, h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
            H[j, i] = H[i, j]
    return H


full = np.append(th_em, np.log(lam_em / (1 - lam_em)))
negll = lambda x: -mix_loglik(x[:-1], 1 / (1 + np.exp(-x[-1])))
H = num_hessian(negll, full)
V = np.linalg.inv(H)
se = np.sqrt(np.diag(V))
OUT["em"] = {n: float(v) for n, v in zip(NAMES, th_em)}
OUT["em"].update(lam=float(lam_em), se_b3=float(se[6]), se_a1=float(se[1]),
                 s1=float(np.exp(th_em[7])), s2=float(np.exp(th_em[8])),
                 loglik=float(mix_loglik(th_em, lam_em)))
Ihat = (w_em > 0.5).astype(int)
OUT["em"]["misclassified_weeks"] = int((Ihat != I).sum())
OUT["em"]["uncertain_weeks_01_09"] = int(((w_em > 0.1) & (w_em < 0.9)).sum())
OUT["em"]["implied_theta_porter"] = float(th_em[1] * (np.exp(-th_em[6]) - 1.0))
OUT["true_implied_theta_porter"] = float(TRUE["a1"] * (np.exp(-TRUE["b3"]) - 1.0))

# plug-in variant from the best start
th_pl, lam_pl, w_pl, path_pl, _ = run_em(starts[best], th_start, exact=False)
OUT["plugin_variant"] = dict(b3=float(th_pl[6]), lam=float(lam_pl),
                             mixture_loglik=float(mix_loglik(th_pl, lam_pl)),
                             max_abs_diff_w=float(np.max(np.abs(w_pl - w_em))))


# ------------------------------------------------------------------ Markov switching (Hamilton filter)
def hamilton(x, return_probs=False):
    th, g1, g0 = x[:10], x[10], x[11]
    p11 = 1 / (1 + np.exp(-g1))   # Pr(I_{t+1}=1 | I_t=1)
    p01 = 1 / (1 + np.exp(-g0))   # Pr(I_{t+1}=1 | I_t=0)
    try:
        l1, l0 = logh(th)
    except np.linalg.LinAlgError:
        return -1e10
    if not (np.all(np.isfinite(l1)) and np.all(np.isfinite(l0))):
        return -1e10
    m = np.maximum(l1, l0)
    h1, h0 = np.exp(l1 - m), np.exp(l0 - m)   # scaled densities (log-sum-exp)
    xi_pred = p01 / (1 - p11 + p01)  # stationary probability of collusion
    ll = 0.0
    filt, pred = np.empty(T), np.empty(T)
    for t in range(T):
        pred[t] = xi_pred
        f = xi_pred * h1[t] + (1 - xi_pred) * h0[t]
        ll += m[t] + np.log(f)
        xi = xi_pred * h1[t] / f
        filt[t] = xi
        xi_pred = p11 * xi + p01 * (1 - xi)
    if not return_probs:
        return ll if np.isfinite(ll) else -1e10
    # Kim smoother for Pr(I_t = 1 | all data)
    smooth = np.empty(T)
    smooth[-1] = filt[-1]
    for t in range(T - 2, -1, -1):
        # Pr(I_t=1 | data) = filt_t * [p11 * smooth_{t+1}/pred_{t+1} + (1-p11) * (1-smooth_{t+1})/(1-pred_{t+1})]
        a = filt[t] * (p11 * smooth[t + 1] / pred[t + 1] + (1 - p11) * (1 - smooth[t + 1]) / (1 - pred[t + 1]))
        b = (1 - filt[t]) * (p01 * smooth[t + 1] / pred[t + 1] + (1 - p01) * (1 - smooth[t + 1]) / (1 - pred[t + 1]))
        smooth[t] = a / (a + b)
    return ll, filt, smooth, p11, p01


x0 = np.append(th_em, [np.log(0.95 / 0.05), np.log(0.1 / 0.9)])
res = optimize.minimize(lambda x: -hamilton(x), x0, method="BFGS", options={"gtol": 1e-8, "maxiter": 5000})
x_ms = res.x
ll_ms, filt, smooth, p11_hat, p01_hat = hamilton(x_ms, return_probs=True)
Hm = num_hessian(lambda x: -hamilton(x), x_ms)
se_ms = np.sqrt(np.diag(np.linalg.inv(Hm)))
OUT["markov"] = {n: float(v) for n, v in zip(NAMES, x_ms[:10])}
OUT["markov"].update(g1=float(x_ms[10]), g0=float(x_ms[11]), p11=float(p11_hat), p01=float(p01_hat),
                     se_g1=float(se_ms[10]), se_g0=float(se_ms[11]), se_b3=float(se_ms[6]),
                     loglik=float(ll_ms),
                     misclassified_filtered=int(((filt > 0.5).astype(int) != I).sum()),
                     misclassified_smoothed=int(((smooth > 0.5).astype(int) != I).sum()),
                     uncertain_smoothed_01_09=int(((smooth > 0.1) & (smooth < 0.9)).sum()))

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.75), gridspec_kw={"width_ratios": [1, 1.7]})
ax = axes[0]
styles = {"median split of supply residual": ("-", BLUE), "lowest quartile = war": ("--", ORANGE),
          "random classification": (":", INK)}
for name, r in em_runs.items():
    ls, col = styles[name]
    it = np.arange(len(r["path"]))
    ax.plot(it, r["path"], ls, color=col, lw=1.4, label={"median split of supply residual": "median split",
                                                         "lowest quartile = war": "lowest quartile = war",
                                                         "random classification": "random"}[name])
lo = min(r["path"][0] for r in em_runs.values())
hi = max(r["path"][-1] for r in em_runs.values())
ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
ax.set_xlim(0, 40)
ax.set_xlabel("EM iteration")
ax.set_ylabel("mixture log-likelihood")
ax.set_title("(a) Log-likelihood by iteration", loc="left")
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.52), title="starting classification",
          title_fontsize=7)
ax.grid(axis="y", color=GRID, lw=0.5)

ax = axes[1]
# shade true price-war weeks
war = (I == 0)
t0 = None
for t in range(T + 1):
    if t < T and war[t] and t0 is None:
        t0 = t
    if (t == T or not war[t]) and t0 is not None:
        ax.axvspan(t0 - 0.5, t - 0.5, color=SHADE, lw=0)
        t0 = None
ax.plot(week, w_em, color=BLUE, lw=1.0, label=r"EM posterior $w_t$ (i.i.d. regimes)")
ax.plot(week, smooth, color=ORANGE, lw=1.0, ls="--", label=r"smoothed Markov probability")
ax.set_ylim(-0.03, 1.12)
ax.set_xlim(0, T - 1)
ax.set_yticks([0, 0.5, 1])
ax.axhline(0.5, color=INK2, lw=0.5, ls=":")
ax.set_xlabel("week")
ax.set_ylabel("Pr(collusive regime)")
ax.set_title("(b) Regime probabilities, with true price wars shaded", loc="left")
ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIGDIR, "em_switching.pdf"))
fig.savefig(os.path.join(HERE, "em_switching_preview.png"), dpi=170)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


with open(os.path.join(HERE, "em_switching_numbers.json"), "w") as f:
    json.dump(clean(OUT), f, indent=2)
print(json.dumps(clean(OUT), indent=2))
