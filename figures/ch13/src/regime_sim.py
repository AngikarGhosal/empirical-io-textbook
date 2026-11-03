"""
Regime-switching simulations for Chapter 13, Sections 13.2.4-13.2.6.

What the script does
  1. Simulates weekly data (T = 328, the length of the JEC sample) from the switching
     supply-demand model of Section 13.2.2, calibrated to Porter (1983, Table 3).
  2. Estimates the model (a) with the regimes known, (b) by the EM algorithm with i.i.d.
     regimes (Kiefer 1980, Porter 1983), and (c) by EM with first-order Markov regimes
     (Hamilton filter forward, smoother backward).
  3. Computes the likelihood-ratio statistic for "no switching" and its parametric-bootstrap
     null distribution (Section 13.2.6).
  4. Runs a Monte Carlo on how classification accuracy and the estimates depend on the
     separation ratio b3 / s2 (exercise on EM).
  5. Applies the Berry-Briggs test of first order against second order to a simulated
     first-order Markov regime series and to a Green-Porter series with wars of fixed length.

Outputs
  figures/ch13/regime_sim.pdf     (fig:13-gp-path)
  figures/ch13/em_mc.pdf          (fig:13-em-mc)
  figures/ch13/src/regime_sim_output.txt   (every number quoted in the text)
Run:  python regime_sim.py      (about 15 to 25 minutes, mostly the bootstrap and Monte Carlo)

Model (weekly data t = 1..T)
    log Q_t = a0 + a1 log p_t + a2 L_t + U1_t                     (demand)
    log p_t = b0 + b1 log Q_t + b3 I_t + U2_t                     (supply relation)
(U1, U2) ~ N(0, Sigma) i.i.d., I_t in {0, 1}. I_t is excluded from demand and L_t from
supply, so each equation is exactly identified. ML can therefore be carried out on the
unrestricted reduced form y_t = C z_t + V_t, z_t = (1, L_t, I_t), V_t ~ N(0, Omega), and
the structural parameters follow from a1 = C[0,2] / C[1,2] and b1 = C[1,1] / C[0,1].
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
OUT = os.path.join(HERE, "regime_sim_output.txt")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


# ---------------------------------------------------------------- calibration
T = 328                 # weeks: week 1 of 1880 to week 16 of 1886
LAM = 0.62              # stationary share of collusive weeks (mean of PO is .6189, Porter Table 2)
Q00 = 0.90              # war continues w.p. 0.9, so the mean war lasts 1 / (1 - 0.9) = 10 weeks
Q11 = 1.0 - (1.0 - Q00) * (1.0 - LAM) / LAM   # gives stationary share LAM
# Intercepts reproduce Porter (1983, Table 4) at PN = 0, LAKES = 0: price .1673, quantity 38,680.
LOGQ0, LOGP0 = np.log(38680.0), np.log(0.1673)

CALIBS = {
    "ML": dict(a1=-0.800, a2=-0.430, b1=0.090, b3=0.545, s1=0.399, s2=0.109),    # Table 3, right
    "2SLS": dict(a1=-0.742, a2=-0.437, b1=0.251, b3=0.382, s1=0.398, s2=0.243),  # Table 3, left
}


def structural_to_rf(c):
    a0 = LOGQ0 - c["a1"] * LOGP0
    b0 = LOGP0 - c["b1"] * LOGQ0
    B = np.array([[1.0, -c["a1"]], [-c["b1"], 1.0]])
    G = np.array([[a0, c["a2"], 0.0], [b0, 0.0, c["b3"]]])
    C = np.linalg.solve(B, G)
    Sig = np.diag([c["s1"] ** 2, c["s2"] ** 2])
    Bi = np.linalg.inv(B)
    Om = Bi @ Sig @ Bi.T
    return C, Om, a0, b0


def rf_to_structural(C, Om):
    a1 = C[0, 2] / C[1, 2]
    b1 = C[1, 1] / C[0, 1]
    B = np.array([[1.0, -a1], [-b1, 1.0]])
    G = B @ C
    S = B @ Om @ B.T
    return dict(a0=G[0, 0], a1=a1, a2=G[0, 1], b0=G[1, 0], b1=b1, b3=G[1, 2],
                s1=np.sqrt(S[0, 0]), s2=np.sqrt(S[1, 1]), s12=S[0, 1])


def lakes_series(T):
    week = np.arange(T) % 52 + 1
    return ((week >= 16) & (week <= 45)).astype(float)   # open 30 of 52 weeks (mean .577)


def markov_chain(T, q11, q00, rng, start=1):
    I = np.empty(T, dtype=int)
    I[0] = start
    for t in range(1, T):
        stay = q11 if I[t - 1] == 1 else q00
        I[t] = I[t - 1] if rng.random() < stay else 1 - I[t - 1]
    return I


def gp_chain(T, g, Tw, rng):
    """Green-Porter regimes: a war starts w.p. g after a collusive week and lasts exactly Tw weeks."""
    I = np.ones(T, dtype=int)
    t = 1
    while t < T:
        if I[t - 1] == 1 and rng.random() < g:
            I[t:t + Tw] = 0
            t += Tw
            if t < T:
                I[t] = 1
                t += 1
        else:
            t += 1
    return I


def simulate(C, Om, L, I, rng):
    Z = np.column_stack([np.ones(len(L)), L, I])
    V = rng.standard_normal((len(L), 2)) @ np.linalg.cholesky(Om).T
    return Z @ C.T + V


def logphi(Y, M, Om):
    Oi = np.linalg.inv(Om)
    R = Y - M
    q = np.einsum("ti,ij,tj->t", R, Oi, R)
    return -np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(Om)) - 0.5 * q


def mstep(Y, L, w):
    """Exact M-step: normal equations with E[I_t] = w_t and E[I_t^2] = w_t (not w_t^2)."""
    T = len(L)
    X = np.column_stack([np.ones(T), L])
    Szz = np.zeros((3, 3))
    Szz[:2, :2] = X.T @ X
    Szz[:2, 2] = X.T @ w
    Szz[2, :2] = X.T @ w
    Szz[2, 2] = w.sum()
    Szy = np.vstack([X.T @ Y, w @ Y])
    C = np.linalg.solve(Szz, Szy).T
    M1 = np.column_stack([X, np.ones(T)]) @ C.T
    M0 = np.column_stack([X, np.zeros(T)]) @ C.T
    R1, R0 = Y - M1, Y - M0
    Om = (np.einsum("t,ti,tj->ij", w, R1, R1) + np.einsum("t,ti,tj->ij", 1 - w, R0, R0)) / T
    return C, Om, M1, M0


def em_iid(Y, L, w0, tol=1e-8, maxit=3000):
    w = w0.copy()
    ll_path = []
    for it in range(maxit):
        lam = w.mean()
        C, Om, M1, M0 = mstep(Y, L, w)
        a = np.log(lam) + logphi(Y, M1, Om)
        b = np.log(1 - lam) + logphi(Y, M0, Om)
        m = np.maximum(a, b)
        lt = m + np.log(np.exp(a - m) + np.exp(b - m))
        ll = lt.sum()
        w = np.exp(a - lt)
        if ll_path and ll < ll_path[-1] - 1e-7:
            raise RuntimeError("EM decreased the likelihood")
        ll_path.append(ll)
        if len(ll_path) > 1 and abs(ll_path[-1] - ll_path[-2]) < tol:
            break
    return dict(ll=ll_path[-1], C=C, Om=Om, lam=w.mean(), w=w, it=it + 1, path=ll_path)


def forward_backward(l1, l0, P, pi1):
    """Hamilton filter forward and smoother backward (scaled). State 0 = war, 1 = collusion.
    P[i, j] = Pr(I_{t+1} = j | I_t = i)."""
    T = len(l1)
    m = np.maximum(l1, l0)
    d0 = np.exp(l0 - m).tolist()
    d1 = np.exp(l1 - m).tolist()
    p00, p01, p10, p11 = float(P[0, 0]), float(P[0, 1]), float(P[1, 0]), float(P[1, 1])
    f0, f1, c = [0.0] * T, [0.0] * T, [0.0] * T
    pr0, pr1 = 1.0 - pi1, pi1
    for t in range(T):                       # prediction -> update (Hamilton filter)
        n0, n1 = pr0 * d0[t], pr1 * d1[t]
        c[t] = n0 + n1
        f0[t], f1[t] = n0 / c[t], n1 / c[t]
        pr0 = f0[t] * p00 + f1[t] * p10
        pr1 = f0[t] * p01 + f1[t] * p11
    b0, b1 = [1.0] * T, [1.0] * T
    for t in range(T - 2, -1, -1):           # backward pass
        e0, e1 = d0[t + 1] * b0[t + 1], d1[t + 1] * b1[t + 1]
        b0[t] = (p00 * e0 + p01 * e1) / c[t + 1]
        b1[t] = (p10 * e0 + p11 * e1) / c[t + 1]
    filt = np.column_stack([f0, f1])
    beta = np.column_stack([b0, b1])
    smooth = filt * beta
    smooth /= smooth.sum(axis=1, keepdims=True)
    dens = np.column_stack([d0, d1])
    x = filt[:-1, :, None] * P[None, :, :] * (dens[1:] * beta[1:])[:, None, :]
    xi = x / x.sum(axis=(1, 2), keepdims=True)
    return float(np.log(c).sum() + m.sum()), filt, smooth, xi


def em_markov(Y, L, w0, tol=1e-8, maxit=3000):
    w = w0.copy()
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi1 = 0.5
    ll_old = -np.inf
    for it in range(maxit):
        C, Om, M1, M0 = mstep(Y, L, w)
        ll, filt, smooth, xi = forward_backward(logphi(Y, M1, Om), logphi(Y, M0, Om), P, pi1)
        w = smooth[:, 1]
        S = xi.sum(axis=0)
        P = S / S.sum(axis=1, keepdims=True)
        pi1 = w[0]
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    return dict(ll=ll, C=C, Om=Om, P=P, w=w, filt=filt[:, 1], it=it + 1)


def null_fit(Y, L):
    X = np.column_stack([np.ones(len(L)), L])
    C0 = np.linalg.lstsq(X, Y, rcond=None)[0].T
    R = Y - X @ C0.T
    Om = R.T @ R / len(L)
    return logphi(Y, X @ C0.T, Om).sum(), C0, Om


def starts(Y, L, rng, n_random=3):
    X = np.column_stack([np.ones(len(L)), L])
    r = Y[:, 1] - X @ np.linalg.lstsq(X, Y[:, 1], rcond=None)[0]   # price residual
    out = [(r > np.quantile(r, q)).astype(float) * 0.9 + 0.05 for q in (0.5, 0.3, 0.7)]
    out += [rng.uniform(0.05, 0.95, size=len(L)) for _ in range(n_random)]
    return out


def best_em(Y, L, rng, n_random=3, maxit=2000):
    best = None
    for w0 in starts(Y, L, rng, n_random):
        try:
            res = em_iid(Y, L, w0, maxit=maxit)
        except (np.linalg.LinAlgError, RuntimeError, FloatingPointError, ValueError):
            continue
        if best is None or res["ll"] > best["ll"]:
            best = res
    return best


def label_fix(res):
    """Label regimes so that b3 > 0 (collusion raises the supply intercept)."""
    st = rf_to_structural(res["C"], res["Om"])
    if st["b3"] < 0:
        return None
    return st


def berry_briggs(I):
    """Berry-Briggs test of first order against second order.
    p(a, b) = Pr(I_t = 1 | I_{t-2} = a, I_{t-1} = b). H0: p(0,0) = p(1,0)."""
    I = np.asarray(I)
    p, n = {}, {}
    for a in (0, 1):
        for b in (0, 1):
            mask = (I[:-2] == a) & (I[1:-1] == b)
            n[(a, b)] = int(mask.sum())
            p[(a, b)] = I[2:][mask].mean() if mask.sum() > 0 else np.nan
    var = {k: p[k] * (1 - p[k]) / n[k] for k in p}
    stat = (p[(0, 0)] - p[(1, 0)]) ** 2 / (var[(0, 0)] + var[(1, 0)])
    return stat, p, n


# ---------------------------------------------------------------- 1. one simulated sample
rng = np.random.default_rng(1875)
L = lakes_series(T)
I = markov_chain(T, Q11, Q00, rng, start=1)
E = rng.standard_normal((T, 2))          # common standard-normal draws for both calibrations
weeks = np.arange(1, T + 1)
nwars = int(((I[1:] == 0) & (I[:-1] == 1)).sum())
say(f"Regime path: T = {T}, q11 = {Q11:.4f}, q00 = {Q00}, collusive share {I.mean():.3f}, "
    f"number of wars {nwars}")

results = {}
for name, c in CALIBS.items():
    C, Om, a0, b0 = structural_to_rf(c)
    Y = np.column_stack([np.ones(T), L, I]) @ C.T + E @ np.linalg.cholesky(Om).T
    say("")
    say(f"Calibration {name}: a0 = {a0:.3f}, b0 = {b0:.3f}, " +
        ", ".join(f"{k} = {v}" for k, v in c.items()) +
        f"; separation b3/s2 = {c['b3'] / c['s2']:.2f}")
    for Iv in (0, 1):
        for Lv in (0, 1):
            yq, yp = C @ np.array([1.0, Lv, Iv])
            say(f"  model cell I={Iv}, LAKES={Lv}: price {np.exp(yp):.4f}, quantity {np.exp(yq):,.0f}")
    Z3 = np.column_stack([np.ones(T), L, I])
    Ck = np.linalg.lstsq(Z3, Y, rcond=None)[0].T
    Rk = Y - Z3 @ Ck.T
    sk = rf_to_structural(Ck, Rk.T @ Rk / T)
    say("  known regimes (ML = 2SLS here): " + ", ".join(f"{k} = {v:.3f}" for k, v in sk.items()))
    r_iid = best_em(Y, L, rng, n_random=5)
    st = rf_to_structural(r_iid["C"], r_iid["Om"])
    mis = int(((r_iid["w"] > 0.5).astype(int) != I).sum())
    say(f"  EM, i.i.d. regimes (best of 8 starts): loglik = {r_iid['ll']:.2f}, lambda = {r_iid['lam']:.3f}, "
        f"iterations = {r_iid['it']}, misclassified weeks = {mis}")
    say("     " + ", ".join(f"{k} = {v:.3f}" for k, v in st.items()))
    rT = em_iid(Y, L, I * 0.9 + 0.05)
    stT = rf_to_structural(rT["C"], rT["Om"])
    say(f"  EM, i.i.d., started at the true regimes: loglik = {rT['ll']:.2f}, lambda = {rT['lam']:.3f}, "
        f"misclassified = {int(((rT['w'] > 0.5).astype(int) != I).sum())}, a1 = {stT['a1']:.3f}, "
        f"b1 = {stT['b1']:.3f}, b3 = {stT['b3']:.3f}")
    r_mk = em_markov(Y, L, starts(Y, L, rng, 0)[0])
    stm = rf_to_structural(r_mk["C"], r_mk["Om"])
    mis_mk = int(((r_mk["w"] > 0.5).astype(int) != I).sum())
    say(f"  EM, Markov regimes: loglik = {r_mk['ll']:.2f}, q11 = {r_mk['P'][1, 1]:.3f}, "
        f"q00 = {r_mk['P'][0, 0]:.3f}, misclassified weeks = {mis_mk}")
    say("     " + ", ".join(f"{k} = {v:.3f}" for k, v in stm.items()))
    say(f"     implied mean spells: collusion {1 / (1 - r_mk['P'][1, 1]):.1f} weeks, "
        f"war {1 / (1 - r_mk['P'][0, 0]):.1f} weeks")
    ll0, C0, Om0 = null_fit(Y, L)
    say(f"  LR statistic 2(l1 - l0), i.i.d. mixture against no switching: {2 * (r_iid['ll'] - ll0):.1f}")
    say(f"  EM log-likelihood path (first 6 iterations from the median start): "
        + ", ".join(f"{x:.2f}" for x in em_iid(Y, L, starts(Y, L, rng, 0)[0])["path"][:6]))
    results[name] = dict(Y=Y, w_iid=r_iid["w"], w_mk=r_mk["w"], C0=C0, Om0=Om0)

# ---------------------------------------------------------------- 2. bootstrap of the LR
C0, Om0 = results["ML"]["C0"], results["ML"]["Om0"]
B_BOOT = 300
rngb = np.random.default_rng(2024)
LRb = []
X2 = np.column_stack([np.ones(T), L])
for b in range(B_BOOT):
    Yb = X2 @ C0.T + rngb.standard_normal((T, 2)) @ np.linalg.cholesky(Om0).T
    l0b, _, _ = null_fit(Yb, L)
    rb = best_em(Yb, L, rngb, n_random=1, maxit=600)
    LRb.append(max(2 * (rb["ll"] - l0b), 0.0))
LRb = np.array(LRb)
q90, q95, q99 = np.quantile(LRb, [0.90, 0.95, 0.99])
say("")
say(f"Parametric bootstrap under H0 ({B_BOOT} replications, T = {T}, null fit of the ML calibration): "
    f"quantiles 90%: {q90:.2f}, 95%: {q95:.2f}, 99%: {q99:.2f}; chi2(1): 2.71, 3.84, 6.63")
say(f"   share of bootstrap LR above 3.84: {(LRb > 3.84).mean():.3f}; mean {LRb.mean():.2f}")

# ---------------------------------------------------------------- 3. Monte Carlo on separation
ratios = [1.0, 1.5, 2.0, 3.0, 5.0]
R_MC = 60
rngm = np.random.default_rng(99)
base = dict(CALIBS["ML"])
mc = {r: dict(iid=[], mk=[], a1_iid=[], b3_iid=[], a1_known=[], b3_known=[]) for r in ratios}
for r in ratios:
    c = dict(base)
    c["s2"] = c["b3"] / r
    C, Om, _, _ = structural_to_rf(c)
    for rep in range(R_MC):
        Ir = markov_chain(T, Q11, Q00, rngm, start=1)
        Yr = simulate(C, Om, L, Ir, rngm)
        Z3 = np.column_stack([np.ones(T), L, Ir])
        Ck = np.linalg.lstsq(Z3, Yr, rcond=None)[0].T
        Rk = Yr - Z3 @ Ck.T
        sk = rf_to_structural(Ck, Rk.T @ Rk / T)
        mc[r]["a1_known"].append(sk["a1"])
        mc[r]["b3_known"].append(sk["b3"])
        ri = best_em(Yr, L, rngm, n_random=1, maxit=1000)
        wi = ri["w"]
        sti = rf_to_structural(ri["C"], ri["Om"])
        if sti["b3"] < 0:          # relabel so that b3 > 0
            wi = 1 - wi
        mc[r]["iid"].append(((wi > 0.5).astype(int) != Ir).mean())
        mc[r]["a1_iid"].append(sti["a1"])
        mc[r]["b3_iid"].append(abs(sti["b3"]))
        rm = em_markov(Yr, L, starts(Yr, L, rngm, 0)[0], tol=1e-6, maxit=1000)
        wm = rm["w"]
        if rf_to_structural(rm["C"], rm["Om"])["b3"] < 0:
            wm = 1 - wm
        mc[r]["mk"].append(((wm > 0.5).astype(int) != Ir).mean())
say("")
say(f"Monte Carlo ({R_MC} replications per ratio, ML calibration with s2 = b3 / ratio, Markov regimes "
    f"with q11 = {Q11:.4f}, q00 = {Q00}):")
say("  ratio | misclass. iid EM | misclass. Markov EM | median a1: known, iid EM | median b3: known, iid EM")
for r in ratios:
    d = mc[r]
    say(f"  {r:4.1f}  | {np.mean(d['iid']):.3f}            | {np.mean(d['mk']):.3f}               | "
        f"{np.median(d['a1_known']):.3f}, {np.median(d['a1_iid']):.3f}           | "
        f"{np.median(d['b3_known']):.3f}, {np.median(d['b3_iid']):.3f}")
    say(f"         sd a1: known {np.std(d['a1_known']):.3f}, iid EM {np.std(d['a1_iid']):.3f}; "
        f"IQR a1 iid EM [{np.quantile(d['a1_iid'], 0.25):.3f}, {np.quantile(d['a1_iid'], 0.75):.3f}]")
def bayes_error(D, lam):
    """Error rate of the rule w > 1/2 with known parameters, i.i.d. regimes with Pr(I=1) = lam,
    and Mahalanobis distance D between the two component means (D = b3 / s2 here)."""
    k = np.log(lam / (1 - lam)) / D
    return lam * stats.norm.cdf(-D / 2 - k) + (1 - lam) * stats.norm.cdf(-D / 2 + k)


say(f"  Bayes-rule floor (known parameters, i.i.d. regimes, lambda = {LAM}): " +
    ", ".join(f"ratio {r}: {bayes_error(r, LAM):.3f}" for r in ratios))

# ---------------------------------------------------------------- 4. Berry-Briggs on simulated series
rng2 = np.random.default_rng(7)
I_m = markov_chain(T, Q11, Q00, rng2)
g = (1 - LAM) / (LAM * 10)
I_gp = gp_chain(T, g, 10, rng2)
say("")
say(f"Green-Porter chain: war-start probability g = {g:.4f}, wars of exactly 10 weeks")
for name, series in (("first-order Markov", I_m), ("Green-Porter, wars of 10 weeks", I_gp)):
    stat, pp, nn = berry_briggs(series)
    cells = "; ".join(f"p{a}{b} = {pp[(a, b)]:.3f} (n = {nn[(a, b)]})" for (a, b) in pp)
    say(f"Berry-Briggs test, {name}: chi2(1) = {stat:.2f}, p = {1 - stats.chi2.cdf(stat, 1):.3f}; {cells}")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#d9d8d3"


def shade(ax, I):
    start = None
    for t in range(T):
        if I[t] == 0 and start is None:
            start = t
        if (I[t] == 1 or t == T - 1) and start is not None:
            end = t if I[t] == 1 else t + 1
            ax.axvspan(weeks[start] - 0.5, weeks[end - 1] + 0.5, color=GREY, lw=0)
            start = None


fig, axes = plt.subplots(3, 1, figsize=(6.3, 6.4), sharex=True,
                         gridspec_kw={"height_ratios": [1.1, 0.8, 0.8], "hspace": 0.32})
rA, rB = results["ML"], results["2SLS"]
ax = axes[0]
shade(ax, I)
ax.plot(weeks, np.exp(rA["Y"][:, 1]), color="#0b0b0b", lw=1.0)
ax.set_ylabel("price $p_t$\n(dollars per 100 lbs)")
ax.set_title(r"(a) Simulated price, ML calibration (shaded: true price wars, $I_t = 0$)",
             fontsize=8.5, loc="left")
for ax, r, ttl in ((axes[1], rA, r"(b) Posterior probability of collusion, ML calibration, $\frac{\beta_3}{\sigma_2} = 5.0$"),
                   (axes[2], rB, r"(c) Posterior probability of collusion, 2SLS calibration, $\frac{\beta_3}{\sigma_2} = 1.6$")):
    shade(ax, I)
    ax.plot(weeks, r["w_iid"], color=BLUE, lw=1.1, label=r"EM with i.i.d. regimes: $w_t$")
    ax.plot(weeks, r["w_mk"], color=ORANGE, lw=1.1, ls="--",
            label=r"EM with Markov regimes: smoothed $\Pr(I_t = 1 \mid \mathcal{Y}_T)$")
    ax.axhline(0.5, color="#8a8984", lw=0.6, ls=":")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", r"$\frac{1}{2}$", "1"])
    ax.set_title(ttl, fontsize=8.5, loc="left")
axes[2].set_xlabel("week")
axes[2].set_xlim(0.5, T + 0.5)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5, frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(bottom=0.11)
fig.savefig(os.path.join(FIGDIR, "regime_sim.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "regime_sim_preview.png"), dpi=110, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"wspace": 0.32})
ax = axes[0]
bins = np.linspace(0, max(16, LRb.max() + 0.5), 33)
ax.hist(LRb, bins=bins, density=True, color=BLUE, alpha=0.55, edgecolor="white", lw=0.5,
        label="bootstrap LR under $H_0$")
xs = np.linspace(0.05, bins[-1], 400)
ax.plot(xs, stats.chi2.pdf(xs, 1), color="#0b0b0b", lw=1.1, label=r"$\chi^2_1$ density")
ax.axvline(3.84, color="#0b0b0b", lw=0.8, ls=":")
ax.axvline(q95, color=BLUE, lw=0.9, ls="--")
ax.text(3.84 + 0.2, 0.26, r"$\chi^2_1$: 3.84", fontsize=7.5)
ax.text(q95 + 0.2, 0.19, f"bootstrap: {q95:.2f}", fontsize=7.5, color=BLUE)
ax.set_ylim(0, 0.5)
ax.set_xlabel(r"$LR = 2(\ell_1 - \ell_0)$")
ax.set_ylabel("density")
ax.set_title("(a) Null distribution of the LR statistic", fontsize=8.5, loc="left")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax = axes[1]
ax.plot(ratios, [np.mean(mc[r]["iid"]) for r in ratios], "o-", color=BLUE, lw=1.1, ms=3.5,
        label="EM, i.i.d. regimes")
ax.plot(ratios, [np.mean(mc[r]["mk"]) for r in ratios], "s--", color=ORANGE, lw=1.1, ms=3.5,
        label="EM, Markov regimes")
ax.set_xlabel(r"separation $\frac{\beta_3}{\sigma_2}$")
ax.set_ylabel("share of weeks misclassified")
ax.set_ylim(0, None)
ax.set_title("(b) Classification accuracy", fontsize=8.5, loc="left")
ax.legend(fontsize=7, frameon=False)
fig.savefig(os.path.join(FIGDIR, "em_mc.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "em_mc_preview.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
say("")
say("Figures saved to figures/ch13/regime_sim.pdf and figures/ch13/em_mc.pdf")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
