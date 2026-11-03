"""Simulated market worked example for Chapter 8, Section 8.11 (sec:08-worked).

The script regenerates every number, table entry and figure of Section 8.11 from the
primitives in sim_market_primitives.csv (characteristics, ownership, xi, omega, market sizes)
and fixed random seeds.

Steps
  1. Equilibrium prices in 100 markets by the Morrow-Skerlos fixed point (prop:08-ms), with
     checks against a Newton-type root finder and the plain markup iteration.
  2. Realized shares from M_t simulated consumers.
  3. Logit demand by OLS and 2SLS.
  4. BLP contraction with Sigma known.
  5. 2SLS on the recovered mean utilities.
  6. Full nested fixed point GMM estimation of Sigma with three instrument sets,
     simulation-corrected standard errors, Hansen J, and the objective as a function of
     sigma_price (fig:08-profile).
  7. Elasticities and diversion ratios in market 0 (tab:08-sim-elas, fig:08-ownelas).

Outputs: figures/ch08/sim_profile.pdf, figures/ch08/sim_ownelas.pdf,
         src/worked_example_numbers.txt, src/worked_example_results.json
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize, root
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
DATA = HERE / "sim_market_primitives.csv"
TXT = HERE / "worked_example_numbers.txt"
JSN = HERE / "worked_example_results.json"

# ----------------------------------------------------------------------------- design
BETA = np.array([-1.0, 0.2, 0.15])        # constant, sugar, organic
ALPHA = 3.0                               # mean price coefficient
SIG = np.array([0.5, 0.2, 0.3, 0.5])      # sd of tastes: constant, sugar, organic, price
GAMMA1 = np.array([0.2, 0.25, 0.3])       # marginal cost: constant, sugar, organic
GAMMA2 = 0.5                              # marginal cost: milk price
XC = ["X0", "X1", "X2"]

B_FIRM, SEED_FIRM = 100, 2026             # consumer types behind the firms' expected demand
SEED_CONS = 2027                          # M_t consumers per market (realized shares)
SEED_STARTS, N_STARTS = 2028, 20          # random starting prices for the equilibrium checks
NS_EST, SEED_EST = 1000, 2030             # estimation draws per market
SEED_SIMCORR, R_SIMCORR = 2040, 30        # redraws for the simulation-error correction
NS_TRUE, SEED_TRUE = 200_000, 2050        # draws for population ("true") elasticities
SEED_OPT = 2060                           # random starting values of the outer search
TOL_P = 1e-12                             # price fixed point tolerance (sup norm)
TOL_IN = 1e-12                            # BLP contraction tolerance (sup norm)

BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7a5cc2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

lines = []
RES = {}


def out(s=""):
    print(s, flush=True)
    lines.append(str(s))


# ============================================================================ helpers
def choice_probs(v):
    """Logit probabilities for utilities v (..., J); outside utility 0; overflow safe."""
    m = np.maximum(v.max(axis=-1, keepdims=True), 0.0)
    ev = np.exp(v - m)
    return ev / (np.exp(-m) + ev.sum(axis=-1, keepdims=True))


def probs_market(p, x, xi, nu, beta=BETA, alpha=ALPHA, sig=SIG):
    """s_ij for one market at prices p (draws in rows)."""
    xr = np.column_stack([x, p])
    delta = x @ beta - alpha * p + xi
    return choice_probs(delta[None, :] + (nu * sig) @ xr.T)


def lam_gam(p, x, xi, nu):
    """Shares, Lambda (vector) and Gamma with ds/dp' = Gamma - diag(Lambda)."""
    sij = probs_market(p, x, xi, nu)
    w = (ALPHA - SIG[3] * nu[:, 3])[:, None] * sij
    return sij.mean(0), w.mean(0), w.T @ sij / len(nu)


def solve_zeta(mc, own, x, xi, nu, p0, maxit=10_000):
    """Morrow-Skerlos iteration p <- mc + Lambda^{-1}[s + (Omega o Gamma)(p - mc)]."""
    p = p0.copy()
    for it in range(1, maxit + 1):
        s, lam, gam = lam_gam(p, x, xi, nu)
        pn = mc + (s + (own * gam) @ (p - mc)) / lam
        if not np.all(np.isfinite(pn)):
            return pn, it, False
        if np.max(np.abs(pn - p)) < TOL_P:
            return pn, it, True
        p = pn
    return p, maxit, False


def solve_markup(mc, own, x, xi, nu, p0, maxit=2_000):
    """Plain markup iteration p <- mc - [(Omega o D)']^{-1} s."""
    p = p0.copy()
    for it in range(1, maxit + 1):
        s, lam, gam = lam_gam(p, x, xi, nu)
        d = gam - np.diag(lam)
        try:
            pn = mc - np.linalg.solve(own * d.T, s)
        except np.linalg.LinAlgError:
            return p, it, False
        if not np.all(np.isfinite(pn)) or np.max(np.abs(pn)) > 1e6:
            return pn, it, False
        if np.max(np.abs(pn - p)) < TOL_P:
            return pn, it, True
        p = pn
    return p, maxit, False


def foc(p, mc, own, x, xi, nu):
    s, lam, gam = lam_gam(p, x, xi, nu)
    return s + (own * (gam - np.diag(lam)).T) @ (p - mc)


def markup_gap(p, mc, own, x, xi, nu):
    s, lam, gam = lam_gam(p, x, xi, nu)
    with np.errstate(all="ignore"):
        return (s + (own * gam) @ (p - mc)) / lam - (p - mc)


def markup_residual(p, mc, own, x, xi, nu):
    r = np.max(np.abs(markup_gap(p, mc, own, x, xi, nu)))
    return float(r) if np.isfinite(r) else np.inf


def ownership(owner):
    return (owner[:, None] == owner[None, :]).astype(float)


def tsls(y, x, z):
    xh = z @ np.linalg.lstsq(z, x, rcond=None)[0]
    b = np.linalg.solve(xh.T @ x, xh.T @ y)
    return b, y - x @ b, xh


def cluster_vcov(xh, e, groups):
    """c (Xh'Xh)^-1 [sum_g Xh_g'e_g e_g'Xh_g] (Xh'Xh)^-1, c = G/(G-1)(N-1)/(N-K)."""
    bread = np.linalg.inv(xh.T @ xh)
    sc = pd.DataFrame(xh * e[:, None]).groupby(groups).sum().to_numpy()
    n, k = xh.shape
    g = sc.shape[0]
    c = g / (g - 1) * (n - 1) / (n - k)
    return c * bread @ (sc.T @ sc) @ bread


def cluster_S(z, e, groups):
    """(1/N) sum_t (Z_t'e_t)(Z_t'e_t)'."""
    sc = pd.DataFrame(z * e[:, None]).groupby(groups).sum().to_numpy()
    return sc.T @ sc / z.shape[0]


def hansen_j(y, x, z, groups):
    """Two-step GMM with the market-clustered weight from 2SLS residuals."""
    _, e1, _ = tsls(y, x, z)
    w = np.linalg.inv(cluster_S(z, e1, groups))
    a = x.T @ z @ w @ z.T @ x
    b2 = np.linalg.solve(a, x.T @ z @ w @ z.T @ y)
    g = z.T @ (y - x @ b2) / len(y)
    j = len(y) * g @ w @ g
    df = z.shape[1] - x.shape[1]
    return float(j), int(df), float(stats.chi2.sf(j, df)), b2


def fmt(b, se):
    return f"{b:.3f} ({se:.3f})"


# ============================================================================ step 1-2
t0 = time.time()
df = pd.read_csv(DATA).sort_values(["market_id", "product_id"]).reset_index(drop=True)
df["owner"] = df["owner"].astype(int)
df["mc"] = df[XC].to_numpy() @ GAMMA1 + GAMMA2 * df["Z"] + df["omega"]
markets = np.sort(df["market_id"].unique())
T = len(markets)
idx = [np.flatnonzero(df["market_id"].to_numpy() == m) for m in markets]

rng_firm = np.random.default_rng(SEED_FIRM)
nu_firm = rng_firm.standard_normal((T, B_FIRM, 4))
rng_starts = np.random.default_rng(SEED_STARTS)
rng_cons = np.random.default_rng(SEED_CONS)

price = np.zeros(len(df))
s_exp = np.zeros(len(df))
qty = np.zeros(len(df), dtype=int)
iters, focres = [], []
ms = {"zeta": [0, 0, 0.0], "markup": [0, 0, 0.0], "root": [0, 0, 0.0]}  # solved, failed, maxdev
markup_fail_examples = []
for t, rows in enumerate(idx):
    g = df.iloc[rows]
    x, xi, mc = g[XC].to_numpy(), g["xi"].to_numpy(), g["mc"].to_numpy()
    own = ownership(g["owner"].to_numpy())
    p, it, ok = solve_zeta(mc, own, x, xi, nu_firm[t], mc.copy())
    assert ok
    iters.append(it)
    focres.append(np.max(np.abs(foc(p, mc, own, x, xi, nu_firm[t]))))
    # equilibrium checks from random starts
    for _ in range(N_STARTS):
        p0 = rng_starts.uniform(0.0, 4.0, size=len(mc))
        runs = {
            "zeta": solve_zeta(mc, own, x, xi, nu_firm[t], p0)[0],
            "markup": solve_markup(mc, own, x, xi, nu_firm[t], p0)[0],
            "root": root(foc, p0, args=(mc, own, x, xi, nu_firm[t]), method="hybr",
                         tol=1e-14).x,
        }
        for key, ps in runs.items():
            if markup_residual(ps, mc, own, x, xi, nu_firm[t]) < 1e-10:
                ms[key][0] += 1
                ms[key][2] = max(ms[key][2], float(np.max(np.abs(ps - p))))
            else:
                ms[key][1] += 1
    # realized shares from M_t consumers
    m = int(g["M"].iloc[0])
    delta = x @ BETA - ALPHA * p + xi
    xr = np.column_stack([x, p])
    nu_c = rng_cons.standard_normal((m, 4))
    eps = rng_cons.gumbel(size=(m, len(delta) + 1))
    u = eps.copy()
    u[:, 1:] += delta[None, :] + (nu_c * SIG) @ xr.T
    counts = np.bincount(u.argmax(axis=1), minlength=len(delta) + 1)
    price[rows] = p
    s_exp[rows] = probs_market(p, x, xi, nu_firm[t]).mean(0)
    qty[rows] = counts[1:]

df["price"], df["s_exp"], df["q"] = price, s_exp, qty
df["s"] = df["q"] / df["M"]
df["delta_true"] = df[XC].to_numpy() @ BETA - ALPHA * df["price"] + df["xi"]
s0 = 1.0 - df.groupby("market_id")["s"].transform("sum")
df["s0"] = s0
iters = np.array(iters)
markup = df["price"] - df["mc"]

out("=== Step 1: equilibrium ===")
out(f"N obs {len(df)}, markets {T}, products per market {df.groupby('market_id').size().min()}"
    f"-{df.groupby('market_id').size().max()} (mean {df.groupby('market_id').size().mean():.2f})")
out(f"owners: {sorted(df['owner'].unique())}; products by owner: "
    f"{df.groupby('owner')['product_id'].unique().apply(sorted).to_dict()}")
out(f"M_t range {df['M'].min()}-{df['M'].max()}, mean {df.groupby('market_id')['M'].first().mean():.1f}")
out(f"zeta iterations min {iters.min()} max {iters.max()} mean {iters.mean():.2f} (market 0: {iters[0]})")
out(f"max |FOC| at solution {max(focres):.2e}")
for k, v in ms.items():
    out(f"random starts [{k}]: solved {v[0]} of {v[0] + v[1]}, max price dev from baseline {v[2]:.1e}")
out(f"markups p - mc: min {markup.min():.3f} max {markup.max():.3f} mean {markup.mean():.3f}; "
    f"mean Lerner {(markup / df['price']).mean():.3f}")
out(f"prices: min {df['price'].min():.3f} max {df['price'].max():.3f} mean {df['price'].mean():.3f}")
out(f"min alpha_i over firm draws {(ALPHA - SIG[3] * nu_firm[..., 3]).min():.3f}")
out("=== Step 2: realized shares ===")
gap = df["s"] - df["s_exp"]
out(f"consumers {df.groupby('market_id')['M'].first().sum()}, zero shares {(df['q'] == 0).sum()}, "
    f"min realized share {df['s'].min():.4f} (min buyers {df['q'].min()})")
out(f"mean |realized - expected| {gap.abs().mean():.4f}, max {gap.abs().max():.4f}, "
    f"corr {np.corrcoef(df['s'], df['s_exp'])[0, 1]:.3f}")
out(f"mean outside share realized {s0.groupby(df['market_id']).first().mean():.3f}, "
    f"min {s0.min():.3f}")
RES["step1"] = {"iters": [int(iters.min()), int(iters.max()), float(iters.mean()), int(iters[0])],
                "focmax": float(max(focres)), "multistart": ms,
                "markup": [float(markup.min()), float(markup.max()), float(markup.mean())],
                "lerner_mean": float((markup / df["price"]).mean())}

m0 = df[df["market_id"] == markets[0]].copy()
out("market 0 table: product owner mc p p-mc s_exp s_real q delta_true")
for _, r in m0.iterrows():
    out(f"  {int(r.product_id):2d} & {int(r.owner)} & {r.mc:.4f} & {r.price:.4f} & {r.price - r.mc:.4f} & "
        f"{r.s_exp:.4f} & {r.s:.4f} & {int(r.q)} & {r.delta_true:.3f}")
out(f"  outside: s_exp {1 - m0.s_exp.sum():.4f} s_real {1 - m0.s.sum():.4f}; M_0 = {int(m0.M.iloc[0])}")

# ============================================================================ step 3
groups = df["market_id"].to_numpy()
firm = df.groupby(["market_id", "owner"])
mkt = df.groupby("market_id")
for col in ("X1", "X2"):
    fs = firm[col].transform("sum")
    df[col.lower() + "_own"] = fs - df[col]
    df[col.lower() + "_rival"] = mkt[col].transform("sum") - fs
fn = firm["X0"].transform("size")
df["n_own"] = fn - 1
df["n_rival"] = mkt["X0"].transform("size") - fn
IV_BLP = ["Z", "x1_own", "x1_rival", "x2_own", "x2_rival", "n_own", "n_rival"]

X = np.column_stack([df[XC].to_numpy(), df["price"].to_numpy()])  # theta_1 = (beta, -alpha)
N = len(df)
Zb = np.column_stack([df[XC].to_numpy(), df[IV_BLP].to_numpy()])
y_logit = np.log(df["s"].to_numpy()) - np.log(df["s0"].to_numpy())

b_ols, e_ols, _ = tsls(y_logit, X, X)
V_ols = cluster_vcov(X, e_ols, groups)
b_iv, e_iv, xh = tsls(y_logit, X, Zb)
V_iv = cluster_vcov(xh, e_iv, groups)
# first stage
fs_x = np.column_stack([df[XC].to_numpy(), df[IV_BLP].to_numpy()])
b_fs, e_fs, _ = tsls(df["price"].to_numpy(), fs_x, fs_x)
V_fs = cluster_vcov(fs_x, e_fs, groups)
ex = np.arange(3, fs_x.shape[1])
F_all = b_fs[ex] @ np.linalg.solve(V_fs[np.ix_(ex, ex)], b_fs[ex]) / len(ex)
ex2 = np.arange(4, fs_x.shape[1])
F_comp = b_fs[ex2] @ np.linalg.solve(V_fs[np.ix_(ex2, ex2)], b_fs[ex2]) / len(ex2)
J_l, Jdf_l, Jp_l, _ = hansen_j(y_logit, X, Zb, groups)
Zz = np.column_stack([df[XC].to_numpy(), df["Z"].to_numpy()])
b_ivz, _, _ = tsls(y_logit, X, Zz)
out("=== Step 3: logit ===")
names = ["Constant", "Sugar", "Organic", "alpha"]
for i, nm in enumerate(names):
    sgn = -1 if i == 3 else 1
    out(f"  {nm}: OLS {fmt(sgn * b_ols[i], np.sqrt(V_ols[i, i]))}  2SLS {fmt(sgn * b_iv[i], np.sqrt(V_iv[i, i]))}")
out(f"  first stage: milk coef {fmt(b_fs[3], np.sqrt(V_fs[3, 3]))}; cluster F (7 IVs) {F_all:.2f}; "
    f"F of six competition IVs {F_comp:.2f}")
out(f"  Hansen J {J_l:.2f} df {Jdf_l} p {Jp_l:.3f}; 2SLS with milk only alpha {-b_ivz[3]:.3f}")
RES["logit"] = {"ols": [float(v) for v in b_ols], "ols_se": np.sqrt(np.diag(V_ols)).tolist(),
                "iv": [float(v) for v in b_iv], "iv_se": np.sqrt(np.diag(V_iv)).tolist(),
                "F": float(F_all), "F_comp": float(F_comp), "J": [J_l, Jdf_l, Jp_l],
                "alpha_milk_only": float(-b_ivz[3])}

# ============================================================================ padded arrays
counts_t = np.array([len(r) for r in idx])
JM = counts_t.max()
mask = np.zeros((T, JM), dtype=bool)
for t, n in enumerate(counts_t):
    mask[t, :n] = True
XR = np.zeros((T, JM, 4))
XR[mask] = np.column_stack([df[XC].to_numpy(), df["price"].to_numpy()])
LOGS = np.zeros((T, JM))
LOGS[mask] = np.log(df["s"].to_numpy())
D0 = np.zeros((T, JM))
D0[mask] = y_logit


def mu_of(sig, nu):
    return np.einsum("tjk,tik->tij", XR * sig, nu)


def probs_pad(delta, mu):
    v = np.where(mask[:, None, :], delta[:, None, :] + mu, -np.inf)
    mx = np.maximum(v.max(axis=-1, keepdims=True), 0.0)
    ev = np.exp(v - mx)
    return ev / (np.exp(-mx) + ev.sum(axis=-1, keepdims=True))


def contraction(delta0, mu, tol=TOL_IN, maxit=50_000):
    d = delta0.copy()
    for it in range(1, maxit + 1):
        s = probs_pad(d, mu).mean(axis=1)
        dn = d + np.where(mask, LOGS - np.log(np.where(mask, s, 1.0)), 0.0)
        if np.max(np.abs(dn - d)) < tol:
            return dn, it
        d = dn
    raise RuntimeError("contraction did not converge")


def contraction_iters(delta0, mu, tol=TOL_IN):
    """Per-market iteration counts (each market iterated until its own sup change < tol)."""
    d = delta0.copy()
    done = np.zeros(T, dtype=bool)
    its = np.zeros(T, dtype=int)
    for it in range(1, 50_001):
        s = probs_pad(d, mu).mean(axis=1)
        dn = d + np.where(mask, LOGS - np.log(np.where(mask, s, 1.0)), 0.0)
        ch = np.max(np.abs(dn - d), axis=1)
        newly = (~done) & (ch < tol)
        its[newly] = it
        done |= newly
        d = dn
        if done.all():
            return d, its
    raise RuntimeError


def ddelta_dsig(delta, sig, nu):
    """d delta / d sigma' by the implicit function theorem, padded (T, JM, 4)."""
    P = probs_pad(delta, mu_of(sig, nu))
    ns = P.shape[1]
    s = P.mean(axis=1)
    dsdd = -np.einsum("tij,tik->tjk", P, P) / ns
    dsdd[:, np.arange(JM), np.arange(JM)] += s
    xbar = np.einsum("tim,tmk->tik", P, XR)
    dsds = (np.einsum("tij,tik->tjk", P, nu) * XR
            - np.einsum("tij,tik->tjk", P, nu * xbar)) / ns
    inv = ~mask
    dsdd[inv[:, :, None] | inv[:, None, :]] = 0.0
    tt, jj = np.nonzero(inv)
    dsdd[tt, jj, jj] = 1.0
    dsds[inv] = 0.0
    return -np.linalg.solve(dsdd, dsds)


# ============================================================================ step 4-5
rng_est = np.random.default_rng(SEED_EST)
NU = rng_est.standard_normal((T, NS_EST, 4))
mu_known = mu_of(SIG, NU)
d_known, its_known = contraction_iters(D0, mu_known)
s_chk = probs_pad(d_known, mu_known).mean(axis=1)
err_known = np.max(np.abs(np.log(s_chk[mask]) - LOGS[mask]))
delta_rc = d_known[mask]
b_rco, e_rco, _ = tsls(delta_rc, X, X)
V_rco = cluster_vcov(X, e_rco, groups)
b_rc, e_rc, xh = tsls(delta_rc, X, Zb)
V_rc = cluster_vcov(xh, e_rc, groups)
J_rc, Jdf_rc, Jp_rc, _ = hansen_j(delta_rc, X, Zb, groups)
rterm = y_logit - delta_rc
b_r, _, _ = tsls(rterm, X, Zb)
out("=== Step 4: contraction with Sigma known ===")
out(f"  iterations per market min {its_known.min()} max {its_known.max()} mean {its_known.mean():.2f};"
    f" market 0 {its_known[0]}; max |log s - log s_obs| {err_known:.1e}")
d0 = d_known[0, :counts_t[0]]
out("  market 0: product buyers logit_start recovered true")
for j, r in enumerate(m0.itertuples()):
    out(f"  {r.product_id:2d} & {r.q} & {D0[0, j]:.3f} & {d0[j]:.3f} & {r.delta_true:.3f}")
# simulation part of the market-0 gap, with 100,000 draws
nu_big = np.random.default_rng(2032).standard_normal((1, 100_000, 4))
XR0 = XR[:1]
mu0 = np.einsum("tjk,tik->tij", XR0 * SIG, nu_big)


def contraction_single(d, mu, logs, msk, tol=TOL_IN):
    for it in range(1, 100_000):
        v = np.where(msk[:, None, :], d[:, None, :] + mu, -np.inf)
        mx = np.maximum(v.max(axis=-1, keepdims=True), 0.0)
        ev = np.exp(v - mx)
        s = (ev / (np.exp(-mx) + ev.sum(axis=-1, keepdims=True))).mean(axis=1)
        dn = d + np.where(msk, logs - np.log(np.where(msk, s, 1.0)), 0.0)
        if np.max(np.abs(dn - d)) < tol:
            return dn
        d = dn


d_big = contraction_single(d_known[:1].copy(), mu0, LOGS[:1], mask[:1])[0, :counts_t[0]]
out("  market 0 simulation part (1000 vs 100000 draws): " + ", ".join(f"{v:.3f}" for v in d0 - d_big))
out("  market 0 sampling part (100000 draws vs truth): "
    + ", ".join(f"{v:.3f}" for v in d_big - m0.delta_true.to_numpy()))
out("=== Step 5: 2SLS on recovered delta ===")
for i, nm in enumerate(names):
    sgn = -1 if i == 3 else 1
    out(f"  {nm}: OLS {fmt(sgn * b_rco[i], np.sqrt(V_rco[i, i]))}  2SLS {fmt(sgn * b_rc[i], np.sqrt(V_rc[i, i]))}")
out(f"  Hansen J {J_rc:.2f} df {Jdf_rc} p {Jp_rc:.3f}")
out(f"  alpha gap RC - logit {(-b_rc[3]) - (-b_iv[3]):.3f}; 2SLS coefficient of price in r-term "
    f"regression {b_r[3]:.3f}; corr(r, p) {np.corrcoef(rterm, df['price'])[0, 1]:.3f}")
r0 = rterm[idx[0]]
p0m = df["price"].to_numpy()[idx[0]]
out(f"  market 0 r-term: cheapest product {r0[np.argmin(p0m)]:.3f}, most expensive {r0[np.argmax(p0m)]:.3f}")
RES["rc_known"] = {"iters": [int(its_known.min()), int(its_known.max()), float(its_known.mean())],
                   "ols": b_rco.tolist(), "iv": b_rc.tolist(), "iv_se": np.sqrt(np.diag(V_rc)).tolist(),
                   "J": [J_rc, Jdf_rc, Jp_rc], "corr_r_p": float(np.corrcoef(rterm, df["price"])[0, 1])}

# ============================================================================ step 6
# instrument sets
pf = df["price"].to_numpy()
# (b) differentiation instruments (Gandhi and Houde 2019)
phat_x = np.column_stack([df[XC].to_numpy(), df["Z"].to_numpy()])
phat = phat_x @ np.linalg.lstsq(phat_x, pf, rcond=None)[0]   # exogenous predicted price
df["phat"] = phat
dx1_own, dx1_riv, org_own, org_riv, dp_own, dp_riv = (np.zeros(N) for _ in range(6))
for rows in idx:
    g = df.iloc[rows]
    x1, x2, ow, ph = g["X1"].to_numpy(), g["X2"].to_numpy(), g["owner"].to_numpy(), g["phat"].to_numpy()
    for a, r in enumerate(rows):
        oth = np.arange(len(rows)) != a
        same = (ow == ow[a]) & oth
        riv = ow != ow[a]
        dx1_own[r] = np.sum((x1[same] - x1[a]) ** 2)
        dx1_riv[r] = np.sum((x1[riv] - x1[a]) ** 2)
        org_own[r] = np.sum(x2[same] == x2[a])
        org_riv[r] = np.sum(x2[riv] == x2[a])
        dp_own[r] = np.sum((ph[same] - ph[a]) ** 2)
        dp_riv[r] = np.sum((ph[riv] - ph[a]) ** 2)
Zd = np.column_stack([df[XC].to_numpy(), df["Z"].to_numpy(), dx1_own, dx1_riv, org_own, org_riv,
                      dp_own, dp_riv])


def rank_info(Z):
    sv = np.linalg.svd(Z / np.linalg.norm(Z, axis=0), compute_uv=False)
    return int(np.linalg.matrix_rank(Z)), float(sv[0] / sv[-1])


out("=== Step 6: full NFXP GMM ===")
out(f"  rank / condition number: BLP {rank_info(Zb)}, differentiation {rank_info(Zd)}")


class NFXP:
    """Nested fixed point GMM for theta_2 = sigma (4 nonnegative entries)."""

    def __init__(self, Z, nu=NU):
        self.Z, self.nu = Z, nu
        self.last = D0.copy()
        self.nevals = 0
        self.nits = []

    def delta(self, sig):
        d, it = contraction(self.last, mu_of(sig, self.nu))
        self.last = d
        self.nits.append(it)
        return d

    def linear(self, dvec, W):
        Z = self.Z
        A = X.T @ Z @ W @ Z.T @ X
        th1 = np.linalg.solve(A, X.T @ Z @ W @ Z.T @ dvec)
        return th1, dvec - X @ th1, A

    def obj(self, sig, W, grad=True):
        self.nevals += 1
        dpad = self.delta(sig)
        dvec = dpad[mask]
        th1, xi, A = self.linear(dvec, W)
        g = self.Z.T @ xi / N
        Q = N * g @ W @ g
        if not grad:
            return Q
        dd = ddelta_dsig(dpad, sig, self.nu)[mask]          # (N, 4)
        Z = self.Z
        dxi = dd - X @ np.linalg.solve(A, X.T @ Z @ W @ Z.T @ dd)
        dQ = 2.0 * N * g @ W @ (Z.T @ dxi / N)
        return Q, dQ

    def fit(self, W, starts):
        best = None
        sols = []
        for s0_ in starts:
            r = minimize(self.obj, s0_, args=(W,), jac=True, method="L-BFGS-B",
                         bounds=[(0.0, 5.0)] * 4, options={"ftol": 1e-14, "gtol": 1e-9,
                                                          "maxiter": 500})
            sols.append((float(r.fun), r.x.copy(), int(r.nit)))
            if best is None or r.fun < best.fun:
                best = r
        return best, sols

    def at(self, sig, W):
        dpad = self.delta(sig)
        th1, xi, _ = self.linear(dpad[mask], W)
        return dpad, th1, xi


def estimate(label, Z, starts, R=R_SIMCORR):
    est = NFXP(Z)
    L = Z.shape[1]
    W1 = np.linalg.inv(Z.T @ Z / N)
    t1 = time.time()
    r1, sols1 = est.fit(W1, starts)
    _, th1a, xi1 = est.at(r1.x, W1)
    W2 = np.linalg.inv(cluster_S(Z, xi1, groups))
    starts2 = [r1.x] + list(starts[:3])
    r2, sols2 = est.fit(W2, starts2)
    sig = r2.x
    dpad, th1, xi = est.at(sig, W2)
    J = float(r2.fun) if L > 8 else 0.0
    # Jacobian of moments wrt theta = (theta_1, sigma)
    dd = ddelta_dsig(dpad, sig, NU)[mask]
    G = Z.T @ np.column_stack([-X, dd]) / N
    S2 = cluster_S(Z, xi, groups)
    # simulation error in the moments: redraw the consumer draws R times
    rng = np.random.default_rng(SEED_SIMCORR)
    gs = []
    for _ in range(R):
        nu_r = rng.standard_normal((T, NS_EST, 4))
        d_r, _ = contraction(dpad, mu_of(sig, nu_r))
        gs.append(Z.T @ (d_r[mask] - X @ th1) / N)
    V3 = np.cov(np.array(gs).T)
    W = W2
    bread = np.linalg.inv(G.T @ W @ G)
    V_nosim = bread @ G.T @ W @ (S2 / N) @ W @ G @ bread
    V_sim = bread @ G.T @ W @ (S2 / N + V3) @ W @ G @ bread
    se0, se1 = np.sqrt(np.diag(V_nosim)), np.sqrt(np.diag(V_sim))
    out(f"--- {label}: L = {L}, outer evals {est.nevals}, time {time.time() - t1:.0f}s, "
        f"mean inner iterations {np.mean(est.nits):.1f}")
    out("  step-1 solutions (Q, sigma): " + "; ".join(
        f"{q:.3f} [{', '.join(f'{v:.3f}' for v in x_)}]" for q, x_, _ in sorted(sols1, key=lambda a: a[0])))
    out("  step-2 solutions (Q, sigma): " + "; ".join(
        f"{q:.3f} [{', '.join(f'{v:.3f}' for v in x_)}]" for q, x_, _ in sorted(sols2, key=lambda a: a[0])))
    theta = np.concatenate([th1, sig])
    lab = ["Constant", "Sugar", "Organic", "alpha", "s_const", "s_sugar", "s_org", "s_price"]
    for i, nm in enumerate(lab):
        sg = -1 if i == 3 else 1
        out(f"  {nm}: {sg * theta[i]:.3f} (se no-sim {se0[i]:.3f}, sim-corrected {se1[i]:.3f})")
    if L > 8:
        out(f"  Hansen J {J:.2f} df {L - 8} p {stats.chi2.sf(J, L - 8):.3f}")
    eig = np.linalg.eigvalsh(G.T @ W @ G)
    out(f"  eigenvalues of G'WG: min {eig.min():.2e} max {eig.max():.2e}")
    q_true = float(est.obj(SIG, W2, grad=False))
    est.last = dpad.copy()
    out(f"  objective at the true Sigma (step-2 weight): {q_true:.2f}; minimum {float(r2.fun):.2f}; "
        f"difference {q_true - float(r2.fun):.2f} (chi2_4 0.95 quantile {stats.chi2.ppf(0.95, 4):.2f})")
    return {"label": label, "sig": sig, "th1": th1, "se0": se0, "se1": se1, "J": J, "L": L,
            "W2": W2, "est": est, "dpad": dpad, "theta": theta, "q_true": q_true, "q_min": float(r2.fun),
            "sols1": [(q, x_.tolist()) for q, x_, _ in sols1],
            "sols2": [(q, x_.tolist()) for q, x_, _ in sols2]}


rng_opt = np.random.default_rng(SEED_OPT)
starts = [np.full(4, 0.5)] + [rng_opt.uniform(0.05, 1.5, size=4) for _ in range(5)]
fitA = estimate("(a) BLP instruments", Zb, starts)
fitB = estimate("(b) differentiation instruments", Zd, starts)

# (c) approximate optimal instruments (Reynaert and Verboven 2014). The derivative of delta
# with respect to sigma_k vanishes at sigma_k = 0 (prop:08-signsigma), and (b) puts several
# sigmas at zero, so the instruments are evaluated at the round preliminary value sigma0 = 1
# with the linear parameters concentrated out under the differentiation instruments.
SIG0 = np.ones(4)
estB0 = NFXP(Zd)
_, th1B, _ = estB0.at(SIG0, np.linalg.inv(Zd.T @ Zd / N))
sigB = SIG0
out(f"  preliminary values for (c): sigma0 = {SIG0.tolist()}, theta1(sigma0) = {np.round(th1B, 3).tolist()}")
phat2_x = np.column_stack([df[XC].to_numpy(), df["Z"].to_numpy(), Zd[:, 4:]])
phat2 = phat2_x @ np.linalg.lstsq(phat2_x, pf, rcond=None)[0]
Xhat = np.column_stack([df[XC].to_numpy(), phat2])
dhat = np.zeros((T, JM))
dhat[mask] = Xhat @ th1B                      # mean utility with xi = 0 at predicted prices
XR_save = XR.copy()
XR[mask] = Xhat                               # random-coefficient characteristics at phat
dd_hat = ddelta_dsig(dhat, sigB, NU)[mask]
dd_zero = ddelta_dsig(dhat, np.array([1.0, 0.0, 1.0, 1.0]), NU)[mask]
out(f"  check: max |d delta / d sigma_sugar| at sigma_sugar = 0: {np.max(np.abs(dd_zero[:, 1])):.2e} "
    f"(simulation noise) versus {np.max(np.abs(dd_hat[:, 1])):.2e} at sigma_sugar = 1")
XR[:] = XR_save
Zo = np.column_stack([df[XC].to_numpy(), phat2, dd_hat])
out(f"  rank / condition number: optimal {rank_info(Zo)}")
fitC = estimate("(c) approximate optimal instruments", Zo, starts[:3])

# profile of the objective in sigma_price
grid = np.linspace(0.0, 1.5, 31)
prof = {}
for fit in (fitA, fitB, fitC):
    est = NFXP(fit["est"].Z)
    W = fit["W2"]
    vals = []
    for v in grid:
        sg = fit["sig"].copy()
        sg[3] = v
        vals.append(est.obj(sg, W, grad=False))
    prof[fit["label"]] = np.array(vals)
    out(f"  profile {fit['label']}: Q(0) {vals[0]:.2f}, min on grid {min(vals):.2f} at "
        f"{grid[int(np.argmin(vals))]:.2f}, Q(1.5) {vals[-1]:.2f}")
    rel = np.array(vals) - min(vals)
    above = np.flatnonzero((rel > stats.chi2.ppf(0.95, 1)) & (grid > grid[int(np.argmin(vals))]))
    if above.size:
        k = above[0]
        cross = grid[k - 1] + (grid[k] - grid[k - 1]) * (stats.chi2.ppf(0.95, 1) - rel[k - 1]) / (rel[k] - rel[k - 1])
        out(f"    objective exceeds its minimum by the chi2_1 0.95 quantile above sigma_p = {cross:.2f}")
    below0 = rel[0] > stats.chi2.ppf(0.95, 1)
    out(f"    excess at sigma_p = 0: {rel[0]:.2f} (above the quantile: {below0}); at true 0.5: "
        f"{rel[int(np.argmin(np.abs(grid - 0.5)))]:.2f}")
# true profile for (a): re-optimize the other three sigmas at each sigma_price
gridp = np.linspace(0.0, 1.5, 16)
estA = NFXP(Zb)


def prof_obj(s3, v, W):
    sg = np.append(s3, v)
    Q, dQ = estA.obj(sg, W)
    return Q, dQ[:3]


full_prof = []
start3 = fitA["sig"][:3].copy()
for v in gridp:
    r = minimize(prof_obj, start3, args=(v, fitA["W2"]), jac=True, method="L-BFGS-B",
                 bounds=[(0.0, 5.0)] * 3, options={"ftol": 1e-12, "gtol": 1e-8})
    full_prof.append(float(r.fun))
    start3 = r.x
full_prof = np.array(full_prof)
out("  fully profiled Q for (a) on sigma_price grid: " + ", ".join(f"{v:.2f}" for v in full_prof))

RES["nfxp"] = {}
for fit in (fitA, fitB, fitC):
    RES["nfxp"][fit["label"]] = {"theta": fit["theta"].tolist(), "se0": fit["se0"].tolist(),
                                 "se1": fit["se1"].tolist(), "J": fit["J"], "L": fit["L"],
                                 "sols1": fit["sols1"], "sols2": fit["sols2"]}
RES["profile"] = {"grid": grid.tolist(), "values": {k: v.tolist() for k, v in prof.items()},
                  "grid_full": gridp.tolist(), "full_BLP": full_prof.tolist()}

# figure: profile
fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
cols = {fitA["label"]: BLUE, fitB["label"]: ORANGE, fitC["label"]: AQUA}
short = {fitA["label"]: "BLP instruments", fitB["label"]: "Differentiation instruments",
         fitC["label"]: "Approximate optimal instruments"}
for lab, vals in prof.items():
    ax.plot(grid, vals - vals.min(), color=cols[lab], lw=2, label=short[lab])
ax.plot(gridp, full_prof - full_prof.min(), color=BLUE, lw=1.4, ls="--",
        label="BLP instruments, other $\\sigma$ re-optimized")
ax.axhline(stats.chi2.ppf(0.95, 1), color=INK2, lw=1, ls=":")
ax.text(0.02, stats.chi2.ppf(0.95, 1) + 0.15, r"0.95 quantile of $\chi^2_{1}$", ha="left",
        va="bottom", fontsize=8, color=INK2)
ax.axvline(SIG[3], color=INK2, lw=1, ls="-.")
ax.text(SIG[3] + 0.02, 0.55 * ax.get_ylim()[1], r"true $\sigma_p = 0.5$", fontsize=8, color=INK2)
ax.set_xlabel(r"$\sigma_p$ (standard deviation of the price coefficient)", fontsize=9, color=INK2)
ax.set_ylabel(r"$N\,\bar g(\theta)^{\top} W\,\bar g(\theta)$ minus its minimum", fontsize=9, color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(labelsize=8, colors=INK2)
fig.legend(fontsize=8, frameon=False, loc="outside lower center", ncol=2)
fig.savefig(FIG / "sim_profile.pdf")
fig.savefig(HERE / "sim_profile_preview.png", dpi=150)
plt.close(fig)

# ============================================================================ step 7
p0v = XR[0, :counts_t[0], 3]
x0v = XR[0, :counts_t[0], :3]
s0v = np.exp(LOGS[0, :counts_t[0]])
prods = m0["product_id"].to_numpy()


def rc_elas(delta, sig, alpha, nu, xr):
    sij = choice_probs(delta[None, :] + (nu * sig) @ xr.T)
    ai = (alpha - sig[3] * nu[:, 3])[:, None]
    w = ai * sij
    D = w.T @ sij / len(nu) - np.diag(w.mean(0))
    si0 = 1.0 - sij.sum(1)
    d0_ = (w * si0[:, None]).mean(0)
    s_ = sij.mean(0)
    s0_ = si0.mean()
    E = D * xr[:, 3][None, :] / s_[:, None]
    e0 = d0_ * xr[:, 3] / s0_
    div0 = d0_ / (-np.diag(D))
    return E, e0, div0


def logit_elas(alpha, s, p):
    s0_ = 1 - s.sum()
    E = alpha * np.outer(np.ones(len(s)), p * s)
    E[np.diag_indices(len(s))] = -alpha * p * (1 - s)
    e0 = alpha * p * s
    div0 = s0_ / (1 - s)
    return E, e0, div0


xr0 = np.column_stack([x0v, p0v])
models = {}
models["Logit (2SLS)"] = logit_elas(-b_iv[3], s0v, p0v)
models["RC, Sigma known"] = rc_elas(d_known[0, :counts_t[0]], SIG, -b_rc[3], NU[0], xr0)
best_fit = fitC
models["RC, full GMM"] = rc_elas(best_fit["dpad"][0, :counts_t[0]], best_fit["sig"], -best_fit["th1"][3],
                                 NU[0], xr0)
models["RC, BLP-IV GMM"] = rc_elas(fitA["dpad"][0, :counts_t[0]], fitA["sig"], -fitA["th1"][3],
                                   NU[0], xr0)
nu_true = np.random.default_rng(SEED_TRUE).standard_normal((NS_TRUE, 4))
models["True"] = rc_elas(m0["delta_true"].to_numpy(), SIG, ALPHA, nu_true, xr0)
out("=== Step 7: market 0 elasticities (own reported as positive) ===")
out(f"  products {prods.tolist()}; prices {np.round(p0v, 3).tolist()}")
for k, (E, e0, div0) in models.items():
    out(f"  {k}: own " + ", ".join(f"{-v:.2f}" for v in np.diag(E))
        + "; diversion to outside " + ", ".join(f"{100 * v:.1f}" for v in div0))
j4 = int(np.flatnonzero(prods == 4)[0])
for k, (E, e0, div0) in models.items():
    out(f"  {k}: cross wrt p_4 (rows " + ", ".join(str(p) for p in prods) + ", outside): "
        + ", ".join(f"{v:.3f}" for v in E[:, j4]) + f", {e0[j4]:.3f}")
j8 = int(np.flatnonzero(prods == 8)[0])
for k, (E, e0, div0) in models.items():
    out(f"  {k}: cross wrt p_8: " + ", ".join(f"{v:.3f}" for v in E[:, j8]) + f", {e0[j8]:.3f}")
RES["elas"] = {k: {"own": (-np.diag(E)).tolist(), "div0": div0.tolist(), "cross4": E[:, j4].tolist(),
                   "cross4_out": float(e0[j4]), "cross8": E[:, j8].tolist(), "cross8_out": float(e0[j8])}
               for k, (E, e0, div0) in models.items()}

GOLD = "#c98a00"
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6), sharey=True, layout="constrained")
ser = [("Logit (2SLS)", "Logit, 2SLS", BLUE, "o", -0.28),
       ("RC, Sigma known", r"RC logit, $\Sigma$ known", ORANGE, "s", -0.14),
       ("RC, full GMM", "RC logit, GMM (approx. optimal IV)", PURPLE, "^", 0.0),
       ("RC, BLP-IV GMM", "RC logit, GMM (BLP IV)", GOLD, "v", 0.14),
       ("True", "RC logit, true parameters", AQUA, "D", 0.28)]
yy = np.arange(len(prods))
others = np.array([k for k in range(len(prods)) if k != j4])
for key, nm, c, mk, dy in ser:
    E = models[key][0]
    axes[0].scatter(-np.diag(E), yy + dy, s=26, marker=mk, color=c, edgecolor=INK,
                    linewidth=0.5, label=nm, zorder=3)
    axes[1].scatter(E[others, j4], yy[others] + dy, s=26, marker=mk, color=c, edgecolor=INK,
                    linewidth=0.5, zorder=3)
axes[0].set_yticks(yy, [f"product {p}" for p in prods])
axes[0].invert_yaxis()
axes[0].set_xlabel(r"own-price elasticity $-\frac{\partial s_j}{\partial p_j}\frac{p_j}{s_j}$",
                   fontsize=9, color=INK2)
axes[1].set_xscale("log")
axes[1].set_xticks([0.03, 0.05, 0.1, 0.2, 0.5, 1.0], ["0.03", "0.05", "0.1", "0.2", "0.5", "1"])
axes[1].minorticks_off()
axes[1].set_xlabel(r"cross-price elasticity $\frac{\partial s_j}{\partial p_4}\frac{p_4}{s_j}$ (log scale)",
                   fontsize=9, color=INK2)
axes[0].set_title("(a) Own-price elasticities", loc="left", fontsize=9, color=INK)
axes[1].set_title("(b) Response to the price of product 4", loc="left", fontsize=9, color=INK)
for ax in axes:
    ax.grid(axis="x", color=GRID, lw=0.8, which="both")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK2, length=0)
fig.legend(fontsize=7.5, frameon=False, loc="outside lower center", ncol=3)
fig.savefig(FIG / "sim_ownelas.pdf")
fig.savefig(HERE / "sim_ownelas_preview.png", dpi=150)
plt.close(fig)

out(f"total time {time.time() - t0:.0f}s")
TXT.write_text("\n".join(lines))
JSN.write_text(json.dumps(RES, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
