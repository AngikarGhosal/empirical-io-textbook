"""Supply-side illustrations for Section 4.5 (sec:04-supply) in the 25-product logit market.

The market is the one of ex:04-logit25 (data_simulations.csv in the course folder):
25 products, x_j = (1, X2_j, X3_j), xi_j, omega_j, five owners (3, 3, 6, 6, 7 products),
beta = (0.1, 0.25, 0.1), alpha = 2, mc_j = x_j gamma + omega_j with gamma = (0.2, 0.25, 0.3),
market size 1. The multi-product Bertrand-Nash equilibrium under the owners in the data
("M") is treated as the observed status quo.

Computations (all numbers quoted in the text come from the output file):
  1. Cost recovery mc_hat = p + (Omega o Delta^T)^{-1} s at the observed prices under three
     conduct assumptions: single-product firms, the true owners, one owner of all 25 products.
  2. Cost regressions of mc_hat on x_j (OLS) under the three assumptions.
  3. Merger simulation: owners 3 and 4 merge. Solvers: markup iteration, best-response
     iteration (Lambert W), Newton's method with the analytic Jacobian of eq:04-dGdp
     (the count quoted in the text), Powell's hybrid method (scipy root, a cross-check),
     Morrow-Skerlos iteration (logit case).
     Unilateral best response of the merged firm at pre-merger rival prices versus the new
     equilibrium. Welfare. 500 random starting points.
  4. Pass-through at the status quo: implicit-function-theorem matrix dp/dc^T, checked by
     re-solving the equilibrium after small cost changes; partial (rivals fixed) values
     1 - s_j, -s_j and 1 - S_f.
Output: figures/ch04/src/supply_examples_numbers.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.special import lambertw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                      # .../ECON_257
DATA = ROOT / "ProblemSet_1" / "data_simulations.csv"
OUT_TXT = HERE / "supply_examples_numbers.txt"

BETA = np.array([0.1, 0.25, 0.1])
ALPHA = 2.0
GAMMA = np.array([0.2, 0.25, 0.3])
TOL = 1e-13

df = pd.read_csv(DATA).sort_values("prod_id").reset_index(drop=True)
X = df[["X1", "X2", "X3"]].to_numpy()
XI = df["xi"].to_numpy()
OMEGA_C = df["omega"].to_numpy()
OWNER = df["owner"].to_numpy()
J = len(df)
MC = X @ GAMMA + OMEGA_C
A = X @ BETA + XI


def shares(p):
    e = np.exp(A - ALPHA * p)
    phi = 1.0 + e.sum()
    return e / phi, 1.0 / phi, phi


def jac(p):
    """Delta[j, k] = d s_j / d p_k."""
    s, _, _ = shares(p)
    D = ALPHA * np.outer(s, s)
    D[np.diag_indices(J)] = -ALPHA * s * (1.0 - s)
    return D


def omega_matrix(owner):
    return (owner[:, None] == owner[None, :]).astype(float)


def G(p, c, Om):
    """Stacked first-order conditions s + (Omega o Delta^T)(p - c)."""
    return shares(p)[0] + (Om * jac(p).T) @ (p - c)


def markup_iteration(c, Om, p0, maxit=100_000):
    p = p0.copy()
    for it in range(1, maxit + 1):
        s, _, _ = shares(p)
        p_new = c - np.linalg.solve(Om * jac(p).T, s)
        d = np.max(np.abs(p_new - p))
        p = p_new
        if d < TOL:
            return p, it
    raise RuntimeError("markup iteration failed")


def br_iteration(c, owner, p0, maxit=100_000):
    v = A - ALPHA * c
    p = p0.copy()
    firms = np.unique(owner)
    for it in range(1, maxit + 1):
        e = np.exp(A - ALPHA * p)
        p_new = p.copy()
        for f in firms:
            own = owner == f
            z = np.exp(v[own]).sum() * np.exp(-1.0) / (1.0 + e[~own].sum())
            p_new[own] = c[own] + (1.0 + lambertw(z).real) / ALPHA
        d = np.max(np.abs(p_new - p))
        p = p_new
        if d < TOL:
            return p, it
    raise RuntimeError("BR iteration failed")


def ms_iteration(c, Om, p0, maxit=100_000):
    """Morrow-Skerlos: p - c = Lambda^{-1}[s + (Omega o Gamma)(p - c)], logit case
    Lambda = alpha diag(s), Gamma = alpha s s^T."""
    p = p0.copy()
    for it in range(1, maxit + 1):
        s, _, _ = shares(p)
        Lam = ALPHA * s
        Gam = ALPHA * np.outer(s, s)
        zeta = (s + (Om * Gam) @ (p - c)) / Lam
        p_new = c + zeta
        d = np.max(np.abs(p_new - p))
        p = p_new
        if d < TOL:
            return p, it
    raise RuntimeError("MS iteration failed")


def newton(c, Om, p0):
    """Powell hybrid method (scipy root, finite-difference/Broyden Jacobian)."""
    sol = root(lambda p: G(p, c, Om), p0, method="hybr", tol=1e-14)
    return sol.x, sol.nfev, sol.success


def dGdp_analytic(p, c, Om):
    """Jacobian of G from eq:04-dGdp: Delta + Omega o Delta^T + H, with
    H[j, l] = sum_k Omega[j, k] m_k d^2 s_k / (dp_j dp_l). For logit,
    d s_k / d p_j = -alpha s_k (1{k=j} - s_j) = Delta[k, j] and
    d^2 s_k / (dp_j dp_l) = -alpha [Delta[k, l] (1{k=j} - s_j) - s_k Delta[j, l]]."""
    s, _, _ = shares(p)
    D = jac(p)                                   # D[k, j] = d s_k / d p_j
    m = p - c
    I = np.eye(J)
    # second derivatives T[k, j, l] = d^2 s_k / (dp_j dp_l)
    T = -ALPHA * (D[:, None, :] * (I - s[None, :])[:, :, None]
                  - s[:, None, None] * D[None, :, :])
    H = np.einsum("jk,k,kjl->jl", Om, m, T)
    return D + Om * D.T + H


def newton_analytic(c, Om, p0, maxit=100):
    """Newton's method p <- p - [dG/dp^T]^{-1} G(p) with the analytic Jacobian."""
    p = p0.copy()
    for it in range(1, maxit + 1):
        step = np.linalg.solve(dGdp_analytic(p, c, Om), G(p, c, Om))
        p = p - step
        if np.max(np.abs(step)) < TOL:
            return p, it
    raise RuntimeError("Newton failed")


def best_response_firm(c, owner, f, p):
    """Exact best response of firm f (common markup, Lambert W) to the other prices in p."""
    v = A - ALPHA * c
    own = owner == f
    e = np.exp(A - ALPHA * p)
    z = np.exp(v[own]).sum() * np.exp(-1.0) / (1.0 + e[~own].sum())
    out = p.copy()
    out[own] = c[own] + (1.0 + lambertw(z).real) / ALPHA
    return out


lines = []
w = lines.append

# ------------------------------------------------------------------ status quo
OmM = omega_matrix(OWNER)
p0, it0 = br_iteration(MC, OWNER, MC + 0.5)
s0, s00, phi0 = shares(p0)
w("Status quo (owners in the data, Bertrand-Nash)")
w(f"  outside share {s00:.4f}, Phi {phi0:.4f}, CS {np.log(phi0) / ALPHA:.4f}, "
  f"total profit {((p0 - MC) * s0).sum():.4f}")
for f in range(5):
    own = OWNER == f
    w(f"  owner {f}: n={own.sum()}, S_f={100 * s0[own].sum():.2f}%, markup {(p0 - MC)[own][0]:.4f}, "
      f"100*profit {100 * ((p0 - MC) * s0)[own].sum():.3f}")
w("")

# ------------------------------------------------------------------ 1. cost recovery
w("1. Cost recovery at the observed (status quo) prices, alpha known")
Delta0 = jac(p0)
structures = {"single-product": np.eye(J), "true owners": OmM, "one owner": np.ones((J, J))}
mc_hat = {}
for name, Om in structures.items():
    mh = p0 + np.linalg.solve(Om * Delta0.T, s0)
    mc_hat[name] = mh
    err = mh - MC
    w(f"  {name:15s}: implied markup range {np.min(p0 - mh):.4f} to {np.max(p0 - mh):.4f}; "
      f"mc_hat - mc: mean {err.mean():+.4f}, min {err.min():+.4f}, max {err.max():+.4f}; "
      f"negative mc_hat: {(mh < 0).sum()} of {J}; mean mc_hat {mh.mean():.4f} (true mean {MC.mean():.4f})")
w(f"  true mc range {MC.min():.4f} to {MC.max():.4f}; observed price range {p0.min():.4f} to {p0.max():.4f}")
w(f"  share-weighted Lerner index at status quo: true {np.sum(s0 * (p0 - MC) / p0) / s0.sum():.4f}")
for name in structures:
    L = (p0 - mc_hat[name]) / p0
    w(f"    implied median Lerner index ({name}): {np.median(L):.4f}")
w("")

# ------------------------------------------------------------------ 2. cost regressions
w("2. OLS of mc_hat on x = (1, X2, X3); true gamma = (0.2, 0.25, 0.3)")
XtX_inv = np.linalg.inv(X.T @ X)
for name in ["true", *structures.keys()]:
    yv = MC if name == "true" else mc_hat[name]
    g = XtX_inv @ X.T @ yv
    res = yv - X @ g
    s2 = res @ res / (J - X.shape[1])
    se = np.sqrt(np.diag(s2 * XtX_inv))
    w(f"  {name:15s}: gamma_hat = ({g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}), "
      f"se = ({se[0]:.4f}, {se[1]:.4f}, {se[2]:.4f}), R2 = {1 - res @ res / np.sum((yv - yv.mean())**2):.3f}")
w(f"  X2 range {X[:, 1].min():.3f} to {X[:, 1].max():.3f}; X3 values {np.unique(X[:, 2])}")
w("")

# ------------------------------------------------------------------ 3. merger
w("3. Merger of owners 3 and 4 (status quo -> merged), mc unchanged")
owner_post = np.where(OWNER == 4, 3, OWNER)
OmP = omega_matrix(owner_post)
start = p0.copy()
pA, itA = markup_iteration(MC, OmP, start)
pB, itB = br_iteration(MC, owner_post, start)
pC, nfC, okC = newton(MC, OmP, start)
pD, itD = ms_iteration(MC, OmP, start)
w(f"  markup iteration: {itA} iterations; BR iteration: {itB}; MS iteration: {itD}; "
  f"Newton (hybr): {nfC} function evaluations, success {okC}")
w(f"  max |p_markup - p_BR| {np.max(np.abs(pA - pB)):.2e}, |p_MS - p_BR| {np.max(np.abs(pD - pB)):.2e}, "
  f"|p_Newton - p_BR| {np.max(np.abs(pC - pB)):.2e}")
pE, itE = newton_analytic(MC, OmP, start)
jac_err = np.max(np.abs(dGdp_analytic(start, MC, OmP)
                        - np.column_stack([(G(start + 1e-6 * e, MC, OmP) - G(start - 1e-6 * e, MC, OmP)) / 2e-6
                                           for e in np.eye(J)])))
w(f"  Newton with analytic Jacobian (eq:04-dGdp): {itE} iterations ({itE} Jacobian and {itE} G evaluations); "
  f"max |p_Newton_analytic - p_BR| {np.max(np.abs(pE - pB)):.2e}; analytic vs numerical Jacobian max error {jac_err:.2e}")
w(f"  max FOC residual at solution {np.max(np.abs(G(pB, MC, OmP))):.2e}")
rng = np.random.default_rng(4)
dev = 0.0
for _ in range(500):
    pr, _ = br_iteration(MC, owner_post, MC + rng.uniform(0, 5, J))
    dev = max(dev, np.max(np.abs(pr - pB)))
w(f"  500 random starts mc + U[0,5]^25: max deviation {dev:.2e}")
p1 = pB
s1, s01, phi1 = shares(p1)
merged = owner_post == 3
# unilateral response: merged firm best-responds to pre-merger rival prices
pU = best_response_firm(MC, owner_post, 3, p0)
sU, s0U, phiU = shares(pU)
w("  Price changes (post minus pre):")
w(f"    merged firm, unilateral BR at pre-merger rival prices: {np.min((pU - p0)[merged]):.4f} "
  f"to {np.max((pU - p0)[merged]):.4f}, mean {np.mean((pU - p0)[merged]):.4f}")
w(f"    merged firm, new equilibrium: {np.min((p1 - p0)[merged]):.4f} to {np.max((p1 - p0)[merged]):.4f}, "
  f"mean {np.mean((p1 - p0)[merged]):.4f}")
for f in [3, 4]:
    own = OWNER == f
    w(f"      former owner {f}: pre markup {(p0 - MC)[own][0]:.4f}, unilateral {(pU - MC)[own][0]:.4f}, "
      f"post markup {(p1 - MC)[own][0]:.4f}, dp {np.mean((p1 - p0)[own]):.4f}")
w(f"    merged common markup post {(p1 - MC)[merged][0]:.4f}; 1/(alpha(1-S)) = {1 / (ALPHA * (1 - s1[merged].sum())):.4f}")
w(f"    rivals (owners 0-2), new equilibrium: {np.min((p1 - p0)[~merged]):.4f} to {np.max((p1 - p0)[~merged]):.4f}, "
  f"mean {np.mean((p1 - p0)[~merged]):.4f}")
for f in [0, 1, 2]:
    own = OWNER == f
    w(f"      owner {f}: pre markup {(p0 - MC)[own][0]:.4f}, post markup {(p1 - MC)[own][0]:.4f}, "
      f"dp {np.mean((p1 - p0)[own]):.4f}")
w("  Shares and profits (pre -> post):")
for f in range(5):
    own = OWNER == f
    w(f"    owner {f}: S {100 * s0[own].sum():.2f}% -> {100 * s1[own].sum():.2f}%, "
      f"100*profit {100 * ((p0 - MC) * s0)[own].sum():.3f} -> {100 * ((p1 - MC) * s1)[own].sum():.3f}")
pm0 = ((p0 - MC) * s0)[merged].sum()
pm1 = ((p1 - MC) * s1)[merged].sum()
pmU = ((pU - MC) * sU)[merged].sum()
pr0 = ((p0 - MC) * s0)[~merged].sum()
pr1 = ((p1 - MC) * s1)[~merged].sum()
w(f"    merging parties combined: 100*profit {100 * pm0:.3f} -> unilateral {100 * pmU:.3f} -> equilibrium {100 * pm1:.3f}")
w(f"    rivals combined: 100*profit {100 * pr0:.3f} -> {100 * pr1:.3f}")
cs0, cs1 = np.log(phi0) / ALPHA, np.log(phi1) / ALPHA
w(f"  outside share {s00:.4f} -> {s01:.4f}")
w(f"  CS {cs0:.4f} -> {cs1:.4f} (change {cs1 - cs0:+.4f}, {100 * (cs1 / cs0 - 1):+.2f}%)")
w(f"  total profit {((p0 - MC) * s0).sum():.4f} -> {((p1 - MC) * s1).sum():.4f} "
  f"(change {((p1 - MC) * s1).sum() - ((p0 - MC) * s0).sum():+.4f})")
w(f"  total surplus {cs0 + ((p0 - MC) * s0).sum():.4f} -> {cs1 + ((p1 - MC) * s1).sum():.4f} "
  f"(change {cs1 + ((p1 - MC) * s1).sum() - cs0 - ((p0 - MC) * s0).sum():+.4f})")
w(f"  share-weighted price {np.sum(s0 * p0) / s0.sum():.4f} -> {np.sum(s1 * p1) / s1.sum():.4f}")
# required efficiency: uniform proportional mc reduction for merging firms that keeps merged
# firm's prices at pre-merger levels? (compensating mc reduction: m_post at pre shares)
S_pre_merged = s0[merged].sum()
m_needed = 1 / (ALPHA * (1 - S_pre_merged))
w(f"  compensating marginal-cost reduction (keeps all prices at pre-merger level): "
  f"merged markup at pre shares {m_needed:.4f}; owner 3 needs {m_needed - (p0 - MC)[OWNER == 3][0]:.4f}, "
  f"owner 4 needs {m_needed - (p0 - MC)[OWNER == 4][0]:.4f}")
c_comp = MC.copy()
c_comp[OWNER == 3] -= m_needed - (p0 - MC)[OWNER == 3][0]
c_comp[OWNER == 4] -= m_needed - (p0 - MC)[OWNER == 4][0]
pcomp, _ = br_iteration(c_comp, owner_post, p0)
w(f"    check: max |p(compensated) - p_pre| = {np.max(np.abs(pcomp - p0)):.2e}")
for f in [3, 4]:
    own = OWNER == f
    red = m_needed - (p0 - MC)[own][0]
    w(f"    owner {f}: mean mc {MC[own].mean():.4f}; reduction / mean mc = {red / MC[own].mean():.3f}; "
      f"reduction / mc range {np.min(red / MC[own]):.3f} to {np.max(red / MC[own]):.3f}")
w("")

# ------------------------------------------------------------------ 4. pass-through
w("4. Pass-through at the status quo (IFT and finite differences)")


def num_jac(fun, x, h=1e-6):
    n = len(x)
    out = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = h
        out[:, k] = (fun(x + e) - fun(x - e)) / (2 * h)
    return out


dGdp = num_jac(lambda p: G(p, MC, OmM), p0)
dGdc = -(OmM * jac(p0).T)
dGdc_num = num_jac(lambda c: G(p0, c, OmM), MC)
w(f"  check dG/dc^T = -(Omega o Delta^T): max error {np.max(np.abs(dGdc - dGdc_num)):.2e}")
PT = -np.linalg.solve(dGdp, dGdc)          # dp/dc^T
# finite-difference check by re-solving
h = 1e-5
PT_fd = np.zeros((J, J))
for k in range(J):
    c_up = MC.copy(); c_up[k] += h
    c_dn = MC.copy(); c_dn[k] -= h
    pu, _ = br_iteration(c_up, OWNER, p0)
    pd_, _ = br_iteration(c_dn, OWNER, p0)
    PT_fd[:, k] = (pu - pd_) / (2 * h)
w(f"  max |IFT - finite-difference re-solve| = {np.max(np.abs(PT - PT_fd)):.2e}")
own_diag = np.diag(PT)
w(f"  own pass-through dp_j/dc_j (equilibrium): range {own_diag.min():.4f} to {own_diag.max():.4f}, mean {own_diag.mean():.4f}")
w(f"  partial own pass-through 1 - s_j: range {(1 - s0).min():.4f} to {(1 - s0).max():.4f}")
# product-specific shock: product 20 (owner 4), largest share of owner 4
j = 20
f = OWNER[j]
own = OWNER == f
others_own = own.copy(); others_own[j] = False
w(f"  product {j} (owner {f}, s_j = {s0[j]:.4f}): partial own 1 - s_j = {1 - s0[j]:.4f}, "
  f"partial on other own products -s_j = {-s0[j]:.4f}")
w(f"    equilibrium: own {PT[j, j]:.4f}; other own products {PT[others_own, j].min():.4f} to {PT[others_own, j].max():.4f}; "
  f"rivals {PT[~own, j].min():.5f} to {PT[~own, j].max():.5f}")
# check partial values numerically: firm f best-responds holding rivals fixed
c_up = MC.copy(); c_up[j] += h
c_dn = MC.copy(); c_dn[j] -= h
pu = best_response_firm(c_up, OWNER, f, p0)
pd_ = best_response_firm(c_dn, OWNER, f, p0)
part = (pu - pd_) / (2 * h)
w(f"    numerical partial (rivals fixed): own {part[j]:.4f}, other own {part[others_own].min():.4f} to {part[others_own].max():.4f}")
# firm-wide shock to owner 4
S4 = s0[own].sum()
resp = PT[:, own].sum(axis=1)
w(f"  firm-wide shock to owner {f} (S_f = {S4:.4f}): partial 1 - S_f = {1 - S4:.4f}")
w(f"    equilibrium: own products {resp[own].min():.4f} to {resp[own].max():.4f}; rivals {resp[~own].min():.4f} to {resp[~own].max():.4f}")
c_up = MC.copy(); c_up[own] += h
c_dn = MC.copy(); c_dn[own] -= h
pu = best_response_firm(c_up, OWNER, f, p0)
pd_ = best_response_firm(c_dn, OWNER, f, p0)
part = (pu - pd_) / (2 * h)
w(f"    numerical partial (rivals fixed): {part[own].min():.4f} to {part[own].max():.4f}")
for g in range(5):
    og = OWNER == g
    r = PT[:, og].sum(axis=1)
    w(f"    firm-wide shock to owner {g} (S_f={s0[og].sum():.4f}): partial {1 - s0[og].sum():.4f}, "
      f"equilibrium own {r[og].mean():.4f}, rivals mean {r[~og].mean():.4f}")
# industry-wide shock
ind = PT.sum(axis=1)
w(f"  industry-wide shock (all products): equilibrium pass-through {ind.min():.4f} to {ind.max():.4f}, "
  f"share-weighted mean {np.sum(s0 * ind) / s0.sum():.4f}")
w("")

# ------------------------------------------------------------------ 5. duopoly closed form
w("5. Symmetric single-product logit duopoly with s1 = s2 = 0.3 (closed form)")
s = 0.3
b = s * s / (1 - s)
w(f"  b = {b:.5f}; own equilibrium pass-through (1-s)/(1-b^2) = {(1 - s) / (1 - b * b):.4f}; "
  f"rival response b(1-s)/(1-b^2) = {b * (1 - s) / (1 - b * b):.4f}; industry-wide (1-s)/(1-b) = {(1 - s) / (1 - b):.4f}")
# numerical check: choose delta so shares are 0.3 at equilibrium, alpha=1, c=1
alpha_d, c_d = 1.0, 1.0
m = 1 / (alpha_d * (1 - s))
pstar = c_d + m
# s = e^{d - p} / (1 + 2 e^{d - p}) = 0.3 -> e^{d-p} = 0.3/(1 - 0.6) = 0.75
dd = np.log(0.75) + pstar


def duo_eq(c):
    p = np.array([pstar, pstar])
    for _ in range(100000):
        e = np.exp(dd - alpha_d * p)
        p_new = np.empty(2)
        for k in range(2):
            z = np.exp(dd - alpha_d * c[k] - 1) / (1 + e[1 - k])
            p_new[k] = c[k] + (1 + lambertw(z).real) / alpha_d
        if np.max(np.abs(p_new - p)) < 1e-14:
            return p_new
        p = p_new
    return p


hh = 1e-6
pu = duo_eq(np.array([c_d + hh, c_d]))
pdn = duo_eq(np.array([c_d - hh, c_d]))
w(f"  numerical: own {(pu[0] - pdn[0]) / (2 * hh):.4f}, rival {(pu[1] - pdn[1]) / (2 * hh):.4f}")
pu = duo_eq(np.array([c_d + hh, c_d + hh]))
pdn = duo_eq(np.array([c_d - hh, c_d - hh]))
w(f"  numerical industry-wide: {(pu[0] - pdn[0]) / (2 * hh):.4f}")

OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
