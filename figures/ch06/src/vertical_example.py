"""Worked numerical example of the vertical model (Section 6.5, Example
ex:06-vnumeric, Table tab:06-vnumeric) and the numbers behind the three
vertical-model figures (fig:06-frontier, fig:06-ulines, fig:06-nuaxis).

Model: u_ij = delta_j - nu_i p_j, nu_i ~ Exponential(lambda = 1), outside
good with delta_0 = p_0 = 0.

Part A inverts observed shares for qualities, computes the share Jacobian,
the elasticity matrix (with the outside-good row), and the markups and
implied marginal costs under three ownership structures. Every analytic
result is checked: the shares implied by the recovered qualities are
recomputed by Monte Carlo and by the closed form, the Jacobian by finite
differences, and the first-order conditions by direct evaluation.

Part B prints the numbers used for the illustrative figures.

No figure is produced. Fixed seed for the Monte Carlo check.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
RNG = np.random.default_rng(20261022)


# ----------------------------------------------------------------- helpers
def F(x):
    """CDF of Exp(1) on (0, inf)."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 1.0 - np.exp(-np.clip(x, 0, None)), 0.0)


def f(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, np.exp(-np.clip(x, 0, None)), 0.0)


def Finv(u):
    return -np.log1p(-u)


def shares_general(delta, p):
    """Berry-Pakes general cutoffs (min over cheaper, max over dearer).
    Products must be sorted by price; the outside good is (0, 0)."""
    d = np.concatenate([[0.0], delta])
    q = np.concatenate([[0.0], p])
    n = len(d)
    s = np.zeros(n)
    for j in range(n):
        up = np.inf if j == 0 else min((d[j] - d[k]) / (q[j] - q[k]) for k in range(j))
        lo = 0.0 if j == n - 1 else max((d[k] - d[j]) / (q[k] - q[j]) for k in range(j + 1, n))
        lo = max(lo, 0.0)
        if up > lo:
            s[j] = (1.0 if np.isinf(up) else F(up)) - F(lo)
    return s  # s[0] is the outside good


def invert(p, s):
    S = np.cumsum(s[::-1])[::-1]          # S_j = sum_{k >= j} s_k
    nubar = Finv(S)                        # cutoffs
    dp = np.diff(np.concatenate([[0.0], p]))
    delta = np.cumsum(dp * nubar)
    return S, nubar, delta


def jacobian(p, S, nubar):
    dp = np.diff(np.concatenate([[0.0], p]))
    g = f(nubar) * nubar / dp              # g_j = f(nubar_j) nubar_j / (p_j - p_{j-1})
    J = len(p)
    D = np.zeros((J, J))
    for j in range(J):
        D[j, j] = -g[j] - (g[j + 1] if j + 1 < J else 0.0)
        if j + 1 < J:
            D[j, j + 1] = g[j + 1]
            D[j + 1, j] = g[j + 1]
    d0 = np.zeros(J)
    d0[0] = g[0]                            # ds_0 / dp_1
    return g, D, d0


def markups(D, s, owner):
    Om = (owner[:, None] == owner[None, :]).astype(float)
    A = Om * D.T
    m = -np.linalg.solve(A, s)
    return m, A


# ----------------------------------------------------------------- Part A
print("=" * 70)
print("PART A: worked example")
p = np.array([1.0, 1.5, 2.5])
s = np.array([0.20, 0.15, 0.10])
s0 = 1.0 - s.sum()
print("prices", p, " shares", s, " s0 = %.2f" % s0)

S, nubar, delta = invert(p, s)
print("S_j        ", S)
print("cutoffs    ", np.round(nubar, 4), " decreasing:", np.all(np.diff(nubar) < 0))
print("f(cutoff)  ", np.round(f(nubar), 4), " (= 1 - S_j)")
print("delta      ", np.round(delta, 4))
print("increments (p_j - p_{j-1}) nubar_j:", np.round(np.diff(np.concatenate([[0], p])) * nubar, 4))

# check 1: closed-form shares at recovered delta
s_chk = shares_general(delta, p)
print("closed-form shares at recovered delta (s0, s1..):", np.round(s_chk, 6))
# check 2: Monte Carlo
nu = RNG.exponential(1.0, size=2_000_000)
U = np.column_stack([np.zeros_like(nu)] + [delta[j] - nu * p[j] for j in range(3)])
ch = U.argmax(axis=1)
print("Monte Carlo shares (2e6 draws):", np.round(np.bincount(ch, minlength=4) / nu.size, 4))

g, D, d0 = jacobian(p, S, nubar)
print("g_j        ", np.round(g, 4))
print("Jacobian D (rows shares, cols prices):\n", np.round(D, 4))
print("outside row ds0/dp:", np.round(d0, 4))
print("column sums incl. outside:", np.round(D.sum(axis=0) + d0, 12))

# check 3: finite differences holding delta fixed
h = 1e-6
Dfd = np.zeros((3, 3))
d0fd = np.zeros(3)
for k in range(3):
    pp = p.copy(); pp[k] += h
    pm = p.copy(); pm[k] -= h
    sp = shares_general(delta, pp)
    sm = shares_general(delta, pm)
    Dfd[:, k] = (sp[1:] - sm[1:]) / (2 * h)
    d0fd[k] = (sp[0] - sm[0]) / (2 * h)
print("max |D - D_fd| =", np.abs(D - Dfd).max(), "  max |d0 - d0_fd| =", np.abs(d0 - d0fd).max())

# elasticities eps_jk = (ds_j/dp_k)(p_k/s_j)
E = D * p[None, :] / s[:, None]
E0 = d0 * p / s0
print("Elasticity matrix (rows shares, cols prices):\n", np.round(E, 3))
print("outside row:", np.round(E0, 3))

owners = {
    "(a) single-product firms": np.array([1, 2, 3]),
    "(b) products 1 and 2 jointly owned": np.array([1, 1, 3]),
    "(c) one firm owns all": np.array([1, 1, 1]),
    "(d) products 1 and 3 jointly owned": np.array([1, 2, 1]),
}
for name, own in owners.items():
    m, A = markups(D, s, own)
    mc = p - m
    foc = s + A @ m
    print(name)
    print("   markups      ", np.round(m, 4))
    print("   mc           ", np.round(mc, 4))
    print("   Lerner       ", np.round(m / p, 4))
    print("   FOC residual ", np.abs(foc).max())

# check 4: direct profit maximization for (a): numerical best responses
def profit_j(j, pj, pvec, mc):
    pp = pvec.copy(); pp[j] = pj
    sh = shares_general(delta, pp)[1:]
    return (pj - mc[j]) * sh[j]

m_a, _ = markups(D, s, owners["(a) single-product firms"])
mc_a = p - m_a
for j in range(3):
    grid = np.linspace(p[j] - 0.2, p[j] + 0.2, 4001)
    vals = [profit_j(j, x, p, mc_a) for x in grid]
    print("   best response check (a), product %d: argmax %.4f vs p %.4f" % (j + 1, grid[int(np.argmax(vals))], p[j]))

# ----------------------------------------------------------------- Part B
print("=" * 70)
print("PART B: illustrative figure numbers")
pf = np.array([1.0, 2.0, 3.2])
df = np.array([1.6, 2.4, 2.9])
cut = np.array([df[0] / pf[0], (df[1] - df[0]) / (pf[1] - pf[0]), (df[2] - df[1]) / (pf[2] - pf[1])])
print("cutoffs", np.round(cut, 4))
print("dominated point (2.6, 2.5): chord value at 2.6 =",
      round(df[1] + (2.6 - pf[1]) * cut[2], 4))
print("outright-dominated point (1.5, 1.4): delta below product 1? ", 1.4 < df[0])
sh = np.array([F(cut[2]), F(cut[1]) - F(cut[2]), F(cut[0]) - F(cut[1]), 1 - F(cut[0])])
print("Exp(1) shares s3, s2, s1, s0:", np.round(sh, 4))
# general-cutoff check including the two dominated products
pa = np.array([1.0, 1.5, 2.0, 2.6, 3.2])
da = np.array([1.6, 1.4, 2.4, 2.5, 2.9])
print("general-cutoff shares with dominated products (s0, then by price):",
      np.round(shares_general(da, pa), 4))
