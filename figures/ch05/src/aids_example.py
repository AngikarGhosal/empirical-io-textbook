"""Running numerical example for Chapter 5 (AIDS).

Three goods, alpha_0 = 0, alpha = (0.5, 0.3, 0.2), beta = (-0.10, 0.04, 0.06),
Gamma symmetric with zero row and column sums, base prices p = (1, 1, 1), ln I = 1.

The script computes budget shares, expenditure elasticities, Marshallian and
Hicksian price elasticities, the Deaton-Muellbauer negativity matrix K and its
eigenvalues, checks the adding-up, homogeneity, symmetry, Engel and Cournot
identities, checks the analytic elasticities against numerical derivatives of
the demand functions, and reports the range of ln I (at unit prices) over which
shares stay in [0, 1] and K stays negative semidefinite.  It also computes the
Cobb-Douglas numbers used in the CV/EV figure.  Output goes to
figures/ch05/src/aids_example_numbers.txt.
"""
import os
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "aids_example_numbers.txt")

alpha0 = 0.0
alpha = np.array([0.5, 0.3, 0.2])
beta = np.array([-0.10, 0.04, 0.06])
Gamma = np.array([[-0.20, 0.12, 0.08],
                  [0.12, -0.22, 0.10],
                  [0.08, 0.10, -0.18]])
J = 3


def lnP(lnp):
    """Translog price index ln P(p)."""
    return alpha0 + alpha @ lnp + 0.5 * lnp @ Gamma @ lnp


def b_of(lnp):
    """b(p) = prod_k p_k^beta_k (beta_0 = 1)."""
    return np.exp(beta @ lnp)


def shares(lnp, lnI):
    """Marshallian budget shares."""
    return alpha + Gamma @ lnp + beta * (lnI - lnP(lnp))


def quantities(lnp, lnI):
    return shares(lnp, lnI) * np.exp(lnI) / np.exp(lnp)


def ln_e(lnp, u):
    """AIDS log expenditure function."""
    return lnP(lnp) + u * b_of(lnp)


def indirect_u(lnp, lnI):
    return (lnI - lnP(lnp)) / b_of(lnp)


def hicksian(lnp, u):
    """Hicksian demands h_i = w_i e / p_i, with Hicksian shares."""
    w = alpha + Gamma @ lnp + beta * u * b_of(lnp)
    return w * np.exp(ln_e(lnp, u)) / np.exp(lnp)


def elasticities(lnp, lnI):
    w = shares(lnp, lnI)
    r = lnI - lnP(lnp)                       # ln(I/P)
    eta = 1.0 + beta / w
    dlnP = alpha + Gamma @ lnp               # d lnP / d ln p_j = w_j - beta_j ln(I/P)
    dw = Gamma - np.outer(beta, dlnP)        # d w_i / d ln p_j
    eps_m = -np.eye(J) + dw / w[:, None]
    eps_h = eps_m + np.outer(eta, w)
    K = Gamma + np.outer(beta, beta) * r - np.diag(w) + np.outer(w, w)
    return w, eta, dw, eps_m, eps_h, K


def fmt(M):
    return "\n".join("  [" + ", ".join(f"{x: .3f}" for x in row) + "]" for row in np.atleast_2d(M))


lines = []
P = lambda s="": lines.append(s)

# ------------------------------------------------------------- restrictions
P("Restrictions")
P(f"  sum alpha = {alpha.sum():.6f}, sum beta = {beta.sum():.6f}")
P(f"  column sums of Gamma = {Gamma.sum(axis=0)}, row sums = {Gamma.sum(axis=1)}")
P(f"  symmetric: {np.allclose(Gamma, Gamma.T)}")

lnp0 = np.zeros(J)
lnI0 = 1.0
w, eta, dw, eps_m, eps_h, K = elasticities(lnp0, lnI0)
P()
P("Base point: p = (1,1,1), ln I = 1, ln P = %.3f, ln(I/P) = %.3f" % (lnP(lnp0), lnI0 - lnP(lnp0)))
P("  shares w = " + ", ".join(f"{x:.4f}" for x in w))
P("  expenditure elasticities eta = " + ", ".join(f"{x:.4f}" for x in eta))
P("  share derivatives dw_i/dln p_j:")
P(fmt(dw))
P("  Marshallian elasticities:")
P(fmt(eps_m))
P("  Hicksian elasticities:")
P(fmt(eps_h))
P("  Hicksian elasticities as k_ij / w_i (check):")
P(fmt(K / w[:, None]))
P("  K matrix:")
P(fmt(K))
ev = np.linalg.eigvalsh(K)
P("  eigenvalues of K = " + ", ".join(f"{x:.4f}" for x in sorted(ev)))
P(f"  K 1 = {K.sum(axis=1)}")

# ------------------------------------------------------------- identities
P()
P("Identities")
P(f"  Engel aggregation sum_i w_i eta_i = {w @ eta:.6f}")
P("  Cournot aggregation sum_i w_i eps_ij + w_j = " + ", ".join(f"{x:.2e}" for x in (w @ eps_m + w)))
P("  Homogeneity sum_j eps_ij + eta_i = " + ", ".join(f"{x:.2e}" for x in (eps_m.sum(axis=1) + eta)))
P("  Hicksian row sums = " + ", ".join(f"{x:.2e}" for x in eps_h.sum(axis=1)))
S_share = w[:, None] * eps_h
P(f"  Symmetry w_i eps^h_ij symmetric: {np.allclose(S_share, S_share.T)}")

# ------------------------------------------------------------- numerical derivative check
h = 1e-6
num_m = np.zeros((J, J))
num_h = np.zeros((J, J))
u0 = indirect_u(lnp0, lnI0)
q0 = quantities(lnp0, lnI0)
hq0 = hicksian(lnp0, u0)
for j in range(J):
    d = np.zeros(J); d[j] = h
    num_m[:, j] = (np.log(quantities(lnp0 + d, lnI0)) - np.log(quantities(lnp0 - d, lnI0))) / (2 * h)
    num_h[:, j] = (np.log(hicksian(lnp0 + d, u0)) - np.log(hicksian(lnp0 - d, u0))) / (2 * h)
num_eta = (np.log(quantities(lnp0, lnI0 + h)) - np.log(quantities(lnp0, lnI0 - h))) / (2 * h)
P()
P("Numerical checks (central differences)")
P(f"  max |Marshallian analytic - numeric| = {np.abs(num_m - eps_m).max():.2e}")
P(f"  max |Hicksian analytic - numeric| = {np.abs(num_h - eps_h).max():.2e}")
P(f"  max |eta analytic - numeric| = {np.abs(num_eta - eta).max():.2e}")
P(f"  Marshallian = Hicksian demand at u0: {np.allclose(q0, hq0)}  (u0 = {u0:.4f})")

# Shephard's lemma check: dlne/dlnp_i = w_i at (p0, u0)
sh = np.array([(ln_e(lnp0 + np.eye(J)[i] * h, u0) - ln_e(lnp0 - np.eye(J)[i] * h, u0)) / (2 * h) for i in range(J)])
P("  Shephard share check dln e/dln p_i = " + ", ".join(f"{x:.4f}" for x in sh))

# ------------------------------------------------------------- ranges in ln I at unit prices
grid = np.linspace(-6, 8, 140001)
ok_share = np.array([np.all((shares(lnp0, g) >= 0) & (shares(lnp0, g) <= 1)) for g in grid[::100]])
g_share = grid[::100][ok_share]
P()
P("Ranges at unit prices")
P(f"  shares in [0,1] for ln I in [{g_share.min():.3f}, {g_share.max():.3f}]")
# exact endpoints
roots = [(-alpha[i]) / beta[i] for i in range(J)]
P("  zero-share points ln I = -alpha_i/beta_i: " + ", ".join(f"{x:.3f}" for x in roots))


def max_eig(lnI):
    _, _, _, _, _, KK = elasticities(lnp0, lnI)
    e = np.linalg.eigvalsh(KK)
    e = e[np.argsort(np.abs(e))][1:]     # drop the structural zero eigenvalue
    return e.max()


sub = grid[::50]
sub = sub[(sub > -3.333) & (sub < 5.0)]
me = np.array([max_eig(g) for g in sub])
conc = sub[me <= 1e-12]
P(f"  K negative semidefinite (nonzero eigenvalues <= 0) for ln I in [{conc.min():.3f}, {conc.max():.3f}] within the valid-share range")
# locate the boundaries precisely
f = lambda g: max_eig(g)
if me[0] > 0:
    lo = brentq(f, sub[0], conc.min() + 0.01)
    P(f"    lower concavity boundary ln I = {lo:.4f}")
if me[-1] > 0:
    hi = brentq(f, conc.max() - 0.01, sub[-1])
    P(f"    upper concavity boundary ln I = {hi:.4f}")
for g in [0.0, 1.0, 2.0, 3.0, 4.0]:
    _, _, _, _, _, KK = elasticities(lnp0, g)
    P(f"    ln I = {g:.1f}: eigenvalues of K = " + ", ".join(f"{x:.4f}" for x in sorted(np.linalg.eigvalsh(KK))))

# ------------------------------------------------------------- Marshallian choke price of good 3 (for reference)
fz = lambda x: shares(np.array([0.0, 0.0, x]), lnI0)[2]
lnpv_m = brentq(fz, 0.0, 5.0)
fzh = lambda x: alpha[2] + Gamma[2, 2] * x + beta[2] * u0 * b_of(np.array([0.0, 0.0, x]))
lnpv_h = brentq(fzh, 0.0, 5.0)
P()
P("Choke (virtual) prices of good 3, other prices at 1")
P(f"  Marshallian (income fixed): ln p3V = {lnpv_m:.3f}, p3V = {np.exp(lnpv_m):.2f}")
P(f"  Hicksian (utility fixed):   ln p3V = {lnpv_h:.3f}, p3V = {np.exp(lnpv_h):.2f}")
cv = np.exp(ln_e(np.array([0.0, 0.0, lnpv_h]), u0)) - np.exp(lnI0)
P(f"  CV of removing good 3 = {cv:.3f} = {100 * cv / np.exp(lnI0):.1f} percent of income")

# ------------------------------------------------------------- Cobb-Douglas numbers for the CV/EV figure
# u = q1^(1/2) q2^(1/2), I = 10, p2 = 1, p1 rises from 1 to 2.
I = 10.0
a = 0.5
u0cd = (a * I / 1.0) ** a * ((1 - a) * I / 1.0) ** (1 - a)
u1cd = (a * I / 2.0) ** a * ((1 - a) * I / 1.0) ** (1 - a)
e = lambda p1, u: u * (p1 / a) ** a * (1.0 / (1 - a)) ** (1 - a)
CV = e(2.0, u0cd) - I
EV = I - e(1.0, u1cd)
dCS = a * I * np.log(2.0)
P()
P("Cobb-Douglas illustration (a = 1/2, I = 10, p2 = 1, p1: 1 -> 2)")
P(f"  u0 = {u0cd:.4f}, u1 = {u1cd:.4f}")
P(f"  CV = {CV:.4f}, Marshallian surplus loss = {dCS:.4f}, EV = {EV:.4f}")
P(f"  Hicksian at u0: q1 = {u0cd:.3f} / sqrt(p1); at u1: q1 = {u1cd:.3f} / sqrt(p1); Marshallian q1 = 5 / p1")

text = "\n".join(lines)
print(text)
with open(OUT, "w") as fh:
    fh.write(text + "\n")
