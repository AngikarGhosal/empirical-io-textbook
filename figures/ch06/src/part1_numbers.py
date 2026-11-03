"""Numbers used in the text of Chapter 6, Part 1 (sections 6.1-6.3).

1. Two-goods example (delta_1 = 0.5 -> 1.3, delta_2 = 1.0) in two versions.
   (a) Logit version: the outside good has its own shock, u_0 = eps_0, and
       eps_0, eps_1, eps_2 are independent standard Gumbel. Shares, proportional
       losses, change in expected maximum utility, and the correlation (1/2) of
       the differenced shocks eps_j - eps_0. Checked by Monte Carlo (fixed seed).
   (b) Deterministic outside option, u_0 = 0, with independent standard Gumbel
       shocks on the two inside goods only. Shares from the truncated-Gumbel
       integrals s_0 = F(-d1) F(-d2) and
       s_1 = int_{-d1}^inf f(e) F(e + d1 - d2) de. Checked by Monte Carlo.
2. Sampling error of observed shares: relative standard deviation of an
   observed share computed from M independent consumers.
3. Numerical check of the WDZ theorem (gradient of the social surplus function)
   for correlated normal shocks.
"""
import numpy as np
from scipy.integrate import quad

rng = np.random.default_rng(20261022)

# 1. Two-goods example -----------------------------------------------------
def logit_shares(d):
    e = np.exp(np.concatenate(([0.0], d)))
    return e / e.sum()

d0 = np.array([0.5, 1.0])
d1 = np.array([1.3, 1.0])
s0, s1 = logit_shares(d0), logit_shares(d1)
print("shares before (s0,s1,s2):", np.round(s0, 4))
print("shares after  (s0,s1,s2):", np.round(s1, 4))
print("ratio after/before for goods 0 and 2:", np.round(s1[0] / s0[0], 4),
      np.round(s1[2] / s0[2], 4))
G0 = np.log(1 + np.exp(d0).sum()) + np.euler_gamma
G1 = np.log(1 + np.exp(d1).sum()) + np.euler_gamma
print("E max before %.4f after %.4f change %.4f; ln(s0 before/s0 after) %.4f"
      % (G0, G1, G1 - G0, np.log(s0[0] / s1[0])))

n = 2_000_000
eps = rng.gumbel(size=(n, 3))
u = eps + np.concatenate(([0.0], d0))
ch = u.argmax(axis=1)
print("MC shares before:", np.round(np.bincount(ch, minlength=3) / n, 4),
      " MC E max %.4f" % u.max(axis=1).mean())
# conditional mean of max utility by chosen alternative
print("MC E[max | choice j]:",
      np.round([u.max(axis=1)[ch == j].mean() for j in range(3)], 4))

# differenced shocks eps_j - eps_0 in the logit version (separate generator so
# that the random stream used by the other checks is unchanged)
rng_b = np.random.default_rng(20261103)
e3 = rng_b.gumbel(size=(n, 3))
dif = e3[:, 1:] - e3[:, [0]]
print("logit version: corr(eps_1 - eps_0, eps_2 - eps_0) = %.4f (theory 0.5);"
      " var = %.4f, %.4f (theory pi^2/3 = %.4f)"
      % (np.corrcoef(dif.T)[0, 1], dif[:, 0].var(), dif[:, 1].var(), np.pi ** 2 / 3))

# (b) deterministic outside option u_0 = 0, Gumbel shocks on inside goods only
def gF(x):
    return np.exp(-np.exp(-x))


def gf(x):
    return np.exp(-x) * gF(x)


def det_shares(d):
    a1, a2 = d
    s_0 = gF(-a1) * gF(-a2)
    s_1 = quad(lambda e: gf(e) * gF(e + a1 - a2), -a1, np.inf)[0]
    return np.array([s_0, s_1, 1.0 - s_0 - s_1])


t0, t1 = det_shares(d0), det_shares(d1)
print("deterministic outside: shares before:", np.round(t0, 4),
      " s0 = exp(-e^d1 - e^d2) = %.5f" % np.exp(-np.exp(d0).sum()))
print("deterministic outside: shares after :", np.round(t1, 4),
      " s0 = %.5f" % np.exp(-np.exp(d1).sum()))
print("deterministic outside: ratio after/before for goods 0 and 2:",
      np.round(t1[0] / t0[0], 4), np.round(t1[2] / t0[2], 4))
e2 = rng_b.gumbel(size=(n, 2))
for dd in (d0, d1):
    uu = np.column_stack([np.zeros(n), e2 + dd])
    print("  MC shares (u_0 = 0) at delta =", dd, ":",
          np.round(np.bincount(uu.argmax(axis=1), minlength=3) / n, 4))

# 2. Sampling error ----------------------------------------------------------
for M, s in [(1_000_000, 0.0001), (1_000_000, 0.01), (10_000, 0.01)]:
    sd = np.sqrt(s * (1 - s) / M)
    print("M=%d s=%.4f sd=%.2e relative sd=%.4f buyers=%.0f" % (M, s, sd, sd / s, M * s))

# 3. WDZ numerical check with correlated normal shocks -------------------------
J1 = 3
Om = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.5]])
L = np.linalg.cholesky(Om)
z = rng.standard_normal((1_000_000, J1)) @ L.T
V = np.array([0.0, 0.4, -0.2])
def G(Vv):
    return (z + Vv).max(axis=1).mean()
P = np.bincount((z + V).argmax(axis=1), minlength=J1) / z.shape[0]
h = 1e-2
grad = np.array([(G(V + h * np.eye(J1)[j]) - G(V - h * np.eye(J1)[j])) / (2 * h)
                 for j in range(J1)])
print("probit P:", np.round(P, 4), " numerical grad G:", np.round(grad, 4))
