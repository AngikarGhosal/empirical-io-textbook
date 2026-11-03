"""Numbers used in the text of Chapter 6, Part 1 (sections 6.1-6.3).

1. Two-goods example with independent standard Gumbel shocks
   (delta_1 = 0.5 -> 1.3, delta_2 = 1.0): shares, proportional losses,
   change in expected maximum utility. Checked by Monte Carlo (fixed seed).
2. Sampling error of observed shares: relative standard deviation of an
   observed share computed from M independent consumers.
3. Numerical check of the WDZ theorem (gradient of the social surplus function)
   for correlated normal shocks.
"""
import numpy as np

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
