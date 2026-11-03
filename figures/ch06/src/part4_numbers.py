"""Numbers used in the text of Chapter 6, Part 4 (tastes for products and the
pure characteristics model).

1. Bounded variety gains in the logit. With an outside good and entrants whose
   mean utilities are V_j = -j^2, expected maximum utility is
   gamma_E + ln(1 + sum_{j<=J} exp(-j^2)), which converges as J grows.
2. Symmetric oligopoly with normal tastes for products. N single-product firms,
   utility -p_j + sigma Z_ij with Z_ij i.i.d. standard normal, no outside good.
   The symmetric markup is sigma / m_N with m_N = E[max_{j<=N} Z_j]. We compute
   m_N by quadrature, check it by Monte Carlo, check the identity
   m_N = N (N - 1) int phi^2 Phi^(N-2), and check by a grid search that the
   markup is a best response when rivals charge it (N = 2, 10, 100).
3. A kink in a simulated pure characteristics share. One draw, outside good
   (0, 0), products with prices (1, 2) and qualities (1 + t, 2), and a price
   coefficient alpha ~ Exp(1). Then s_1(t) = max{exp(-(1 - t)) - exp(-(1 + t)), 0}.
4. The large Monte Carlo design of Berry and Pakes (2007). Probability that
   delta_j = 2 + x_j + xi_j is not positive, under two readings of the
   "range" that scales xi_j: the megahertz range, and the range divided by ten
   (the units of x). Analytic and Monte Carlo (fixed seed).
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr
from scipy.stats import norm

rng = np.random.default_rng(20261103)

# 1. Bounded variety gains ---------------------------------------------------
print("1. Logit with V_j = -j^2 and an outside good")
for J in (1, 2, 5, 10, 1000):
    S = np.exp(-np.arange(1, J + 1, dtype=float) ** 2).sum()
    print("  J=%5d  sum exp(V_j)=%.6f  E max utility=%.6f  share of product J=%.3e"
          % (J, S, np.euler_gamma + np.log1p(S), np.exp(-J ** 2) / (1 + S)))

# 2. Normal tastes for products ---------------------------------------------
def m_N(N):
    """E[max of N standard normals] = int_0^inf (1 - Phi^N) - int_-inf^0 Phi^N."""
    pos = quad(lambda z: -np.expm1(N * log_ndtr(z)), 0, np.inf, limit=400)[0]
    neg = quad(lambda z: np.exp(N * log_ndtr(z)), -np.inf, 0, limit=400)[0]
    return pos - neg


def theta_integral(N):
    """N (N - 1) int phi(z)^2 Phi(z)^(N - 2) dz."""
    f = lambda z: np.exp(2 * norm.logpdf(z) + (N - 2) * log_ndtr(z))
    return N * (N - 1) * quad(f, -np.inf, np.inf, limit=400, points=None)[0]


def share(pj, pbar, N, sigma=1.0):
    """Share of a firm charging pj when the N - 1 rivals charge pbar."""
    a = (pbar - pj) / sigma
    f = lambda z: np.exp(norm.logpdf(z) + (N - 1) * log_ndtr(z + a))
    return quad(f, -np.inf, np.inf, limit=400)[0]


print("2. Normal tastes for products, sigma = 1, no outside good")
for N in (2, 10, 100, 1000, 10 ** 6):
    m = m_N(N)
    line = "  N=%8d  m_N=%.4f  markup=%.4f  bound 1/sqrt(2 ln N)=%.4f" % (
        N, m, 1 / m, 1 / np.sqrt(2 * np.log(N)))
    if N <= 1000:
        line += "  N(N-1)int phi^2 Phi^(N-2)=%.4f" % theta_integral(N)
    print(line)
print("  m_2 = 1/sqrt(pi) = %.4f" % (1 / np.sqrt(np.pi)))
for N in (2, 10, 100):
    mc = rng.standard_normal((200_000, N)).max(axis=1).mean()
    c, pbar = 0.0, 1 / m_N(N)
    grid = np.linspace(pbar * 0.5, pbar * 1.5, 401)
    prof = np.array([(p - c) * share(p, pbar, N) for p in grid])
    print("  N=%3d  MC m_N=%.4f  candidate markup=%.4f  grid best response=%.4f"
          "  share at candidate=%.4f (1/N=%.4f)"
          % (N, mc, pbar, grid[prof.argmax()], share(pbar, pbar, N), 1 / N))

# 3. Kink in a simulated pure characteristics share --------------------------
def s1(t):
    return max(np.exp(-(1 - t)) - np.exp(-(1 + t)), 0.0)


h = 1e-6
print("3. Kink: s1(0)=%.6f  left derivative=%.6f  right derivative=%.6f  2/e=%.6f"
      % (s1(0), (s1(0) - s1(-h)) / h, (s1(h) - s1(0)) / h, 2 / np.e))

# 4. Large Monte Carlo design of Berry and Pakes -----------------------------
print("4. Large design: P(delta_j = 2 + x_j + xi_j <= 0)")
years = {1: (25, 133), 2: (25, 200), 3: (33, 240)}
for t, (lo, hi) in years.items():
    xlo, xhi = lo / 10, hi / 10
    for label, R in (("MHz range", hi - lo), ("range / 10", (hi - lo) / 10)):
        # xi = R (2u - 1) is uniform on [-R, R]; P(xi <= -(2 + x)) given x
        cond = lambda x: np.clip((R - 2 - x) / (2 * R), 0, 1)
        p_an = quad(cond, xlo, xhi)[0] / (xhi - xlo)
        x = rng.uniform(xlo, xhi, 1_000_000)
        xi = R * (2 * rng.uniform(size=x.size) - 1)
        p_mc = np.mean(2 + x + xi <= 0)
        print("  year %d  x in [%.1f, %.1f]  %-10s R=%6.1f  P=%.4f  MC=%.4f"
              % (t, xlo, xhi, label, R, p_an, p_mc))
