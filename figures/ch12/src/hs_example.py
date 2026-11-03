"""Worked example for the Harrington and Skrzypacz (2011) lysine strategy profile
(Section 12.7.3). No figure; prints every number quoted in the text.

Notation (book): N firms, market demand m uniform on {m_lo, ..., m_hi}, mean Em,
static Nash price p^N(c) = c + kappa0 (constant markup), collusive price
p_hat = p^N(c) + Delta, per-unit transfer z = (N-1)/N * Delta, breakdown function
  phi(x) = beta (m_hi - x)(1 - delta)   for x <= m_hi
  phi(x) = (1 - delta)**omega0          for x >  m_hi.
On path each customer picks a firm with probability 1/N (binomial split), which
satisfies the belief assumptions A2 and A3 of the paper (their footnote 9).

Checks, for beta = beta_min (the bound (4) of the paper) and for a larger beta:
  (a) underreporting ICCs for every q and every r < q;
  (b) overreporting ICCs (including the extra payment) for every q and r > q;
  (c) no-report deviation; (d) phi in [0, 1].
The checks are run for the on-path split (s = 1/N) and for splits s in a grid,
which stand in for the beliefs after a deviant own price.
"""
import numpy as np
from scipy.stats import binom

N = 5
m_lo, m_hi = 80, 100
ms = np.arange(m_lo, m_hi + 1)
pm = np.full(ms.size, 1.0 / ms.size)
Em = float(pm @ ms)
Delta = 1.0                       # p_hat - p^N(c), a free normalization
z = (N - 1) / N * Delta           # Proposition (effective marginal cost)
omega0 = 0.5
qbar = m_hi

print(f"N = {N}, m uniform on {{{m_lo},...,{m_hi}}}, E[m] = {Em:.4f}, "
      f"mbar - E[m] = {m_hi - Em:.4f}")
print(f"Corollary condition E[m]/(mbar-E[m]) = {Em/(m_hi-Em):.4f} > N-1 = {N-1}")
print(f"Equivalent: E[m]/mbar = {Em/m_hi:.4f} > (N-1)/N = {(N-1)/N:.4f}")
print(f"Uniform form: mbar/m_lo = {m_hi/m_lo:.4f} < N/(N-2) = {N/(N-2):.4f}")
retained = 1 - (N - 1) * (m_hi - Em) / Em
print(f"Retained share of first-best collusive gain = {retained:.6f} (5/9 = {5/9:.6f})")


def beta_min(delta):
    return N * z / (delta * (Em * Delta - N * z * (m_hi - Em)))


def phi(x, beta, delta):
    x = np.asarray(x)
    return np.where(x <= m_hi, beta * (m_hi - x) * (1 - delta),
                    (1 - delta) ** omega0)


def gain(beta, delta):
    """V - V^N from equation (7) of the paper."""
    return (Delta * Em / N) / (1 - delta + delta * beta * (m_hi - Em) * (1 - delta))


def beliefs(q, s):
    w = pm * binom.pmf(q, ms, s)
    return w / w.sum()


def check(beta, delta, s_grid):
    G = gain(beta, delta)
    worst = np.inf
    for s in s_grid:
        for q in range(0, qbar + 1):
            sig = beliefs(q, s)
            if not np.isfinite(sig).all():
                continue
            # payoff of report r, up to terms that do not depend on r
            def val(r):
                return -z * r - delta * G * (sig @ phi(ms - q + r, beta, delta))
            v_true = val(q)
            v_none = -delta * G * 1.0 - 0.0   # no report: certain breakdown, no payment
            others = [val(r) for r in range(0, qbar + 1) if r != q]
            # relative comparison with the same r-independent terms
            slack = v_true - max(max(others), v_none + 0.0 * q)
            worst = min(worst, slack)
    return worst


def by_type(beta, delta, s=None):
    """Smallest slack of the underreporting and overreporting ICCs, with arg (q, r)."""
    s = 1.0 / N if s is None else s
    G = gain(beta, delta)
    wu, wo = (np.inf, None), (np.inf, None)
    for q in range(0, qbar + 1):
        sig = beliefs(q, s)
        vals = np.array([-z * r - delta * G * (sig @ phi(ms - q + r, beta, delta))
                         for r in range(0, qbar + 1)])
        for r in range(0, qbar + 1):
            sl = vals[q] - vals[r]
            if r < q and sl < wu[0]:
                wu = (sl, (q, r))
            if r > q and sl < wo[0]:
                wo = (sl, (q, r))
    return wu, wo


for delta, mult in ((0.95, 1.0), (0.95, 1.5), (0.99, 1.5)):
    wu, wo = by_type(mult * beta_min(delta), delta)
    print(f"delta = {delta}, beta = {mult} beta_min: smallest under-report slack "
          f"{wu[0]:.4f} at (q, r) = {wu[1]}; smallest over-report slack {wo[0]:.4f} at {wo[1]}")

for delta in (0.95, 0.99):
    b = beta_min(delta)
    G = gain(b, delta)
    hazard = b * (m_hi - Em) * (1 - delta)
    print(f"\ndelta = {delta}: beta_min = {b:.6f} (= 0.08/delta = {0.08/delta:.6f})")
    print(f"  expected breakdown probability per period = {hazard:.6f}; "
          f"formula 0.8(1-delta)/delta = {0.8*(1-delta)/delta:.6f}")
    print(f"  expected length of the collusive phase = {1/hazard:.2f} periods")
    print(f"  largest on-path breakdown probability (m = m_lo) = "
          f"{float(phi(m_lo, b, delta)):.6f}")
    print(f"  phi(0) = {float(phi(0, b, delta)):.6f}, phi(mbar+1) = {(1-delta)**omega0:.6f}")
    print(f"  underreport-by-one ICC: delta (V-V^N) beta (1-delta) = "
          f"{delta*G*b*(1-delta):.6f} vs z = {z:.6f}")
    norm = (1 - delta) * G
    print(f"  normalized gain (1-delta)(V-V^N) = {norm:.6f}; formula (1/N)[E[m]Delta - Nz(mbar-E[m])] "
          f"= {(Em*Delta - N*z*(m_hi-Em))/N:.6f}; first best E[m]Delta/N = {Em*Delta/N:.6f}")
    s_grid = [1.0 / N]
    print(f"  min slack of all reporting ICCs, on-path split: {check(b, delta, s_grid):.3e}")
    print(f"  same with beta = 1.5 beta_min: {check(1.5*b, delta, s_grid):.3e}")
    s_grid = np.linspace(0.05, 0.6, 12)
    print(f"  min slack over deviant splits s in [0.05, 0.6], beta_min: "
          f"{check(b, delta, s_grid):.3e}")

# Uniform-demand thresholds N/(N-2)
print("\nN, required E[m]/mbar, uniform bound N/(N-2):")
for n in range(2, 8):
    ub = "any m_lo > 0" if n == 2 else f"{n/(n-2):.4f}"
    print(f"  {n}: {(n-1)/n:.4f}, {ub}")
