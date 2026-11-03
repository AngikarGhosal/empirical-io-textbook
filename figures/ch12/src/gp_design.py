"""Symmetric trigger-price equilibria in the original Green-Porter (1984) model (Section 12.5.3, example).

Two firms, inverse demand P(Q) = 1 - Q, zero marginal cost. The observed price is p = eta P(Q), where
ln(eta) ~ N(-s^2/2, s^2) so that E[eta] = 1. Cournot output 1/3 per firm, Cournot profit 1/9.
Restricted output y per firm in normal periods. Expected normal-period profit of a firm producing r
when its rival produces y:  pibar(r) = r (1 - r - y).

Trigger price pbar, reversion lasting T - 1 periods. With k = pbar / P(2y) and z = (ln k + s^2/2) / s,
the probability of a reversion in a normal period is Phi(z). The value of a firm producing r is (GP (3))
   V(r) = piN / (1 - d) + (pibar(r) - piN) / (1 - d + (d - d^T) Phi(z_r)),
   z_r = (ln(pbar / (1 - r - y)) + s^2 / 2) / s.
The equilibrium condition is GP (6) at r = y, plus the global condition V(r) <= V(y) for all r (GP (4)).

For each (y, T) the script finds the trigger z solving (6) with the smallest reversion probability,
checks (4) on a grid of r, and reports the cartel's best (y, T, pbar). Prints every number quoted.
"""
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

piN = 1.0 / 9.0
SQ2PI = math.sqrt(2 * math.pi)


def phi(z):
    return np.exp(-0.5 * z * z) / SQ2PI


def foc(z, y, T, d, s):
    """GP (6) for the symmetric linear model as a function of z (vectorized in z)."""
    A = d - d ** T
    lhs = (1 - 3 * y) * (1 - d + A * ndtr(z))
    rhs = A * phi(z) / s * (y * (1 - 2 * y) - piN) / (1 - 2 * y)
    return lhs - rhs


RS = np.linspace(0.0, 0.66, 6601)


def V_of_r(r, y, pbar, T, d, s):
    P = 1 - r - y
    zr = (np.log(pbar / P) + s * s / 2) / s
    return piN / (1 - d) + (r * (1 - r - y) - piN) / (1 - d + (d - d ** T) * ndtr(zr))


ZS = np.linspace(-6, 3, 9001)


def equilibrium(y, T, d, s):
    """Lowest-reversion-probability trigger solving (6); returns dict or None."""
    vals = foc(ZS, y, T, d, s)
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    z = brentq(lambda zz: float(foc(zz, y, T, d, s)), ZS[i], ZS[i + 1])
    k = math.exp(z * s - s * s / 2)
    pbar = k * (1 - 2 * y)
    Vy = float(V_of_r(y, y, pbar, T, d, s))
    Vr = V_of_r(RS, y, pbar, T, d, s)
    glob = Vr.max() <= Vy + 1e-9
    return dict(z=z, G=float(ndtr(z)), pbar=pbar, k=k, V=Vy, norm=(1 - d) * Vy, glob=glob,
                rbest=RS[Vr.argmax()])


def run(d, s, Ts, ys):
    print(f"\n=== delta = {d}, sigma = {s} ===")
    print(f"Cournot normalized value = {piN:.5f}; monopoly sharing pibar = 0.12500")
    print("At the monopoly share y = 0.25, by T:")
    for T in Ts:
        e = equilibrium(0.25, T, d, s)
        if e is None:
            print(f"  T = {T:5d}: no trigger solves (6)")
            continue
        print(f"  T = {T:5d}: pbar = {e['pbar']:.4f}, Pr(reversion) = {e['G']:.4f}, "
              f"(1-d)V = {e['norm']:.5f}, global = {e['glob']}, argmax_r V(r) = {e['rbest']:.4f}")
    print("Best y for each T (global condition imposed):")
    best = None
    for T in Ts:
        bestT = None
        for yy in ys:
            e = equilibrium(yy, T, d, s)
            if e is None or not e["glob"]:
                continue
            if bestT is None or e["V"] > bestT[1]["V"]:
                bestT = (yy, e)
        if bestT:
            yy, e = bestT
            share = (T - 1) * e["G"] / (1 + (T - 1) * e["G"])
            print(f"  T = {T:5d}: y* = {yy:.4f}, pbar = {e['pbar']:.4f}, Pr(reversion) = {e['G']:.4f}, "
                  f"share of reversion periods = {share:.4f}, (1-d)V = {e['norm']:.5f}")
            if best is None or e["V"] > best[2]["V"]:
                best = (yy, T, e)
    yy, T, e = best
    print(f"Cartel optimum on the grid: y = {yy}, T = {T}, pbar = {e['pbar']:.4f}, "
          f"Pr(reversion) = {e['G']:.4f}, (1-d)V = {e['norm']:.5f}")


INF = 10 ** 6      # d ** INF == 0.0: reversion forever
Ts = [2, 4, 8, 16, 50, INF]
ys = np.round(np.arange(0.2000, 0.3301, 0.0025), 4)
run(0.95, 0.1, Ts, ys)
for T in range(2, 41):
    e = equilibrium(0.25, T, 0.95, 0.1)
    if e is not None and e["glob"]:
        print(f"Shortest T sustaining the monopoly share y = 0.25: T = {T} "
              f"(pbar = {e['pbar']:.4f}, Pr(reversion) = {e['G']:.4f}, (1-d)V = {e['norm']:.5f})")
        break
