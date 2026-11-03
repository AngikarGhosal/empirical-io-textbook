"""Deterministic four-period demand cycle (Section 12.4.3, Haltiwanger-Harrington logic).

Demand D_t(p) = A_t (1 - p), c = 0, with A_t cycling through (4, 8, 12, 8):
low (L), rising middle (M_up), high (H), falling middle (M_dn). Monopoly profit
pi^m_t = A_t / 4 cycles through (1, 2, 3, 2). Bertrand reversion forever.

1. Critical discount factor at which monopoly pricing just satisfies the date-t
   incentive constraint (N-1) pi^m_t <= sum_{s>=1} delta^s pi^m_{t+s}, holding all
   other dates at monopoly.
2. Most collusive profits: the greatest solution of
   v_t = min{ pi^m_t, (1/(N-1)) sum_{s>=1} delta^s v_{t+s} }, found by iterating
   from v = pi^m (a decreasing sequence), and the implied prices.
No figure. Prints every number quoted in the text.
"""
import numpy as np
from scipy.optimize import brentq

A = np.array([4.0, 8.0, 12.0, 8.0])
names = ["L", "M_up", "H", "M_dn"]
pim = A / 4.0


def future(v, t, delta, S=2000):
    return sum(delta ** s * v[(t + s) % 4] for s in range(1, S))


def future_closed(v, t, delta):
    # periodic sum: sum_{s=1}^{4} delta^s v_{t+s} / (1 - delta^4)
    return sum(delta ** s * v[(t + s) % 4] for s in range(1, 5)) / (1 - delta ** 4)


for N in (2, 3):
    print(f"N = {N}: critical delta for monopoly at each date (others at monopoly)")
    for t in range(4):
        f = lambda d: future_closed(pim, t, d) - (N - 1) * pim[t]
        print(f"   {names[t]:5s} {brentq(f, 1e-6, 0.999999):.4f}")

for N, delta in ((2, 0.52), (2, 0.58)):
    v = pim.copy()
    for _ in range(100000):
        vn = np.array([min(pim[t], future_closed(v, t, delta) / (N - 1)) for t in range(4)])
        if np.max(np.abs(vn - v)) < 1e-14:
            v = vn
            break
        v = vn
    p = 0.5 * (1 - np.sqrt(1 - 4 * v / A))
    print(f"N = {N}, delta = {delta}: most collusive profits {np.round(v, 4)}, "
          f"prices {np.round(p, 4)} (monopoly price 0.5)")
    print(f"   difference of continuation values at M_up and M_dn (monopoly profits): "
          f"{future_closed(pim, 1, delta) - future_closed(pim, 3, delta):.4f}; "
          f"formula delta (pi_H - pi_L) / (1 + delta^2) = {delta * (3 - 1) / (1 + delta ** 2):.4f}")
