"""Collusion in the open auction (Section 14.12): a worked example.

Participants: one logger with V_L ~ U[0, 1] and two mills with V_M1, V_M2 ~ U[0, 2], reserve r = 0.
Competitive open auction: price = second-highest of the three values.
Mill-collusive open auction (ALS Section II.E): only the highest-value mill bids, so the
price = second-highest of {max mill value, logger value}.
Price reduction = (Y - V_L)^+ with Y = min(V_M1, V_M2), the second-highest mill value.

Outputs: figures/ch14/src/collusion_example_numbers.txt
"""
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
OUT_TXT = HERE / "collusion_example_numbers.txt"

l, y, t = sp.symbols("l y t", nonnegative=True)
f_L = 1                      # density of V_L on [0, 1]
f_Y = 1 - y / 2              # density of min of two U[0, 2] on [0, 2]
reduction = sp.integrate(sp.integrate((y - l) * f_Y, (y, l, 2)) * f_L, (l, 0, 1))
p_two_mills_top = sp.integrate(sp.integrate(f_Y, (y, l, 2)) * f_L, (l, 0, 1))

# Competitive price: E[second-highest of {V_L, V_M1, V_M2}] = int_0^2 Pr(second-highest > t) dt.
F_L = sp.Piecewise((t, t <= 1), (1, True))
F_M = t / 2
# Pr(second-highest <= t) = Pr(at most one value above t)
P_le = F_L * F_M**2 + (1 - F_L) * F_M**2 + F_L * 2 * F_M * (1 - F_M)
price_comp = sp.integrate(sp.simplify(1 - P_le), (t, 0, 2))
price_coll = sp.simplify(price_comp - reduction)

# Monte Carlo check
rng = np.random.default_rng(3)
M = 2_000_000
vl = rng.uniform(0, 1, M)
vm = rng.uniform(0, 2, (M, 2))
allv = np.column_stack([vl, vm])
second = np.sort(allv, axis=1)[:, -2]
coll_set = np.column_stack([vl, vm.max(axis=1)])
second_coll = coll_set.min(axis=1)

lines = [
    "One logger U[0,1], two mills U[0,2], r = 0, open auction",
    f"Competitive expected price = {sp.nsimplify(price_comp)} = {float(price_comp):.4f}  (MC {second.mean():.4f})",
    f"Collusive expected price   = {sp.nsimplify(price_coll)} = {float(price_coll):.4f}  (MC {second_coll.mean():.4f})",
    f"Expected reduction         = {sp.nsimplify(reduction)} = {float(reduction):.4f}",
    f"Relative reduction         = {float(reduction / price_comp):.4f}",
    f"Pr(top two values are both mills) = {sp.nsimplify(p_two_mills_top)} = {float(p_two_mills_top):.4f}"
    f"  (MC {np.mean(np.minimum(vm[:,0], vm[:,1]) > vl):.4f})",
]
OUT_TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
