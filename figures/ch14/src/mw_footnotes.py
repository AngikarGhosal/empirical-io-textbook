"""Checks of two examples in the footnotes of Milgrom and Weber (1982), used in Sections 14.8-14.9.

Footnote 21 (discontinuous first-price bid). n = 2, V_i = S_i. With probability 1/2 both signals are
i.i.d. U[0,1]; with probability 1/2 both are i.i.d. U[1,2]. The equilibrium bid is s/2 on [0,1)
and (s+1)/2 on (1,2], so it jumps from 1/2 to 1 at s = 1.

Footnote 27 (an irregular reserve/entry-fee pair). n = 2, V_i = S_i. With probability 1/2 both
signals are i.i.d. U[0,2]; with probability 1/2 both are i.i.d. U[1,2]. Reserve price 0, entry fee
e = 0.32. The screening level solves s F(s|s) = e. We compute F(s|s) on both sides of s = 1 by
Bayes' rule and the set of types above the screening level who prefer not to enter.
"""
import numpy as np
from scipy import optimize

# ----------------------------------------------------------- footnote 21
def bid_fn21(s):
    # conditional on the regime (known from s except at s = 1), bid E[Y | Y < s, regime]
    return s / 2 if s < 1 else 1 + (s - 1) / 2

print("fn 21: bid just below 1 = %.4f, just above 1 = %.4f" % (bid_fn21(1 - 1e-9), bid_fn21(1 + 1e-9)))


# ----------------------------------------------------------- footnote 27
def post_regime1(s):
    """Posterior probability of regime 1 (U[0,2]) given S_1 = s."""
    d1 = 0.5 if 0 <= s <= 2 else 0.0      # density of U[0,2]
    d2 = 1.0 if 1 <= s <= 2 else 0.0      # density of U[1,2]
    return 0.5 * d1 / (0.5 * d1 + 0.5 * d2)


def F_ss(s):
    """Pr(Y_1 < s | S_1 = s)."""
    p1 = post_regime1(s)
    return p1 * (s / 2) + (1 - p1) * np.clip(s - 1, 0, 1)


e = 0.32
print("fn 27: F(s|s) just below 1 = %.4f, just above 1 = %.4f" % (F_ss(1 - 1e-9), F_ss(1 + 1e-9)))
# expected payoff from entering and bidding (reserve 0, SPA or FPA give the same entry decision
# at the margin): for a type with s, the net payoff of participating against entrants above s*
s_star = optimize.brentq(lambda s: s * F_ss(s) - e, 0.01, 0.999)
print("fn 27: screening level s* = %.4f" % s_star)
# For s > 1 the gross expected payoff of a participant facing rivals who enter iff above s* is
# E[(V_1 - price) 1{win}]; at the margin of the screening definition we use s F(s|s) as in the
# definition of the screening level (payoff of a bidder who wins only against non-entrants).
grid = np.linspace(1.0001, 2.0, 20000)
payoff = grid * np.array([F_ss(x) for x in grid])
below = grid[payoff < e]
print("fn 27: types above 1 with s F(s|s) < e: from %.4f to %.4f" % (below.min(), below.max()))
