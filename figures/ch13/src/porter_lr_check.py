"""Compare Porter's (1983) statistic 2T(L1 - L0), with L = log|det B| - 0.5 log|Sigma|
(equivalently -0.5 log|Omega| in the reduced form), against the mixture LR 2(l1 - l0).
Reuses the EM code of regime_sim.py (imported without running its main part).
Used in Section 13.2.6 (Testing for regime switching). Output: porter_lr_check_output.txt.
"""
import os
import types

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
code = open(os.path.join(SRC, "regime_sim.py"), encoding="utf-8").read()
cut = code.index("# ---------------------------------------------------------------- 1. one simulated sample")
mod = types.ModuleType("rs")
mod.__dict__["__file__"] = os.path.join(SRC, "regime_sim.py")
exec(code[:cut], mod.__dict__)
rs = mod

T = rs.T
L = rs.lakes_series(T)


def porter_stat(res, Om0):
    # Porter: L = log|det B| - 0.5 log|Sigma| = -0.5 log|Omega| (per observation)
    return T * (np.log(np.linalg.det(Om0)) - np.log(np.linalg.det(res["Om"])))


rng = np.random.default_rng(7)
C, Om, a0, b0 = rs.structural_to_rf(rs.CALIBS["ML"])

# (1) switching sample, ML calibration (same construction as the chapter)
I = rs.markov_chain(T, rs.Q11, rs.Q00, rng, start=1)
Y = rs.simulate(C, Om, L, I, rng)
l0, C0, Om0 = rs.null_fit(Y, L)
res = rs.best_em(Y, L, rng)
print("switching sample: mixture LR =", round(2 * (res["ll"] - l0), 1),
      " Porter stat =", round(porter_stat(res, Om0), 1), " lambda =", round(res["lam"], 3))
lam = res["lam"]
print("   2T*h(lambda) =", round(-2 * T * (lam * np.log(lam) + (1 - lam) * np.log(1 - lam)), 1))

# (2) null samples
mix, por = [], []
for r in range(60):
    Y = rs.simulate(C, Om, L, np.zeros(T), rng)
    l0, C0, Om0 = rs.null_fit(Y, L)
    res = rs.best_em(Y, L, rng, n_random=2, maxit=1500)
    if res is None:
        continue
    mix.append(2 * (res["ll"] - l0))
    por.append(porter_stat(res, Om0))
mix, por = np.array(mix), np.array(por)
print("null: mixture LR quantiles 50/90/95:", np.round(np.quantile(mix, [.5, .9, .95]), 2))
print("null: Porter stat quantiles 50/90/95:", np.round(np.quantile(por, [.5, .9, .95]), 2))
print("null: Porter stat max:", round(por.max(), 1))
