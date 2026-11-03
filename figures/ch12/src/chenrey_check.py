"""Checks of the Chen-Rey (2013) leniency algebra used in Section 12.8.

Notation of the book (Chen-Rey notation in brackets):
  B      per-firm collusive stake (B)
  F      fine (F)
  varrho probability that an investigation uncovers the cartel (rho)
  ell    leniency (amnesty) rate of the first informant (q)
  iota   probability that an investigation is launched (alpha)
  vsig   probability that a launched investigation succeeds (p)
  delta  discount factor (delta)

The script verifies symbolically:
  * the thresholds (1)-(3), the optimum (4)-(5), the N-firm extension;
  * the first-informant remark and the "report once" result (Prop. 2),
    including the stage-0 participation condition V_O >= 0;
  * the scrutiny formula for ell_hat(T) (Section 3.4, Prop. 3);
  * the open-investigation thresholds, eq. (9)-(14) and Prop. 6;
and prints every number quoted in the text (figure values, the
Hinloopen-Soetevent case, the region boundaries for delta = 0.8).
Output: printed to screen and to chenrey_check_output.txt.
"""
import os
import sympy as sp
import numpy as np

out_lines = []


def say(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    out_lines.append(s)


B, F, rho, ell, d = sp.symbols("B F varrho ell delta", positive=True)
N = sp.symbols("N", positive=True)

# ---------------------------------------------------------------- (1)-(3)
VN = (B - rho * F) / (1 - d)
B_low = sp.solve(sp.Eq(VN, 2 * B - rho * F), B)[0]
say("(1) B_low =", sp.simplify(B_low))
BNr = sp.solve(sp.Eq(VN, 2 * B - (1 - ell) * F), B)[0]
say("(2) B_N^r =", sp.factor(sp.simplify(BNr)))
VR = (B - (1 - ell / 2) * F) / (1 - d)
BR = sp.solve(sp.Eq(VR, 2 * B - (1 - ell / 2) * F), B)[0]
say("(3) B_R =", sp.factor(sp.simplify(BR)))
say("B_N^r - B_low =", sp.factor(sp.simplify(BNr - B_low)))

ell_hat = sp.solve(sp.Eq(BNr, BR), ell)[0]
say("(4) ell_hat =", sp.simplify(ell_hat), " check (1-rho)/(1-delta/2):",
    sp.simplify(ell_hat - (1 - rho) / (1 - d / 2)))
B_hat = sp.simplify(BR.subs(ell, ell_hat))
say("(5) B_hat =", sp.factor(B_hat), " check:",
    sp.simplify(B_hat - d * (1 - d + rho) * F / ((2 * d - 1) * (2 - d))))
say("B_hat - B_low =", sp.factor(sp.simplify(B_hat - B_low)))
VRhat = sp.simplify(VR.subs(ell, ell_hat))
say("V_N - V_R(ell_hat) =", sp.factor(sp.simplify(VN - VRhat)))

# ---------------------------------------------------------------- N firms
VNN = (B - rho * F) / (1 - d)
B_lowN = sp.solve(sp.Eq(VNN, N * B - rho * F), B)[0]
BNrN = sp.solve(sp.Eq(VNN, N * B - (1 - ell) * F), B)[0]
VRN = (B - (1 - ell / N) * F) / (1 - d)
BRN = sp.solve(sp.Eq(VRN, N * B - (1 - ell / N) * F), B)[0]
ell_hatN = sp.solve(sp.Eq(BNrN, BRN), ell)[0]
say("N firms: B_low =", sp.factor(sp.simplify(B_lowN)))
say("N firms: B_N^r =", sp.factor(sp.simplify(BNrN)))
say("N firms: B_R =", sp.factor(sp.simplify(BRN)))
say("N firms: ell_hat =", sp.factor(sp.simplify(ell_hatN)))
B_hatN = sp.simplify(BRN.subs(ell, ell_hatN))
say("N firms: B_hat =", sp.factor(B_hatN))

# ---------------------------------------------------------------- first informant
BRprime = sp.solve(sp.Eq((B - (1 - ell) * F) / (1 - d), 2 * B - (1 - ell) * F), B)[0]
say("both informants: B_R' =", sp.factor(sp.simplify(BRprime)),
    " at ell = 1 - rho:", sp.simplify(BRprime.subs(ell, 1 - rho)))

# ---------------------------------------------------------------- report once
VO = B - (1 - ell / 2) * F + d * VN
BO = sp.solve(sp.Eq(VO, 2 * B - (1 - ell / 2) * F), B)[0]
say("report once: threshold =", sp.factor(sp.simplify(BO)))
# participation at stage 0: refusing yields zero, so V_O >= 0 is needed
BOpart = sp.solve(sp.Eq(VO, 0), B)[0]
BOpart_text = ((1 - d) * (1 - ell / 2) + d * rho) * F
say("report once: participation V_O >= 0 iff B >=", sp.factor(sp.simplify(BOpart)),
    " check vs text:", sp.simplify(BOpart - BOpart_text))
ellO = sp.solve(sp.Eq(BOpart, B_low), ell)[0]
ellO_text = 2 * (1 - 2 * d * rho / (2 * d - 1))
say("report once: ell_O =", sp.simplify(ellO), " check vs text:", sp.simplify(ellO - ellO_text))
say("report once: ell_O - (1 - rho) =", sp.factor(sp.simplify(ellO - (1 - rho))),
    "(nonpositive iff varrho >= (2delta-1)/(2delta+1))")
for (dv, rv, lv, bv) in [(0.8, 0.3, 0.2, None), (0.9, 0.1, 1.2, 0.12)]:
    vv = {d: dv, rho: rv, F: 1, ell: lv}
    Blv = float(B_low.subs(vv))
    Bv = Blv if bv is None else bv
    say(f"  delta = {dv}, varrho = {rv}, ell = {lv}: B_low = {Blv:.4f}, ell_low = {1 - rv:.2f},"
        f" ell_O = {float(ellO.subs(vv)):.4f}, (2delta-1)/(2delta+1) = {(2 * dv - 1) / (2 * dv + 1):.4f}")
    say(f"    at B = {Bv:.4f}: V_N = {float(VN.subs(vv).subs(B, Bv)):.4f},"
        f" V_O = {float(VO.subs(vv).subs(B, Bv)):.4f}, B_N^r = {float(BNr.subs(vv)):.4f},"
        f" B_O = {float(BOpart.subs(vv)):.4f}")

# ---------------------------------------------------------------- scrutiny
beta, gam = sp.symbols("beta gamma", positive=True)
BNrT = (rho - (1 - gam) * (1 - ell)) * F / (2 * gam - 1)
BRT = beta * (1 - ell / 2) * F / (2 * beta - 1)
ellT = sp.solve(sp.Eq(BNrT, BRT), ell)[0]
ellT_cr = 2 * (1 - rho) * (beta + d - 1) / ((2 * beta - 1) + (2 * gam - 1) * (1 - beta))
diff = sp.simplify((ellT - ellT_cr).subs(gam, (1 - rho) * d + rho * beta))
say("scrutiny: ell_hat(T) (our solve) minus CR formula, with gamma substituted:", diff)
# check the value recursions behind gamma
V = sp.symbols("V")
VNrec = sp.solve(sp.Eq(V, B - rho * F + (1 - rho) * d * V + rho * beta * V), V)[0]
say("scrutiny: V_N(T) =", sp.simplify(VNrec))
dval, rval = 0.9, 0.2
say("scrutiny numbers (delta = 0.9, rho = 0.2, F = 1):")
for T in range(0, 6):
    b = dval ** (T + 1)
    g = (1 - rval) * dval + rval * b
    if b <= 0.5:
        say(f"  T = {T}: beta = {b:.4f} <= 1/2, collude-and-report not sustainable")
        continue
    lT = 2 * (1 - rval) * (b + dval - 1) / ((2 * b - 1) + (2 * g - 1) * (1 - b))
    BT = b * (1 - lT / 2) / (2 * b - 1)
    say(f"  T = {T}: beta = {b:.4f}, gamma = {g:.4f}, ell_hat = {lT:.4f}, B_hat = {BT:.4f}")

# ---------------------------------------------------------------- open investigations
al, p = sp.symbols("iota varsigma", positive=True)  # alpha, p in CR
qb, qa = sp.symbols("ell_b ell_a")
VNo = (B - al * p * F) / (1 - d)
BNi = sp.solve(sp.Eq(VNo, 2 * B - al * (1 - qa) * F), B)[0]
say("open: B_N^i =", sp.factor(sp.simplify(BNi)))
BNc = sp.solve(sp.Eq(-p * F + d * VNo, -(1 - qa) * F), B)[0]
say("open: B_N^c =", sp.factor(sp.simplify(BNc)))
VI = (B - al * (1 - qa / 2) * F) / (1 - d)
BIr = sp.solve(sp.Eq(VI, 2 * B - (1 - qb) * F), B)[0]
BIi = sp.solve(sp.Eq(VI, 2 * B - al * (1 - qa / 2) * F), B)[0]
say("open: B_I^r =", sp.factor(sp.simplify(BIr)))
say("open: B_I^i =", sp.factor(sp.simplify(BIi)))
BRo = d * (1 - qb / 2) * F / (2 * d - 1)
BNro = (al * p - (1 - d) * (1 - qb)) * F / (2 * d - 1)

# policy 1: B_N^i(qa) = B_I^r(qb, qa) = B_R(qb)
sol1 = sp.solve([sp.Eq(BNi, BIr), sp.Eq(BIr, BRo)], [qb, qa], dict=True)[0]
qb1 = sp.simplify(sol1[qb])
qa1 = sp.simplify(sol1[qa])
qb1_cr = 2 * ((1 - al) * (2 - d) + al * (1 - p)) / (2 * (1 - d) ** 2 + (2 - d))
qa1_cr = (2 / al) * (d * (1 - d) * (1 - al) + al * (1 - p) * (2 - d)) / (2 * (1 - d) ** 2 + 2 - d)
say("open: (10) q_b^1 check:", sp.simplify(qb1 - qb1_cr), " q_a^1 check:", sp.simplify(qa1 - qa1_cr))
B1 = sp.simplify(BRo.subs(qb, qb1))
B1_cr = (2 * (1 - d) ** 2 + al * (1 - d + p)) / (2 * (1 - d) ** 2 + 2 - d) * d * F / (2 * d - 1)
say("open: (11) B_1 check:", sp.simplify(B1 - B1_cr))
# (9) p_tilde_1
ptil1 = (al * (2 - d) - d) / (2 * al * (1 - d))
say("open: (9) p1 =", ptil1)
# B_1 >= B_hat(alpha p) iff p <= p1 ?
Bhat_ap = (1 - d + al * p) / (2 - d) * d * F / (2 * d - 1)
expr = sp.factor(sp.simplify(B1 - Bhat_ap))
say("open: B_1 - B_hat(alpha p) =", expr)

# policy 2: B_N^c(qa) = B_I^r(qb, qa) = B_R(qb)
sol2 = sp.solve([sp.Eq(BNc, BIr), sp.Eq(BIr, BRo)], [qb, qa], dict=True)[0]
qb2 = sp.simplify(sol2[qb])
qa2 = sp.simplify(sol2[qa])
den2 = 2 * (2 * d - 1) * (1 - d) * (2 - d) + al * d ** 2
qb2_cr = 2 * (2 * (1 - d) * (2 * d - 1) + al * (3 * d ** 2 - 3 * d + 1)
              - (2 * d - 1) * (1 - d + al * d) * al * p) / den2
qa2_cr = 2 * ((1 - d) * (5 * d - d ** 2 - 2) + al * d ** 2
              - (2 * d - 1) * (2 - d) * (1 - d + al * d) * p) / den2
say("open: (13) q_b^2 check:", sp.simplify(qb2 - qb2_cr), " q_a^2 check:", sp.simplify(qa2 - qa2_cr))
B2 = sp.simplify(BRo.subs(qb, qb2))
B2_cr = (2 * (1 - d) ** 2 + al * (1 - d) + (1 - d + al * d) * al * p) / den2 * d * F
say("open: (14) B_2 check:", sp.simplify(B2 - B2_cr))
ptil2 = (1 - d) * (3 * d - 2) / ((2 * d - 1) * (2 - d) + 2 * al * d * (1 - d))
expr2 = sp.factor(sp.simplify(B2 - Bhat_ap))
say("open: B_2 - B_hat(alpha p) =", expr2)
say("open: (12) p2 =", ptil2)

# p_tilde (B1 >= B2 iff p >= p_tilde)
ptil = (2 * (1 - d) * (2 * d - 1) + al * (2 * d ** 2 - 1) - al ** 2 * d) / (
    al * (2 * d - 1) * (3 - 2 * d) + 4 * al ** 2 * (1 - d) ** 2)
expr3 = sp.factor(sp.simplify(B1 - B2))
say("open: B_1 - B_2 =", expr3)
say("open: alpha_1 root of numerator of p_tilde:",
    sp.solve(sp.Eq(2 * (1 - d) * (2 * d - 1) + al * (2 * d ** 2 - 1) - al ** 2 * d, 0), al))

# Prop. 6
ph1 = (1 - d) * (al ** 2 + d * (1 - al)) / (al * (2 - d - al))
ph2 = (1 - d) / (2 * d - 1) * (d * (1 - d) + al * (2 * d - 1)) / ((1 - d + al * d) * (2 - d - al))
say("Prop. 6: q_a^1 - q_b^1 numerator sign check at p = p_hat_1:",
    sp.simplify((qa1 - qb1).subs(p, ph1)))
say("Prop. 6: q_a^2 - q_b^2 at p = p_hat_2:", sp.simplify((qa2 - qb2).subs(p, ph2)))

# ---------------------------------------------------------------- numbers
def num(expr, **kw):
    return float(expr.subs({sp.Symbol(k, positive=True): v for k, v in kw.items()}))


say("\nFigure fig:12-leniency (delta = 0.8, varrho = 0.3, F = 1):")
vals = {d: 0.8, rho: 0.3, F: 1}
say("  B_low =", float(B_low.subs(vals)))
say("  ell_low = 1 - rho =", 0.7, " ell_bar = 2(1 - rho) =", 1.4)
say("  ell_hat =", float(ell_hat.subs(vals)), " B_hat =", float(B_hat.subs(vals)))
say("  B_N^r(0) =", float(BNr.subs(vals).subs(ell, 0)), " B_N^r(2) =", float(BNr.subs(vals).subs(ell, 2)))
say("  B_R(0) =", float(BR.subs(vals).subs(ell, 0)), " B_R(2) =", float(BR.subs(vals).subs(ell, 2)))
vals2 = {d: 0.8, rho: 0.4, F: 1}
say("Panel (b) varrho' = 0.4: B_low =", float(B_low.subs(vals2)), " ell_low =", 0.6,
    " ell_hat =", float(ell_hat.subs(vals2)), " B_hat =", float(B_hat.subs(vals2)),
    " ell_bar =", 1.2)
say("Hinloopen-Soetevent: B_R(1.8) =", float(BR.subs(vals2).subs(ell, 1.8)),
    " B_R(1) =", float(BR.subs(vals2).subs(ell, 1)))
say("  V_N - V_R(ell_hat) per unit F at (0.8, 0.3):", float(((1 - rho) / (2 - d)).subs(vals)))
say("N-firm example (delta = 0.9, varrho = 0.3):")
for n in [2, 3, 4]:
    v = {d: 0.9, rho: 0.3, F: 1, N: n}
    say(f"  N = {n}: ell_hat = {float(ell_hatN.subs(v)):.4f}, B_hat = {float(B_hatN.subs(v)):.4f},"
        f" B_low = {float(B_lowN.subs(v)):.4f}, ell_bar = {n * 0.7:.2f}")

say("\nRegion boundaries for delta = 0.8:")
dv = 0.8
iota_low = dv / (2 - dv)
alpha1 = (2 * dv - 1) / dv
say("  iota_low = delta/(2-delta) =", iota_low, " iota_1 = (2delta-1)/delta =", alpha1)
f1 = sp.lambdify(al, ptil1.subs(d, dv))
f2 = sp.lambdify(al, ptil2.subs(d, dv))
f0 = sp.lambdify(al, ptil.subs(d, dv))
from scipy.optimize import brentq
a0 = brentq(lambda a: f1(a) - f2(a), iota_low + 1e-9, 0.99)
say("  common point iota_0 =", round(a0, 4), " varsigma_0 =", round(f1(a0), 4),
    " p_tilde(iota_0) =", round(f0(a0), 4))
say("  p2(0) =", round(f2(1e-12), 4), " p1(1) =", round(f1(1.0), 4))
for dv2 in [0.7, 0.9]:
    g2 = sp.lambdify(al, ptil2.subs(d, dv2))
    g1 = sp.lambdify(al, ptil1.subs(d, dv2))
    il = dv2 / (2 - dv2)
    a0b = brentq(lambda a: g1(a) - g2(a), il + 1e-9, 0.999)
    say(f"  delta = {dv2}: p2(0) = {g2(1e-12):.4f}, iota_low = {il:.4f}, iota_0 = {a0b:.4f}, varsigma_0 = {g1(a0b):.4f}")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "chenrey_check_output.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out_lines) + "\n")
