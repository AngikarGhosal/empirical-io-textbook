"""Chapter 14, Part 2: symbolic and numerical checks for every computed number.

Run:  python part2_numbers.py
Prints each check; writes part2_numbers.txt next to this script and
selfselection_curve.txt (coordinates used by the TikZ redraw of MW Figure 1).
"""
import os
import numpy as np
import sympy as sp
from scipy import integrate, optimize

out_lines = []


def report(label, value):
    line = f"{label}: {value}"
    print(line)
    out_lines.append(line)


v, t, z, r, b = sp.symbols("v t z r b", positive=True)
n = sp.symbols("n", integer=True, positive=True)
lam, gam, a, k = sp.symbols("lambda gamma a k", positive=True)


def rs_bid(F, lower, nn):
    """Riley-Samuelson bid v - int_lower^v F^(n-1) / F(v)^(n-1)."""
    num = sp.integrate(F.subs(v, t) ** (nn - 1), (t, lower, v))
    return sp.simplify(v - num / F ** (nn - 1))


# ------------------------------------------------------------- 14.5 examples
for nn in [2, 3, 5, 10]:
    report(f"uniform bid n={nn}", rs_bid(v, 0, nn))
# general n uniform: check (n-1)/n v solves the ODE  beta' = (n-1)(v-beta)/v
beta = (n - 1) / n * v
report("uniform ODE residual", sp.simplify(sp.diff(beta, v) - (n - 1) * (v - beta) / v))

# uniform with reserve
for nn in [2, 3]:
    report(f"uniform reserve bid n={nn}", sp.simplify(rs_bid(v, r, nn)))
bid_r = rs_bid(v, sp.Rational(1, 2), 2)
report("uniform n=2 r=1/2 bid", sp.simplify(bid_r))
report("  beta(1/2), beta(1)", (bid_r.subs(v, sp.Rational(1, 2)), bid_r.subs(v, 1)))
rev_fpa_r = sp.integrate(bid_r * 2 * v, (v, sp.Rational(1, 2), 1))
report("FPA revenue n=2 r=1/2", rev_fpa_r)

# power distribution
for nn in [2, 3]:
    report(f"power bid n={nn}", sp.simplify(rs_bid(v ** gam, 0, nn)))

# exponential, n = 2
Fexp = 1 - sp.exp(-lam * v)
bexp = rs_bid(Fexp, 0, 2)
report("exponential bid n=2", sp.simplify(bexp))
target = 1 / lam - v * sp.exp(-lam * v) / (1 - sp.exp(-lam * v))
report("  difference from closed form", sp.simplify(bexp - target))
report("  limit v->oo", sp.limit(target, v, sp.oo))
report("  limit v->0", sp.limit(target, v, 0))
rev_exp = sp.integrate(sp.simplify(target * 2 * Fexp * sp.diff(Fexp, v)), (v, 0, sp.oo))
report("  FPA revenue (should be 1/(2 lambda))", sp.simplify(rev_exp))

# uniform revenue and surplus, general n
EVmax = sp.integrate(v * n * v ** (n - 1), (v, 0, 1))
report("E[V_(n:n)]", sp.simplify(EVmax))
report("FPA revenue uniform", sp.simplify((n - 1) / n * EVmax))
report("bidder surplus per bidder", sp.simplify(sp.integrate(v ** n / n, (v, 0, 1))))

# mean value-bid relation, uniform
for nn in [2, 3, 5]:
    EB = sp.Rational(nn - 1, nn) * sp.Rational(1, 2)
    bbar = sp.Rational(nn - 1, nn)
    report(f"mean relation n={nn} (should be 1/2)", sp.Rational(nn - 2, nn - 1) * EB + bbar / (nn - 1))

# multiplicity: beta_k = (n-1)/n v - k/v^(n-1)
for nn in [2, 3, 4]:
    bk = sp.Rational(nn - 1, nn) * v - k / v ** (nn - 1)
    resid = sp.simplify(sp.diff(bk, v) - (nn - 1) * (v - bk) / v)
    payoff = sp.simplify((v - bk) * v ** (nn - 1))
    exante = sp.integrate(payoff, (v, 0, 1))
    report(f"multiplicity n={nn}: ODE residual, interim payoff, ex ante payoff", (resid, payoff, exante))
    # MW's printed slope n/(n+1) fails the ODE
    bmw = sp.Rational(nn, nn + 1) * v - k / v ** (nn - 1)
    report(f"  printed slope n/(n+1) ODE residual", sp.simplify(sp.diff(bmw, v) - (nn - 1) * (v - bmw) / v))

# CRRA: U(x) = x^a, uniform values: beta = (n-1)/(n-1+a) v
c = (n - 1) / (n - 1 + a)
bc = c * v
# FOC: U(v-b)/U'(v-b) = (v-b)/a = G/((n-1) g) = b/(n-1) when G(b)=b/c
report("CRRA FOC residual", sp.simplify((v - bc) / a - bc / (n - 1)))
rev_crra = sp.simplify(c * EVmax)
report("CRRA FPA revenue", rev_crra)
report("CRRA FPA/SPA ratio", sp.simplify(rev_crra / ((n - 1) / (n + 1))))
report("CRRA n=2 a=1/2 revenue", rev_crra.subs({n: 2, a: sp.Rational(1, 2)}))

# ------------------------------------------------------------- 14.6
# all-pay and third-price, uniform
m_v = sp.integrate(t * (n - 1) * t ** (n - 2), (t, 0, v))
report("uniform m(v)", sp.simplify(m_v))
# third-price general formula beta = v + F/((n-2) f); uniform
report("third-price uniform", sp.simplify(v + v / (n - 2)))
# check E[beta(Y2) | Y1 = v] = v for uniform third price, n=4
nn = 4
cc = sp.Rational(nn - 1, nn - 2)
EY2 = sp.integrate(cc * t * (nn - 2) * t ** (nn - 3) / v ** (nn - 2), (t, 0, v))
report("third-price n=4: E[beta(Y2)|Y1=v]", sp.simplify(EY2))

# interdependent values V = (S1+S2)/2, two bidders
s, s2, cst = sp.symbols("s s2 c", positive=True)
pay_spa = sp.integrate(((s + s2) / 2 - s2), (s2, 0, b))
report("SPA interdependent: d payoff/d b", sp.simplify(sp.diff(pay_spa, b)))
pay_fpa = sp.integrate(((s + s2) / 2 - b), (s2, 0, b / cst))
foc = sp.diff(pay_fpa, b).subs(b, cst * s)
report("FPA interdependent: c solving FOC", sp.solve(sp.simplify(foc), cst))
u1, u2 = sp.symbols("u1 u2", positive=True)
rev_spa_iv = sp.integrate(sp.integrate(sp.Min(u1, u2), (u1, 0, 1)), (u2, 0, 1))
report("SPA interdependent revenue E[min]", sp.nsimplify(rev_spa_iv))
report("FPA interdependent revenue E[max]/2", sp.Rational(2, 3) / 2)
report("interim surplus SPA", sp.simplify(sp.integrate((s - s2) / 2, (s2, 0, s))))
report("interim surplus FPA", sp.simplify(sp.integrate((s + s2) / 2 - s / 2, (s2, 0, s))))

# ------------------------------------------------------------- 14.7
# uniform reserve table
report("--- reserve table (uniform [0,1], v0 = 0, r* = 1/2) ---", "")
rows = []
for nn in range(1, 6):
    rev0 = sp.Rational(nn - 1, nn + 1)
    revstar = sp.integrate((2 * v - 1) * nn * v ** (nn - 1), (v, sp.Rational(1, 2), 1))
    nosale = sp.Rational(1, 2 ** nn)
    dwl = sp.integrate(v * nn * v ** (nn - 1), (v, 0, sp.Rational(1, 2)))
    bk = sp.Rational(nn, nn + 2)
    closed = rev0 + sp.Rational(1, (nn + 1) * 2 ** nn)
    assert sp.simplify(revstar - closed) == 0
    rows.append((nn, rev0, revstar, revstar - rev0, nosale, dwl, bk))
    report(f"n={nn}", f"rev0={rev0}={float(rev0):.4f}, rev*={revstar}={float(revstar):.4f}, "
                      f"gain={revstar-rev0}={float(revstar-rev0):.4f}, Pr(no sale)={nosale}={float(nosale):.4f}, "
                      f"DWL={dwl}={float(dwl):.4f}, English(n+1)={bk}={float(bk):.4f}")

# SPA with r = 1/2, n = 2, direct
direct = sp.Rational(1, 4) * sp.Rational(2, 3) + sp.Rational(1, 2) * sp.Rational(1, 2)
report("SPA n=2 r=1/2 direct", direct)
# entry fee equivalent
fee = sp.Rational(1, 2) * sp.Rational(1, 2)
rev_fee = 2 * sp.Rational(1, 2) * fee + sp.Rational(1, 4) * sp.Rational(2, 3)
report("entry fee 1/4 revenue n=2", rev_fee)
# E[psi(V)] = lower bound of support (uniform)
report("E[psi(V)] uniform", sp.integrate(2 * v - 1, (v, 0, 1)))

# exponential optimal reserve: psi = v - 1/lambda
psi_exp = sp.simplify(v - (1 - Fexp) / sp.diff(Fexp, v))
report("psi exponential", psi_exp)

# asymmetric optimal auction: w ~ U[0,1], s ~ U[0,2], v0 = 0
vw, vs = sp.symbols("vw vs", positive=True)
psi_w = 2 * vw - 1
psi_s = 2 * vs - 2
dens = sp.Rational(1, 2)
# regions
p_none = sp.integrate(sp.integrate(dens, (vs, 0, 1)), (vw, 0, sp.Rational(1, 2)))
p_w = sp.integrate(sp.integrate(dens, (vs, 0, vw + sp.Rational(1, 2))), (vw, sp.Rational(1, 2), 1))
p_s = 1 - p_none - p_w
p_w_lower = sp.integrate(sp.integrate(dens, (vs, vw, vw + sp.Rational(1, 2))), (vw, sp.Rational(1, 2), 1))
rev_w = sp.integrate(sp.integrate(psi_w * dens, (vs, 0, vw + sp.Rational(1, 2))), (vw, sp.Rational(1, 2), 1))
# s wins: vs in (1, 2) and vs > vw + 1/2
rev_s = sp.integrate(sp.integrate(psi_s * dens, (vw, 0, sp.Min(1, vs - sp.Rational(1, 2)))), (vs, 1, 2))
rev_s = sp.integrate(sp.integrate(psi_s * dens, (vw, 0, vs - sp.Rational(1, 2))), (vs, 1, sp.Rational(3, 2))) + \
    sp.integrate(sp.integrate(psi_s * dens, (vw, 0, 1)), (vs, sp.Rational(3, 2), 2))
rev_opt = sp.simplify(rev_w + rev_s)
report("asym optimal: Pr(no sale), Pr(w wins), Pr(s wins), Pr(w wins with lower value)",
       (p_none, p_w, p_s, p_w_lower))
report("asym optimal revenue", (rev_opt, float(rev_opt)))
# SPA, no reserve: E[min(Vw, Vs)]
spa_asym = sp.integrate((1 - t) * (1 - t / 2), (t, 0, 1))
report("asym SPA revenue E[min]", (spa_asym, float(spa_asym)))
# SPA with common reserve 1/2 (for comparison)
# total surplus under optimal: E[V_winner]
ts_w = sp.integrate(sp.integrate(vw * dens, (vs, 0, vw + sp.Rational(1, 2))), (vw, sp.Rational(1, 2), 1))
ts_s = sp.integrate(sp.integrate(vs * dens, (vw, 0, vs - sp.Rational(1, 2))), (vs, 1, sp.Rational(3, 2))) + \
    sp.integrate(sp.integrate(vs * dens, (vw, 0, 1)), (vs, sp.Rational(3, 2), 2))
ts_opt = sp.simplify(ts_w + ts_s)
ts_eff = sp.integrate(sp.integrate(sp.Max(vw, vs) * dens, (vs, 0, 2)), (vw, 0, 1))
report("asym optimal total surplus", (ts_opt, float(ts_opt)))
report("asym efficient total surplus", (sp.nsimplify(ts_eff), float(ts_eff)))

# Monte Carlo cross-check of the asymmetric numbers
rng = np.random.default_rng(0)
N = 2_000_000
W = rng.uniform(0, 1, N)
S = rng.uniform(0, 2, N)
pw, ps = 2 * W - 1, 2 * S - 2
best = np.maximum(np.maximum(pw, ps), 0)
report("MC asym optimal revenue", float(best.mean()))
report("MC asym Pr(w wins)", float(((pw > ps) & (pw > 0)).mean()))

# ------------------------------------------------------------- MW Figure 1 redraw
def e_curve(p):
    return 0.1 * p + 2.0 * p ** 2 - 4.5 * p ** 3 + 3.2 * p ** 4


def e_prime(p):
    return 0.1 + 4.0 * p - 13.5 * p ** 2 + 12.8 * p ** 3


grid = np.linspace(0, 1, 4001)
E = e_curve(grid)
report("e(p) increasing on [0,1]", bool(np.all(np.diff(E) > 0)))


def chosen_p(val):
    return grid[np.argmax(val * grid - E)]


# slope at which two points tie (jump in p*): find v with two maximizers
def gap(val):
    u = val * grid - E
    i = np.argmax(u)
    # best payoff on the other side of the concave region
    left = grid < 0.35
    return np.max(u[left]) - np.max(u[~left])


vhat = optimize.brentq(gap, 0.2, 1.5)
u = vhat * grid - E
left = grid < 0.35
pL = grid[left][np.argmax(u[left])]
pR = grid[~left][np.argmax(u[~left])]
report("MW redraw: jump slope vhat, tangency points", (round(vhat, 4), round(pL, 4), round(pR, 4)))
for val in [0.3, vhat - 0.05, vhat + 0.05, 1.2]:
    report(f"  p*({val:.3f})", round(float(chosen_p(val)), 4))
v1 = 0.30
p1 = chosen_p(v1)
report("MW redraw: v=0.30 tangency p, e, intercept", (round(p1, 4), round(e_curve(p1), 4), round(e_curve(p1) - v1 * p1, 4)))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfselection_curve.txt"), "w") as fh:
    fh.write(f"vhat={vhat:.6f} pL={pL:.6f} pR={pR:.6f} eL={e_curve(pL):.6f} eR={e_curve(pR):.6f}\n")
    fh.write(f"v1={v1} p1={p1:.6f} e1={e_curve(p1):.6f}\n")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "part2_numbers.txt"), "w") as fh:
    fh.write("\n".join(out_lines) + "\n")
