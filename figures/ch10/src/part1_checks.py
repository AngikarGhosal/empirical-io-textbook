"""Numerical checks for Chapter 10, Part 1 (Sections 10.2-10.4).

Every number that Part 1 computes itself (rather than quoting from a paper)
is produced here. Run with:  python part1_checks.py
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

np.set_printoptions(precision=6, suppress=True)


def header(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)


# ---------------------------------------------------------------------------
# 1. CES variety index (Feenstra 1994) for Apple-Cinnamon Cheerios.
#    Inputs: eta = 2.13 (Hausman 1997, p. 231); ACC family-segment share
#    6.19 percent (Hausman 1997, Table 5.2).
# ---------------------------------------------------------------------------
header("1. CES variety index with Hausman's eta and ACC segment share")
eta, w_n = 2.13, 0.0619
ratio = (1.0 - w_n) ** (1.0 / (eta - 1.0))     # c_1 / c_0
print(f"c1/c0                      = {ratio:.5f}")
print(f"fall in segment COL index  = {100 * (1 - ratio):.3f} percent")
print(f"CV / segment expenditure   = {100 * (1 / ratio - 1):.3f} percent")
print(f"EV / segment expenditure   = {100 * (1 - ratio):.3f} percent")
# first-order approximation w_n / (eta - 1)
print(f"first-order w/(eta-1)      = {100 * w_n / (eta - 1):.3f} percent")

# Sato-Vartia identity check with random CES data
rng = np.random.default_rng(10)
J = 6
a = rng.uniform(0.5, 2.0, J)
p0 = rng.uniform(0.5, 2.0, J)
p1 = p0 * rng.uniform(0.8, 1.3, J)
et = 3.0
c = lambda p, idx: (np.sum(a[idx] * p[idx] ** (1 - et))) ** (1 / (1 - et))
idx_all = np.arange(J)
s0 = a * p0 ** (1 - et); s0 /= s0.sum()
s1 = a * p1 ** (1 - et); s1 /= s1.sum()
L = np.where(np.isclose(s1, s0), s0, (s1 - s0) / (np.log(s1) - np.log(s0)))
om = L / L.sum()
sv = np.prod((p1 / p0) ** om)
print(f"Sato-Vartia check: c ratio = {c(p1, idx_all) / c(p0, idx_all):.10f}, "
      f"SV index = {sv:.10f}")
# Feenstra check: good 5 new in period 1, good 0 disappears in period 1
I0 = np.array([0, 1, 2, 3, 4]); I1 = np.array([1, 2, 3, 4, 5]); I = np.array([1, 2, 3, 4])
sh = lambda p, idx: a[idx] * p[idx] ** (1 - et) / np.sum(a[idx] * p[idx] ** (1 - et))
sI0 = sh(p0, I); sI1 = sh(p1, I)
LI = np.where(np.isclose(sI1, sI0), sI0, (sI1 - sI0) / (np.log(sI1) - np.log(sI0)))
svI = np.prod((p1[I] / p0[I]) ** (LI / LI.sum()))
lam0 = np.sum(a[I] * p0[I] ** (1 - et)) / np.sum(a[I0] * p0[I0] ** (1 - et))
lam1 = np.sum(a[I] * p1[I] ** (1 - et)) / np.sum(a[I1] * p1[I1] ** (1 - et))
print(f"Feenstra check: exact = {c(p1, I1) / c(p0, I0):.10f}, "
      f"formula = {svI * (lam1 / lam0) ** (1 / (et - 1)):.10f}")

# ---------------------------------------------------------------------------
# 2. Three canonical shapes through (p, q) = (1, 1) with elasticity 2.
# ---------------------------------------------------------------------------
header("2. Linear, AIDS (beta = 0) and constant-elasticity demand through (1,1)")
eps = 2.0
lin = lambda t: max(0.0, 1 - eps * (t - 1))
aids = lambda t: max(0.0, (1 - np.log(t)) / t)       # w = a(1 - ln p), y a = 1
ce = lambda t: t ** (-eps)
print("slopes at p=1:", (lin(1.0001) - lin(0.9999)) / 2e-4,
      (aids(1.0001) - aids(0.9999)) / 2e-4, (ce(1.0001) - ce(0.9999)) / 2e-4)
print("chokes: linear", 1 + 1 / eps, " AIDS", np.exp(1 / (eps - 1)), " CE inf")
print("CS linear =", quad(lin, 1, 1 + 1 / eps)[0], " formula pq/(2 eps) =", 1 / (2 * eps))
print("CS AIDS   =", quad(aids, 1, np.e)[0], " formula pq/(2(eps-1)) =", 1 / (2 * (eps - 1)))
print("CS CE     =", quad(ce, 1, np.inf)[0], " formula pq/(eps-1) =", 1 / (eps - 1))
# convexity of AIDS demand (beta=0): q'' = y p^-3 (2 w - 3 gamma)
pp = np.linspace(1, np.e - 1e-6, 5)
print("AIDS q'' on [1,e]:", [(aids(t + 1e-4) - 2 * aids(t) + aids(t - 1e-4)) / 1e-8 for t in pp])

# ---------------------------------------------------------------------------
# 3. Example: linear demand with income effects (Hausman 1981).
#    q = a + b p + c y, observed p = 1, y = 100, q = 2, b = -2, c = 0.02.
# ---------------------------------------------------------------------------
header("3. Linear demand with income effects: exact CV versus Marshallian CS")
b, cc, y, p1_, q1 = -2.0, 0.02, 100.0, 1.0, 2.0
aa = q1 - b * p1_ - cc * y
print(f"a = {aa}")
e_fun = lambda p, u: u * np.exp(cc * p) - (b * p + b / cc + aa) / cc
u1 = np.exp(-cc * p1_) * (y + (b * p1_ + b / cc + aa) / cc)
print(f"u1 = {u1:.4f}; check e(p1,u1) = {e_fun(p1_, u1):.6f}")
hick = lambda p, u: cc * u * np.exp(cc * p) - b / cc
pV = np.log(b / (cc ** 2 * u1)) / cc
print(f"Hicksian virtual price pV = {pV:.6f}; h(pV) = {hick(pV, u1):.2e}")
D = -np.log(1 + cc * q1 / b) / cc
print(f"pV - p1 from closed form  = {D:.6f}")
pM = -(aa + cc * y) / b
print(f"Marshallian choke pM      = {pM:.6f}")
CV = e_fun(pV, u1) - y
CV_closed = (abs(b) * D - q1) / cc
CV_int = quad(lambda t: hick(t, u1), p1_, pV)[0]
CS = quad(lambda t: aa + b * t + cc * y, p1_, pM)[0]
print(f"CV = {CV:.6f} (closed form {CV_closed:.6f}, integral {CV_int:.6f})")
print(f"Marshallian CS = {CS:.6f}; gap (CV-CS)/CS = {100 * (CV - CS) / CS:.4f} percent")
print(f"series approx q^2/(2|b|) + c q^3/(3 b^2) = {q1**2/(2*abs(b)) + cc*q1**3/(3*b*b):.6f}")
eta_y = cc * y / q1
print(f"income elasticity at observed point = {eta_y}; Willig approx eta CS/(2y) = "
      f"{100 * eta_y * CS / (2 * y):.4f} percent")
# EV (reference u0, rationed utility): y = e(pV(u0), u0) with zero demand
# u0 solves: income y reaches utility u0 when good unavailable, i.e. e(pV(u0),u0)=y
def ev_resid(u):
    pv = np.log(b / (cc ** 2 * u)) / cc
    return e_fun(pv, u) - y
u0 = brentq(ev_resid, u1 * 1.5, u1 * 0.999)
EV = y - e_fun(p1_, u0)
print(f"u0 = {u0:.4f}; EV = {EV:.6f}  (EV <= CS <= CV: {EV <= CS <= CV})")

# ---------------------------------------------------------------------------
# 4. Linear offset: multiproduct monopolist with symmetric B.
# ---------------------------------------------------------------------------
header("4. Linear demand: incumbent prices unchanged by a new product")
for trial in range(3):
    Jn = 4
    M = rng.normal(size=(Jn, Jn))
    B = M @ M.T + Jn * np.eye(Jn)            # symmetric positive definite
    a_vec = rng.uniform(5, 10, Jn) + B @ np.ones(Jn) * 2
    cvec = rng.uniform(0.5, 1.0, Jn)
    p_after = 0.5 * (np.linalg.solve(B, a_vec) + cvec)
    O, n = np.arange(Jn - 1), Jn - 1
    Bt = B[np.ix_(O, O)] - np.outer(B[O, n], B[n, O]) / B[n, n]
    at = a_vec[O] - B[O, n] * a_vec[n] / B[n, n]
    p_before = 0.5 * (np.linalg.solve(Bt, at) + cvec[O])
    q_after = a_vec - B @ p_after
    print(f"trial {trial}: max |p_before - p_after| = "
          f"{np.max(np.abs(p_before - p_after[O])):.2e}; q_n after = {q_after[n]:.3f}")
# two-product closed form
bb, dd, a1, a2, c1 = 2.0, 1.0, 10.0, 8.0, 1.0
print("2x2 closed form p1 =", 0.5 * ((bb * a1 + dd * a2) / (bb ** 2 - dd ** 2) + c1))

# ---------------------------------------------------------------------------
# 5. Logit: symmetric single-product equilibrium markups by number of products.
#    alpha = 1, v - alpha c = 2 (so e^{v - alpha c} = e^2).
# ---------------------------------------------------------------------------
header("5. Symmetric single-product logit Bertrand equilibrium")
alpha, vc = 1.0, 2.0
rows = []
for Jn in [1, 2, 3, 5, 10, 20, 50]:
    g = lambda m: alpha * m - 1 - np.exp(vc - alpha * m) / (1 + (Jn - 1) * np.exp(vc - alpha * m))
    # FOC alpha m (1 - s) = 1 with s = x/(1 + J x): 1 - s = (1 + (J-1) x)/(1 + J x)
    g2 = lambda m: alpha * m * (1 + (Jn - 1) * np.exp(vc - alpha * m)) / (1 + Jn * np.exp(vc - alpha * m)) - 1
    m = brentq(g2, 1e-9, 50)
    x = np.exp(vc - alpha * m)
    s = x / (1 + Jn * x)
    rows.append((Jn, m, s, Jn * s))
    print(f"J = {Jn:3d}: markup = {m:.4f}, share each = {s:.4f}, total inside = {Jn*s:.4f}, "
          f"check 1/(alpha(1-s)) = {1/(alpha*(1-s)):.4f}")

# Asymmetric entry: incumbents' equilibrium prices fall (fixed-point iteration)
header("5b. Asymmetric logit: entry lowers every incumbent price")
v = np.array([2.0, 1.5, 1.0]); cst = np.array([1.0, 0.8, 0.5])
def br(p, v, cst, j):
    A = 1 + np.sum(np.exp(v - alpha * p)) - np.exp(v[j] - alpha * p[j])
    f = lambda m: alpha * m - 1 - np.exp(v[j] - alpha * (cst[j] + m)) / A
    return cst[j] + brentq(f, 1e-12, 100)
def eq(v, cst):
    p = cst + 2.0
    for _ in range(500):
        p_new = np.array([br(p, v, cst, j) for j in range(len(v))])
        if np.max(np.abs(p_new - p)) < 1e-13:
            break
        p = p_new
    return p
p_pre = eq(v, cst)
p_post = eq(np.append(v, 1.2), np.append(cst, 0.7))
print("pre-entry prices :", p_pre)
print("post-entry prices:", p_post)

# Multiproduct logit firm: uniform markup 1/(alpha(1-S_f)); adding a product raises it
header("5c. Multiproduct logit monopolist: adding a product raises markups")
for A in [np.exp(2.0), np.exp(2.0) + np.exp(1.5)]:
    m = brentq(lambda m: alpha * m - 1 - A * np.exp(-alpha * m), 1e-9, 50)
    s0 = 1 / (1 + A * np.exp(-alpha * m))
    print(f"A = {A:.4f}: markup = {m:.4f}, outside share = {s0:.4f}, 1/(alpha s0) = {1/(alpha*s0):.4f}")

# ---------------------------------------------------------------------------
# 6. Linear residual demand: CS at the monopoly price is half the profit.
# ---------------------------------------------------------------------------
header("6. Linear demand: consumer surplus = profit / 2 at the monopoly price")
bq, pV_, c_ = 3.0, 10.0, 2.0
pstar = (pV_ + c_) / 2; qstar = bq * (pV_ - pstar)
print("profit =", (pstar - c_) * qstar, " CS =", 0.5 * (pV_ - pstar) * qstar)
