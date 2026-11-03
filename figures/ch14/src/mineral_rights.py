"""Mineral rights example with uniform noise (Example A in Sections 14.8-14.9).

Pure common value V, signals S_i = V + eps_i with eps_i i.i.d. uniform on [-w, w], n bidders.
Diffuse-prior idealization: V uniform on a wide interval [0, K] and attention to signals far from
its ends. Claims verified here:
  (a) v(s, s) = E[V | S_1 = s, Y_1 = s] = s - w (n - 2) / n            (second-price bid)
  (b) first-price bid from the Milgrom-Weber formula equals s - w in the interior
  (c) English price E[V | S_1 = Y_1 = y1, Y_2 = y2, ..., Y_{n-1}] = (y1 + y_{n-1}) / 2
  (d) best-response checks: bidding as if one's own signal is optimal in the FPA and SPA
  (e) expected bidder rents (V minus expected price):
        FPA 2w/(n+1), SPA 2w(n-1)/(n(n+1)), English w/(n+1)
Output: figures/ch14/mw_mineral_rents.pdf
"""
from pathlib import Path

import numpy as np
from scipy import integrate

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "mw_mineral_rents.pdf"

W = 1.0       # half-width of the noise
K = 30.0      # prior support [0, K]


def cdf_eps(x):
    """CDF of S given V evaluated at S - V = x."""
    return np.clip((x + W) / (2 * W), 0.0, 1.0)


def pdf_eps(x):
    return np.where(np.abs(x) <= W, 1.0 / (2 * W), 0.0)


def post_grid(s, m=4001):
    """Grid over V in the posterior support given S_1 = s (prior uniform on [0, K])."""
    lo, hi = max(0.0, s - W), min(K, s + W)
    return np.linspace(lo, hi, m)


def v_ss(s, n):
    """E[V | S_1 = s, Y_1 = s] with the proper prior: weight f(s|V) (n-1) F(s|V)^{n-2} f(s|V)."""
    V = post_grid(s)
    wgt = pdf_eps(s - V) ** 2 * cdf_eps(s - V) ** (n - 2)
    return integrate.trapezoid(V * wgt, V) / integrate.trapezoid(wgt, V)


def ratio_ss(s, n):
    """f_{Y1}(s|s) / F_{Y1}(s|s) with the proper prior."""
    V = post_grid(s)
    post = pdf_eps(s - V)                       # prior flat, so posterior of V given S_1 = s
    F = integrate.trapezoid(post * cdf_eps(s - V) ** (n - 1), V)
    f = integrate.trapezoid(post * (n - 1) * cdf_eps(s - V) ** (n - 2) * pdf_eps(s - V), V)
    return f / F


def fpa_bid(s, n, m=3000):
    """beta(s) = v(lo,lo) L(lo|s) + int_lo^s v(a,a) dL(a|s), lo = lowest signal (-W)."""
    lo = -W + 1e-6
    a = np.linspace(lo, s, m)
    r = np.array([ratio_ss(x, n) for x in a])
    # cumulative integral of r from a to s
    cum = integrate.cumulative_trapezoid(r, a, initial=0.0)
    Lw = np.exp(-(cum[-1] - cum))
    vv = np.array([v_ss(x, n) for x in a])
    # Stieltjes integral with an atom L(lo|s) at the bottom
    return vv[0] * Lw[0] + integrate.trapezoid(vv * np.gradient(Lw, a), a)


print("(a) second-price bid v(s,s) at s = 15 (interior):")
for n in (2, 3, 5, 8):
    print("   n = %d: numeric %.4f, formula s - w(n-2)/n = %.4f" % (n, v_ss(15.0, n), 15.0 - W * (n - 2) / n))
print("    f(s|s)/F(s|s) at s = 15, n = 3: %.4f (formula n/(2w) = %.4f)" % (ratio_ss(15.0, 3), 3 / (2 * W)))

print("(b) first-price bid from the MW formula, n = 3:")
for s in (2.0, 5.0, 10.0):
    print("   s = %.1f: numeric %.4f, diffuse formula s - w = %.4f" % (s, fpa_bid(s, 3), s - W))

print("(c) English price at an interior realization:")
rng = np.random.default_rng(1982)
for n in (3, 5):
    y = np.sort(15.0 + rng.uniform(-0.6, 0.6, n - 1))[::-1]     # y1 >= ... >= y_{n-1}
    sig = np.concatenate(([y[0]], y))                             # S_1 set equal to Y_1
    V = np.linspace(12.0, 18.0, 600001)
    like = np.prod([pdf_eps(x - V) for x in sig], axis=0)
    price = integrate.trapezoid(V * like, V) / integrate.trapezoid(like, V)
    print("   n = %d: numeric %.4f, (y1 + y_{n-1})/2 = %.4f" % (n, price, (y[0] + y[-1]) / 2))


# (d) best responses under the diffuse prior: given S_1 = s, V ~ U[s-w, s+w];
# others' signals i.i.d. U[V-w, V+w]; bidder 1 bids as if his signal were x = s + d.
def payoff_fpa(d, n, m=4001):
    V = np.linspace(-W, W, m)                   # s normalized to 0
    win = cdf_eps(d - V) ** (n - 1)             # Pr(Y_1 < x | V)
    bid = d - W
    return integrate.trapezoid((V - bid) * win, V) / (2 * W)


def payoff_spa(d, n, m=2001):
    # payment v(Y1, Y1) = Y1 - w(n-2)/n when Y1 < x
    V = np.linspace(-W, W, m)
    tot = 0.0
    out = []
    for v in V:
        y = np.linspace(v - W, min(v + W, d), m)
        if y[-1] <= y[0]:
            out.append(0.0)
            continue
        dens = (n - 1) * cdf_eps(y - v) ** (n - 2) * pdf_eps(y - v)
        out.append(integrate.trapezoid((v - (y - W * (n - 2) / n)) * dens, y))
    return integrate.trapezoid(np.array(out), V) / (2 * W)


print("(d) best-response checks (argmax over deviations d, should be 0):")
ds = np.linspace(-0.5, 0.5, 101)
for n in (2, 4):
    pf = [payoff_fpa(d, n) for d in ds]
    print("   FPA n = %d: argmax d = %.3f, payoff at d=0: %.4f (formula 2w/(n(n+1)) = %.4f)"
          % (n, ds[int(np.argmax(pf))], payoff_fpa(0.0, n), 2 * W / (n * (n + 1))))
ds2 = np.linspace(-0.3, 0.3, 31)
for n in (3,):
    ps = [payoff_spa(d, n) for d in ds2]
    print("   SPA n = %d: argmax d = %.3f" % (n, ds2[int(np.argmax(ps))]))

# (e) Monte Carlo of rents with V = 0 (translation invariance)
print("(e) expected bidder rents, V - E[price], in units of w:")
R = 400_000
ns = np.arange(2, 11)
rent = {"FPA": [], "SPA": [], "ENG": []}
for n in ns:
    eps = rng.uniform(-W, W, size=(R, n))
    eps.sort(axis=1)
    top, second, low = eps[:, -1], eps[:, -2], eps[:, 0]
    p_fpa = top - W
    p_spa = second - W * (n - 2) / n
    p_eng = (second + low) / 2 if n > 2 else second
    mc = (-p_fpa.mean(), -p_spa.mean(), -p_eng.mean())
    th = (2 * W / (n + 1), 2 * W * (n - 1) / (n * (n + 1)), W / (n + 1))
    rent["FPA"].append(th[0]); rent["SPA"].append(th[1]); rent["ENG"].append(th[2])
    print("   n = %2d: FPA %.4f (%.4f), SPA %.4f (%.4f), English %.4f (%.4f)"
          % (n, mc[0], th[0], mc[1], th[1], mc[2], th[2]))

# (f) seller discloses S_0 = V + eps_0, eps_0 ~ U[-w, w] independent of the bidders' noise.
# SPA price w(Y1, Y1, S0): posterior of V given S_1 = Y_1 = y and S_0 = s0 is proportional to
# (y + w - V)^{n-2} on [max(y, s0) - w, min(y, s0) + w].  English price: posterior given all
# signals (winner's set to Y1) and S0 is uniform on [max - w, min + w].
def mean_power(a, b, k):
    """Mean of the density proportional to t^k on [a, b] (k >= 0)."""
    return (k + 1) / (k + 2) * (b ** (k + 2) - a ** (k + 2)) / (b ** (k + 1) - a ** (k + 1))


print("(f) disclosure of S_0 = V + eps_0: expected rents with S_0 public (units of w):")
rent_disc = {}
for n in (2, 3, 4, 5):
    eps = rng.uniform(-W, W, size=(R, n))
    eps.sort(axis=1)
    s0 = rng.uniform(-W, W, R)
    y1, ylow = eps[:, -2], eps[:, 0]
    lo_t = y1 - np.minimum(y1, s0)                  # t = y + w - V ranges over [lo_t, hi_t]
    hi_t = 2 * W - (np.maximum(y1, s0) - y1)
    if n == 2:
        Et = (lo_t + hi_t) / 2
    else:
        Et = mean_power(lo_t, hi_t, n - 2)
    p_spa = y1 + W - Et
    if n == 2:
        p_eng = p_spa
    else:
        p_eng = (np.maximum(y1, s0) + np.minimum(ylow, s0)) / 2
    rent_disc[n] = (-p_spa.mean(), -p_eng.mean())

    # numerical integration of the SPA rent over (Y_1, S_0); English rent in closed form
    def integrand(z0, y, n=n):
        Fy = (y + W) / (2 * W)
        dens = n * (n - 1) * Fy ** (n - 2) * (1 - Fy) / (2 * W) / (2 * W)
        a, b = y - min(y, z0), 2 * W - (max(y, z0) - y)
        Et_ = (a + b) / 2 if n == 2 else mean_power(a, b, n - 2)
        return (y + W - Et_) * dens

    v1 = integrate.dblquad(integrand, -W, W, lambda y: -W, lambda y: y, epsabs=1e-11, epsrel=1e-11)[0]
    v2 = integrate.dblquad(integrand, -W, W, lambda y: y, lambda y: W, epsabs=1e-11, epsrel=1e-11)[0]
    print("   n = %d: SPA %.4f (Monte Carlo %.4f; without disclosure %.4f), English %.4f "
          "(formula n w/((n+1)(n+2)) = %.4f; without %.4f)"
          % (n, -(v1 + v2), -p_spa.mean(), 2 * W * (n - 1) / (n * (n + 1)), -p_eng.mean(),
             n * W / ((n + 1) * (n + 2)), W / (n + 1)))

# ---------------------------------------------------------------- figure
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})
fig, ax = plt.subplots(figsize=(5.2, 3.5))
ax.plot(ns, rent["FPA"], color=ORANGE, marker="o", mfc="white", mew=1.5, lw=1.6,
        label=r"first price: $\frac{2\omega}{n+1}$")
ax.plot(ns, rent["SPA"], color=BLUE, marker="s", mfc="white", mew=1.5, lw=1.6,
        label=r"second price: $\frac{2\omega(n-1)}{n(n+1)}$")
ax.plot(ns, rent["ENG"], color=GREEN, marker="D", mfc="white", mew=1.5, lw=1.6,
        label=r"English: $\frac{\omega}{n+1}$")
ax.set_xticks(ns)
ax.set_xlabel(r"number of bidders $n$")
ax.set_ylabel(r"$V-\mathbb{E}[\mathrm{price}]$ (units of $\omega$)")
ax.set_ylim(0, 0.72)
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
