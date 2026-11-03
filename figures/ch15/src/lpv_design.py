"""Figure fig:15-lpv-design (Chapter 15, Section 15.5.1).

The Monte Carlo design of Li, Perrigne and Vuong (2002, Section 3), taken from
Kagel, Harstad and Levin (1987): v_i = y + u_i with y ~ U[0.25, 0.75] and
u_i ~ U[-0.25, 0.25], so that values lie on [0, 1], have pairwise correlation
1/2, and a triangular marginal density.

Panel (a): the triangular marginal density f(v) = 4v on [0, 1/2], 4 - 4v on
[1/2, 1].
Panel (b): bid shading v - b(v) in the affiliated private value (APV)
equilibrium and in the independent private value (IPV) equilibrium with the
same marginal density, for n = 2 and n = 3.

APV bids:
  * v <= 1/2: b = n v / (n + 1) (LPV eq. (10)).
  * n = 2, v >= 1/2: b = 2 v / 3 (LPV eq. (11)).
  * n = 3, v >= 1/2: closed form of LPV eq. (12). We check it against a
    numerical solution of the first-order ODE (LPV eq. (1)),
    s'(v) = (v - s(v)) f_{Y1|V1}(v|v) / F_{Y1|V1}(v|v),
    with F_{Y1|V1} computed by quadrature over the common component y.
IPV bids: b(v) = v - int_0^v F(x)^{n-1} dx / F(v)^{n-1}, with F the
triangular CDF.

No random numbers are used. Output: figures/ch15/lpv_design.pdf and a text
file lpv_design_numbers.txt with the checks.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "lpv_design.pdf"
NUM = HERE / "lpv_design_numbers.txt"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.8,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
A, B, EPS = 0.25, 0.75, 0.25


# ---------------------------------------------------------------- primitives
def f_marg(v):
    v = np.asarray(v, dtype=float)
    return np.where(v <= 0.5, 4 * v, 4 - 4 * v)


def F_marg(v):
    v = np.asarray(v, dtype=float)
    return np.where(v <= 0.5, 2 * v ** 2, 1 - 2 * (1 - v) ** 2)


def cond_cdf_Y1(x, v, n):
    """F_{Y1|V1}(x | v): Pr(max of the other n-1 values <= x | V_1 = v)."""
    lo, hi = max(A, v - EPS), min(B, v + EPS)   # support of y given v_1 = v

    def integrand(y):
        p = np.clip((x - y + EPS) / (2 * EPS), 0.0, 1.0)
        return p ** (n - 1)

    val, _ = quad(integrand, lo, hi, limit=200)
    return val / (hi - lo)


def cond_pdf_Y1(x, v, n, h=1e-5):
    return (cond_cdf_Y1(x + h, v, n) - cond_cdf_Y1(x - h, v, n)) / (2 * h)


# ---------------------------------------------------------------- APV bids
def apv_bid_closed(v, n):
    v = np.asarray(v, dtype=float)
    out = n / (n + 1) * v
    if n == 2:
        return out
    hi = v > 0.5

    def C(a):
        return (((a - 0.25) / 0.5) ** 2 + 0.75) ** (-1.5) * np.exp(
            -np.sqrt(3) * np.arctan((a - 0.25) / (0.25 * np.sqrt(3))))

    const = np.exp(np.sqrt(3) * np.arctan(1 / np.sqrt(3))) / 8
    vals = []
    for vv in v[hi]:
        I, _ = quad(lambda a: 1.0 / C(a), 0.5, vv)
        vals.append(vv - C(vv) * (const + I))
    out = out.copy()
    out[hi] = vals
    return out


def apv_bid_ode(vgrid, n):
    """Solve LPV eq. (1) from v = 1/2 (where b = n/(n+1) v) upward."""
    def rhs(v, s):
        ratio = cond_pdf_Y1(v, v, n) / cond_cdf_Y1(v, v, n)
        return (v - s) * ratio

    sol = solve_ivp(rhs, (0.5, vgrid[-1]), [n / (n + 1) * 0.5], t_eval=vgrid,
                    rtol=1e-8, atol=1e-10)
    return sol.y[0]


# ---------------------------------------------------------------- IPV bids
def ipv_bid(v, n):
    out = []
    for vv in np.atleast_1d(v):
        if vv <= 0:
            out.append(0.0)
            continue
        num, _ = quad(lambda x: F_marg(x) ** (n - 1), 0, vv, points=[0.5])
        out.append(vv - num / F_marg(vv) ** (n - 1))
    return np.array(out)


# ---------------------------------------------------------------- checks
lines = []
for n in (2, 3):
    # f/F at the diagonal for v <= 1/2 should equal n / v
    for v in (0.2, 0.4):
        r = cond_pdf_Y1(v, v, n) / cond_cdf_Y1(v, v, n)
        lines.append(f"n={n}, v={v}: f/F = {r:.5f}, n/v = {n / v:.5f}")
vg = np.linspace(0.5, 0.95, 10)
b_closed = apv_bid_closed(vg, 3)
b_ode = apv_bid_ode(vg, 3)
lines.append("n=3, v >= 1/2: closed form (LPV eq. 12) versus ODE solution")
for v, bc, bo in zip(vg, b_closed, b_ode):
    lines.append(f"  v={v:.2f}  closed={bc:.5f}  ode={bo:.5f}")
b2_ode = apv_bid_ode(vg, 2)
lines.append("n=2, v >= 1/2: 2v/3 versus ODE solution")
for v, bo in zip(vg, b2_ode):
    lines.append(f"  v={v:.2f}  2v/3={2 * v / 3:.5f}  ode={bo:.5f}")
for n in (2, 3):
    for v in (0.3, 0.75, 1.0):
        lines.append(f"n={n}, v={v}: APV bid={apv_bid_closed(np.array([v]), n)[0]:.4f}, "
                     f"IPV bid={ipv_bid(v, n)[0]:.4f}")
NUM.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- figure
v = np.linspace(0.0, 1.0, 201)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.75), constrained_layout=True)

ax1.plot(v, f_marg(v), color=COLORS[0], lw=2)
ax1.set_xlabel(r"value $v$")
ax1.set_ylabel(r"density $f(v)$")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 2.2)
ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.set_xticklabels(["0", r"$\frac{1}{4}$", r"$\frac{1}{2}$", r"$\frac{3}{4}$", "1"])
ax1.set_title("(a) marginal density of values", loc="left")

series = []
for k, n in enumerate((2, 3)):
    b_apv = apv_bid_closed(v, n)
    b_ipv = ipv_bid(v, n)
    ax2.plot(v, v - b_apv, color=COLORS[2 * k], lw=2, ls="-")
    ax2.plot(v, v - b_ipv, color=COLORS[2 * k + 1], lw=2, ls="--")
    series.append((n, b_apv, b_ipv))
ax2.set_xlabel(r"value $v$")
ax2.set_ylabel(r"shading $v-\beta(v)$")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 0.52)
ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_xticklabels(["0", r"$\frac{1}{4}$", r"$\frac{1}{2}$", r"$\frac{3}{4}$", "1"])
ax2.set_title("(b) bid shading, APV and IPV", loc="left")
ax2.legend(["APV, $n=2$", "IPV, $n=2$", "APV, $n=3$", "IPV, $n=3$"],
           frameon=False, loc="upper left")
fig.savefig(OUT)
print("saved", OUT)
