"""Figure fig:08-contraction (Chapter 8, section 8.5) and the numbers quoted in the text.

Three simulated markets with J = 10 inside products and an outside good. Random coefficients
on the constant (sd 1.0), on a characteristic x (sd 1.0) and on price (sd 0.5), nu ~ N(0, I_3),
n_s = 1,000 pseudo-random draws held fixed. The mean utilities are delta_j = a_t + b_j, with b_j
fixed across markets and the market shifter a_t chosen so that the outside share is about 0.9,
0.5 or 0.1. The "observed" shares are the model shares at the true delta*, so that the error
||delta^h - delta*|| is known exactly.

For each market we start from the logit inversion delta^0_j = ln s_j - ln s_0 and run
  (a) the plain BLP iteration delta^{h+1} = delta^h + ln s_obs - ln s(delta^h);
  (b) SQUAREM (scheme S3 of Varadhan and Roland 2008) on the same map.
We record the sup-norm error against the number of evaluations of the share function, and the
modulus kappa = max_j sum_k dPhi_j/d delta_k = max_j [1 - (1/s_j) int s_ij s_i0 dF] at delta*.

Output: figures/ch08/fig_contraction.pdf, src/contraction_numbers.txt
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "fig_contraction.pdf"
PNG = HERE / "fig_contraction_preview.png"
TXT = HERE / "contraction_numbers.txt"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

lines = []


def out(s=""):
    print(s)
    lines.append(s)


rng = np.random.default_rng(20261022)
J, NS = 10, 1000
x = np.sort(rng.uniform(0.0, 2.0, J))
price = 1.0 + 0.8 * x + rng.uniform(0.0, 0.5, J)
XR = np.column_stack([np.ones(J), x, price])     # characteristics with random coefficients
SIGMA = np.array([1.0, 1.0, 0.5])
NU = rng.standard_normal((NS, 3))
MU = NU @ (XR * SIGMA).T                         # (NS, J), mu_ij
b = 1.0 * x - 2.0 * price + rng.normal(0.0, 0.3, J)


def ind_shares(delta):
    v = delta[None, :] + MU
    m = np.maximum(0.0, v.max(axis=1, keepdims=True))
    ev = np.exp(v - m)
    e0 = np.exp(-m)
    den = e0 + ev.sum(axis=1, keepdims=True)
    return ev / den, (e0 / den)[:, 0]


def shares(delta):
    sij, si0 = ind_shares(delta)
    return sij.mean(axis=0), si0.mean()


def phi(delta, ln_sobs):
    s, _ = shares(delta)
    return delta + ln_sobs - np.log(s)


def modulus(delta):
    sij, si0 = ind_shares(delta)
    s = sij.mean(axis=0)
    rows = 1.0 - (sij * si0[:, None]).mean(axis=0) / s
    # check against the full Jacobian of Phi
    ds = (sij[:, :, None] * (np.eye(J)[None, :, :] - sij[:, None, :])).mean(axis=0)
    jac = np.eye(J) - ds / s[:, None]
    assert np.all(jac > -1e-14)
    assert np.allclose(jac.sum(axis=1), rows)
    diag_share = np.mean(np.diag(jac) / rows)
    return rows.max(), rows.min(), diag_share


def plain(delta0, ln_sobs, dstar, maxeval=20000, tol=1e-13):
    d = delta0.copy()
    errs = [np.abs(d - dstar).max()]
    nstop = None
    for h in range(1, maxeval + 1):
        dn = phi(d, ln_sobs)
        step = np.abs(dn - d).max()
        d = dn
        errs.append(np.abs(d - dstar).max())
        if nstop is None and step < 1e-12:
            nstop = h
        if errs[-1] < tol:
            break
    return np.array(errs), np.arange(len(errs)), nstop


def squarem(delta0, ln_sobs, dstar, maxeval=20000, tol=1e-13):
    d = delta0.copy()
    errs, evals = [np.abs(d - dstar).max()], [0]
    n = 0
    nstop = None
    while n < maxeval:
        d1 = phi(d, ln_sobs)
        d2 = phi(d1, ln_sobs)
        r = d1 - d
        v = d2 - 2.0 * d1 + d
        nv = np.linalg.norm(v)
        if nv == 0.0:
            d = d2
            n += 2
        else:
            a = -np.linalg.norm(r) / nv
            a = min(a, -1.0)
            dprime = d - 2.0 * a * r + a * a * v
            if not np.all(np.isfinite(dprime)):
                dprime = d2
            dn = phi(dprime, ln_sobs)
            n += 3
            step = np.abs(dn - d).max()
            d = dn
            if nstop is None and step < 1e-12:
                nstop = n
        errs.append(np.abs(d - dstar).max())
        evals.append(n)
        if errs[-1] < tol:
            break
    return np.array(errs), np.array(evals), nstop


targets = [0.9, 0.5, 0.1]
results = []
out("Design: J = 10, n_s = 1000, sigma = (1.0, 1.0, 0.5) on (constant, x, price)")
for s0_target in targets:
    a = brentq(lambda a: shares(a + b)[1] - s0_target, -30, 30)
    dstar = a + b
    s, s0 = shares(dstar)
    ln_sobs = np.log(s)
    delta0 = np.log(s) - np.log(s0)
    kmax, kmin, dshare = modulus(dstar)
    e_p, n_p, stop_p = plain(delta0, ln_sobs, dstar)
    e_s, n_s, stop_s = squarem(delta0, ln_sobs, dstar)
    # empirical linear rate of the plain iteration over the middle of the path
    lo, hi = int(0.3 * len(e_p)), int(0.7 * len(e_p))
    rate = np.exp(np.polyfit(n_p[lo:hi], np.log(e_p[lo:hi]), 1)[0])
    out(f"\nTarget s0 = {s0_target}: realized s0 = {s0:.4f}, inside shares sum = {s.sum():.4f}")
    out(f"  largest inside share = {s.max():.4f}, smallest = {s.min():.5f}")
    out(f"  modulus at delta*: max row sum = {kmax:.4f}, min row sum = {kmin:.4f}")
    out(f"  average share of the diagonal term in the row sum = {dshare:.3f}")
    out(f"  initial error (logit start) = {e_p[0]:.4f}")
    out(f"  plain: evaluations to error < 1e-13 = {n_p[-1]}, step < 1e-12 at h = {stop_p}, "
        f"empirical rate = {rate:.4f}")
    out(f"  SQUAREM: evaluations to error < 1e-13 = {n_s[-1]}, step < 1e-12 after {stop_s} evaluations")
    results.append((s0_target, s0, e_p, n_p, e_s, n_s, kmax))

# --------------------------------------------------------------------------- inner-error bound check
s0_target, s0, e_p, n_p, e_s, n_s, kmax = results[2]
out("\nCheck of the bound ||delta^H - delta*|| <= kappa/(1-kappa) * ||delta^H - delta^{H-1}||, s0 = 0.1:")
a = brentq(lambda a: shares(a + b)[1] - 0.1, -30, 30)
dstar = a + b
s, s0v = shares(dstar)
ln_sobs = np.log(s)
d = np.log(s) - np.log(s0v)
for tol_in in [1e-4, 1e-8]:
    dd = d.copy()
    while True:
        dn = phi(dd, ln_sobs)
        step = np.abs(dn - dd).max()
        dd = dn
        if step < tol_in:
            break
    err = np.abs(dd - dstar).max()
    out(f"  tolerance {tol_in:.0e}: last step {step:.2e}, true error {err:.2e}, "
        f"bound kappa*step/(1-kappa) = {kmax * step / (1 - kmax):.2e}")

# --------------------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(6.0, 3.4))
cols = [BLUE, ORANGE, AQUA]
for (s0_target, s0, e_p, n_p, e_s, n_s, kmax), c in zip(results, cols):
    ax.semilogy(n_p, np.maximum(e_p, 1e-16), color=c, lw=1.6,
                label=rf"$s_0 \approx {s0_target}$, plain ($\kappa = {kmax:.3f}$)")
    ax.semilogy(n_s, np.maximum(e_s, 1e-16), color=c, lw=1.6, ls="--",
                label=rf"$s_0 \approx {s0_target}$, SQUAREM")
ax.set_xlabel("Evaluations of the share function")
ax.set_ylabel(r"$\Vert \delta^{h} - \delta^{*} \Vert_{\infty}$")
ax.set_ylim(1e-14, 10)
xmax = max(r[3][-1] for r in results)
ax.set_xlim(0, xmax * 1.02)
ax.grid(True, which="major", color=GRID, lw=0.6)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper right")
fig.tight_layout()
fig.savefig(OUT)
fig.savefig(PNG, dpi=130)
TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("saved", OUT)
