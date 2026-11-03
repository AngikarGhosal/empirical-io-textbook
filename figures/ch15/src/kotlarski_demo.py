"""Deconvolution of log bids with Kotlarski's formulas (Section 15.7.3).

Design (our simulation):
    T = 1,000 auctions with n = 4 bidders of one type. Log bids  Q_jt = Y_3t + Y_jt, where
        Y_3 = ln(omega) = 0.3 (Gamma(4, 1) - 4) / 2       (common component: s.d. 0.3, right-skewed)
        Y_j = ln(B0_j)  = -0.2 (Gamma(2, 1) - 2) / sqrt(2) (homogenized log bid: s.d. 0.2, left-skewed)
    Both components are ordinary smooth in the sense of Fan (1991), as Krasnokutskaya's (D5) requires,
    and their characteristic functions never vanish. E[Y_j] = 0 holds by construction.

Estimator (Li and Vuong 1998; Krasnokutskaya 2011), averaging over all ordered pairs (a, b) of bids
within an auction:
    Psi-hat(t1, t2) = mean of exp(i t1 Q_a + i t2 Q_b),  Psi1-hat(t1, t2) = mean of i Q_a exp(i t1 Q_a + i t2 Q_b),
    phi3-hat(t) = exp( int_0^t Psi1-hat(0, u) / Psi-hat(0, u) du ),  phi1-hat(t) = Psi-hat(t, 0) / phi3-hat(t),
    density-hat(x) = (1 / 2 pi) int_{-Tc}^{Tc} (1 - |t| / Tc) exp(-i t x) phi-hat(t) dt   (damping factor).
    Tc is chosen on the grid 3, 4, ..., 25 to match the first two moments of each component
    (Krasnokutskaya's criterion): mean(Y_3) = mean(Q), Var(Y_3) = Cov(Q_a, Q_b), Var(Y_j) = Var(Q) - Cov(Q_a, Q_b).
    Moments of the estimated densities are computed on [-1.2, 1.2]; negative values are set to zero
    only for plotting.
R = 200 replications; we plot the pointwise median and 5-95 percent band.

Outputs: figures/ch15/kotlarski_demo.pdf and figures/ch15/src/kotlarski_demo_numbers.txt
Seed: 15703.
"""
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_FIG = HERE.parent / "kotlarski_demo.pdf"
OUT_TXT = HERE / "kotlarski_demo_numbers.txt"

SEED = 15703
T = 1000
NB = 4
K3, K1 = 4.0, 2.0
S3 = 0.3
S1 = 0.2
R = 200
TC_GRID = np.arange(3.0, 25.01, 1.0)

rng = np.random.default_rng(SEED)

t_grid = np.linspace(0.0, 25.0, 1251)         # nonnegative frequencies; cf(-t) = conj(cf(t))
x_grid = np.linspace(-1.2, 1.2, 241)


def simulate():
    y3 = S3 * (rng.gamma(K3, 1.0, T) - K3) / np.sqrt(K3)
    y = -S1 * (rng.gamma(K1, 1.0, (T, NB)) - K1) / np.sqrt(K1)
    return y3[:, None] + y


def cf_pieces(Q):
    """Psi(0,u), Psi1(0,u), Psi(u,0) on t_grid, averaged over all ordered pairs (a, b), a != b.

    Every bid appears equally often in each position, so Psi(0,u) = Psi(u,0) = mean of exp(i u Q),
    and Psi1(0,u) = mean over (t, b) of exp(i u Q_bt) * i * (sum_{a != b} Q_at) / (n - 1).
    """
    q = Q.ravel()
    others = ((Q.sum(axis=1, keepdims=True) - Q) / (NB - 1)).ravel()
    E = np.exp(1j * np.outer(t_grid, q))
    psi0u = E.mean(axis=1)
    psi1_0u = (1j * others[None, :] * E).mean(axis=1)
    return psi0u, psi1_0u, psi0u.copy()


def cumtrapz(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


EXP_XT = np.exp(-1j * np.outer(x_grid, t_grid))
DT = t_grid[1] - t_grid[0]
TRAP = np.full(t_grid.size, DT)
TRAP[0] = TRAP[-1] = DT / 2.0


def invert(phi, tc):
    """Damped inverse Fourier transform using phi on t >= 0 and conjugate symmetry:
    int_{-tc}^{tc} = 2 Re int_0^{tc} (trapezoid rule; the damping weight is zero at t = tc)."""
    w = np.clip(1.0 - t_grid / tc, 0.0, None) * TRAP
    return 2.0 * np.real(EXP_XT @ (w * phi)) / (2.0 * np.pi)


def moments(dens):
    mass = np.trapezoid(dens, x_grid)
    mu = np.trapezoid(x_grid * dens, x_grid) / mass
    var = np.trapezoid((x_grid - mu) ** 2 * dens, x_grid) / mass
    return mu, var


est3 = np.zeros((R, x_grid.size))
est1 = np.zeros((R, x_grid.size))
tc_chosen = np.zeros(R)
var3_hat = np.zeros(R)
var1_hat = np.zeros(R)

for r in range(R):
    Q = simulate()
    psi0u, psi1_0u, psiu0 = cf_pieces(Q)
    phi3 = np.exp(cumtrapz(psi1_0u / psi0u, t_grid))
    phi1 = psiu0 / phi3
    # moment targets
    m_target = Q.mean()
    qc = Q - Q.mean()
    v3_target = np.mean([np.mean(qc[:, a] * qc[:, b]) for a in range(NB) for b in range(NB) if a != b])
    v1_target = np.mean(qc**2) - v3_target
    best = None
    for tc in TC_GRID:
        d3 = invert(phi3, tc)
        d1 = invert(phi1, tc)
        mu3, v3 = moments(d3)
        mu1, v1 = moments(d1)
        crit = (abs(mu3 - m_target) + abs(v3 - v3_target)) / v3_target + \
               (abs(mu1 - 0.0) + abs(v1 - v1_target)) / v1_target
        if best is None or crit < best[0]:
            best = (crit, tc, d3, d1, v3, v1)
    _, tc_chosen[r], est3[r], est1[r], var3_hat[r], var1_hat[r] = best

from scipy import stats

sc3 = S3 / np.sqrt(K3)          # Y_3 = sc3 * (G - K3), G ~ Gamma(K3)
sc1 = S1 / np.sqrt(K1)          # Y_j = -sc1 * (G - K1), G ~ Gamma(K1)
true3 = stats.gamma.pdf(x_grid / sc3 + K3, K3) / sc3
true1 = stats.gamma.pdf(-x_grid / sc1 + K1, K1) / sc1
# density of one log bid Q = Y_3 + Y_j (numerical convolution), for comparison
zz = np.linspace(-3.0, 3.0, 6001)
f3z = stats.gamma.pdf(zz / sc3 + K3, K3) / sc3
single = np.array([np.trapezoid(f3z * stats.gamma.pdf(-(x - zz) / sc1 + K1, K1) / sc1, zz) for x in x_grid])

lines = [
    f"seed = {SEED}, T = {T}, n = {NB}, sd(ln omega) = {S3} (Gamma shape {K3}), sd(ln B0) = {S1} (Gamma shape {K1}), R = {R}",
    f"Tc chosen: median = {np.median(tc_chosen):.2f}, 5th pct = {np.percentile(tc_chosen, 5):.2f},"
    f" 95th pct = {np.percentile(tc_chosen, 95):.2f}",
    f"Var(ln omega): true {S3**2:.4f}; median estimate from deconvolved density {np.median(var3_hat):.4f}",
    f"Var(ln B0): true {S1**2:.4f}; median estimate from deconvolved density {np.median(var1_hat):.4f}",
]
OUT_TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

C_TRUE = "#1f3b73"
C_EST = "#c4512b"
C_ALT = "#7a7a7a"

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
panels = [
    (axes[0], true3, np.maximum(est3, 0.0), r"(a) Common component $\ln\omega$", "true density"),
    (axes[1], true1, np.maximum(est1, 0.0), r"(b) Homogenized log bid $\ln B^0$", "true density"),
]
for ax, tr, es, title, lab in panels:
    ax.fill_between(x_grid, np.percentile(es, 5, axis=0), np.percentile(es, 95, axis=0),
                    color=C_EST, alpha=0.18, lw=0)
    ax.plot(x_grid, tr, color=C_TRUE, lw=2.0, label=lab)
    ax.plot(x_grid, np.median(es, axis=0), color=C_EST, lw=2.0, label="Kotlarski estimate")
    ax.plot(x_grid, single, color=C_ALT, lw=1.1, ls="--", label="density of one log bid")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("log scale")
    ax.set_ylabel("density")
    ax.set_xlim(-1.2, 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.0, loc="upper left")
axes[0].set_ylim(0, 2.2)
axes[1].set_ylim(0, 2.6)
fig.tight_layout()
fig.savefig(OUT_FIG)
print("saved", OUT_FIG)
