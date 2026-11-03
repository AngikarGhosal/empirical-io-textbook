"""
Nonparametric regression, partially linear model and deconvolution.
Appendix A, Section A.11 (fig:A-npreg, fig:A-deconv).

Design 1 (fig:A-npreg): Nadaraya-Watson versus local linear.
    x on [0, 1] with non-uniform density f(x) = (2/3)(1 + x);
    m(x) = 2 exp(-5x) + 0.5 x (steep near the left boundary, m'(0) = -9.5);
    y = m(x) + e, e ~ N(0, 0.25^2), n = 400; Epanechnikov kernel, common bandwidth h = 0.10.
    Panel (a): one sample with both fits. Panel (b): pointwise bias over 1,000 samples,
    with the asymptotic boundary approximation (3/8) h m'(0) for Nadaraya-Watson at x = 0.

Design 2 (Robinson partially linear model, numbers only):
    w ~ U(0, 1), x = sin(2 pi w) + v, v ~ N(0, 1), y = beta x + 2 sin(2 pi w) + e, e ~ N(0, 1),
    beta = 1, n = 500, 1,000 samples. Estimators: OLS of y on (1, x, w) (linear control for w)
    and Robinson's double-residual estimator with local linear first stages (Epanechnikov,
    h = 0.08); heteroskedasticity-robust s.e. and coverage of nominal 95 percent intervals.

Design 3 (fig:A-deconv): Kotlarski / Li-Vuong deconvolution.
    Y_1 = U + e_1, Y_2 = U + e_2, U ~ Gamma(shape 4, scale 0.25) (mean 1, sd 0.5),
    e_j ~ N(0, 0.4^2), n = 1,000 pairs, 200 samples. Characteristic functions estimated by sample
    averages, Kotlarski's formula integrated by the trapezoid rule on [0, T] (801 points),
    densities by the inverse Fourier transform with the damping factor (1 - |t| / T).
    Average integrated squared error recorded for several T.

Output: figures/appA/npreg.pdf, figures/appA/deconv.pdf, figures/appA/src/npreg_demo.json,
        previews *_preview.png
"""
import os
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
SEED = 20261022
rng = np.random.default_rng(SEED)
OUT = {"seed": SEED}

BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7b5cd6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral"})


def epa(z):
    return np.where(np.abs(z) <= 1, 0.75 * (1 - z**2), 0.0)


def nw(x0, x, y, h):
    K = epa((x[None, :] - x0[:, None]) / h)
    return (K * y[None, :]).sum(1) / K.sum(1)


def loclin(x0, x, y, h):
    d = x[None, :] - x0[:, None]
    K = epa(d / h)
    S0, S1, S2 = K.sum(1), (K * d).sum(1), (K * d**2).sum(1)
    T0, T1 = (K * y[None, :]).sum(1), (K * d * y[None, :]).sum(1)
    return (S2 * T0 - S1 * T1) / (S0 * S2 - S1**2)


# ------------------------------------------------------------------ design 1
def m_fun(x):
    return 2 * np.exp(-5 * x) + 0.5 * x


def draw_x(nn):
    # f(x) = (2/3)(1 + x) on [0, 1]; CDF (2/3)(x + x^2 / 2); inverse by the quadratic formula
    uu = rng.uniform(size=nn)
    return -1 + np.sqrt(1 + 3 * uu)


n1, h1, sig1, R1 = 400, 0.10, 0.25, 1000
xg = np.linspace(0, 1, 201)
NWm = np.zeros(xg.size)
LLm = np.zeros(xg.size)
NW2 = np.zeros(xg.size)
LL2 = np.zeros(xg.size)
for r in range(R1):
    x = draw_x(n1)
    y = m_fun(x) + rng.normal(0, sig1, n1)
    a, b = nw(xg, x, y, h1), loclin(xg, x, y, h1)
    NWm += a
    LLm += b
    NW2 += a**2
    LL2 += b**2
NWm /= R1
LLm /= R1
sd_nw = np.sqrt(NW2 / R1 - NWm**2)
sd_ll = np.sqrt(LL2 / R1 - LLm**2)
bias_nw, bias_ll = NWm - m_fun(xg), LLm - m_fun(xg)
mp0 = -10 + 0.5
OUT["npreg"] = {
    "n": n1, "h": h1, "sigma": sig1, "reps": R1, "m_prime_0": mp0,
    "bias_nw_x0": float(bias_nw[0]), "bias_ll_x0": float(bias_ll[0]),
    "approx_bias_nw_x0": float(3 / 8 * h1 * mp0),
    "sd_nw_x0": float(sd_nw[0]), "sd_ll_x0": float(sd_ll[0]),
    "bias_nw_x05": float(bias_nw[100]), "bias_ll_x05": float(bias_ll[100]),
    "bias_nw_x1": float(bias_nw[-1]), "bias_ll_x1": float(bias_ll[-1]),
    "max_abs_bias_nw_interior": float(np.max(np.abs(bias_nw[(xg >= h1) & (xg <= 1 - h1)]))),
    "max_abs_bias_ll_interior": float(np.max(np.abs(bias_ll[(xg >= h1) & (xg <= 1 - h1)]))),
}
x = draw_x(n1)
y = m_fun(x) + rng.normal(0, sig1, n1)
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
a = ax[0]
a.scatter(x, y, s=3, color=INK2, alpha=0.35, lw=0)
a.plot(xg, m_fun(xg), color=INK, lw=1.0, ls="--", label="true $m(x)$")
a.plot(xg, nw(xg, x, y, h1), color=ORANGE, lw=1.3, label="Nadaraya-Watson")
a.plot(xg, loclin(xg, x, y, h1), color=BLUE, lw=1.3, label="local linear")
a.set_xlim(0, 1)
a.set_xlabel("$x$")
a.set_ylabel("$y$")
a.set_title(f"(a) One sample, $n = {n1}$, $h = {h1}$", loc="left")
a.legend(frameon=False, loc="upper right")
a.grid(color=GRID, lw=0.5)
a = ax[1]
a.axhline(0, color=INK2, lw=0.6)
a.axvspan(0, h1, color=GRID, lw=0)
a.axvspan(1 - h1, 1, color=GRID, lw=0)
a.plot(xg, bias_nw, color=ORANGE, lw=1.3, label="Nadaraya-Watson")
a.plot(xg, bias_ll, color=BLUE, lw=1.3, label="local linear")
a.plot([0], [3 / 8 * h1 * mp0], "o", color=ORANGE, ms=4, mfc="white", mew=0.9,
       label=r"$\frac{3}{8}h\,m'(0)$")
a.set_xlim(0, 1)
a.set_xlabel("$x$")
a.set_ylabel("bias")
a.set_title(f"(b) Pointwise bias, {R1:,} samples", loc="left")
a.legend(frameon=False, loc="lower right")
a.grid(color=GRID, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "npreg.pdf"))
fig.savefig(os.path.join(HERE, "npreg_preview.png"), dpi=170)

# ------------------------------------------------------------------ design 2: Robinson
n2, R2, beta, h2 = 500, 1000, 1.0, 0.08
b_ols, b_rob, cov_ols, cov_rob, se_rob = [], [], [], [], []
for r in range(R2):
    w = rng.uniform(size=n2)
    xx = np.sin(2 * np.pi * w) + rng.normal(size=n2)
    yy = beta * xx + 2 * np.sin(2 * np.pi * w) + rng.normal(size=n2)
    X = np.column_stack([np.ones(n2), xx, w])
    bo = np.linalg.lstsq(X, yy, rcond=None)[0]
    eo = yy - X @ bo
    XtXi = np.linalg.inv(X.T @ X)
    Vo = XtXi @ (X.T * eo**2) @ X @ XtXi
    b_ols.append(bo[1])
    cov_ols.append(abs(bo[1] - beta) <= 1.96 * np.sqrt(Vo[1, 1]))
    ey, ex = loclin(w, w, yy, h2), loclin(w, w, xx, h2)
    ry, rx = yy - ey, xx - ex
    br = (rx @ ry) / (rx @ rx)
    er = ry - br * rx
    se = np.sqrt(np.sum(rx**2 * er**2)) / (rx @ rx)
    b_rob.append(br)
    se_rob.append(se)
    cov_rob.append(abs(br - beta) <= 1.96 * se)
OUT["robinson"] = {
    "n": n2, "reps": R2, "beta": beta, "h": h2,
    "ols_mean": float(np.mean(b_ols)), "ols_sd": float(np.std(b_ols)),
    "ols_coverage": float(np.mean(cov_ols)),
    "rob_mean": float(np.mean(b_rob)), "rob_sd": float(np.std(b_rob)),
    "rob_mean_se": float(np.mean(se_rob)), "rob_coverage": float(np.mean(cov_rob)),
    "infeasible_sd_theory": float(1 / np.sqrt(n2)),
}

# ------------------------------------------------------------------ design 3: deconvolution
kU, thU, sE = 4, 0.25, 0.4
n3, R3, NT = 1000, 200, 801
ug = np.linspace(-0.5, 3.5, 321)
eg = np.linspace(-1.5, 1.5, 241)
fU_true = stats.gamma.pdf(ug, kU, scale=thU)
fE_true = stats.norm.pdf(eg, 0, sE)


def deconv(Y1, Y2, T):
    t = np.linspace(0, T, NT)
    E2 = np.exp(1j * np.outer(t, Y2))
    psi0 = E2.mean(1)                                  # Psi(0, t)
    psi1 = (1j * Y1[None, :] * E2).mean(1)             # d Psi / d t1 at (0, t)
    ratio = psi1 / psi0
    integ = np.concatenate([[0], np.cumsum(0.5 * (ratio[1:] + ratio[:-1]) * np.diff(t))])
    phiU = np.exp(integ)
    psiT0 = np.exp(1j * np.outer(t, Y1)).mean(1)       # Psi(t, 0)
    phiE1 = psiT0 / phiU
    damp = np.clip(1 - t / T, 0, None)

    def inv(phi, grid):
        M = np.real(np.exp(-1j * np.outer(grid, t)) * (phi * damp)[None, :])
        return np.sum(0.5 * (M[:, 1:] + M[:, :-1]) * np.diff(t)[None, :], 1) / np.pi
    return inv(phiU, ug), inv(phiE1, eg)


Ts = [5, 8, 10, 12]
ise = {T: ([], []) for T in Ts}
FU8, FE8, NAIVE = [], [], []
for r in range(R3):
    U = rng.gamma(kU, thU, n3)
    Y1 = U + rng.normal(0, sE, n3)
    Y2 = U + rng.normal(0, sE, n3)
    for T in Ts:
        fu, fe = deconv(Y1, Y2, T)
        ise[T][0].append(np.sum((fu - fU_true) ** 2) * (ug[1] - ug[0]))
        ise[T][1].append(np.sum((fe - fE_true) ** 2) * (eg[1] - eg[0]))
        if T == 8:
            FU8.append(fu)
            FE8.append(fe)
    hY = 1.06 * Y1.std(ddof=1) * n3 ** (-0.2)
    NAIVE.append(stats.norm.pdf((ug[:, None] - Y1[None, :]) / hY).sum(1) / (n3 * hY))
FU8, FE8, NAIVE = np.array(FU8), np.array(FE8), np.array(NAIVE)
OUT["deconv"] = {
    "n": n3, "reps": R3, "U": "Gamma(4, 0.25)", "e": "N(0, 0.4^2)",
    "int_fU2": float(np.sum(fU_true**2) * (ug[1] - ug[0])),
    "int_fE2": float(np.sum(fE_true**2) * (eg[1] - eg[0])),
    "ise_by_T": {str(T): {"U_mean": float(np.mean(ise[T][0])), "U_median": float(np.median(ise[T][0])),
                          "e_mean": float(np.mean(ise[T][1])), "e_median": float(np.median(ise[T][1]))}
                 for T in Ts},
    "naive_ise": float(np.mean(np.sum((NAIVE - fU_true[None, :]) ** 2, 1) * (ug[1] - ug[0]))),
    "sd_U": 0.5, "sd_Y": float(np.sqrt(0.25 + sE**2)),
}
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
a = ax[0]
a.fill_between(ug, np.quantile(FU8, 0.05, 0), np.quantile(FU8, 0.95, 0), color=BLUE, alpha=0.18,
               lw=0, label="5 to 95 percent band")
a.plot(ug, np.median(FU8, 0), color=BLUE, lw=1.4, label=r"median of $\hat f_U$ ($T = 8$)")
a.plot(ug, NAIVE.mean(0), color=ORANGE, lw=1.1, ls="-.", label=r"mean kernel density of $Y_1$")
a.plot(ug, fU_true, color=INK, lw=1.0, ls="--", label="true $f_U$")
a.set_xlim(-0.5, 3.5)
a.set_xlabel("$u$")
a.set_ylabel("density")
a.set_title("(a) Common component $U$", loc="left")
a.legend(frameon=False, loc="upper right")
a.grid(color=GRID, lw=0.5)
a = ax[1]
a.fill_between(eg, np.quantile(FE8, 0.05, 0), np.quantile(FE8, 0.95, 0), color=BLUE, alpha=0.18,
               lw=0, label="5 to 95 percent band")
a.plot(eg, np.median(FE8, 0), color=BLUE, lw=1.4, label=r"median of $\hat f_{\epsilon_1}$ ($T = 8$)")
a.plot(eg, fE_true, color=INK, lw=1.0, ls="--", label=r"true $f_{\epsilon_1}$")
a.set_xlim(-1.5, 1.5)
a.set_xlabel(r"$\epsilon$")
a.set_ylabel("density")
a.set_title(r"(b) Idiosyncratic component $\epsilon_1$", loc="left")
a.legend(frameon=False, loc="upper right")
a.grid(color=GRID, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "deconv.pdf"))
fig.savefig(os.path.join(HERE, "deconv_preview.png"), dpi=170)

with open(os.path.join(HERE, "npreg_demo.json"), "w") as fh:
    json.dump(OUT, fh, indent=2)
print(json.dumps(OUT, indent=2))
