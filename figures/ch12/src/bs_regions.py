"""Pricing regions in the Bagwell-Staiger Markov-growth game (Section 12.4.4).

Parameters: N = 2 firms, delta = 0.9, boom growth g_b = 1.05, recession growth
g_r = 0.30, profit per consumer pi(P) = P (1 - P) (c = 0, p^m = 1/2).
Maintained assumption (BS (15)): delta g_b > (N-1)/N > delta g_r.

For each (rho, lambda) on a grid we compute the most collusive pair (P_b, P_r)
in two ways:
  1. analytically, from BS Lemmas 1 and 2 and the binding constraint in the weak phase;
  2. by brute force, iterating the best-response map on prices from (p^m, p^m)
     (a monotone decreasing sequence that converges to the greatest fixed point).
The two classifications are compared. Analytic boundaries lambda_hat(rho),
rho_hat(lambda), rho_tilde(lambda) (BS (19)-(21)) and the anti-diagonal are overlaid.
Panel (b): price in the weak phase as a function of lambda for two values of rho.
Output: figures/ch12/bs_regions.pdf. Prints every number quoted in the text.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#7a7974"

N, DELTA, GB, GR = 2, 0.9, 1.05, 0.30
PM = 0.5
PIM = PM * (1 - PM)
assert DELTA * GB > (N - 1) / N > DELTA * GR, "assumption (15) fails"

rho_star = (DELTA * GB - (N - 1) / N) / (DELTA * (GB - GR))
lam_star = 1 - rho_star


def coefs(rho, lam):
    D = DELTA / ((1 - (1 - lam) * DELTA * GR) * (1 - (1 - rho) * DELTA * GB)
                 - DELTA ** 2 * lam * GB * rho * GR)
    B = N - 1 - GB * D * (1 - rho - DELTA * GR * (1 - lam - rho))
    R = N - 1 - GR * D * (1 - lam - DELTA * GB * (1 - lam - rho))
    return D, B, R


def lower_root(level):
    """Smallest P in [0, 1/2] with P (1 - P) = level; P^m if level >= pi(P^m)."""
    level = np.clip(level, 0.0, PIM)
    return 0.5 * (1 - np.sqrt(1 - 4 * level))


def analytic(rho, lam):
    D, B, R = coefs(rho, lam)
    xr = D * lam * GB - R   # slack of recession IC at (p^m, p^m), up to the factor pi(p^m)
    xb = D * rho * GR - B   # slack of boom IC at (p^m, p^m)
    Pb = np.full_like(rho, PM)
    Pr = np.full_like(rho, PM)
    mono = np.minimum(xr, xb) >= 0
    comp = (~mono) & (R * B > (D * lam * GB) * (D * rho * GR))
    pro = (~mono) & (~comp) & (xr < xb)
    cnt = (~mono) & (~comp) & (xb <= xr)
    with np.errstate(divide="ignore", invalid="ignore"):
        Pr = np.where(pro, lower_root(PIM * D * lam * GB / R), Pr)
        Pb = np.where(cnt, lower_root(PIM * D * rho * GR / B), Pb)
    Pb = np.where(comp, 0.0, Pb)
    Pr = np.where(comp, 0.0, Pr)
    cls = np.select([mono, pro, cnt, comp], [0, 1, 2, 3])
    return cls, Pb, Pr


def brute(rho, lam, iters=20000):
    D, B, R = coefs(rho, lam)
    Pb = np.full_like(rho, PM)
    Pr = np.full_like(rho, PM)
    for _ in range(iters):
        pib, pir = Pb * (1 - Pb), Pr * (1 - Pr)
        with np.errstate(divide="ignore", invalid="ignore"):
            capb = np.where(B > 0, pir * rho * GR * D / B, np.inf)
            capr = np.where(R > 0, pib * lam * GB * D / R, np.inf)
        Pb_new = lower_root(np.where(np.isinf(capb), PIM, capb))
        Pr_new = lower_root(np.where(np.isinf(capr), PIM, capr))
        if np.max(np.abs(Pb_new - Pb)) + np.max(np.abs(Pr_new - Pr)) < 1e-12:
            Pb, Pr = Pb_new, Pr_new
            break
        Pb, Pr = Pb_new, Pr_new
    tol = 1e-4
    cls = np.select([(Pb > PM - tol) & (Pr > PM - tol), (Pb < tol) & (Pr < tol), Pb > Pr], [0, 3, 1], 2)
    return cls, Pb, Pr


def lam_hat(rho):
    return (1 - (1 - rho) * DELTA * GB) / (1 / lam_star - DELTA * GB)


def rho_hat(lam):
    return (GB - GR) / GB * rho_star + rho_star * GR / (GB * lam_star) * lam


def rho_tilde(lam):
    return (1 - (1 - lam) * DELTA * GR) / (1 / rho_star - DELTA * GR)


print(f"N = {N}, delta = {DELTA}, g_b = {GB}, g_r = {GR}")
print(f"delta g_b = {DELTA * GB:.4f}, (N-1)/N = {(N - 1) / N:.4f}, delta g_r = {DELTA * GR:.4f}")
print(f"rho* = {rho_star:.4f}, lambda* = {lam_star:.4f}")
print(f"lambda_hat(0) = {lam_hat(0):.4f}, rho_hat(0) = {rho_hat(0):.4f}, rho_tilde(1) = {rho_tilde(1):.4f}")
print(f"lambda_hat(rho*) = {lam_hat(rho_star):.4f}, rho_hat(lambda*) = {rho_hat(lam_star):.4f}, "
      f"rho_tilde(lambda*) = {rho_tilde(lam_star):.4f}")

n = 301
r_ax = np.linspace(0.0005, 0.9995, n)
l_ax = np.linspace(0.0005, 0.9995, n)
RHO, LAM = np.meshgrid(r_ax, l_ax)
cls_a, Pb_a, Pr_a = analytic(RHO, LAM)
cls_b, Pb_b, Pr_b = brute(RHO, LAM)
mism = np.mean(cls_a != cls_b)
print(f"grid {n}x{n}: share of points where analytic and brute-force classes differ = {mism:.5f}")
both = (cls_a == cls_b)
print(f"max |P_b analytic - brute| on agreeing points = {np.max(np.abs(Pb_a - Pb_b)[both]):.2e}, "
      f"max |P_r| diff = {np.max(np.abs(Pr_a - Pr_b)[both]):.2e}")
names = ["monopoly", "procyclical", "countercyclical", "competitive"]
for i, nm in enumerate(names):
    print(f"  share of unit square: {nm:16s} {np.mean(cls_a == i):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw=dict(width_ratios=[1.05, 1]))
ax = axes[0]
cmap = ListedColormap(["#d6e6f8", "#fbdccd", "#cdeee1", "#ecebe7"])
ax.pcolormesh(r_ax, l_ax, cls_a, cmap=cmap, vmin=-0.5, vmax=3.5, shading="nearest", rasterized=True)
x = np.linspace(0, 1, 200)
ax.plot(x, 1 - x, color="black", lw=1.0)
rr = np.linspace(0, rho_star, 100)
ax.plot(rr, lam_hat(rr), color=BLUE, lw=1.6)
ll = np.linspace(0, 1, 200)
rh = rho_hat(ll)
ok = rh <= 1
ax.plot(rh[ok], ll[ok], color=ORANGE, lw=1.6)
lt = np.linspace(lam_star, 1, 100)
ax.plot(rho_tilde(lt), lt, color=AQUA, lw=1.6)
ax.plot([rho_star], [lam_star], "ko", ms=3.5)
ax.text(0.13, 0.62, "monopoly", fontsize=8.5)
ax.text(0.03, 0.055, "procyclical", fontsize=8.5, rotation=0)
ax.text(0.80, 0.80, "counter-\ncyclical", fontsize=8.5, ha="center")
ax.text(0.86, 0.45, "competitive", fontsize=8.5, ha="center")
ax.text(0.30, 0.225, r"$\hat\lambda(\rho)$", color=BLUE, fontsize=9)
ax.text(0.555, 0.04, r"$\hat\rho(\lambda)$", color=ORANGE, fontsize=9)
ax.text(rho_tilde(0.93) - 0.12, 0.93, r"$\tilde\rho(\lambda)$", color=AQUA, fontsize=9)
ax.text(0.15, 0.94, r"$\lambda + \rho = 1$", fontsize=8.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xticks([0, rho_star, 1])
ax.set_xticklabels(["0", r"$\rho^*$", "1"])
ax.set_yticks([0, lam_star, 1])
ax.set_yticklabels(["0", r"$\lambda^*$", "1"])
ax.set_xlabel(r"$\rho$ (boom $\to$ recession)")
ax.set_ylabel(r"$\lambda$ (recession $\to$ boom)")
ax.set_title("(a) most collusive pricing", fontsize=9.5)

ax = axes[1]
lam_line = np.linspace(0.0005, 0.9995, 2001)
for rho0, col in ((0.3, BLUE), (0.85, ORANGE)):
    cl, pb, pr = analytic(np.full_like(lam_line, rho0), lam_line)
    weak = np.minimum(pb, pr)
    # break the line at jumps
    jumps = np.where(np.abs(np.diff(weak)) > 0.05)[0]
    segs = np.split(np.arange(lam_line.size), jumps + 1)
    for j, s in enumerate(segs):
        ax.plot(lam_line[s], weak[s], color=col, lw=2,
                label=(rf"$\rho = {rho0}$" if j == 0 else None))
    print(f"rho = {rho0}: classes along lambda grid change at lambda = "
          f"{[round(float(lam_line[i + 1]), 3) for i in np.where(np.diff(cl) != 0)[0]]}")
ax.axhline(PM, color=GRAY, ls="--", lw=1)
ax.text(0.98, PM + 0.012, r"$p^m$", ha="right", fontsize=9, color="#52514e")
ax.set_xlim(0, 1)
ax.set_ylim(-0.01, 0.56)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"weak-phase price $\min\{p_b^*, p_r^*\}$")
ax.set_title(r"(b) weak-phase price", fontsize=9.5)
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.32, 0.62))
ax.grid(axis="y", color="#e6e5e1", lw=0.6)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "bs_regions.pdf")
fig.savefig(out, dpi=300)
print("saved", os.path.abspath(out))
