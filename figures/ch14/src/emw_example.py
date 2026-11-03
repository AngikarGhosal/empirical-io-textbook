"""One informed bidder in a first-price auction (Engelbrecht-Wiggans, Milgrom and Weber 1983;
Hendricks and Porter 1988 with c = 0). Illustration for Section 14.8.5.

The informed bidder's estimate H = E[V | X, Z] is uniform on [0, 1]; reserve price R = 0.2.
Equilibrium: informed bids 0 (abstains) if h < R, R if R <= h <= hhat, E[H | H <= h] = h/2 if
h > hhat, where E[H | H < hhat] = R gives hhat = 2R. The maximum uninformed bid has CDF
G*(b) = F(hhat) on [0, R] and G*(b) = F(tau(b)) = 2b on (R, 1/2].

The script computes the equilibrium objects, the informed bidder's win probability (closed form
of Hendricks and Porter 1988, p. 870), profits and revenue, checks them by simulation with two
uninformed bidders who mix independently, checks best responses on a grid, and draws
figures/ch14/emw_equilibrium.pdf.
"""
from pathlib import Path

import numpy as np
from scipy import integrate

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "emw_equilibrium.pdf"

R = 0.2
HHAT = 2 * R
F = lambda h: np.clip(h, 0.0, 1.0)


def sigma(h):
    h = np.asarray(h, dtype=float)
    return np.where(h < R, 0.0, np.where(h <= HHAT, R, h / 2))


def Gstar(b):
    b = np.asarray(b, dtype=float)
    return np.where(b <= R, HHAT, np.clip(2 * b, 0.0, 1.0))


Fh, FR = F(HHAT), F(R)
win_uncond = 0.5 * (1 - Fh**2) + Fh * (Fh - FR)
p_any_bid = 1 - Fh * FR
win_cond_hp = 0.5 + 0.5 * Fh * (Fh - FR) / (1 - Fh * FR)
print("hhat = %.3f" % HHAT)
print("informed win probability: unconditional %.4f, conditional on a bid %.4f (HP formula %.4f)"
      % (win_uncond, win_uncond / p_any_bid, win_cond_hp))
prof_inf = integrate.quad(lambda h: (h - h / 2) * F(h), HHAT, 1)[0] + Fh * integrate.quad(lambda h: h - R, R, HHAT)[0]
surplus = 0.5 - FR * Fh * (R / 2)  # E[H] minus E[H 1{no sale}] = E[H; H<R] * Pr(no uninformed bid)
print("informed expected profit %.4f, total surplus %.4f, seller revenue %.4f"
      % (prof_inf, surplus, surplus - prof_inf))

# best-response checks for the informed bidder on a grid of types and bids
bgrid = np.linspace(0.0, 0.6, 6001)
worst = 0.0
for h in np.linspace(0.0, 1.0, 101):
    pay = (h - bgrid) * np.where(bgrid < R, 0.0, Gstar(bgrid))
    pay = np.where(bgrid < R, 0.0, pay)                      # bids below R are not allowed
    best = max(pay.max(), 0.0)
    eq = (h - sigma(h)) * (Gstar(sigma(h)) if sigma(h) >= R else 0.0)
    worst = max(worst, best - eq)
print("informed: max gain from deviating on the grid = %.2e" % worst)

# uninformed payoff from bidding b > R: E[(H - b) 1{H < tau(b)}] = int_0^{2b} (h - b) dh
for b in (0.25, 0.35, 0.45):
    print("uninformed payoff from bid %.2f: %.2e" % (b, integrate.quad(lambda h: h - b, 0, 2 * b)[0]))
# bidding exactly R: wins if h < R, ties with informed types in [R, hhat] (split evenly)
pay_R = integrate.quad(lambda h: h - R, 0, R)[0] + 0.5 * integrate.quad(lambda h: h - R, R, HHAT)[0]
print("uninformed payoff from bidding R exactly: %.4f (negative)" % pay_R)

# simulation with two uninformed bidders; each abstains w.p. sqrt(F(hhat)), else draws from
# G*(b)^{1/2} on (R, 1/2]
rng = np.random.default_rng(1988)
M = 2_000_000
H = rng.random(M)
binf = sigma(H)
unif = []
for _ in range(2):
    u = rng.random(M)
    q0 = np.sqrt(HHAT)
    # inverse of sqrt(2b) = u on (q0, 1]  ->  b = u^2 / 2
    unif.append(np.where(u < q0, 0.0, u**2 / 2))
mu = np.maximum(unif[0], unif[1])
inf_bid = binf > 0
uninf_bid = mu > 0
any_bid = inf_bid | uninf_bid
inf_wins = inf_bid & (binf > mu)
print("simulation: informed wins %.4f unconditional, %.4f given a bid"
      % (inf_wins.mean(), inf_wins.sum() / any_bid.sum()))
price = np.where(any_bid, np.maximum(binf, mu), 0.0)
unf_win = uninf_bid & (mu > binf)
print("simulation: revenue %.4f, informed profit %.4f, uninformed profit %.4f"
      % (price.mean(), np.where(inf_wins, H - binf, 0).mean(), np.where(unf_win, H - mu, 0).mean()))
print("simulation: uninformed profit on tracts where informed abstains %.4f, where informed bids %.4f"
      % (np.where(unf_win & ~inf_bid, H - mu, 0).sum() / (~inf_bid).sum(),
         np.where(unf_win & inf_bid, H - mu, 0).sum() / inf_bid.sum()))
print("simulation: Pr(no informed bid) %.3f, Pr(no uninformed bid) %.3f"
      % ((~inf_bid).mean(), (~uninf_bid).mean()))

# ---------------------------------------------------------------- figure
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))

ax = axes[0]
ax.plot([0, R], [0, 0], color=ORANGE, lw=2.0)
ax.plot([R, HHAT], [R, R], color=ORANGE, lw=2.0)
hh = np.linspace(HHAT, 1, 100)
ax.plot(hh, hh / 2, color=ORANGE, lw=2.0, label=r"$\sigma^*(h)$")
ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (4, 3)))
ax.plot([R, R], [0, R], color=INK2, lw=0.6, ls=":")
ax.plot([HHAT, HHAT], [0, R], color=INK2, lw=0.6, ls=":")
ax.set_xticks([0, R, HHAT, 1])
ax.set_xticklabels([r"$0$", r"$r$", r"$\hat h$", r"$1$"])
ax.set_yticks([0, R, 0.5])
ax.set_yticklabels([r"$0$", r"$r$", r"$\frac{1}{2}$"])
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.62)
ax.set_xlabel(r"informed estimate $h$")
ax.set_ylabel(r"informed bid")
ax.set_title(r"(a) informed bid", fontsize=10)

ax = axes[1]
# informed bid CDF
ax.plot([0, R], [FR, FR], color=ORANGE, lw=2.0, label="informed bid")
bb = np.linspace(R, 0.5, 100)
ax.plot(bb, 2 * bb, color=ORANGE, lw=2.0)
ax.plot([0.5, 0.6], [1, 1], color=ORANGE, lw=2.0)
ax.plot([0, R], [Fh, Fh], color=BLUE, lw=2.0, ls=(0, (5, 2)), label="highest uninformed bid")
ax.plot(bb, 2 * bb, color=BLUE, lw=2.0, ls=(0, (5, 2)))
ax.plot([0.5, 0.6], [1, 1], color=BLUE, lw=2.0, ls=(0, (5, 2)))
ax.plot([R], [FR], marker="o", mfc="white", mec=ORANGE, ms=5)
ax.plot([R], [Fh], marker="o", mfc=ORANGE, mec=ORANGE, ms=5)
ax.set_xticks([0, R, 0.5])
ax.set_xticklabels([r"$0$", r"$r$", r"$\frac{1}{2}$"])
ax.set_yticks([0, FR, Fh, 1])
ax.set_yticklabels([r"$0$", r"$F(r)$", r"$F(\hat h)$", r"$1$"])
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"bid $b$")
ax.set_ylabel(r"CDF")
ax.set_title(r"(b) bid distributions", fontsize=10)
ax.legend(frameon=False, loc="upper left", fontsize=8)
for a in axes:
    a.grid(color=GRID, lw=0.6)
    a.set_axisbelow(True)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
