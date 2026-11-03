"""Collusive prices in the two-state iid demand model (Section 12.4.1).

N Bertrand firms, constant marginal cost c = 0, demand D_s(p) = A_s (1 - p) in
state s in {H, L}, each state with probability 1/2, iid. Industry profit
pi_s(p) = A_s p (1 - p), monopoly price 1/2 in both states.

kappa = delta / ((1 - delta)(N - 1)) is the slack factor. The most collusive
prices are (Proposition prop:12-rs-regimes):
  kappa < 1                 : p_H = p_L = 0 (competitive pricing)
  kappa = 1                 : p_L = 1/2, pi_H(p_H) = pi_L^M
  1 < kappa < kappa_bar     : p_L = 1/2, pi_H(p_H) = kappa pi_L^M / (2 - kappa)
  kappa >= kappa_bar        : p_H = p_L = 1/2
with kappa_bar = 2 pi_H^M / (pi_H^M + pi_L^M).
Output: figures/ch12/rs_two_state.pdf. Prints every number quoted in the text.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#7a7974"

N, AH, AL = 2, 1.5, 1.0
piHM, piLM = AH / 4.0, AL / 4.0
kbar = 2 * piHM / (piHM + piLM)
d_low = (N - 1) / N
d_bar = kbar * (N - 1) / (1 + kbar * (N - 1))


def kappa(delta):
    return delta / ((1 - delta) * (N - 1))


def lower_root(level, A):
    """Smallest p in [0, 1/2] with A p (1 - p) = level (level <= A / 4)."""
    return 0.5 * (1 - np.sqrt(1 - 4 * level / A))


def prices(delta):
    k = kappa(delta)
    if k < 1 - 1e-12:
        return 0.0, 0.0
    if k >= kbar:
        return 0.5, 0.5
    target = k * piLM / (2 - k)  # equals piLM at k = 1
    return lower_root(target, AH), 0.5


print(f"N = {N}, A_H = {AH}, A_L = {AL}")
print(f"kappa_bar = {kbar:.6f}")
print(f"delta threshold (N-1)/N = {d_low:.6f}")
print(f"delta_bar = {d_bar:.6f}")
pH1, _ = prices(d_low)
print(f"p_H at kappa = 1: {pH1:.6f}  (closed form 0.5(1 - sqrt(1 - A_L/A_H)) = "
      f"{0.5 * (1 - np.sqrt(1 - AL / AH)):.6f})")
for dd in (0.51, 0.52, 0.53, 0.54):
    print(f"delta = {dd:.2f}: kappa = {kappa(dd):.4f}, p_H = {prices(dd)[0]:.4f}, p_L = {prices(dd)[1]:.2f}")

fig, ax = plt.subplots(figsize=(5.6, 3.5))
lo, hi = 0.44, 0.60
# regime 4: competitive
x4 = np.linspace(lo, d_low, 50)[:-1]
ax.plot(x4, np.zeros_like(x4), color=GRAY, lw=2)
ax.plot([d_low], [0], marker="o", mfc="white", mec=GRAY, ms=6, zorder=5)
# regime 2 and 1
x2 = np.linspace(d_low, hi, 400)
pH = np.array([prices(x)[0] for x in x2])
ax.plot(x2, np.full_like(x2, 0.5), color=ORANGE, lw=3.4, label=r"recession price $p_L$", zorder=2)
ax.plot(x2, pH, color=BLUE, lw=1.7, label=r"boom price $p_H$", zorder=3)
ax.plot([d_low], [pH1], marker="o", color=BLUE, ms=6, zorder=5)
ax.plot([d_low], [0.5], marker="o", color=ORANGE, ms=6, zorder=5)
for xv in (d_low, d_bar):
    ax.axvline(xv, color=GRAY, ls=":", lw=1)
ax.text(0.5 * (lo + d_low), 0.56, "regime 4\n(no collusion)", ha="center", va="bottom", fontsize=8.5)
ax.text(0.5 * (d_low + d_bar), 0.56, "regime 2", ha="center", va="bottom", fontsize=8.5)
ax.text(0.5 * (d_bar + hi), 0.56, "regime 1\n(monopoly)", ha="center", va="bottom", fontsize=8.5)
ax.annotate("regime 3\n" + r"($\kappa = 1$)", xy=(d_low, pH1), xytext=(0.455, 0.27),
            fontsize=8.5, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
ax.text(lo + 0.003, 0.025, "competitive: $p_H = p_L = c = 0$", fontsize=8.5, color="#52514e")
ax.set_xlim(lo, hi)
ax.set_ylim(-0.02, 0.66)
ax.set_xticks([0.44, d_low, d_bar, 0.60])
ax.set_xticklabels(["0.44", r"$\frac{N-1}{N}$", r"$\bar\delta$", "0.60"])
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels(["0", "0.1", "0.2", "0.3", "0.4", r"$\frac{1}{2}$"])
ax.set_xlabel(r"discount factor $\delta$")
ax.set_ylabel("most collusive price")
ax.legend(loc="center right", frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 0.33))
ax.grid(axis="y", color="#e6e5e1", lw=0.6)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "rs_two_state.pdf")
fig.savefig(out)
print("saved", os.path.abspath(out))
