"""Bertrand version of Green and Porter (1984): numbers and a simulated price path (Section 12.5.2).

Two firms, marginal cost c, demand D(p) with probability 1 - gamma and zero with probability gamma (iid).
Trigger strategy: charge p^m until own sales are zero, then charge c for T periods, then collude again.

Incentive constraint (Proposition prop:12-gp-bertrand):
    L(T) = 2 (1 - gamma) delta + (2 gamma - 1) delta^(T + 1) >= 1.
Value of collusion, in units of pi^m:
    V(T) = (1 - gamma) / 2 / (1 - (1 - gamma) delta - gamma delta^(T + 1)).
Long-run fraction of periods spent in price wars: gamma T / (1 + gamma T).

Output: figures/ch12/gp_path.pdf. Prints every number quoted in the text.
"""
import os
import math
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


def L(T, d, g):
    return 2 * (1 - g) * d + (2 * g - 1) * d ** (T + 1)


def V(T, d, g):
    """Value of collusion in units of pi^m (T = inf allowed)."""
    tail = 0.0 if math.isinf(T) else g * d ** (T + 1)
    return (1 - g) / 2 / (1 - (1 - g) * d - tail)


def T_min(d, g):
    """Shortest punishment satisfying the IC, or None if no T works."""
    if g >= 0.5 or 2 * (1 - g) * d <= 1:
        return None
    x = (2 * (1 - g) * d - 1) / (1 - 2 * g)      # need d^(T+1) <= x
    t1 = math.log(x) / math.log(d)              # need T + 1 >= t1
    T = max(1, math.ceil(t1 - 1 - 1e-12))
    while L(T, d, g) < 1:
        T += 1
    while T > 1 and L(T - 1, d, g) >= 1:
        T -= 1
    return T, x, t1


print("=== Example ex:12-gp-numbers: delta = 0.9, gamma = 0.1 ===")
d, g = 0.9, 0.1
print(f"T = inf condition 2 delta (1 - gamma) = {2 * d * (1 - g):.4f}")
Tm, x, t1 = T_min(d, g)
print(f"need delta^(T+1) <= {x:.4f}; ln(x)/ln(delta) = {t1:.4f}; minimal T = {Tm}")
for T in range(0, 6):
    print(f"  T = {T}: L(T) = {L(T, d, g):.4f}, 0.9^(T+1) = {d ** (T + 1):.4f}, V/pi^m = {V(T, d, g):.4f}")
print(f"  T = inf: V/pi^m = {V(math.inf, d, g):.4f}")
print(f"perfect monitoring (gamma = 0): V/pi^m = {0.5 / (1 - d):.4f}")
print(f"long-run fraction in wars at T = {Tm}: {g * Tm / (1 + g * Tm):.4f}")
print(f"normalized payoff (1 - delta) V / pi^m at T = {Tm}: {(1 - d) * V(Tm, d, g):.4f}")
print(f"long-run average profit per firm / pi^m: {(1 - g) * 0.5 / (1 + g * Tm):.4f}")

print("\n=== Figure fig:12-gp-ic: delta = 0.9 ===")
for gg in (0.05, 0.1, 0.3):
    res = T_min(d, gg)
    print(f"gamma = {gg}: limit 2(1-gamma)delta = {2 * (1 - gg) * d:.4f}, minimal T = {res[0]}, "
          f"L at T=0: {L(0, d, gg):.4f}")
    print("   L(T), T=0..15:", " ".join(f"{L(T, d, gg):.4f}" for T in range(16)))

print("\n=== Exercise check: delta = 0.95, gamma = 0.2 ===")
res = T_min(0.95, 0.2)
print(f"minimal T = {res[0]}, fraction in wars = {0.2 * res[0] / (1 + 0.2 * res[0]):.4f}")

# ---------------------------------------------------------------- simulation
rng = np.random.default_rng(55)
Tsim, n = 2, 60
zero = rng.random(n) < g          # zero-demand periods
price = np.empty(n)
war_left = 0
starts = []
for t in range(n):
    if war_left > 0:
        price[t] = 0.0             # price = c (normalized to 0)
        war_left -= 1
    else:
        price[t] = 1.0             # price = p^m (normalized to 1)
        if zero[t]:
            war_left = Tsim
            starts.append(t)
print("\n=== Simulated path (seed 55, gamma = 0.1, T = 2, 60 periods) ===")
print("zero-demand periods:", [int(t) + 1 for t in np.where(zero)[0]])
print("wars triggered after periods:", [s + 1 for s in starts])
print("share of periods at c:", float(np.mean(price == 0.0)))

fig, ax = plt.subplots(figsize=(6.0, 2.6))
tt = np.arange(1, n + 1)
ax.step(tt, price, where="mid", color=BLUE, lw=1.6)
ax.plot(tt[zero], np.full(zero.sum(), -0.12), linestyle="none", marker="v", color=ORANGE, ms=5,
        clip_on=False, label="zero-demand period")
ax.set_yticks([0, 1])
ax.set_yticklabels([r"$c$", r"$p^m$"])
ax.set_ylim(-0.2, 1.45)
ax.set_xlim(0.5, n + 0.5)
ax.set_xlabel(r"period $t$")
ax.set_ylabel("price")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.04), frameon=False, fontsize=9)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "gp_path.pdf")
fig.savefig(out)
print("saved", os.path.abspath(out))
