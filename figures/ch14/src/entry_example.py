"""Entry examples for Section 14.11.

Levin-Smith timing (values learned after paying K), uniform values on [0, 1],
second-price (or any standard) auction, no reserve:
    pi(k)  = 1 / (k (k + 1))                 profit of a participant facing k - 1 rivals
    Pi(p)  = sum_{k=1}^{N} C(N-1, k-1) p^{k-1} (1-p)^{N-k} pi(k)
    p*     solves Pi(p) = K (p* = 1 if Pi(1) >= K)
    revenue = E[(X-1)/(X+1) 1{X >= 2}], X ~ Bin(N, p*);  welfare = E[X/(X+1)] - N p* K.

Samuelson timing (values known before entry), same primitives, reserve r:
    cutoff v* solves (v* - r) v*^(N-1) = K.

Outputs: figures/ch14/entry.pdf and figures/ch14/src/entry_example_numbers.txt
"""
from math import comb
from pathlib import Path

import numpy as np
from scipy import optimize

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_FIG = HERE.parent / "entry.pdf"
OUT_TXT = HERE / "entry_example_numbers.txt"


def pi(k):
    return 1.0 / (k * (k + 1))


def Pi(p, N):
    return sum(comb(N - 1, k - 1) * p ** (k - 1) * (1 - p) ** (N - k) * pi(k) for k in range(1, N + 1))


def p_star(N, K):
    if Pi(1.0, N) >= K:
        return 1.0
    if Pi(0.0, N) <= K:
        return 0.0
    return optimize.brentq(lambda p: Pi(p, N) - K, 0.0, 1.0, xtol=1e-14)


def binom_pmf(N, p):
    return np.array([comb(N, k) * p**k * (1 - p) ** (N - k) for k in range(N + 1)])


def outcomes(N, K):
    p = p_star(N, K)
    w = binom_pmf(N, p)
    ks = np.arange(N + 1)
    gross = np.sum(w * ks / (ks + 1.0))                     # E[max of X uniforms], 0 if X = 0
    revenue = np.sum(w[2:] * (ks[2:] - 1) / (ks[2:] + 1.0))  # second-highest value if X >= 2
    welfare = gross - N * p * K
    rents = N * p * (Pi(p, N) - K)                          # expected net bidder profits
    return dict(p=p, EN=N * p, gross=gross, revenue=revenue, welfare=welfare, rents=rents,
                P_none=w[0], P_one=w[1])


lines = ["Levin-Smith example: uniform values, N = 5 potential bidders, no reserve"]
lines.append("pi(k) = " + ", ".join(f"pi({k}) = {pi(k):.5f}" for k in range(1, 6)))
lines.append(f"Pi(0) = {Pi(0.0, 5):.5f}, Pi(1) = {Pi(1.0, 5):.5f} (1/30 = {1/30:.5f})")
for K in (0.05, 0.10):
    o = outcomes(5, K)
    lines.append(
        f"K = {K:.2f}: p* = {o['p']:.4f}, expected entrants = {o['EN']:.4f}, Pr(no entrant) = {o['P_none']:.4f},"
        f" revenue = {o['revenue']:.4f}, welfare = {o['welfare']:.4f}, net rents = {o['rents']:.2e}"
    )
lines.append("")
lines.append("Revenue and welfare as a function of N")
for K in (0.05, 0.10):
    for N in range(1, 11):
        o = outcomes(N, K)
        lines.append(f"K = {K:.2f}, N = {N:2d}: p* = {o['p']:.4f}, E[n] = {o['EN']:.3f}, revenue = {o['revenue']:.4f},"
                     f" welfare = {o['welfare']:.4f}, rents = {o['rents']:.4f}")
    best_det = max(range(1, 11), key=lambda n: n / (n + 1) - n * K)
    lines.append(f"K = {K:.2f}: deterministic n maximizing n/(n+1) - nK is n = {best_det}"
                 f" with welfare {best_det/(best_det+1) - best_det*K:.4f};"
                 f" n-1 gives {(best_det-1)/best_det - (best_det-1)*K:.4f}, n+1 gives {(best_det+1)/(best_det+2) - (best_det+1)*K:.4f}")
lines.append("")
lines.append("Samuelson timing: cutoff v* solves (v* - r) v*^(N-1) = K, uniform values, N = 5")
for K in (0.05, 0.10):
    for r in (0.0, 0.25):
        vs = optimize.brentq(lambda v: (v - r) * v**4 - K, r, 1.0)
        lines.append(f"K = {K:.2f}, r = {r:.2f}: v* = {vs:.4f}, Pr(enter) = {1 - vs:.4f}, expected entrants = {5*(1-vs):.4f}")

OUT_TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# ----------------------------------------------------------------- figure
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))

ax = axes[0]
pp = np.linspace(0, 1, 400)
ax.plot(pp, [Pi(x, 5) for x in pp], color=INK, lw=2, label=r"$\Pi(p)$, $N=5$")
for K, col in ((0.05, BLUE), (0.10, ORANGE)):
    ps = p_star(5, K)
    ax.axhline(K, color=col, lw=1.2, ls="--")
    ax.plot([ps, ps], [0, K], color=col, lw=1, ls=":")
    ax.plot(ps, K, "o", color=col, ms=6, mec="white", mew=1.2, zorder=5)
    ax.text(0.02, K + 0.012, rf"$K={K:.2f}$", color=INK2, fontsize=8)
    ax.text(ps + 0.025, K + (0.015 if K > 0.07 else 0.008), rf"$p^*={ps:.3f}$", color=INK2, fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.52)
ax.set_xlabel(r"entry probability of rivals $p$")
ax.set_ylabel("expected profit of an entrant")
ax.set_title("(a) Entry indifference", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper right")

ax = axes[1]
Ns = np.arange(1, 11)
for K, col, mk in ((0.05, BLUE, "o"), (0.10, ORANGE, "s")):
    rev = [outcomes(int(N), K)["revenue"] for N in Ns]
    ax.plot(Ns, rev, color=col, lw=2, marker=mk, ms=6, mec="white", mew=1.0, label=rf"$K={K:.2f}$")
ax.set_xlim(0.5, 10.5)
ax.set_ylim(0, 0.65)
ax.set_xticks(Ns)
ax.set_xlabel(r"number of potential bidders $N$")
ax.set_ylabel("expected revenue")
ax.set_title("(b) Revenue with equilibrium entry", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(OUT_FIG)
fig.savefig(HERE / "entry_preview.png", dpi=150)
print("wrote", OUT_FIG)
