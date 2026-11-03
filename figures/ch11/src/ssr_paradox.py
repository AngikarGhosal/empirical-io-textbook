"""The merger paradox of Salant, Switzer and Reynolds (1983).

Linear inverse demand P = a - Q, N symmetric Cournot firms with constant marginal cost c. A merger
of M firms (with no cost savings) leaves N - M + 1 symmetric firms. Each firm earns
(a - c)^2 / (n + 1)^2 when n firms are active, so the insiders' joint profit changes by the factor

    R(M, N) = (N + 1)^2 / (M (N - M + 2)^2).

The figure plots 100 (R - 1), the percentage change in the insiders' joint profit, against the
fraction M / N of firms that merge, for N = 5, 10 and 20. The script also prints the smallest
profitable merger for N = 2, ..., 1000 and checks that it always involves at least 80 percent of
the firms.

Output: figures/ch11/ssr_paradox.pdf
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "ssr_paradox.pdf"

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STYLES = ["-", "--", "-."]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})


def ratio(M, N):
    return (N + 1) ** 2 / (M * (N - M + 2) ** 2)


# smallest profitable merger (weak inequality)
mins = []
for N in range(2, 1001):
    Mstar = min(M for M in range(2, N + 1) if ratio(M, N) >= 1 - 1e-12)
    mins.append((N, Mstar, Mstar / N))
worst = min(mins, key=lambda t: t[2])
print("smallest profitable M for N = 2..12:", [(N, M) for N, M, _ in mins[:11]])
print("N = 20:", mins[18], " N = 100:", mins[98], " N = 1000:", mins[-1])
print("minimum fraction over N <= 1000:", worst)
assert worst[2] >= 0.8 - 1e-12

fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.axhline(0.0, color=INK2, lw=0.8)
for N, col, ls in zip([5, 10, 20], COLORS, STYLES):
    Ms = np.arange(2, N + 1)
    y = 100 * (ratio(Ms, N) - 1)
    ax.plot(Ms / N, y, color=col, ls=ls, lw=2.0, marker="o", ms=4.5, mec="white", mew=0.8)
    # direct label near the left end of each curve
    ax.text(Ms[0] / N - 0.012, y[0], rf"$N = {N}$", fontsize=9, color=INK, ha="right", va="center")
ax.axvline(0.8, color=GRID, lw=1.0, zorder=0)
ax.text(0.79, 160, "80 percent", fontsize=8.5, color=INK2, ha="right")
ax.set_xlim(0.0, 1.03)
ax.set_ylim(-80, 230)
ax.set_xlabel(r"fraction of firms that merge, $\frac{M}{N}$")
ax.set_ylabel("change in insiders' joint profit (%)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
fig.savefig(HERE / "ssr_paradox_preview.png", dpi=150)
print("saved", OUT)
