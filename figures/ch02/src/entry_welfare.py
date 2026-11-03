"""Free entry versus the second-best number of firms in the linear Cournot market.

Mankiw and Whinston (1986), Example 1: inverse demand P(Q) = a - bQ, constant marginal cost c,
set-up cost F, symmetric Cournot competition after entry. We use a - c = 8, b = 1, F = 1, so that the
second-best number of firms is N^o = 3 and the free-entry number is N^e = 7.

Left panel: total surplus W(N) and industry profit N * pi_N against a continuous N.
Right panel: per-firm profit pi_N (the private value of entry) and the marginal social value of
entry W'(N). Their gap is the business-stealing term N [P - c] dq_N/dN < 0.

The script also prints the welfare losses for the three cases of the paper's Example 1
(N^o = 3, 5, 8), which are reported in tab:02-mw.

Output: figures/ch02/entry_welfare.pdf (run from any directory).
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "entry_welfare.pdf"

AMC = 8.0  # a - c
B = 1.0


def total_output(n):
    return AMC * n / (B * (n + 1.0))


def welfare(n, f):
    q = total_output(n)
    return AMC * q - 0.5 * B * q ** 2 - n * f


def profit(n, f):
    return AMC ** 2 / (B * (n + 1.0) ** 2) - f


def welfare_prime(n, f):
    return AMC ** 2 / (B * (n + 1.0) ** 3) - f


# ---------------------------------------------------------------- table numbers
max_surplus = AMC ** 2 / (2 * B)
print("maximal surplus (a-c)^2/(2b) =", max_surplus)
for n_opt in (3, 5, 8):
    f = AMC ** 2 / (B * (n_opt + 1) ** 3)
    n_e_cont = (n_opt + 1) ** 1.5 - 1
    n_e_int = int(np.floor(n_e_cont + 1e-9))
    w_opt = welfare(n_opt, f)
    w_e_int = welfare(n_e_int, f)
    w_e_cont = welfare(n_e_cont, f)
    print(
        f"N^o={n_opt}: F={f:.5f}, N^e(cont)={n_e_cont:.3f}, N^e(int)={n_e_int}, "
        f"W(N^o)={w_opt:.4f}, W(N^e int)={w_e_int:.4f}, loss int={100 * (w_opt - w_e_int) / max_surplus:.2f}%, "
        f"W(N^e cont)={w_e_cont:.4f}, loss cont={100 * (w_opt - w_e_cont) / max_surplus:.2f}%"
    )
    # Check that N^o is the integer maximiser of W
    grid = np.arange(1, 60)
    print("   integer argmax of W:", grid[np.argmax(welfare(grid, f))],
          " profit at N^e int:", round(profit(n_e_int, f), 4), " at N^e int + 1:", round(profit(n_e_int + 1, f), 4))

# ---------------------------------------------------------------- figure
F = 1.0
N_OPT, N_E = 3.0, 7.0

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
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

n = np.linspace(1.0, 12.0, 400)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.9))

# Left panel: W(N) and N pi_N
ax1.plot(n, welfare(n, F), color=COLORS[0], lw=2)
ax1.plot(n, n * profit(n, F), color=COLORS[1], lw=2)
ax1.axhline(max_surplus, color=INK2, lw=0.8, ls=(0, (4, 3)))
ax1.text(12.0, max_surplus + 0.6, r"maximal surplus $\frac{(a-c)^2}{2b} = 32$", ha="right", va="bottom",
         fontsize=8, color=INK2)
ax1.text(11.0, welfare(11.0, F) - 1.4, r"$W(N)$", color=INK, fontsize=9, ha="center", va="top")
ax1.text(3.6, 3.6 * profit(3.6, F) + 1.0, r"$N\pi_N$", color=INK, fontsize=9, ha="left", va="bottom")
ax1.axhline(0.0, color=INK2, lw=0.8)
for x, lab in ((N_OPT, r"$N^o = 3$"), (N_E, r"$N^e = 7$")):
    ax1.plot([x], [welfare(x, F)], marker="o", ms=6, color=COLORS[0], mec="white", mew=1.2, zorder=4)
w3, w7 = welfare(N_OPT, F), welfare(N_E, F)
ax1.annotate("", xy=(N_E, w7), xytext=(N_E, w3),
             arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
ax1.plot([N_OPT, N_E], [w3, w3], color=INK2, lw=0.6, ls=":")
ax1.text(7.4, 28.2, r"$W(N^o) - W(N^e) = 2.5$", fontsize=8, color=INK, va="bottom")
ax1.text(N_OPT, w3 + 1.0, r"$N^o$", fontsize=9, ha="center", va="bottom", color=INK)
ax1.text(N_E, w7 - 1.2, r"$N^e$", fontsize=9, ha="center", va="top", color=INK)
ax1.set_xlim(1, 12)
ax1.set_ylim(-8, 36)
ax1.set_xticks(range(1, 13))
ax1.set_xlabel(r"Number of firms $N$")
ax1.set_ylabel("Surplus")
ax1.grid(axis="y", color=GRID, lw=0.6)

# Right panel: pi_N and W'(N)
ax2.plot(n, profit(n, F), color=COLORS[1], lw=2)
ax2.plot(n, welfare_prime(n, F), color=COLORS[2], lw=2)
ax2.axhline(0.0, color=INK2, lw=0.8)
ax2.text(4.6, profit(4.6, F) + 0.25, r"$\pi_N$ (private value)", fontsize=8.5, color=INK, va="bottom")
ax2.text(4.6, welfare_prime(4.6, F) - 0.35, r"$W'(N)$ (social value)", fontsize=8.5, color=INK, va="top")
ax2.plot([N_OPT], [0], marker="o", ms=6, color=COLORS[2], mec="white", mew=1.2, zorder=4)
ax2.plot([N_E], [0], marker="o", ms=6, color=COLORS[1], mec="white", mew=1.2, zorder=4)
ax2.text(N_OPT, 0.35, r"$N^o$", fontsize=9, ha="center", va="bottom", color=INK)
ax2.text(N_E, 0.35, r"$N^e$", fontsize=9, ha="center", va="bottom", color=INK)
ax2.set_xlim(1, 12)
ax2.set_ylim(-1.5, 4.0)
ax2.set_xticks(range(1, 13))
ax2.set_xlabel(r"Number of firms $N$")
ax2.set_ylabel("Value of one more entrant")
ax2.grid(axis="y", color=GRID, lw=0.6)

for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(OUT)
fig.savefig(HERE / "entry_welfare_preview.png", dpi=150)
print("wrote", OUT)
