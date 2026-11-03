"""
Chapter 5, Figure fig:05-cgj-decomposition.

Decomposition of the consumer welfare loss (compensating variation, Rs. billion
per year) in Chaudhuri, Goldberg and Jia (2006), Table 8 (NBER WP 10159 rev.
July 2006, PDF p. 49; identical to the published Table 8).  Columns of Table 8
are cumulative:
    col1 = pure loss of variety                         E(u0,P1) - E(u0,P0)
    col2 = col1 + cross-segment expenditure switching   E(u0,P2) - E(u0,P0)
    col3 = col2 + within-segment price adjustment       E(u0,P') - E(u0,P0)
The three components drawn are
    variety   = col1                (positive, blue)
    switching = col2 - col1         (negative, orange, drawn left of zero)
    prices    = col3 - col2         (positive, aqua, stacked after variety)
so that variety + switching + prices = col3, the total loss printed at the bar end.

Output: figures/ch05/cgj_welfare_decomposition.pdf and a numbers file.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "font.size": 9,
})

scenarios = ["Only ciprofloxacin", "Only ofloxacin", "Cipro, oflo and norflo",
             "Cipro, oflo and sparflo", "All four domestic groups"]
T8 = np.array([
    [4.98, 4.92, 7.32],
    [0.08, 0.08, 0.23],
    [7.52, 7.40, 12.53],
    [6.14, 6.03, 10.58],
    [11.76, 11.35, 17.81],
])
variety = T8[:, 0]
switching = T8[:, 1] - T8[:, 0]
prices = T8[:, 2] - T8[:, 1]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"

lines = ["Decomposition of Table 8 (Rs. billion per year): our arithmetic",
         "scenario | variety | switching | price adjustment | total | shares of total"]
for s, v, w, p, t in zip(scenarios, variety, switching, prices, T8[:, 2]):
    lines.append("%-26s %6.2f %6.2f %6.2f %6.2f | %5.1f%% %5.1f%% %5.1f%%"
                 % (s, v, w, p, t, 100 * v / t, 100 * w / t, 100 * p / t))
with open(os.path.join(HERE, "cgj_welfare_decomposition_numbers.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(6.2, 3.6))
y = np.arange(len(scenarios))[::-1]
h = 0.56
gap = 0.06  # surface gap between stacked fills (data units)
ax.barh(y, variety - gap / 2, left=0, height=h, color=BLUE, label="Pure loss of variety")
ax.barh(y, prices - gap / 2, left=variety + gap / 2, height=h, color=AQUA,
        label="Price adjustment of remaining groups")
ax.barh(y, switching, left=0, height=h, color=ORANGE,
        label="Expenditure switching (reduces the loss)")
ax.axvline(0, color=INK2, lw=0.8)

for yi, v, w, p, t in zip(y, variety, switching, prices, T8[:, 2]):
    ax.text(v + p + 0.35, yi, "%.2f" % t, va="center", ha="left", color=INK, fontsize=8.5,
            fontweight="bold")
    if v > 1.5:
        ax.text(v / 2, yi, "%.2f" % v, va="center", ha="center", color="white", fontsize=8)
    if p > 1.5:
        ax.text(v + p / 2, yi, "%.2f" % p, va="center", ha="center", color=INK, fontsize=8)
    if abs(w) > 0.3:
        ax.text(w - 0.12, yi, ("%.2f" % w).replace("-", "−"), va="center", ha="right",
                color=INK2, fontsize=7.5)

ax.set_yticks(y)
ax.set_yticklabels(scenarios)
ax.set_xlim(-2.8, 20.5)
ax.set_xlabel("Consumer welfare loss (Rs. billion per year)")
ax.grid(axis="x", color="#e4e3df", lw=0.6)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
path = os.path.join(FIGDIR, "cgj_welfare_decomposition.pdf")
fig.savefig(path)
fig.savefig(os.path.join(HERE, "cgj_welfare_decomposition_preview.png"), dpi=130)
print("wrote", path)
