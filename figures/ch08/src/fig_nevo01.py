"""Redraw Table VIII of Nevo (2001): median predicted price-cost margins by conduct model.

Data: Nevo (2001, Econometrica 69(2)), Table VIII, journal p. 333 (PDF p. 27), checked against
the PDF. Margins are (p - mc)/p, medians over 27,862 brand-city-quarter observations, with 95%
confidence intervals for the medians in parentheses (analytical for the logit, 1,500 parametric
bootstrap draws for the full model). The logit column uses Table V column (ix); the full model
uses Table VI. The full model has no entries for the last two rows of the table.

Reference lines (Nevo 2001, p. 333 and Table III, p. 312): the accounting gross margin relative
to the retail price, 46.0%, and the lower bound from national-brand versus private-label price
gaps, about 31% (from Wongtrakool 1994, as reported by Nevo).

The script also writes the decomposition numbers quoted in the text.

Output: figures/ch08/nevo01_margins.pdf and figures/ch08/src/nevo01_numbers.txt
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "nevo01_margins.pdf"
TXT = HERE / "nevo01_numbers.txt"

ROWS = ["Single-product\nfirms", "Current ownership\n(25 brands)", "Joint ownership\n(25 brands)",
        "Current ownership\n(all brands)", "Monopoly or\nperfect collusion"]
# (median, lower, upper) in percent, Table VIII
LOGIT = [(33.6, 31.8, 35.6), (35.8, 33.9, 38.0), (41.9, 39.7, 44.4), (37.2, 35.2, 39.4),
         (54.0, 51.1, 57.3)]
FULL = [(35.8, 24.4, 46.4), (42.2, 29.1, 55.8), (72.6, 62.2, 97.2), None, None]
OBSERVED = 46.0
LOWER = 31.0

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})

fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(ROWS))
w = 0.36
for k, (vals, col, lab, dx) in enumerate([(LOGIT, ORANGE, "Logit (Table V, col. ix)", -w / 2 - 0.01),
                                          (FULL, BLUE, "Random coefficients (Table VI)", w / 2 + 0.01)]):
    for i, v in enumerate(vals):
        if v is None:
            ax.text(x[i] + dx, 2.0, "n.a.", ha="center", va="bottom", fontsize=7.5, color=INK2)
            continue
        m, lo, hi = v
        ax.bar(x[i] + dx, m, width=w, color=col, edgecolor="white", linewidth=1.0,
               label=lab if i == 0 else None, zorder=2)
        ax.errorbar(x[i] + dx, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK,
                    elinewidth=1.0, capsize=3, zorder=3)
        ax.text(x[i] + dx, hi + 1.2, f"{m:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK,
                zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

ax.axhline(OBSERVED, color=INK, linestyle="--", linewidth=1.0, zorder=1,
           label="Accounting margin, 46.0 (Table III)")
ax.axhline(LOWER, color=INK2, linestyle=":", linewidth=1.2, zorder=1,
           label="Private-label lower bound, 31")

ax.set_xticks(x)
ax.set_xticklabels(ROWS, fontsize=8)
ax.set_ylabel("Median margin, percent of price")
ax.set_ylim(0, 105)
ax.set_xlim(-0.6, len(ROWS) - 0.4)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT)

lines = []
for name, vals in [("logit", LOGIT), ("full", FULL)]:
    single, current, joint = vals[0][0], vals[1][0], vals[2][0]
    lines.append(f"{name}: differentiation (single) = {single:.1f}")
    lines.append(f"{name}: portfolio effect (current - single) = {current - single:.1f} points")
    lines.append(f"{name}: collusion bound (joint - current) = {joint - current:.1f} points")
    lines.append(f"{name}: share of current margin due to differentiation = {single / current:.3f}")
# Retail-price basis of the Census margin: manufacturer price is 80.0% of the retail price
# (Table III), so a margin of m percent of the manufacturer price is 0.8 m percent of retail price.
lines.append(f"Census 64.4% of manufacturer price on retail basis = {0.80 * 64.4:.1f}%")
lines.append(f"Check: 57.5% of manufacturer price on retail basis = {0.80 * 57.5:.1f}% (Table III: 46.0)")
# Logit implied combined shares from the closed-form markup 1/(alpha p (1 - S_f)):
# ratio of joint to single margins gives (1 - s_j)/(1 - S). Ratio of medians only.
r25 = LOGIT[2][0] / LOGIT[0][0]
rall = LOGIT[4][0] / LOGIT[0][0]
lines.append(f"logit ratio joint25/single = {r25:.3f}; implied 1 - S25 = {1 / r25:.3f}")
lines.append(f"logit ratio monopoly/single = {rall:.3f}; implied 1 - S_all = {1 / rall:.3f}")
TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"wrote {OUT}")
