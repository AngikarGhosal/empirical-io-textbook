"""Redraw Table 1 of Borenstein (1989): yields at six dominated airports.

Data: Borenstein (1989, RAND Journal of Economics 20(3), Table 1, journal p. 356), average fare per
mile (yield, cents per mile) by nonstop route distance, DB1A third quarter 1987. Every number was
checked against the article.

Panel (a): the "All" column for three groups at each hub (dominant carrier to/from the hub, the
same carrier elsewhere in its system, other carriers to/from the hub).
Panel (b): within each distance band, the percentage difference between the dominant carrier's
yield to/from its hub and its yield elsewhere, 100 * (hub / elsewhere - 1).

Output: figures/ch09/borenstein_yields.pdf, and a text summary of within-band comparisons
(figures/ch09/src/borenstein_yields_numbers.txt) used in the chapter text.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "borenstein_yields.pdf"
TXT = HERE / "borenstein_yields_numbers.txt"

BANDS = ["<200", "200-500", "500-1000", "1000-1500", "1500-2000", ">2000"]
ALL_DOMESTIC = [37.96, 25.82, 17.46, 11.42, 9.12, 7.16, 12.91]

# hub label, enplanement share, origination share,
# rows: dominant carrier not to/from hub, to/from hub, others to/from hub (6 bands + All).
# None marks the cell printed as a dash in the paper.
HUBS = [
    ("TWA\nSt. Louis", 82, 60,
     [34.37, 28.63, 18.50, 12.63, 9.32, 8.37, 10.86],
     [62.08, 33.71, 19.27, 14.70, 11.68, 7.79, 18.50],
     [None, 25.37, 17.98, 11.00, 8.61, 7.61, 14.41]),
    ("Northwest\nMinneapolis", 77, 65,
     [58.57, 32.76, 20.47, 12.59, 9.36, 7.85, 14.41],
     [60.67, 38.89, 22.13, 13.91, 10.82, 8.90, 17.55],
     [45.59, 33.15, 18.70, 11.10, 8.38, 7.88, 14.89]),
    ("USAir\nPittsburgh", 84, 63,
     [45.54, 28.39, 17.31, 11.59, 9.48, 7.98, 17.42],
     [53.49, 31.69, 15.22, 13.26, 9.60, 8.72, 20.32],
     [55.84, 29.04, 18.19, 12.70, 7.84, 6.87, 12.61]),
    ("American\nDallas", 60, 47,
     [48.62, 25.26, 20.56, 12.00, 9.89, 8.51, 12.15],
     [30.28, 20.08, 20.77, 15.49, 15.55, 9.11, 17.24],
     [29.50, 21.73, 18.43, 11.86, 11.32, 6.41, 14.51]),
    ("United\nO'Hare", 50, 43,
     [45.88, 24.36, 16.87, 12.15, 9.88, 7.50, 10.08],
     [64.94, 33.10, 20.06, 9.32, 8.78, 6.05, 14.19],
     [68.88, 33.30, 21.37, 10.35, 8.15, 7.14, 14.99]),
    ("Delta\nAtlanta", 55, 61,
     [36.73, 28.84, 18.62, 11.97, 10.10, 6.17, 12.68],
     [60.06, 36.75, 21.92, 13.88, 10.46, 8.23, 19.79],
     [50.34, 31.20, 19.17, 12.92, 8.99, 8.38, 17.44]),
]

# Reference categorical palette (fixed order) used across the book's figures.
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "hatch.linewidth": 0.6,
})

fig = plt.figure(figsize=(6.3, 6.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.62)
ax = fig.add_subplot(gs[0])
axh = fig.add_subplot(gs[1])

# ---------------------------------------------------------------- panel (a)
n = len(HUBS)
x = np.arange(n)
w = 0.26
series = [
    ("Dominant carrier, to/from hub", 4, BLUE, ""),
    ("Dominant carrier, elsewhere", 3, ORANGE, "////"),
    ("Other carriers, to/from hub", 5, GREEN, "...."),
]
for k, (lab, idx, col, hatch) in enumerate(series):
    vals = [h[idx][-1] for h in HUBS]
    ax.bar(x + (k - 1) * w, vals, width=w - 0.03, color=col, edgecolor="white",
           linewidth=0.8, hatch=hatch, label=lab, zorder=3)
    for xi, v in zip(x + (k - 1) * w, vals):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6, color=INK2,
                rotation=90, zorder=5,
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))
ax.axhline(ALL_DOMESTIC[-1], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2,
           label="All domestic flights (12.91)")
ax.set_xticks(x)
ax.set_xticklabels([f"{h[0]}\n{h[1]}% / {h[2]}%" for h in HUBS], fontsize=8)
ax.set_ylim(0, 25)
ax.set_ylabel("Yield (cents per mile)")
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=7.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2),
          handlelength=1.8, columnspacing=1.5)
ax.set_title("(a) Average yield over all distances", fontsize=9.5, loc="left", pad=34)

# ---------------------------------------------------------------- panel (b)
diff = np.array([[100.0 * (h[4][b] / h[3][b] - 1.0) for b in range(6)] for h in HUBS])
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ee", "#eb6834"])
norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=85)
axh.imshow(diff, cmap=cmap, norm=norm, aspect="auto")
for i in range(n):
    for j in range(6):
        v = diff[i, j]
        axh.text(j, i, f"{v:+.0f}".replace("-", "−"), ha="center", va="center", fontsize=7.8, color=INK)
axh.set_xticks(range(6))
axh.set_xticklabels(BANDS, fontsize=8)
axh.set_yticks(range(n))
axh.set_yticklabels([h[0].replace("\n", " ") for h in HUBS], fontsize=8)
axh.set_xlabel("Nonstop route distance (miles)")
axh.tick_params(length=0)
for s in axh.spines.values():
    s.set_visible(False)
axh.set_xticks(np.arange(-0.5, 6, 1), minor=True)
axh.set_yticks(np.arange(-0.5, n, 1), minor=True)
axh.grid(which="minor", color="white", lw=2)
axh.tick_params(which="minor", length=0)
axh.set_title(r"(b) Dominant carrier: yield to/from hub relative to elsewhere, "
              r"$100\left(\frac{\mathrm{hub}}{\mathrm{elsewhere}}-1\right)$",
              fontsize=9.5, loc="left")

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(HERE / "borenstein_yields_preview.png", dpi=130, bbox_inches="tight")

# ---------------------------------------------------------------- numbers used in the text
lines = []
pos = int((diff > 0).sum())
lines.append(f"Within-band cells with hub yield > elsewhere yield (dominant carrier): {pos} of {diff.size}")
for h, row in zip(HUBS, diff):
    lines.append(f"{h[0].replace(chr(10), ' ')}: " + ", ".join(f"{v:+.1f}" for v in row)
                 + f" | All: {100 * (h[4][-1] / h[3][-1] - 1):+.1f}%"
                 + f" | vs others All: {100 * (h[4][-1] / h[5][-1] - 1):+.1f}%")
# dominant carrier vs other carriers at the hub, within bands
cnt = tot = 0
for h in HUBS:
    for b in range(6):
        if h[5][b] is None:
            continue
        tot += 1
        cnt += h[4][b] > h[5][b]
    lines.append(f"  {h[0].replace(chr(10), ' ')} dominant vs others by band: "
                 + ", ".join("--" if h[5][b] is None else f"{100 * (h[4][b] / h[5][b] - 1):+.1f}"
                             for b in range(6)))
lines.append(f"Within-band cells with dominant > others at hub: {cnt} of {tot}")
TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("wrote", OUT)
