"""
Chapter 5, Figure fig:05-cgj-heatmap.

Heat map of the 7 x 7 block of unconditional price elasticities in the Northern
region, Chaudhuri, Goldberg and Jia (2006), Table 6(a) (NBER WP 10159 rev. July
2006, PDF p. 46; identical to the published Table 6A).  Rows: product group
whose quantity responds.  Columns: product group whose price changes.  The
diagonal (own-price elasticities) is masked from the color scale and printed in
gray.  Diverging blue (positive, substitutes) / red (negative) scale centred at
zero with a neutral gray midpoint; colors saturate at |e| = 1.2 so that the
domestic-domestic block is visible.  Every cell is labelled with its value and
the paper's significance mark (* 5 percent, dagger 10 percent).

Output: figures/ch05/cgj_table6a_heatmap.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "font.size": 9,
})

labels = ["Foreign\ncipro", "Foreign\nnorflo", "Foreign\noflo",
          "Domestic\ncipro", "Domestic\nnorflo", "Domestic\noflo", "Domestic\nsparflo"]
# values and significance marks copied from Table 6(a)
E = np.array([
    [-5.57, -0.13, -0.15, 4.01, 0.11, 0.11, 0.16],
    [-4.27, -0.45, -4.27, 3.50, -6.02, 4.51, 4.65],
    [-0.11, -0.10, -1.38, -0.09, 0.09, 0.23, 0.11],
    [0.18, 0.01, -0.01, -1.68, 0.08, 0.08, 0.10],
    [0.04, -0.03, 0.04, 0.58, -2.23, 0.42, 0.40],
    [0.05, 0.05, 0.11, 0.77, 0.74, -3.42, 0.74],
    [0.07, 0.04, 0.07, 1.15, 0.63, 0.63, -2.88],
])
D = "†"
SIG = [
    ["*", D, "*", "*", D, D, "*"],
    [D, "", D, D, "", "*", "*"],
    ["*", D, "*", "", D, "", "*"],
    ["*", "*", "", "*", "*", "*", "*"],
    ["*", "", "*", "*", "*", "*", "*"],
    ["*", "*", "", "*", "*", "*", "*"],
    ["*", "*", "*", "*", "*", "*", "*"],
]

cmap = LinearSegmentedColormap.from_list(
    "bluered",
    [(0.0, "#a12a2b"), (0.25, "#e34948"), (0.42, "#f2c0be"), (0.5, "#f0efec"),
     (0.58, "#b7d3f6"), (0.75, "#2a78d6"), (1.0, "#104281")])
VMAX = 1.2
norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0.0, vmax=VMAX)

n = E.shape[0]
M = np.ma.masked_array(E, mask=np.eye(n, dtype=bool))

fig, ax = plt.subplots(figsize=(6.2, 4.6))
im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")

for i in range(n):
    for j in range(n):
        v = E[i, j]
        txt = ("%.2f" % v).replace("-", "−") + SIG[i][j]
        if i == j:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white",
                                   edgecolor="#9a9994", lw=0.6, hatch="////", alpha=0.35))
            ax.text(j, i, txt, ha="center", va="center", color="#52514e",
                    fontsize=8.5, style="italic")
        else:
            c = min(abs(v), VMAX) / VMAX
            color = "white" if c > 0.62 else "#0b0b0b"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8.5)

# white 2pt gaps between cells
for k in range(n + 1):
    ax.axhline(k - 0.5, color="white", lw=1.6)
    ax.axvline(k - 0.5, color="white", lw=1.6)
# foreign / domestic block separators
ax.axhline(2.5, color="#52514e", lw=1.0, ls="--")
ax.axvline(2.5, color="#52514e", lw=1.0, ls="--")

ax.set_xticks(range(n))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_yticks(range(n))
ax.set_yticklabels([l.replace("\n", " ") for l in labels], fontsize=8.5)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_xlabel("Price that changes", fontsize=9, labelpad=6)
ax.set_ylabel("Quantity that responds", fontsize=9)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
cb.set_label("Cross-price elasticity (colors saturate beyond ±1.2)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
cb.outline.set_visible(False)

fig.tight_layout()
path = os.path.join(FIGDIR, "cgj_table6a_heatmap.pdf")
fig.savefig(path)
fig.savefig(os.path.join(HERE, "cgj_table6a_heatmap_preview.png"), dpi=130)
print("wrote", path)
