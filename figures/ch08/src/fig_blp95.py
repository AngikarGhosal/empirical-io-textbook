"""Figures for the BLP (1995) case study (Section 8.12), redrawn from the paper's tables.

Outputs (in Textbook/figures/ch08/):
  blp95_semielas.pdf  heat map of Table VI (semi-elasticities)
  blp95_outside.pdf   Table VII, diversion to the outside good, logit vs BLP
  blp95_markup.pdf    Table VIII markups vs price, with the IV-logit markup band (p. 874)
  blp95_ownelas.pdf   Table V own-price elasticities vs price, with logit benchmarks (Table III)
The markup and own-elasticity plots are drawn at their printed size (full text width), so their
annotation fonts (8 pt) keep their size on the page.
Run:  python fig_blp95.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from blp95_data import (CARS, SHORT, PRICE, OWN_ELAS, SEMI, OUT_LOGIT, OUT_BLP, MARKUP,
                        ALPHA_OLS, ALPHA_IV, IVLOGIT_MARKUP_RANGE)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8c8b86", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})

p = np.array(PRICE)
S = np.array(SEMI)
n = len(CARS)


def semielas():
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    off = np.where(off <= 0, 5e-4, off)          # entries printed as 0.000
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    im = ax.imshow(off, cmap="Blues", norm=LogNorm(vmin=5e-4, vmax=12), aspect="equal")
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="#e9e8e4", zorder=2))
        ax.text(i, i, f"{S[i, i]:.0f}", ha="center", va="center", fontsize=6.5,
                color=INK, zorder=3)
        for j in range(n):
            if i != j and S[i, j] >= 0.5:
                ax.text(j, i, f"{S[i, j]:.1f}", ha="center", va="center", fontsize=5.5,
                        color="white" if S[i, j] > 2 else INK, zorder=3)
    ax.set_xticks(range(n))
    ax.set_xticklabels(SHORT, rotation=60, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(SHORT)
    ax.set_xlabel("car whose price rises by \\$1,000 (column $j$)")
    ax.set_ylabel("car whose share responds (row $i$)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("cross semi-elasticity (percent per \\$1,000)")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "blp95_semielas.pdf"))
    plt.close(fig)


def outside():
    y = np.arange(n)[::-1]
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.barh(y + h / 2, OUT_LOGIT, height=h, color=ORANGE, label="Logit", edgecolor="white",
            linewidth=0.8)
    ax.barh(y - h / 2, OUT_BLP, height=h, color=BLUE, label="BLP", edgecolor="white",
            linewidth=0.8)
    for yy, a, b in zip(y, OUT_LOGIT, OUT_BLP):
        ax.text(a + 0.8, yy + h / 2, f"{a:.1f}", va="center", fontsize=6.5, color=MUTED)
        ax.text(b + 0.8, yy - h / 2, f"{b:.1f}", va="center", fontsize=6.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(CARS)
    ax.set_xlim(0, 100)
    ax.set_xlabel("percent of those who leave the car who choose the outside good")
    ax.xaxis.grid(True, color="#e3e2de", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "blp95_outside.pdf"))
    plt.close(fig)


# The markup and own-elasticity plots are drawn at their printed size (the full text width of 6.2 in,
# included at width=\textwidth), so the font sizes below are the sizes on the page.
FINAL_W, FINAL_H = 6.2, 4.3
FS_LABEL, FS_TICK, FS_NOTE = 9.5, 8.5, 8.0


def _final_size_axes(ax):
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)


def _label_points(ax, xs, ys, place):
    """Write each car's name at an offset (points) from its marker; far labels get a leader line."""
    for c, x, yv in zip(CARS, xs, ys):
        dx, dy, ha = place[c]
        far = abs(dx) > 12 or abs(dy) > 12
        ax.annotate(c, (x, yv), xytext=(dx, dy), textcoords="offset points", fontsize=FS_NOTE,
                    color=MUTED, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5, shrinkA=1, shrinkB=3)
                    if far else None)


def markup():
    mk = np.array(MARKUP) / 1000.0
    lo, hi = (v / 1000.0 for v in IVLOGIT_MARKUP_RANGE)
    fig, ax = plt.subplots(figsize=(FINAL_W, FINAL_H))
    ax.axhspan(lo, hi, color=ORANGE, alpha=0.35, lw=0)
    ax.text(0.8, hi + 0.3, "IV logit, all models: \\$4,630 to \\$4,805",
            ha="left", va="bottom", fontsize=FS_NOTE + 0.5, color=INK)
    ax.plot(p, mk, color=BLUE, lw=1.2, alpha=0.6, zorder=2)
    ax.scatter(p, mk, s=30, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3,
               label="BLP (Table VIII)")
    # (dx, dy, ha): offset of the label in points and its horizontal alignment.
    place = {"Mazda 323": (-4, -2, "right"), "Nissan Sentra": (8, -9, "left"),
             "Ford Escort": (9, 1, "left"), "Chevy Cavalier": (-8, 20, "center"),
             "Honda Accord": (8, -10, "left"), "Ford Taurus": (-30, 24, "right"),
             "Buick Century": (7, -4, "left"), "Nissan Maxima": (7, -4, "left"),
             "Acura Legend": (7, -11, "left"), "Lincoln Town Car": (7, -3, "left"),
             "Cadillac Seville": (7, -3, "left"), "Lexus LS400": (7, -3, "left"),
             "BMW 735i": (-7, 6, "right")}
    _label_points(ax, p, mk, place)
    ax.set_xlabel("price (thousands of 1983 dollars)")
    ax.set_ylabel("markup $p-mc$ (thousands of 1983 dollars)")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 12)
    ax.yaxis.grid(True, color="#e3e2de", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=FS_TICK)
    _final_size_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "blp95_markup.pdf"))
    plt.close(fig)


def ownelas():
    eps = np.array(OWN_ELAS)
    grid = np.linspace(0, 40, 200)
    fig, ax = plt.subplots(figsize=(FINAL_W, FINAL_H))
    ax.plot(grid, ALPHA_IV * grid, color=ORANGE, lw=2, label=r"IV logit: $0.216\,p$")
    ax.plot(grid, ALPHA_OLS * grid, color=ORANGE, lw=1.4, ls="--",
            label=r"OLS logit: $0.089\,p$")
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    ax.text(39.5, 1.15, "unit elasticity", ha="right", fontsize=FS_NOTE, color=MUTED)
    ax.scatter(p, eps, s=30, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3,
               label="BLP (Table V)")
    # (dx, dy, ha): offset of the label in points and its horizontal alignment.
    place = {"Mazda 323": (-4, -3, "right"), "Nissan Sentra": (-4, 30, "center"),
             "Ford Escort": (7, -9, "left"), "Chevy Cavalier": (34, 30, "left"),
             "Honda Accord": (-7, 0, "right"), "Ford Taurus": (7, -9, "left"),
             "Buick Century": (7, 3, "left"), "Nissan Maxima": (7, 3, "left"),
             "Acura Legend": (6, -12, "left"), "Lincoln Town Car": (6, 9, "left"),
             "Cadillac Seville": (7, -1, "left"), "Lexus LS400": (7, 8, "left"),
             "BMW 735i": (0, 12, "center")}
    _label_points(ax, p, eps, place)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 9)
    ax.set_xlabel("price (thousands of 1983 dollars)")
    ax.set_ylabel(r"own-price elasticity $\varepsilon_j$")
    ax.yaxis.grid(True, color="#e3e2de", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=FS_TICK)
    _final_size_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "blp95_ownelas.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    semielas()
    outside()
    markup()
    ownelas()
    print("done")
