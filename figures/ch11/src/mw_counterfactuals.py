"""Redraw of Miller and Weinberg (2017), Table X (counterfactual analysis, 2011, RCNL-1 demand).

Inputs are the published entries of Table X (Econometrica 85(6), p. 1788). Prices and brewer markups are volume-weighted means for the
12-pack flagship brands of ABI, Miller and Coors. Welfare entries are percent changes relative to the
"No Merger" scenario (v).

The script prints the derived quantities quoted in Section 11.8:
  * percent price changes relative to no merger,
  * price differences that isolate the unilateral, efficiency and coordinated effects,
  * implied marginal costs (price minus markup),
  * the consumer-surplus weight w in TS% = w CS% + (1 - w) PS% implied by each column.
and draws figures/ch11/mw_counterfactuals.pdf (fig:11-mwdecomp).
Run: python mw_counterfactuals.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "mw_counterfactuals.pdf"
OUT_PNG = HERE / "mw_counterfactuals_preview.png"
OUT_TXT = HERE / "mw_counterfactuals_output.txt"

cols = ["(i)", "(ii)", "(iii)", "(iv)", "(v)"]
price = {"ABI": [10.03, 10.14, 9.38, 9.55, 9.43],
         "Miller": [8.94, 9.37, 8.28, 8.72, 8.19],
         "Coors": [10.18, 10.85, 9.56, 10.22, 9.26]}
markup = {"ABI": [4.45, 4.56, 3.81, 3.97, 3.84],
          "Miller": [4.52, 4.32, 3.83, 3.63, 3.05],
          "Coors": [4.25, 4.06, 3.61, 3.41, 2.68]}
welfare = {"Consumer surplus": [-3.7, -5.3, -0.2, -2.1],
           "Producer surplus": [22.1, 19.1, 10.3, 8.2],
           "Total surplus": [1.3, -0.6, 1.8, -0.1]}

lines = []


def say(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    lines.append(s)


say("Percent price change relative to no merger (v):")
pct = {}
for b, v in price.items():
    v = np.array(v)
    pct[b] = 100 * (v[:4] / v[4] - 1)
    say(f"  {b:7s}", " ".join(f"{c} {x:+.1f}" for c, x in zip(cols[:4], pct[b])))

say("Price differences (dollars):")
for b, v in price.items():
    v = np.array(v)
    say(f"  {b:7s} unilateral (iv)-(v) {v[3]-v[4]:+.2f} | efficiencies (iii)-(iv) {v[2]-v[3]:+.2f} | "
        f"coordination (ii)-(iv) {v[1]-v[3]:+.2f} | coordination given efficiencies (i)-(iii) {v[0]-v[2]:+.2f} | "
        f"total (i)-(v) {v[0]-v[4]:+.2f}")

say("Implied marginal cost = price - markup:")
for b in price:
    mc = np.array(price[b]) - np.array(markup[b])
    say(f"  {b:7s}", " ".join(f"{c} {x:.2f}" for c, x in zip(cols, mc)))
    say(f"  {b:7s} markup / price:", " ".join(f"{c} {100*m/p:.1f}%" for c, m, p in
                                             zip(cols, markup[b], price[b])))

say("Consumer-surplus weight w solving TS% = w CS% + (1-w) PS%:")
for k in range(4):
    cs, ps, ts = welfare["Consumer surplus"][k], welfare["Producer surplus"][k], welfare["Total surplus"][k]
    w = (ps - ts) / (ps - cs)
    say(f"  {cols[k]}: w = {w:.3f}")

# ------------------------------------------------------------------ figure
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, YELLOW, GREEN = "#4a3aa7", "#eda100", "#008300"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "mathtext.fontset": "cm"})

scen = ["(i) coordinated, unilateral,\nefficiencies (observed)", "(ii) coordinated, unilateral,\nno efficiencies",
        "(iii) unilateral,\nefficiencies", "(iv) unilateral,\nno efficiencies"]
y = np.arange(4)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.7), sharey=True)
h = 0.25


def panel(ax, data, colors, title, xlabel, xlim):
    for k, ((name, vals), col) in enumerate(zip(data.items(), colors)):
        pos = y + (1 - k) * h
        ax.barh(pos, vals, height=h - 0.03, color=col, label=name, edgecolor="white", linewidth=0.8)
        for yy, v in zip(pos, vals):
            ax.text(v + (0.25 if v >= 0 else -0.25), yy, f"{v:+.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=6.8, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", ncol=1, frameon=False, fontsize=7.2, handlelength=1.2)


panel(axes[0], {b: pct[b] for b in ["ABI", "Miller", "Coors"]}, [BLUE, ORANGE, AQUA],
      "(a) Flagship 12-pack prices", "percent change relative to no merger", (-3, 24))
panel(axes[1], {"consumer (CS)": welfare["Consumer surplus"], "producer (PS)": welfare["Producer surplus"],
                "total (TS)": welfare["Total surplus"]}, [VIOLET, YELLOW, GREEN], "(b) Surplus", "percent change relative to no merger", (-9, 34))
axes[0].set_yticks(y)
axes[0].set_yticklabels(scen, fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=160)
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
say("wrote", OUT_PDF)
