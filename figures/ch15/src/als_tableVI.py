"""Grouped bar chart of Athey, Levin and Seira (2011), Table VI (Northern sales).

Actual and predicted (competitive) sale prices, in 1982 dollars per thousand board feet, by the
number of participating mills, separately for sealed and open sales. Numbers are typed from
ALS Table VI (QJE 126(1), p. 251). No random numbers are used (no seed needed).
Output: figures/ch15/als_tableVI.pdf
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "als_tableVI.pdf"

groups = ["Zero mills", "One mill", "Two or more mills"]
data = {
    "Sealed bid sales": {"N": [181, 70, 88],
                         "Actual": [51.7, 66.8, 108.1],
                         "Predicted (bidding only)": [51.4, 64.6, 112.1],
                         "Predicted (bidding + entry)": [51.4, 66.9, 112.2]},
    "Open auction sales": {"N": [321, 150, 261],
                           "Actual": [49.8, 50.0, 87.5],
                           "Predicted (bidding only)": [50.5, 52.2, 98.5],
                           "Predicted (bidding + entry)": [47.1, 59.5, 98.0]},
}
series = ["Actual", "Predicted (bidding only)", "Predicted (bidding + entry)"]
colors = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3
hatches = ["", "////", "...."]                       # secondary encoding for print and CVD

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "hatch.linewidth": 0.6, "hatch.color": "white"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
x = np.arange(len(groups))
width = 0.26
for ax, (panel, d) in zip(axes, data.items()):
    for k, (s, c, h) in enumerate(zip(series, colors, hatches)):
        vals = d[s]
        bars = ax.bar(x + (k - 1) * (width + 0.02), vals, width, color=c, hatch=h,
                      edgecolor="white", linewidth=0.8, label=s, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.3, color="#222222", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n($N$ = {n})" for g, n in zip(groups, d["N"])], fontsize=8)
    ax.set_title(panel, fontsize=9.5)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_ylim(0, 135)
    ax.tick_params(axis="both", length=0)
axes[0].set_ylabel("Sale price (1982 dollars per mbf)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(HERE / "als_tableVI_preview.png", dpi=150, bbox_inches="tight")
print("saved", OUT)
