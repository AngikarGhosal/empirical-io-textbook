"""Hub premia of Berry, Carnall and Spiller (Figure fig:09-bcs-premia).

Grouped bars of the fare, marginal-cost and markup hub premia in the bottom panel of
Table 6 of Berry, Carnall and Spiller (NBER WP 5561, 1996), for small and large hub
markets and the two consumer types. The marginal-cost premium is the hub carrier's
cost ADVANTAGE, [MC(N/Hub) - MC(Hub)] / MC(N/Hub), as defined in the notes to Table 6.

The script also recomputes the premia from the rounded averages printed in Table 6
(the printed premia use unrounded averages) and the composition facts used in the text.

Outputs: figures/ch09/bcs_hubpremia.pdf, figures/ch09/src/bcs_hubpremia_numbers.txt
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "bcs_hubpremia.pdf"
TXT = HERE / "bcs_hubpremia_numbers.txt"

# Table 6, bottom panel, as printed (percent).
PRINTED = {
    ("Small", 1): {"Fare": 0.1, "MC": -1.1, "Markup": -0.8},
    ("Small", 2): {"Fare": 3.7, "MC": -4.5, "Markup": 3.3},
    ("Large", 1): {"Fare": 5.0, "MC": 16.8, "Markup": 29.9},
    ("Large", 2): {"Fare": 19.0, "MC": 26.1, "Markup": 63.4},
}
# Table 6, average fare and MC ($) for hub (H) and non-hub (N) carriers.
AVG = {
    ("Small", 1): {"FH": 206, "FN": 206, "MH": 96, "MN": 95},
    ("Small", 2): {"FH": 306, "FN": 295, "MH": 111, "MN": 106},
    ("Large", 1): {"FH": 215, "FN": 205, "MH": 91, "MN": 109},
    ("Large", 2): {"FH": 319, "FN": 268, "MH": 98, "MN": 133},
}
# Table 6: type shares of market ("Pct of Mkt") and split of each type across carriers ("Pct of Typ").
PCT_MKT = {"Small": (0.732, 0.268), "Large": (0.704, 0.296)}
PCT_TYP_HUB = {"Small": (0.400, 0.393), "Large": (0.623, 0.729)}


def main():
    out = []
    out.append("Premia recomputed from rounded Table 6 averages (percent):")
    for key, a in AVG.items():
        fare = 100 * (a["FH"] - a["FN"]) / a["FN"]
        mcadv = 100 * (a["MN"] - a["MH"]) / a["MN"]
        mk = 100 * ((a["FH"] - a["MH"]) - (a["FN"] - a["MN"])) / (a["FN"] - a["MN"])
        # decomposition as shares of the non-hub fare
        mk_share = 100 * ((a["FH"] - a["MH"]) - (a["FN"] - a["MN"])) / a["FN"]
        mc_share = 100 * (a["MN"] - a["MH"]) / a["FN"]
        lerner_h = 100 * (a["FH"] - a["MH"]) / a["FH"]
        lerner_n = 100 * (a["FN"] - a["MN"]) / a["FN"]
        out.append(f"{key}: fare {fare:6.2f} | MC adv {mcadv:6.2f} | markup {mk:6.2f} | "
                   f"printed {PRINTED[key]} | decomposition {mk_share:5.2f} - {mc_share:5.2f} = "
                   f"{mk_share - mc_share:5.2f} | Lerner H {lerner_h:4.1f} N {lerner_n:4.1f} | "
                   f"$ markup H {a['FH'] - a['MH']} N {a['FN'] - a['MN']}")
    out.append("\nComposition of hub-market passengers:")
    for m in ("Small", "Large"):
        t1, t2 = PCT_MKT[m]
        h1, h2 = PCT_TYP_HUB[m]
        hub = t1 * h1 + t2 * h2
        out.append(f"{m}: hub carriers carry {hub:.3f} of passengers; type-2 share among hub "
                   f"passengers {t2 * h2 / hub:.3f}; among non-hub {t2 * (1 - h2) / (1 - hub):.3f}")
    TXT.write_text("\n".join(out) + "\n")
    print("\n".join(out))

    # ------------------------------------------------------------------ figure
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
    })
    measures = ["Fare", "MC", "Markup"]
    labels = ["Fare\npremium", "Marginal-cost\nadvantage", "Markup\npremium"]
    x = np.arange(3)
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, m, title in zip(axes, ("Small", "Large"), ("(a) Small-hub markets", "(b) Large-hub markets")):
        v1 = [PRINTED[(m, 1)][k] for k in measures]
        v2 = [PRINTED[(m, 2)][k] for k in measures]
        b1 = ax.bar(x - w / 2 - 0.01, v1, w, color=BLUE, label="Type 1 (tourist)", zorder=3)
        b2 = ax.bar(x + w / 2 + 0.01, v2, w, color=ORANGE, label="Type 2 (business)", zorder=3)
        for bars, vals in ((b1, v1), (b2, v2)):
            for rect, v in zip(bars, vals):
                off = 2 if v >= 0 else -2
                ax.annotate(f"{v:.1f}".replace("-", "−"), xy=(rect.get_x() + rect.get_width() / 2, v),
                            xytext=(0, off), textcoords="offset points", ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
        ax.axhline(0, color=INK2, lw=0.8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Hub carrier vs. non-hub carriers (%)")
    axes[0].set_ylim(-10, 70)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
