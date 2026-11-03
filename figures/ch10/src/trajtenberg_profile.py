"""Time profile of the gains from innovation in CT scanners and of CT R&D
(Figure fig:10-ct-profile, Section sec:10-trajtenberg).

Data: Trajtenberg (1985), NBER Working Paper 1724 (the working-paper version of Trajtenberg 1989,
JPE 97(2)); every number was read from the working paper.

  Table 11, column (1): incremental ex ante gains per buyer, Delta W_t, $M (1982 dollars).
      1973 is the indirect estimate K0 / k = 2.99 from the diffusion equation (Table 10);
      1974-1982 equal the incremental ex ante overall gains of Table 9 (periods 1973-74 ... 1981-82).
  Table 11, column (9): total gains TW_t, $M.
  Table 12, columns (1)-(2): R&D by U.S. firms and total R&D on CT, constant 1982 $M.
      (The 1968-71 entry is a four-year aggregate and is not plotted.)

The 3-year centered moving average of the incremental gains is our computation (Trajtenberg's
Figure 2 also shows a 3-year moving average). The script also recomputes the capitalized benefit and
cost sums behind the rate of return (Trajtenberg 1985, pp. 72-73 of the typed text).

Outputs: figures/ch10/trajtenberg_profile.pdf, figures/ch10/src/trajtenberg_profile_numbers.txt,
         figures/ch10/src/trajtenberg_profile_preview.png
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "trajtenberg_profile.pdf"
TXT = HERE / "trajtenberg_profile_numbers.txt"

YEARS = np.arange(1973, 1983)
DW = np.array([2.9900, 8.7126, 1.5090, 4.7758, .9399, .1154, .1397, .0743, .1844, .1951])
TW = np.array([638.3, 6926.3, 1502.9, 5959.1, 997.1, 78.6, 73.2, 30.0, 79.3, 87.2])
RD_YEARS = np.arange(1972, 1982)
RD_US = np.array([.22, .82, 7.79, 28.63, 58.18, 46.64, 37.03, 33.70, 29.58, 22.58])
RD_TOT = np.array([5.42, 9.00, 22.62, 59.68, 96.08, 79.68, 64.33, 56.05, 46.40, 37.94])
RD_6871_TOT = 6.22          # 1968-71 aggregate, total R&D (no U.S. entry)

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
})


def main():
    ma = np.convolve(DW, np.ones(3) / 3, mode="valid")      # centered, 1974-1981
    ma_years = YEARS[1:-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.1), gridspec_kw={"wspace": 0.34})

    ax1.plot(YEARS, DW, color=BLUE, lw=1.6, marker="o", ms=3.5, label="Yearly", zorder=3)
    ax1.plot(ma_years, ma, color=INK2, lw=1.1, ls=(0, (4, 2.5)), marker=".", ms=3,
             label="3-year moving average", zorder=2)
    # Annotation text positions are chosen so that no label touches a curve.
    notes = {1973: ("first\nscanner", (1972.62, 0.45)),
             1974: ("second scanner:\nresolution doubled", (1974.5, 8.6)),
             1975: ("first body\nscanners", (1975.45, 5.55)),
             1976: ("body speed and\nnumber of models up", (1977.2, 3.2))}
    for yr, (txt, xy) in notes.items():
        y = DW[yr - 1973]
        ax1.annotate(txt, xy=(yr, y), xytext=xy, fontsize=6.8, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=2.5,
                                     relpos=(0.0, 0.0) if yr == 1975 else (0.5, 0.5)))
    ax1.set_xlim(1972.5, 1982.5)
    ax1.set_ylim(0, 9.5)
    ax1.set_xticks(range(1973, 1983, 2))
    ax1.set_ylabel("Incremental gain per buyer ($M, 1982)")
    ax1.set_title("(a) Incremental ex ante gains", fontsize=9.5, loc="left")
    ax1.legend(frameon=False, fontsize=7.2, loc="upper right", bbox_to_anchor=(1.03, 0.89))

    ax2.plot(YEARS, TW, color=BLUE, lw=1.6, marker="o", ms=3.5, label="Total gains")
    ax2.plot(RD_YEARS, RD_TOT, color=ORANGE, lw=1.4, marker="s", ms=3, label="Total R&D")
    ax2.plot(RD_YEARS, RD_US, color=ORANGE, lw=1.0, ls=(0, (4, 2.5)), label="R&D by U.S. firms")
    ax2.set_yscale("log")
    ax2.set_xlim(1971.5, 1982.5)
    ax2.set_xticks(range(1972, 1983, 2))
    ax2.set_ylim(0.15, 2e4)
    ax2.set_ylabel("$M, 1982 (log scale)")
    ax2.set_title("(b) Total gains and R&D", fontsize=9.5, loc="left")
    ax2.legend(frameon=False, fontsize=7.4, loc="lower right")

    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Year")

    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(HERE / "trajtenberg_profile_preview.png", dpi=130, bbox_inches="tight")

    # ------------------------------------------------------------ numbers used in the text
    r = 0.05
    disc = lambda yrs: (1 + r) ** (-(yrs - 1973.0))
    B = float((TW * disc(YEARS)).sum())
    C_us = float((RD_US * disc(RD_YEARS)).sum())
    C_tot = float((RD_TOT * disc(RD_YEARS)).sum()
                  + RD_6871_TOT / 4 * disc(np.arange(1968, 1972)).sum())
    lines = [
        f"Capitalized total gains 1973-82 at r=0.05 (1973 base): {B:.1f}",
        f"Capitalized U.S. R&D 1972-81: {C_us:.1f}",
        f"Capitalized total R&D 1968-81 (1968-71 spread evenly): {C_tot:.1f}",
        f"r*B/C U.S.: {r * B / C_us:.3f}   paper: 0.05*14813/214 = {0.05 * 14813 / 214:.3f}",
        f"r*B/C total: {r * B / C_tot:.3f}   paper: 0.05*14813/397 = {0.05 * 14813 / 397:.3f}",
        f"average of the two paper ratios: {(0.05 * 14813 / 214 + 0.05 * 14813 / 397) / 2:.3f}",
        f"Undiscounted sum of TW: {TW.sum():.1f}",
        f"TW 1973-77 / total R&D 1968-76: {TW[:5].sum():.1f} / "
        f"{RD_6871_TOT + RD_TOT[:5].sum():.2f} = {TW[:5].sum() / (RD_6871_TOT + RD_TOT[:5].sum()):.1f}",
        f"TW 1978-82 / total R&D 1977-81: {TW[5:].sum():.1f} / {RD_TOT[5:].sum():.2f} = "
        f"{TW[5:].sum() / RD_TOT[5:].sum():.2f}",
        f"Share of 1973-82 total gains in 1973-77: {TW[:5].sum() / TW.sum():.4f}",
        "3-year moving average (1974-81): " + ", ".join(f"{v:.3f}" for v in ma),
        f"K0/k with rounded Table 10 values: {0.074 / 0.025:.3f}; ceiling 1982: "
        f"{0.074 + 0.025 * 16.65:.4f}",
    ]
    TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
