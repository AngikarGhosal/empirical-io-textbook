"""Annual Lerner indices and ASRC market share, U.S. sugar refining, 1887-1914.

Source of every number: Genesove and Mullin (1995), NBER Working Paper 5314,
Table 9 ("Lerner Indices by Year"), printed page 32 (PDF page 34). The numbers
were typed in by hand and checked twice against a 220-dpi render of the table.

Output: figures/ch03/gm_lerner_time.pdf (two panels with a shared year axis;
no dual y-axis). Shaded bands mark the two post-entry price-war regimes of the
paper's Table 13: January 1890 to February 1892 (Spreckels) and September 1898
to April 1900 (Arbuckle and Doscher).

Run from the Textbook folder:  python figures/ch03/src/gm_lerner_time.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "gm_lerner_time.pdf"

# Year: (unadjusted mean, unadjusted s.d., adjusted (log-linear) mean, adjusted s.d.)
LERNER = {
    1890: (0.00, 0.01, 0.00, 0.02),
    1891: (0.05, 0.04, 0.09, 0.08),
    1892: (0.11, 0.07, 0.19, 0.13),
    1893: (0.12, 0.03, 0.22, 0.08),
    1894: (0.10, 0.05, 0.17, 0.11),
    1895: (0.09, 0.03, 0.16, 0.04),
    1896: (0.09, 0.05, 0.14, 0.05),
    1897: (0.10, 0.01, 0.17, 0.06),
    1898: (0.03, 0.04, 0.05, 0.07),
    1899: (-0.02, 0.02, -0.05, 0.04),
    1900: (0.02, 0.04, 0.03, 0.06),
    1901: (0.08, 0.01, 0.14, 0.03),
    1902: (0.08, 0.03, 0.13, 0.06),
    1903: (0.07, 0.04, 0.12, 0.07),
    1904: (0.04, 0.04, 0.07, 0.08),
    1905: (0.06, 0.03, 0.11, 0.06),
    1906: (0.05, 0.03, 0.09, 0.07),
    1907: (0.06, 0.03, 0.11, 0.08),
    1908: (0.05, 0.01, 0.08, 0.02),
    1909: (0.02, 0.02, 0.04, 0.05),
    1910: (0.02, 0.01, 0.04, 0.02),
    1911: (0.04, 0.03, 0.08, 0.07),
    1912: (0.04, 0.02, 0.07, 0.05),
    1913: (0.03, 0.02, 0.05, 0.02),
    1914: (0.02, 0.02, 0.05, 0.05),
}
# ASRC market share in percent (Willett and Gray, via WP Table 9).
SHARE = {
    1887: 79.0, 1888: 75.0, 1889: 66.0, 1890: 67.7, 1891: 65.2, 1892: 91.0,
    1893: 85.7, 1894: 77.0, 1895: 76.6, 1896: 77.0, 1897: 71.4, 1898: 69.7,
    1899: 70.3, 1900: 70.1, 1901: 62.0, 1902: 60.9, 1903: 61.5, 1904: 62.3,
    1905: 58.1, 1906: 57.3, 1907: 56.8, 1908: 54.3, 1909: 50.4, 1910: 49.2,
    1911: 50.1, 1912: 45.5, 1913: 44.0, 1914: 43.0,
}

# Price-war regimes as decimal years (start of month).
WARS = [(1890.0, 1892 + 2 / 12), (1898 + 8 / 12, 1900 + 4 / 12)]

BLUE = "#2a78d6"     # categorical slot 1
ORANGE = "#eb6834"   # categorical slot 2
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
SHADE = "#e9e8e4"


def checks():
    """Print the summary statistics that the text quotes."""
    yrs = sorted(LERNER)
    adj = np.array([LERNER[y][2] for y in yrs])
    una = np.array([LERNER[y][0] for y in yrs])
    sh = np.array([SHARE[y] for y in yrs])
    print("mean ASRC share 1890-1914 (paper: 63.1):", round(sh.mean(), 2))
    print("mean of annual adjusted means (paper average .09):", round(adj.mean(), 3))
    print("mean of annual unadjusted means (paper average .05):", round(una.mean(), 3))
    print("corr(adjusted, share) 1890-1914:", round(np.corrcoef(adj, sh)[0, 1], 3))
    nowar = [y for y in yrs if y not in (1890, 1891, 1898, 1899, 1900)]
    a = np.array([LERNER[y][2] for y in nowar])
    s = np.array([SHARE[y] for y in nowar])
    print("corr(adjusted, share) excluding 1890-91, 1898-1900:", round(np.corrcoef(a, s)[0, 1], 3))
    b, c = np.polyfit(s, a, 1)
    print("OLS slope of annual adjusted index on share, non-war years:", round(b, 4), "intercept", round(c, 3))


def main():
    checks()
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.2, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.12},
    )
    yrs = np.array(sorted(LERNER))
    una = np.array([LERNER[y][0] for y in yrs])
    adj = np.array([LERNER[y][2] for y in yrs])

    for ax in (ax1, ax2):
        for a, b in WARS:
            ax.axvspan(a, b, color=SHADE, zorder=0, lw=0)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=1)
        ax.set_axisbelow(True)

    # Plot annual means at mid-year.
    x = yrs + 0.5
    ax1.axhline(0.0, color=INK2, lw=0.8, zorder=2)
    ax1.plot(x, adj, color=BLUE, lw=2.0, marker="o", ms=4.5, zorder=4,
             markeredgecolor="white", markeredgewidth=0.8,
             label=r"Elasticity-adjusted, $L_\varepsilon$ (log-linear demand)")
    ax1.plot(x, una, color=ORANGE, lw=2.0, marker="s", ms=4.2, zorder=3,
             markeredgecolor="white", markeredgewidth=0.8,
             label=r"Unadjusted, $L=\frac{P-c}{P}$")
    ax1.set_ylabel("Annual mean index")
    ax1.set_ylim(-0.12, 0.27)
    ax1.legend(loc="upper right", frameon=False, fontsize=8)
    ax1.text(1891.1, -0.112, "price war", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax1.text(1899.33, -0.112, "price war", ha="center", va="bottom", fontsize=7.5, color=INK2)

    syrs = np.array(sorted(SHARE))
    share = np.array([SHARE[y] for y in syrs])
    ax2.plot(syrs + 0.5, share, color=INK, lw=2.0, marker="o", ms=4.0,
             markeredgecolor="white", markeredgewidth=0.8)
    ax2.set_ylabel("ASRC share (%)")
    ax2.set_ylim(35, 100)
    ax2.set_xlim(1887, 1915)
    ax2.set_xticks(range(1887, 1916, 3))
    ax2.set_xlabel("Year")

    fig.align_ylabels((ax1, ax2))
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
