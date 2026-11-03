"""Hart-Scott-Rodino filings, Second Requests and enforcement actions, fiscal years 1991-2025.

Data:
  FY1991-2002: Ashenfelter and Hosken (2008, NBER WP 13859), Table 1 ("Mergers Filed",
               "Number of Second Requests", "Number of Enforcement Actions"), compiled there from
               Pautler (2003, Table 5) and the HSR annual reports.
  FY2003-2025: compiled from the FTC/DOJ HSR annual reports (table reproduced in the chapter,
               Table tab:11-hsr). The FY2003 and FY2004 rows agree with Ashenfelter and Hosken's
               Table 1, and the FY2006 row agrees with Farrell and Shapiro (2008 version, fn. 5).

Enforcement actions are the combined merger challenges of the FTC and the DOJ Antitrust Division as
printed in each year's joint annual report, with the corrected FY2022 report (FTC 23, DOJ 26).
Combined FY2015-2023 counts and their agency splits:
  FY2015 42, FY2016 47 (FTC 22), FY2017 39 (FTC 21, DOJ 18), FY2018 39 (FTC 22), FY2019 38 (FTC 21),
  FY2020 43 (FTC 28, DOJ 15), FY2021 32 (FTC 18), FY2022 49 (FTC 23, DOJ 26), FY2023 28 (FTC 16).
The FTC later revised its FY2017 count to 23, which would give a combined 41; we keep the report's 39.
Each report counts an action in the fiscal year of the agency's first public action, so an action
need not concern a transaction filed that year, and some actions concern transactions that were not
reportable (two of the 18 in FY2025). The ratios below are annual activity ratios, not the share of
a filing cohort that met enforcement.

Both ratios are computed as a percentage of all filings in the same year. (Ashenfelter and Hosken's own
percentage column divides by the smaller number of filings for which a Second Request could be issued.)

Two panels share the year axis (no dual y-axis): filings in the top panel, the two rates in the
bottom panel.

Output: figures/ch11/hsr_stats.pdf
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "hsr_stats.pdf"

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})

# year: (filed, second requests, enforcement actions)
AH = {
    1991: (1529, 64, 42), 1992: (1589, 44, 22), 1993: (1846, 71, 32), 1994: (2305, 73, 50),
    1995: (2816, 101, 60), 1996: (3087, 99, 57), 1997: (3702, 122, 57), 1998: (4728, 125, 85),
    1999: (4642, 111, 76), 2000: (4926, 98, 80), 2001: (2376, 70, 55), 2002: (1187, 49, 34),
    2003: (1014, 33, 36), 2004: (1454, 35, 24),
}
HSR = {
    2003: (1014, 33, 36), 2004: (1454, 35, 24), 2005: (1675, 50, 18), 2006: (1768, 45, 32),
    2007: (2201, 63, 34), 2008: (1726, 41, 37), 2009: (716, 31, 31), 2010: (1166, 42, 41),
    2011: (1450, 55, 37), 2012: (1429, 49, 44), 2013: (1326, 47, 38), 2014: (1663, 51, 33),
    2015: (1801, 47, 42), 2016: (1832, 54, 47), 2017: (2052, 51, 39), 2018: (2111, 45, 39),
    2019: (2089, 61, 38), 2020: (1637, 48, 43), 2021: (3520, 65, 32), 2022: (3152, 47, 49),
    2023: (1805, 37, 28), 2024: (2031, 59, 32), 2025: (2006, 41, 18),
}

# consistency checks between sources and of the table totals
for y in (2003, 2004):
    assert AH[y] == HSR[y]
tot = np.array(list(HSR.values())).sum(axis=0)
print("FY2003-2025 totals:", tot, "rates:", np.round(100 * tot[1:] / tot[0], 2))
assert tuple(tot) == (41624, 1097, 812)
ah_tot = np.array(list(AH.values())).sum(axis=0)
print("FY1991-2004 totals (AH Table 1):", ah_tot)
assert tuple(ah_tot) == (37201, 1095, 710)
for y, (f, s, e) in HSR.items():
    print(y, f, s, e, f"{100 * s / f:.1f}%", f"{100 * e / f:.1f}%")

data = dict(AH)
data.update(HSR)
years = np.array(sorted(data))
filed = np.array([data[y][0] for y in years])
sr = np.array([100 * data[y][1] / data[y][0] for y in years])
ea = np.array([100 * data[y][2] / data[y][0] for y in years])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.12))
src_ah = years <= 2002
ax1.bar(years[src_ah], filed[src_ah], width=0.72, color=COLORS[0], alpha=0.55, lw=0)
ax1.bar(years[~src_ah], filed[~src_ah], width=0.72, color=COLORS[0], lw=0)
ax1.set_ylabel("HSR filings")
ax1.set_ylim(0, 5600)
ax1.annotate("threshold raised to\n\\$50 million (2001)", xy=(2001.6, 1500), xytext=(2003.6, 4200),
             fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
ax1.annotate("financial crisis", xy=(2009, 800), xytext=(2010.2, 3000),
             fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
ax1.grid(axis="y", color=GRID, lw=0.6)
ax1.set_axisbelow(True)

ax2.plot(years, sr, color=COLORS[1], lw=2.0, ls="-", marker="o", ms=3.5, mec="white", mew=0.6,
         label="Second Requests")
ax2.plot(years, ea, color=COLORS[2], lw=2.0, ls="--", marker="s", ms=3.2, mec="white", mew=0.6,
         label="enforcement actions")
ax2.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=2, handlelength=2.6)
ax2.set_ylabel("ratio to filings (percent)")
ax2.set_ylim(0, 6.0)
ax2.grid(axis="y", color=GRID, lw=0.6)
ax2.set_axisbelow(True)
ax2.set_xlim(1990.3, 2025.7)
ax2.set_xticks(np.arange(1991, 2026, 2))
ax2.tick_params(axis="x", labelrotation=90)
ax2.set_xlabel("fiscal year")

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(HERE / "hsr_stats_preview.png", dpi=150, bbox_inches="tight")
print("saved", OUT)
