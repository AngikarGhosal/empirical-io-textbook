"""People Express entry: price changes by entry cohort (Figure fig:09-wc-did).

Source: Whinston and Collins (1992), Tables 3 and 4 (journal pp. 459-460).
Each cell is the cross-market mean of the year-to-year percentage change in an
incumbent outcome, with the cross-market standard deviation in parentheses.
Cohorts: markets entered in 1984 (n = 7) and markets entered in 1985 (n = 8).

The script
  1. prints the difference-in-differences contrasts (treated cohort minus
     not-yet-treated or already-treated cohort) for every outcome and period;
  2. recomputes the two-sample t statistics (13 degrees of freedom) under
     three conventions and compares them with the printed values;
  3. draws the figure: grouped bars of the mean percentage change in the mean
     coach price, by period and cohort, with whiskers of +/- one standard error
     of the cohort mean (reported standard deviation divided by sqrt(n)).

Output: figures/ch09/wc_did.pdf and figures/ch09/src/wc_did_numbers.txt
"""
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "wc_did.pdf"
TXT = HERE / "wc_did_numbers.txt"

N84, N85 = 7, 8
PERIODS = ("1984-85", "1985-86")

# (mean, sd) for cohort 1984 and cohort 1985, by outcome and period;
# printed t statistic last.  Table 3 = Newark incumbents, Table 4 = other airport pairs.
TABLES = {
    "Table 3 (Newark incumbents)": {
        "mean price": {"1984-85": ((-33, 26), (2, 24), 2.5), "1985-86": ((0, 15), (-35, 21), 3.5)},
        "sales quantity": {"1984-85": ((108, 101), (36, 40), 0.6),
                           "1985-86": ((12, 22), (86, 107), 0.6)},
        "sd of price": {"1984-85": ((-28, 21), (27, 48), 2.6), "1985-86": ((21, 29), (1, 44), 1.0)},
    },
    "Table 4 (other airport-pair incumbents)": {
        "mean price": {"1984-85": ((-14, 22), (9, 18), 2.1), "1985-86": ((1, 13), (-15, 12), 2.3)},
        "sales quantity": {"1984-85": ((12, 34), (-5, 16), 1.2), "1985-86": ((1, 27), (2, 21), 0.1)},
        "sd of price": {"1984-85": ((2, 28), (52, 42), 2.5), "1985-86": ((6, 16), (-12, 11), 2.6)},
    },
}


def t_pooled(m1, s1, m2, s2, n1=N84, n2=N85, ddof_sd=1):
    """Pooled two-sample t.  ddof_sd = 1: s are sample sds (divisor n - 1).
    ddof_sd = 0: s are population sds (divisor n), converted before pooling."""
    if ddof_sd == 0:
        v1, v2 = n1 * s1 ** 2 / (n1 - 1), n2 * s2 ** 2 / (n2 - 1)
    else:
        v1, v2 = s1 ** 2, s2 ** 2
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    return abs(m1 - m2) / np.sqrt(sp2 * (1 / n1 + 1 / n2))


def t_welch(m1, s1, m2, s2, n1=N84, n2=N85):
    return abs(m1 - m2) / np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)


def main():
    lines = []
    crit = stats.t.ppf(0.975, 13)
    lines.append(f"5% two-sided critical value, t(13): {crit:.3f}")
    for tab, outcomes in TABLES.items():
        lines.append("")
        lines.append(tab)
        lines.append(f"{'outcome':16s} {'period':8s} {'treated':>8s} {'control':>8s} {'DiD':>6s}"
                     f" {'t_pool':>7s} {'t_pool_n':>8s} {'t_welch':>7s} {'printed':>7s}")
        for out, per in outcomes.items():
            for p in PERIODS:
                (m84, s84), (m85, s85), tprint = per[p]
                # Treated cohort is the one entered during the period.
                if p == "1984-85":
                    treat, ctrl = m84, m85
                else:
                    treat, ctrl = m85, m84
                did = treat - ctrl
                lines.append(
                    f"{out:16s} {p:8s} {treat:8.0f} {ctrl:8.0f} {did:6.0f}"
                    f" {t_pooled(m84, s84, m85, s85):7.2f}"
                    f" {t_pooled(m84, s84, m85, s85, ddof_sd=0):8.2f}"
                    f" {t_welch(m84, s84, m85, s85):7.2f} {tprint:7.1f}")
    # Pooled DiD for the mean price (average of the two periods' contrasts).
    for tab in TABLES:
        mp = TABLES[tab]["mean price"]
        d1 = mp["1984-85"][0][0] - mp["1984-85"][1][0]
        d2 = mp["1985-86"][1][0] - mp["1985-86"][0][0]
        lines.append(f"{tab}: price DiD 1984-85 = {d1}, 1985-86 = {d2}, average = {(d1 + d2) / 2}")
    text = "\n".join(lines)
    print(text)
    TXT.write_text(text + "\n")

    # ------------------------------------------------------------------ figure
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
        "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "hatch.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), sharey=True)
    width = 0.36
    x = np.arange(2)
    panels = [("Table 3 (Newark incumbents)", "(a) Entered airport pair (Newark)"),
              ("Table 4 (other airport-pair incumbents)", "(b) Other airports in the city pair")]
    for ax, (tab, title) in zip(axes, panels):
        mp = TABLES[tab]["mean price"]
        for c, (label, color, n, off) in enumerate(
                [("Entered in 1984", BLUE, N84, -width / 2 - 0.01),
                 ("Entered in 1985", ORANGE, N85, width / 2 + 0.01)]):
            means = np.array([mp[p][c][0] for p in PERIODS], float)
            ses = np.array([mp[p][c][1] for p in PERIODS], float) / np.sqrt(n)
            for i in range(2):
                treated = (c == i)  # cohort 1984 treated in period 0, cohort 1985 in period 1
                ax.bar(x[i] + off, means[i], width, color=color if treated else "white",
                       edgecolor=color, linewidth=1.4,
                       label=label if i == 0 else None, zorder=2)
                ax.errorbar(x[i] + off, means[i], yerr=ses[i], color=INK2, lw=0.9,
                            capsize=3, zorder=3)
                va = "top" if means[i] < 0 else "bottom"
                ypos = means[i] - ses[i] - 1.5 if means[i] < 0 else means[i] + ses[i] + 1.5
                ax.text(x[i] + off, ypos, (f"{means[i]:+.0f}".replace("-", "−")) if means[i] != 0 else "0",
                        ha="center", va=va, fontsize=8.5, color=INK)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([r"1984 to 1985", r"1985 to 1986"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-52, 22)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Change in mean incumbent price (%)")
    # Legend: color = cohort, fill = entered during that period.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BLUE, edgecolor=BLUE, label="Entered in 1984"),
               Patch(facecolor=ORANGE, edgecolor=ORANGE, label="Entered in 1985"),
               Patch(facecolor="white", edgecolor=INK2, label="Not entered during the period")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT)
    fig.savefig(HERE / "wc_did_preview.png", dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
