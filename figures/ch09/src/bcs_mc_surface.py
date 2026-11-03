"""Leg marginal cost surface of Berry, Carnall and Spiller (Figure fig:09-bcs-mc).

Redraws the segment (leg) marginal cost implied by the cost panel of Table 5 of
Berry, Carnall and Spiller (NBER WP 5561, 1996), columns II and III, and checks the
density derivative against their Table 7 (column III).

Leg marginal cost, in $100 per passenger (d = leg distance in 1,000 miles,
Q = leg density in 100,000 passengers per quarter, both directions):

  c(d, Q) = chi00 + chi10 d + chi20 d^2 + chi30 d^3 + chi11 d Q + chi21 d^2 Q
            + chi01 Q + chi02 Q^2 + chi03 Q^3 + chi12 d Q^2   (+ congestion terms, set to 0)

where chi00 is the NSEGS coefficient (the per-leg constant). The product-level constant
(the mean of omega) is not reported in Table 5, so levels are shown net of it.

  dc/dQ = chi11 d + chi21 d^2 + chi01 + 2 chi02 Q + 3 chi03 Q^2 + 2 chi12 d Q.

Outputs: figures/ch09/bcs_mc_surface.pdf, figures/ch09/src/bcs_mc_numbers.txt
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "bcs_mc_surface.pdf"
TXT = HERE / "bcs_mc_numbers.txt"

# Table 5, cost panel (key: (power of d, power of Q)).
COEF = {
    "II": {(0, 0): -0.922, (1, 0): 2.965, (2, 0): -1.297, (3, 0): 0.168,
           (1, 1): -3.540, (2, 1): 1.045, (0, 1): 1.818, (0, 2): -0.252,
           (0, 3): -0.102, (1, 2): 0.564},
    "III": {(0, 0): -0.656, (1, 0): 3.181, (2, 0): -1.420, (3, 0): 0.178,
            (1, 1): -3.891, (2, 1): 1.270, (0, 1): 1.584, (0, 2): 0.164,
            (0, 3): -0.238, (1, 2): 0.524},
}

# Table 7 as printed (rows: distance in miles; columns: density).
T7_DIST = [250, 500, 1000, 1500, 2000]
T7_DENS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
T7 = np.array([
    [0.794, 0.808, 0.733, 0.568, -0.27, -0.980],
    [0.125, 0.204, 0.194, 0.096, -0.369, -1.190],
    [-0.737, -0.527, -0.406, -0.373, -0.576, -1.135],
    [-0.964, -0.623, -0.371, -0.207, -0.148, -0.446],
    [-0.556, -0.084, 0.299, 0.593, 0.914, 0.879],
])


def mc(col, d, q):
    c = COEF[col]
    return sum(v * d ** m * q ** n for (m, n), v in c.items())


def dmc_dq(col, d, q):
    c = COEF[col]
    return sum(n * v * d ** m * q ** (n - 1) for (m, n), v in c.items() if n >= 1)


def main():
    lines = []
    d_grid = np.array(T7_DIST) / 1000.0
    comp = np.array([[dmc_dq("III", d, q) for q in T7_DENS] for d in d_grid])
    diff = comp - T7
    lines.append("Table 7 replication (column III): computed dc/dQ")
    lines.append("dist\\dens " + " ".join(f"{q:8.2f}" for q in T7_DENS))
    for i, dm in enumerate(T7_DIST):
        lines.append(f"{dm:5d} mi  " + " ".join(f"{v:8.3f}" for v in comp[i]))
    lines.append("printed minus computed (absolute):")
    for i, dm in enumerate(T7_DIST):
        lines.append(f"{dm:5d} mi  " + " ".join(f"{abs(v):8.3f}" for v in diff[i]))
    mask = np.ones_like(diff, bool)
    mask[0, 4] = False
    lines.append(f"max abs difference excluding (250, 1.50): {np.abs(diff[mask]).max():.4f}")
    lines.append(f"cell (250, 1.50): printed {T7[0, 4]}, computed {comp[0, 4]:.4f}")

    # Selected levels and the zero-derivative locus.
    for col in ("II", "III"):
        lines.append(f"\nColumn {col}: c(0,0) = {mc(col, 0, 0):.3f}; c(1,1) = {mc(col, 1, 1):.3f};"
                     f" c(0.5,0.5) = {mc(col, 0.5, 0.5):.3f}; c(3,2.5) = {mc(col, 3, 2.5):.3f}")
        for dm in (0.25, 0.5, 1.0, 1.5, 2.0):
            qs = np.linspace(0.0, 2.5, 25001)
            g = dmc_dq(col, dm, qs)
            neg = qs[g < 0]
            if neg.size:
                # report the first sign change from + to - and any later return to +
                sign = np.sign(g)
                ch = np.where(np.diff(sign) != 0)[0]
                if ch.size:
                    lines.append(f"  d={dm:4.2f}: sign of dc/dQ changes at Q = "
                                 + ", ".join(f"{qs[k]:.3f}" for k in ch)
                                 + f" (dc/dQ at Q=0: {g[0]:+.3f})")
                else:
                    lines.append(f"  d={dm:4.2f}: dc/dQ<0 on all of [0,2.5]")
            else:
                lines.append(f"  d={dm:4.2f}: dc/dQ>=0 on [0,2.5]")
        # distance band with dc/dQ < 0 for all Q in [0.25, 2.0]
        ds = np.linspace(0.0, 3.0, 3001)
        qs = np.linspace(0.25, 2.0, 351)
        allneg = [dm for dm in ds if np.all(dmc_dq(col, dm, qs) < 0)]
        if allneg:
            lines.append(f"  dc/dQ<0 for every Q in [0.25,2.0] when d in [{min(allneg):.3f}, {max(allneg):.3f}]")
    TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ------------------------------------------------------------------ figure
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
    RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "hatch.color": "#0b0b0b", "hatch.linewidth": 0.45,
    })
    levels = np.arange(-1.0, 4.01, 0.5)
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_seq", RAMP)
    norm = mcolors.BoundaryNorm(levels, cmap.N, extend="both")

    q = np.linspace(0.0, 2.5, 251)
    d = np.linspace(0.0, 3.0, 301)
    QQ, DD = np.meshgrid(q, d)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.15), sharey=True)
    for ax, col in zip(axes, ("II", "III")):
        Z = mc(col, DD, QQ)
        G = dmc_dq(col, DD, QQ)
        cf = ax.contourf(QQ, DD, Z, levels=levels, cmap=cmap, norm=norm, extend="both")
        ax.contourf(QQ, DD, np.where(G < 0, 1.0, np.nan), levels=[0.5, 1.5], colors="none",
                    hatches=["///"])
        ax.contour(QQ, DD, G, levels=[0.0], colors=INK, linewidths=1.2)
        ax.contour(QQ, DD, Z, levels=[0.0], colors=INK2, linewidths=0.8, linestyles=[(0, (2, 2))])
        ax.axhline(0.5, color="white", lw=0.9, ls=(0, (4, 2)))
        if col == "III":
            DG, QG = np.meshgrid(d_grid, T7_DENS, indexing="ij")
            ax.plot(QG.ravel(), DG.ravel(), ls="none", marker="o", ms=3.2, mfc="white",
                    mec=INK, mew=0.7)
        ax.set_title(f"({'a' if col == 'II' else 'b'}) Table 5, column {col}", fontsize=9.5)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 3.0)
    axes[0].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    axes[1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    axes[0].set_ylabel("Leg distance $d_\\ell$ (1,000 miles)")
    axes[1].annotate(r"$\frac{\partial c_\ell}{\partial Q_\ell}<0$", xy=(1.2, 1.05), ha="center",
                     va="center", fontsize=11, color=INK,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    axes[0].annotate(r"$\frac{\partial c_\ell}{\partial Q_\ell}<0$", xy=(1.2, 1.05), ha="center",
                     va="center", fontsize=11, color=INK,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    for ax in axes:
        ax.annotate("500 miles", xy=(0.04, 0.5), xytext=(0, 2), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7.5, color="white")
    fig.supxlabel(r"Leg density $Q_\ell$ (100,000 passengers per quarter)", fontsize=9.5,
                  x=0.47, y=0.02)
    fig.subplots_adjust(left=0.085, right=0.86, bottom=0.17, top=0.9, wspace=0.08)
    cax = fig.add_axes([0.88, 0.17, 0.022, 0.73])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label("Leg marginal cost $c_\\ell$ (\\$100)")
    cb.outline.set_edgecolor(INK2)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
