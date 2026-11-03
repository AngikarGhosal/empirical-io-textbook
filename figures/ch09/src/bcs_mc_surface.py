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

# Largest leg density (DENS, 100,000 passengers per quarter) and largest leg distance
# (DIST, 1,000 miles) in Table 1.
QMAX = 4.553
DMAX = 2.704

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


def dq_coeffs(col, d):
    """Coefficients (a0, a1, a2) of dc/dQ = a0 + a1 Q + a2 Q^2 at distance d."""
    c = COEF[col]
    a0 = c[(0, 1)] + c[(1, 1)] * d + c[(2, 1)] * d ** 2
    a1 = 2 * c[(0, 2)] + 2 * c[(1, 2)] * d
    a2 = 3 * c[(0, 3)]
    return a0, a1, a2


def upper_root(col, d):
    """Larger real root of dc/dQ in Q, or nan if the roots are complex."""
    a0, a1, a2 = dq_coeffs(col, d)
    r = [x.real for x in np.roots([a2, a1, a0]) if abs(x.imag) < 1e-12]
    return max(r) if r else float("nan")


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

    # Selected levels and the zero-derivative locus. dc/dQ is a quadratic in Q, so its sign
    # changes are its real roots. They are reported on the whole observed density range
    # [0, QMAX], where QMAX = 4.553 is the largest leg density in Table 1, which extends
    # beyond both Table 7 (Q <= 2.0) and the plotted range (Q <= 2.5).
    lines.append(f"\nSign changes of dc/dQ are reported for Q in [0, {QMAX}] (largest leg density in Table 1).")
    for col in ("II", "III"):
        c = COEF[col]
        lines.append(f"\nColumn {col}: c(0,0) = {mc(col, 0, 0):.3f}; c(1,1) = {mc(col, 1, 1):.3f};"
                     f" c(0.5,0.5) = {mc(col, 0.5, 0.5):.3f}; c(3,2.5) = {mc(col, 3, 2.5):.3f}")
        for dm in (0.25, 0.5, 1.0, 1.5, 1.7, 2.0):
            a0, a1, a2 = dq_coeffs(col, dm)
            roots = np.roots([a2, a1, a0])
            roots = sorted(r.real for r in roots if abs(r.imag) < 1e-12 and 0.0 <= r.real <= QMAX)
            coeffs = f"dc/dQ = {a0:+.3f} {a1:+.3f} Q {a2:+.3f} Q^2"
            if roots:
                lines.append(f"  d={dm:4.2f}: {coeffs}; sign changes at Q = "
                             + ", ".join(f"{r:.5f}" for r in roots))
            else:
                state = "<0" if a0 < 0 else ">0"
                lines.append(f"  d={dm:4.2f}: {coeffs}; dc/dQ{state} on all of [0, {QMAX}]")
        # distance band with dc/dQ < 0 for all Q in a density range
        ds = np.linspace(0.0, 3.0, 3001)
        for lo, hi in ((0.25, 2.0), (0.0, QMAX)):
            qs = np.linspace(lo, hi, 2001)
            allneg = [dm for dm in ds if np.all(dmc_dq(col, dm, qs) < 0)]
            if allneg:
                lines.append(f"  dc/dQ<0 for every Q in [{lo},{hi}] when d in "
                             f"[{min(allneg):.3f}, {max(allneg):.3f}]")
        # long legs (d >= 1.5): distances at which dc/dQ turns negative at a high density,
        # i.e. the larger root of the quadratic (leading coefficient < 0) lies in (0, hi]
        dl = np.round(np.arange(1.5, DMAX + 1e-9, 0.001), 3)
        for hi in (2.0, 2.5, QMAX):
            hit = [dm for dm in dl if 0.0 < upper_root(col, dm) <= hi]
            if hit:
                lines.append(f"  d in [1.5, {DMAX}]: dc/dQ turns negative at some Q in (0, {hi}] when d in "
                             f"[{min(hit):.3f}, {max(hit):.3f}]")
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
