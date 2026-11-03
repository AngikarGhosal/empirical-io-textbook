"""Identified set for a linear marginal cost when conduct varies freely in [0, 1] (fig:03-rosen-bounds).

Worked example for Section 3.6.3. Inverse demand in market t is P = A_t - Q (slope -1, known).
True marginal cost mc = b0 + b1 Q with (b0, b1) = (2, 0.5). Conduct theta_t differs across
markets. The equilibrium solves A_t - Q - (b0 + b1 Q) = theta_t Q, so
    Q_t = (A_t - b0) / (1 + b1 + theta_t),  P_t = A_t - Q_t,  MR_t = P_t - Q_t.
If all we know is theta_t in [0, 1], each market gives the two linear inequalities
    MR_t <= b0 + b1 Q_t <= P_t,
and the identified set for (b0, b1) is the intersection of these half-planes. We also impose
the economic restrictions b0 >= 0 and b1 >= 0 (nonnegative, nondecreasing marginal cost).

Design A (figure): four markets with interior conduct.
Design B (text only): six markets, one of them competitive and one fully collusive.

Outputs: figures/ch03/rosen_bounds.pdf and figures/ch03/src/rosen_bounds_output.txt
"""
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "rosen_bounds.pdf"
OUT_TXT = HERE / "rosen_bounds_output.txt"

B0, B1 = 2.0, 0.5
DESIGN_A = [(8.0, 0.2), (10.0, 0.9), (12.0, 0.5), (14.0, 0.1)]                         # (A_t, theta_t)
DESIGN_B = [(6.0, 0.1), (8.0, 0.8), (10.0, 0.3), (12.0, 1.0), (14.0, 0.0), (16.0, 0.6)]


def equilibrium(markets):
    rows = []
    for a, th in markets:
        q = (a - B0) / (1.0 + B1 + th)
        p = a - q
        rows.append((a, th, q, p, p - q))
    return rows


def identified_set(rows, sign_restrictions=True):
    """Vertices (b1, b0) of {MR_t <= b0 + b1 Q_t <= P_t for all t} (plus b0, b1 >= 0)."""
    hs = []
    for (_, _, q, p, mr) in rows:
        hs.append([q, 1.0, -p])        # b0 + b1 q - p <= 0
        hs.append([-q, -1.0, mr])      # mr - b0 - b1 q <= 0
    if sign_restrictions:
        hs.append([-1.0, 0.0, 0.0])    # -b1 <= 0
        hs.append([0.0, -1.0, 0.0])    # -b0 <= 0
    hs = np.array(hs)
    # Chebyshev center as an interior point.
    a_mat, b_vec = hs[:, :2], -hs[:, 2]
    norms = np.linalg.norm(a_mat, axis=1)
    lp = linprog([0.0, 0.0, -1.0], A_ub=np.c_[a_mat, norms], b_ub=b_vec,
                 bounds=[(None, None), (None, None), (0.0, None)])
    if lp.x[2] < 1e-9:
        return None, hs
    inter = HalfspaceIntersection(hs, lp.x[:2])
    verts = inter.intersections
    return verts[ConvexHull(verts).vertices], hs


def report(name, rows, poly, lines):
    lines.append("%s" % name)
    lines.append("  market  A_t  theta_t     Q_t      P_t     MR_t   mc_t(true)")
    for i, (a, th, q, p, mr) in enumerate(rows, 1):
        lines.append("  %6d %5.1f %7.2f %8.4f %8.4f %8.4f %8.4f" % (i, a, th, q, p, mr, B0 + B1 * q))
    lines.append("  vertices (b1, b0): " + ", ".join("(%.3f, %.3f)" % (v[0], v[1]) for v in poly))
    lines.append("  b1 in [%.3f, %.3f];  b0 in [%.3f, %.3f]"
                 % (poly[:, 0].min(), poly[:, 0].max(), poly[:, 1].min(), poly[:, 1].max()))
    for i, (_, th, q, p, mr) in enumerate(rows, 1):
        mc_vals = poly[:, 1] + poly[:, 0] * q
        lo, hi = mc_vals.min(), mc_vals.max()
        lines.append("  market %d: mc in [%.3f, %.3f] (naive [%.3f, %.3f]); theta in [%.3f, %.3f], true %.2f"
                     % (i, lo, hi, max(mr, 0.0), p, (p - hi) / q, (p - lo) / q, th))
    lines.append("")


def main():
    lines = []
    rows_a = equilibrium(DESIGN_A)
    poly_a, _ = identified_set(rows_a)
    poly_a_nosign, _ = identified_set(rows_a, sign_restrictions=False)
    report("Design A (with b0, b1 >= 0)", rows_a, poly_a, lines)
    lines.append("Design A without sign restrictions: b1 in [%.3f, %.3f], b0 in [%.3f, %.3f]"
                 % (poly_a_nosign[:, 0].min(), poly_a_nosign[:, 0].max(),
                    poly_a_nosign[:, 1].min(), poly_a_nosign[:, 1].max()))
    lines.append("")
    rows_b = equilibrium(DESIGN_B)
    poly_b, _ = identified_set(rows_b)
    report("Design B (with b0, b1 >= 0)", rows_b, poly_b, lines)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ------------------------------------------------------------------ figure (design A)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    x = np.linspace(-0.4, 1.4, 200)
    for (_, _, q, p, mr) in rows_a:
        ax.plot(x, p - q * x, color="#1f4e79", lw=0.8)
        ax.plot(x, mr - q * x, color="#c55a11", lw=0.8, ls="--")
    ax.axhline(0.0, color="0.3", lw=0.6)
    ax.axvline(0.0, color="0.3", lw=0.6)
    ax.fill(poly_a[:, 0], poly_a[:, 1], color="#9dc3e6", alpha=0.85, zorder=3)
    ax.plot([B1], [B0], "k*", ms=8, zorder=4)
    ax.annotate(r"true $(b_1, b_0)$", xy=(B1, B0), xytext=(0.80, 4.3), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.text(0.62, 6.1, r"solid: $b_0 + b_1 Q_t = P_t$" "\n" r"dashed: $b_0 + b_1 Q_t = P_t - Q_t$",
            fontsize=7.5)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-1.0, 7.0)
    ax.set_xlabel(r"slope of marginal cost $b_1$")
    ax.set_ylabel(r"intercept $b_0$")
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
