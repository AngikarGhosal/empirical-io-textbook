"""Market areas in a two-dimensional characteristics space (fig:04-voronoi).

Consumers have ideal points nu uniformly distributed on the unit square and buy one unit
from the product j that minimizes p_j + tau * ||nu - x_j||^2 (quadratic mismatch cost, which
satisfies Caplin and Nalebuff's linearity assumption A1 after expanding the square, and a
uniform density satisfies A2). Product j's market area is

    A_j = {nu in [0,1]^2 : p_j + tau ||nu - x_j||^2 <= p_k + tau ||nu - x_k||^2 for all k},

an intersection of half-planes, because the two quadratic terms in nu cancel:
    2 tau nu.(x_k - x_j) <= p_k - p_j + tau(||x_k||^2 - ||x_j||^2).
With equal prices A_j is the Voronoi cell of x_j. With unequal prices it is a power-diagram cell.

Cells are computed exactly by clipping the unit square with the half-planes (Sutherland-Hodgman).
Product k is a competitor of j if A_j and A_k share an edge of positive length. Caplin and
Nalebuff's Proposition 9 bounds the average number of competitors per active firm by 6.

Output:
  figures/ch04/voronoi.pdf                 (fig:04-voronoi)
  figures/ch04/src/voronoi_numbers.txt     (numbers quoted in the text and caption)
"""
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, Voronoi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "voronoi.pdf"
OUT_PNG = HERE / "voronoi_preview.png"
OUT_TXT = HERE / "voronoi_numbers.txt"

TAU = 1.0
TOL = 1e-10
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def clip(poly, a, b):
    """Clip a convex polygon (list of vertices, counter-clockwise) by the half-plane a.x <= b."""
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i + 1) % n]
        fP, fQ = a @ P - b, a @ Q - b
        if fP <= TOL:
            out.append(P)
        if (fP < -TOL and fQ > TOL) or (fP > TOL and fQ < -TOL):
            t = fP / (fP - fQ)
            out.append(P + t * (Q - P))
    # drop duplicate consecutive vertices
    clean = []
    for v in out:
        if not clean or np.linalg.norm(v - clean[-1]) > 1e-12:
            clean.append(v)
    if len(clean) > 1 and np.linalg.norm(clean[0] - clean[-1]) <= 1e-12:
        clean.pop()
    return clean


def halfplane(j, k, x, p):
    a = 2.0 * TAU * (x[k] - x[j])
    b = p[k] - p[j] + TAU * (x[k] @ x[k] - x[j] @ x[j])
    return a, b


def cells(x, p, candidates=None):
    """Return list of cell polygons and the neighbour sets (shared edges of positive length)."""
    m = len(x)
    polys, nbrs, on_boundary = [], [], []
    for j in range(m):
        cand = range(m) if candidates is None else candidates[j]
        poly = [v.copy() for v in SQUARE]
        cons = []
        for k in cand:
            if k == j:
                continue
            a, b = halfplane(j, k, x, p)
            cons.append((k, a, b))
            poly = clip(poly, a, b)
            if len(poly) == 0:
                break
        polys.append(np.array(poly))
        nb = set()
        bnd = False
        if len(poly) >= 3:
            n = len(poly)
            for i in range(n):
                P, Q = poly[i], poly[(i + 1) % n]
                if np.linalg.norm(Q - P) < 1e-9:
                    continue
                # edge on the boundary of the square?
                for d in range(2):
                    for val in (0.0, 1.0):
                        if abs(P[d] - val) < 1e-9 and abs(Q[d] - val) < 1e-9:
                            bnd = True
                for k, a, b in cons:
                    scale = np.linalg.norm(a)
                    if abs(a @ P - b) < 1e-8 * scale and abs(a @ Q - b) < 1e-8 * scale:
                        nb.add(k)
        nbrs.append(nb)
        on_boundary.append(bnd)
    return polys, nbrs, on_boundary


def area(poly):
    if len(poly) < 3:
        return 0.0
    xs, ys = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def summary(polys, nbrs, on_boundary):
    active = [j for j in range(len(polys)) if area(polys[j]) > 1e-12]
    C = np.array([len(nbrs[j]) for j in active])
    interior = [j for j in active if not on_boundary[j]]
    Ci = np.array([len(nbrs[j]) for j in interior])
    # symmetry check of the neighbour relation
    sym = all((j in nbrs[k]) for k in active for j in nbrs[k])
    return dict(active=len(active), C=C, mean=C.mean(), interior=len(interior),
                mean_int=Ci.mean() if len(Ci) else np.nan, sym=sym)


lines = []
w = lines.append

# ----------------------------------------------------------------- the 40-product example
M_PROD = 40
rng = np.random.default_rng(1991)
x = rng.uniform(0.0, 1.0, size=(M_PROD, 2))
p_equal = np.full(M_PROD, 1.0)
p_unequal = 1.0 + rng.uniform(0.0, 0.012, size=M_PROD)

polysA, nbrsA, bndA = cells(x, p_equal)
polysB, nbrsB, bndB = cells(x, p_unequal)
SA = summary(polysA, nbrsA, bndA)
SB = summary(polysB, nbrsB, bndB)

# cross-check panel (a) against scipy's Voronoi for interior cells
vor = Voronoi(x)
scipy_nb = {j: set() for j in range(M_PROD)}
for (a_, b_) in vor.ridge_points:
    scipy_nb[a_].add(b_)
    scipy_nb[b_].add(a_)
mismatch = [j for j in range(M_PROD) if not bndA[j] and scipy_nb[j] != nbrsA[j]]

w("40 products, ideal points uniform on [0,1]^2, quadratic mismatch cost, tau = 1, seed 1991")
for lab, S in (("(a) equal prices", SA), ("(b) unequal prices 1 + U[0,0.012]", SB)):
    w(f"{lab}: active firms {S['active']}; average competitors per active firm {S['mean']:.3f}; "
      f"min {S['C'].min()}, max {S['C'].max()}; interior cells {S['interior']}, "
      f"their average {S['mean_int']:.3f}; neighbour relation symmetric: {S['sym']}")
    vals, cnts = np.unique(S["C"], return_counts=True)
    w(f"   distribution of competitor counts: {dict(zip(vals.tolist(), cnts.tolist()))}")
w(f"scipy Voronoi cross-check of interior cells in (a): mismatches {len(mismatch)}")
w(f"prices in (b): {p_unequal.min():.4f} to {p_unequal.max():.4f}")
w("")

# ----------------------------------------------------------------- convergence to six
w("Average number of competitors per firm, equal prices, uniform random locations")
w("   m   reps   all firms (mean over reps)   max over reps   interior cells (mean)")
rng2 = np.random.default_rng(257)
for m, reps in [(10, 400), (20, 400), (40, 400), (80, 200), (160, 100), (320, 50), (640, 25), (1280, 12)]:
    allm, intm, mx = [], [], 0.0
    for _ in range(reps):
        xx = rng2.uniform(0, 1, size=(m, 2))
        tri = Delaunay(xx)
        indptr, ind = tri.vertex_neighbor_vertices
        cand = [ind[indptr[j]:indptr[j + 1]] for j in range(m)]
        P, N, B = cells(xx, np.ones(m), candidates=cand)
        S = summary(P, N, B)
        allm.append(S["mean"])
        intm.append(S["mean_int"])
        mx = max(mx, S["mean"])
    w(f"{m:5d} {reps:6d}   {np.mean(allm):8.3f}                   {mx:8.3f}        {np.nanmean(intm):8.3f}")
w("")

OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

# ----------------------------------------------------------------- figure
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#ffffff"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})

cmin = min(SA["C"].min(), SB["C"].min())
cmax = max(SA["C"].max(), SB["C"].max())
levels = list(range(cmin, cmax + 1))
# spread the available ramp steps over the observed counts (ordinal: start no lighter than step 100)
idx = np.linspace(0, len(RAMP) - 1, len(levels)).round().astype(int)
color_of = {c: RAMP[i] for c, i in zip(levels, idx)}

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.55), constrained_layout=True)
for ax, polys, nbrs, S, title, pr in ((axes[0], polysA, nbrsA, SA, "(a) Equal prices", p_equal),
                                     (axes[1], polysB, nbrsB, SB, "(b) Unequal prices", p_unequal)):
    for j, poly in enumerate(polys):
        if area(poly) <= 1e-12:
            continue
        ax.add_patch(MplPolygon(poly, closed=True, facecolor=color_of[len(nbrs[j])],
                                edgecolor=SURFACE, linewidth=1.4, zorder=1))
    act = np.array([area(pp) > 1e-12 for pp in polys])
    ax.scatter(x[act, 0], x[act, 1], s=9, color=INK, zorder=3, linewidths=0)
    if (~act).any():
        ax.scatter(x[~act, 0], x[~act, 1], s=16, facecolor="white", edgecolor=INK, linewidths=0.8,
                   zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"Characteristic 1 ($\nu_1$)")
    ax.set_title(f"{title}\nmean competitors per firm = {S['mean']:.2f}", fontsize=9.5, color=INK)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
axes[0].set_ylabel(r"Characteristic 2 ($\nu_2$)")

# discrete legend for the number of competitors
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_of[c], edgecolor="none") for c in levels]
fig.legend(handles, [str(c) for c in levels], title="Competitors of the firm", ncol=len(levels),
           loc="lower center", bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=8.5,
           title_fontsize=8.5, handlelength=1.2, columnspacing=1.0)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print("wrote", OUT_PDF)
