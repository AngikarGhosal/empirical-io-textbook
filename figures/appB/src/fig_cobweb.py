"""Figure fig:B-cobweb (Appendix B, Section B.7.2).

Two cobweb diagrams for the fixed-point iteration x^{(h+1)} = Phi(x^{(h)}).
  Left:  Phi(x) = 1 + cos(x)/2, Lipschitz constant 1/2, started at x^{(0)} = 0. The iterates
         converge to the unique fixed point x* (computed here by bisection).
  Right: Phi(x) = 3 - (3/2) x, slope -3/2, fixed point x* = 6/5, started at x^{(0)} = 1. The
         iterates spiral away from x*.
The script prints the iterates and the fixed points so that the caption can quote them.

Run from Textbook/:  python figures/appB/src/fig_cobweb.py
Output: figures/appB/cobweb.pdf, figures/appB/src/cobweb_numbers.txt
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "cobweb.pdf"
PNG = HERE / "cobweb_preview.png"
TXT = HERE / "cobweb_numbers.txt"

# Book palette (same values as the other chapter scripts).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def iterate(phi, x0, n):
    xs = [x0]
    for _ in range(n):
        xs.append(phi(xs[-1]))
    return np.array(xs)


def cobweb_path(xs):
    """Vertices of the cobweb: (x0, 0) -> (x0, x1) -> (x1, x1) -> (x1, x2) -> ..."""
    px, py = [xs[0]], [0.0]
    for h in range(len(xs) - 1):
        px += [xs[h], xs[h + 1]]
        py += [xs[h + 1], xs[h + 1]]
    return np.array(px), np.array(py)


phi_left = lambda x: 1.0 + 0.5 * np.cos(x)
phi_right = lambda x: 3.0 - 1.5 * x

xstar_left = brentq(lambda x: phi_left(x) - x, 0.0, 2.0, xtol=1e-14)
xstar_right = 6.0 / 5.0

xs_left = iterate(phi_left, 0.0, 12)
xs_right = iterate(phi_right, 1.0, 5)

out(f"Left panel: Phi(x) = 1 + cos(x)/2, fixed point x* = {xstar_left:.6f}")
for h, x in enumerate(xs_left):
    out(f"  h = {h:2d}  x = {x:.6f}  |x - x*| = {abs(x - xstar_left):.3e}")
d10 = abs(xs_left[1] - xs_left[0])
out(f"  d(x1, x0) = {d10:.6f}; a priori bound at h = 10 with kappa = 1/2: "
    f"{0.5**10 / 0.5 * d10:.3e}; actual error {abs(xs_left[10] - xstar_left):.3e}")
out(f"  local slope at x*: Phi'(x*) = {-0.5*np.sin(xstar_left):.6f}")
out(f"Right panel: Phi(x) = 3 - 1.5 x, fixed point x* = {xstar_right}")
for h, x in enumerate(xs_right):
    out(f"  h = {h:2d}  x = {x:.6f}  |x - x*| = {abs(x - xstar_right):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))

# ---------------- left panel: contraction
ax = axes[0]
grid = np.linspace(-0.1, 2.1, 400)
ax.plot(grid, grid, color=INK2, lw=1.0, ls="--", label=r"$45^\circ$ line")
ax.plot(grid, phi_left(grid), color=BLUE, lw=2.0,
        label=r"$\Phi(x) = 1 + \frac{1}{2}\cos x$")
px, py = cobweb_path(xs_left)
ax.plot(px, py, color=ORANGE, lw=1.2)
ax.plot(xs_left[:3], np.zeros(3), "o", color=ORANGE, ms=4, clip_on=False, zorder=6)
for h, dx in zip(range(3), (8, 8, -9)):
    ax.annotate(rf"$x^{{({h})}}$", (xs_left[h], 0.0), textcoords="offset points",
                xytext=(dx, 5), ha="center", fontsize=10, color=INK)
ax.plot([xstar_left], [xstar_left], "o", color=INK, ms=5, zorder=5)
ax.annotate(r"$x^*$", (xstar_left, xstar_left), textcoords="offset points",
            xytext=(-18, -14), fontsize=11, color=INK)
ax.set_xlim(-0.1, 2.1)
ax.set_ylim(0.0, 2.1)
ax.set_xlabel(r"$x^{(h)}$")
ax.set_ylabel(r"$\Phi(x^{(h)})$")
ax.grid(color=GRID, lw=0.6)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=9, frameon=False, handlelength=1.6, columnspacing=1.0)

# ---------------- right panel: expansion
ax = axes[1]
grid = np.linspace(-0.5, 3.0, 400)
ax.plot(grid, grid, color=INK2, lw=1.0, ls="--", label=r"$45^\circ$ line")
ax.plot(grid, phi_right(grid), color=BLUE, lw=2.0,
        label=r"$\Phi(x) = 3 - \frac{3}{2}x$")
px, py = cobweb_path(xs_right)
ax.plot(px, py, color=ORANGE, lw=1.2)
ax.plot(xs_right[:4], np.zeros(4), "o", color=ORANGE, ms=4, clip_on=False, zorder=6)
for h, dx in zip(range(4), (9, 10, -10, 12)):
    ax.annotate(rf"$x^{{({h})}}$", (xs_right[h], 0.0), textcoords="offset points",
                xytext=(dx, 5), ha="center", fontsize=10, color=INK)
ax.plot([xstar_right], [xstar_right], "o", color=INK, ms=5, zorder=5)
ax.annotate(r"$x^*$", (xstar_right, xstar_right), textcoords="offset points",
            xytext=(7, 4), fontsize=11, color=INK)
ax.set_xlim(-0.5, 3.0)
ax.set_ylim(0.0, 3.0)
ax.set_xlabel(r"$x^{(h)}$")
ax.set_ylabel(r"$\Phi(x^{(h)})$")
ax.grid(color=GRID, lw=0.6)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=9, frameon=False, handlelength=1.6, columnspacing=1.0)

fig.tight_layout(w_pad=2.0)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(PNG, dpi=150, bbox_inches="tight")
TXT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT}")
