"""Figure fig:B-affiliation (Appendix B, Section B.9.6).

Contours of the standard bivariate normal density with correlation 1/2, the
two unordered points z = (-1, 1) and z' = (1, -1), their join z v z' = (1, 1)
and their meet z ^ z' = (-1, -1). The script computes the two sides of the
affiliation inequality f(z v z') f(z ^ z') >= f(z) f(z') and prints them; the
caption of the figure reports the printed values.

Output: figures/appB/affiliation.pdf and affiliation_numbers.txt (run from any
directory).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "affiliation.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.8,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#0b0b0b"

RHO = 0.5


def f(z1, z2, rho=RHO):
    q = (z1**2 - 2 * rho * z1 * z2 + z2**2) / (1 - rho**2)
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(1 - rho**2))


z, zp = np.array([-1.0, 1.0]), np.array([1.0, -1.0])
join, meet = np.maximum(z, zp), np.minimum(z, zp)
lhs = f(*join) * f(*meet)
rhs = f(*z) * f(*zp)

g = np.linspace(-2.6, 2.6, 401)
Z1, Z2 = np.meshgrid(g, g)
fig, ax = plt.subplots(figsize=(3.6, 3.4), constrained_layout=True)
levels = np.linspace(0.01, 0.17, 9)
cf = ax.contourf(Z1, Z2, f(Z1, Z2), levels=np.concatenate([[0.0], levels, [0.19]]), cmap="Blues", alpha=0.85)
ax.contour(Z1, Z2, f(Z1, Z2), levels=levels, colors="white", linewidths=0.5)
cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
cb.set_label(r"$f(z_1,z_2)$")
cb.outline.set_linewidth(0.5)

# rectangle through the four points
ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=INK, lw=0.7, ls="--", zorder=3)
ax.scatter([z[0], zp[0]], [z[1], zp[1]], s=46, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=4)
ax.scatter([join[0], meet[0]], [join[1], meet[1]], s=46, color=INK, marker="s", edgecolor="white",
           linewidth=1.2, zorder=4)
ax.annotate(r"$z=(-1,1)$", z, xytext=(-1.05, 1.32), ha="center", fontsize=9)
ax.annotate(r"$z^{\prime}=(1,-1)$", zp, xytext=(1.05, -1.45), ha="center", fontsize=9)
ax.annotate(r"$z\vee z^{\prime}=(1,1)$", join, xytext=(1.05, 1.32), ha="center", fontsize=9)
ax.annotate(r"$z\wedge z^{\prime}=(-1,-1)$", meet, xytext=(-1.0, -1.45), ha="center", fontsize=9)
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-2.6, 2.6)
ax.set_aspect("equal")
ax.set_xlabel(r"$z_1$")
ax.set_ylabel(r"$z_2$")
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-2, -1, 0, 1, 2])

fig.savefig(OUT)
fig.savefig(HERE / "affiliation_preview.png", dpi=170)

lines = [
    f"correlation rho_12 = {RHO}",
    f"f(z v z') = f(1,1) = {f(*join):.5f}",
    f"f(z ^ z') = f(-1,-1) = {f(*meet):.5f}",
    f"f(z) = f(-1,1) = {f(*z):.5f}",
    f"f(z') = f(1,-1) = {f(*zp):.5f}",
    f"f(z v z') f(z ^ z') = {lhs:.5f}",
    f"f(z) f(z') = {rhs:.6f}",
    f"ratio = {lhs / rhs:.3f}  (exact exp(8/3) = {np.exp(8 / 3):.3f})",
]
(HERE / "affiliation_numbers.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("wrote", OUT)
