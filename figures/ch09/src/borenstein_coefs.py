"""Redraw the market-power coefficients of Table 2 of Borenstein (1989).

Data: Borenstein (1989, RAND Journal of Economics 20(3), Table 2, journal p. 357), equation (1)
estimated by 2SLS and by OLS for ln FR20, ln FR50 and ln FR80 (3,591 observations). Coefficients
and standard errors were checked against the PDF (p. 15). Intervals are coefficient +/- 1.96 s.e.

The script also prints the magnitude calculations used in the chapter text.

Output: figures/ch09/borenstein_coefs.pdf and figures/ch09/src/borenstein_coefs_numbers.txt.
"""
from math import exp, log
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "borenstein_coefs.pdf"
TXT = HERE / "borenstein_coefs_numbers.txt"

PCT = ["FR20", "FR50", "FR80"]
# (coefficient, standard error) for FR20, FR50, FR80
TSLS = {
    "RUTSHARE": [(0.1120, 0.0606), (0.0288, 0.0731), (0.2238, 0.0784)],
    "ORGSHARE": [(0.0345, 0.0549), (0.1254, 0.0661), (0.1191, 0.0710)],
    "RUTHERF": [(0.1559, 0.0696), (0.1822, 0.0839), (-0.1787, 0.0901)],
    "ORGHERF": [(-0.0699, 0.0623), (-0.0893, 0.0751), (0.0766, 0.0806)],
}
OLS = {
    "RUTSHARE": [(0.0899, 0.0270), (0.0596, 0.0326), (0.1259, 0.0349)],
    "ORGSHARE": [(0.0819, 0.0426), (0.1229, 0.0514), (0.1228, 0.0551)],
    "RUTHERF": [(0.1128, 0.0329), (0.1323, 0.0397), (-0.0120, 0.0425)],
    "ORGHERF": [(-0.0264, 0.0543), (-0.0660, 0.0656), (-0.0033, 0.0702)],
}
PANELS = ["RUTSHARE", "ORGSHARE", "RUTHERF"]

BLUE, ORANGE = "#2a78d6", "#eb6834"
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

fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.9), sharey=True)
x = np.arange(3)
off = 0.13
for ax, var in zip(axes, PANELS):
    for est, col, mk, mfc, dx, lab in [(TSLS, BLUE, "o", BLUE, -off, "2SLS"),
                                        (OLS, ORANGE, "s", "white", off, "OLS")]:
        b = np.array([c for c, _ in est[var]])
        se = np.array([s for _, s in est[var]])
        ax.errorbar(x + dx, b, yerr=1.96 * se, fmt=mk, color=col, mfc=mfc, mec=col, mew=1.6,
                    ms=6.5, elinewidth=1.6, capsize=0, label=lab, zorder=3)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\ln$ " + p for p in PCT], fontsize=8.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_title(var, fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Coefficient and 95% interval")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(w_pad=0.6)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(HERE / "borenstein_coefs_preview.png", dpi=130, bbox_inches="tight")

# ---------------------------------------------------------------- magnitudes used in the text
L = []
def pct(z):
    return 100.0 * (exp(z) - 1.0)

L.append("Route share, 70% vs 10% (difference 0.60):")
for p, (b, _) in zip(PCT, TSLS["RUTSHARE"]):
    L.append(f"  {p}: 0.60*{b:.4f} = {0.60 * b:.4f} log points, exact {pct(0.60 * b):.2f}%")
L.append("Per percentage point of route share: " + ", ".join(f"{b:.4f}%" for b, _ in TSLS["RUTSHARE"]))
L.append("Implied Cournot elasticity 1/phi3: " + ", ".join(f"{1 / b:.1f}" for b, _ in TSLS["RUTSHARE"]))
sd_org = 0.13054
L.append("ORGSHARE one s.d. (0.13054): " + ", ".join(
    f"{p} {sd_org * b:.4f}" for p, (b, _) in zip(PCT, TSLS["ORGSHARE"])))
L.append("ORGSHARE 0.5: " + ", ".join(f"{p} {0.5 * b:.4f}" for p, (b, _) in zip(PCT, TSLS["ORGSHARE"])))
L.append("ORGSHARE ratio high/low: FR80/FR20 = %.2f, FR50/FR20 = %.2f" % (
    TSLS["ORGSHARE"][2][0] / TSLS["ORGSHARE"][0][0], TSLS["ORGSHARE"][1][0] / TSLS["ORGSHARE"][0][0]))
L.append("Direct + max indirect (0.33*phi3) per unit ORGSHARE: " + ", ".join(
    f"{p} {TSLS['ORGSHARE'][k][0] + 0.33 * TSLS['RUTSHARE'][k][0]:.4f}" for k, p in enumerate(PCT)))
h0 = 0.5 ** 2 + 2 * 0.25 ** 2
h1 = 0.5 ** 2 + 0.5 ** 2
dh = h1 - h0
L.append(f"Merger: HHI {h0:.4f} -> {h1:.4f}, change {dh:.4f} = {dh / 0.22374:.2f} s.d.")
for p, (b, _) in zip(PCT, TSLS["RUTHERF"]):
    L.append(f"  {p}: {dh:.3f}*{b:+.4f} = {dh * b:+.4f}, exact {pct(dh * b):+.2f}%")
circ = 1.07538
L.append(f"ln(mean CIRC) = {log(circ):.5f}; effects: " + ", ".join(
    f"{p} {b * log(circ) * 100:.2f}%" for p, b in zip(PCT, [0.1458, 0.1272, 0.2062])))
L.append(f"One plane change vs nonstop at FR80: break-even ln CIRC = {0.0616 / 0.2062:.4f}, "
         f"CIRC = {exp(0.0616 / 0.2062):.3f}; linear approx CIRC-1 = {0.0616 / 0.2062:.3f}")
sd_pot = 0.15919
L.append("POTCOMP one s.d.: " + ", ".join(
    f"{p} {sd_pot * b:.4f}" for p, b in zip(PCT, [-0.0589, -0.0449, -0.0380])))
L.append("Table 3, RUTSHARE excluded, ORGSHARE 0.40 gap: " + ", ".join(
    f"{p} {0.40 * b:.4f} (exact {pct(0.40 * b):.1f}%)" for p, b in zip(PCT, [0.0742, 0.2117, 0.2986])))
L.append("t statistics (2SLS) RUTSHARE: " + ", ".join(f"{b / s:.2f}" for b, s in TSLS["RUTSHARE"]))
L.append("t statistics (2SLS) ORGSHARE: " + ", ".join(f"{b / s:.2f}" for b, s in TSLS["ORGSHARE"]))
L.append("t statistics (2SLS) RUTHERF: " + ", ".join(f"{b / s:.2f}" for b, s in TSLS["RUTHERF"]))
TXT.write_text("\n".join(L) + "\n")
print("\n".join(L))
print("wrote", OUT)
