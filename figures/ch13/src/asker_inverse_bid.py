"""Figure fig:13-asker-vb (Chapter 13, Section 13.8.2): Asker's two-bidder inverse bid function.

Equation (2) of Asker (2010):
    v(b) = b - (1/2) H_r(b) [1 - G(b)] / ( h_r(b) G(b) + H_r(b) g(b) ),
where G (density g) is the CDF of the rival ring member's knockout bid and H_r (density h_r) is the
CDF of the highest non-ring value in the target auction.

Lemma 2 example (Asker 2010, p. 732): H_r lognormal with mu = -2, sigma^2 = 0.08; G lognormal with
mu = -1, sigma^2 = 1. The script checks that v(b) is decreasing on part of the support.
Comparison example (ours): same G, H_r lognormal with mu = -2, sigma^2 = 0.5 (monotone).

Outputs: figures/ch13/asker_vb.pdf and figures/ch13/src/asker_inverse_bid_output.txt
Run from any directory:  python asker_inverse_bid.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
OUT_TXT = os.path.join(HERE, "asker_inverse_bid_output.txt")
OUT_PDF = os.path.join(FIGDIR, "asker_vb.pdf")

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}",
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.8,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"

lines = []


def say(s=""):
    print(s)
    lines.append(s)


def inverse_bid(b, muH, s2H, muG=-1.0, s2G=1.0):
    H = lognorm(s=np.sqrt(s2H), scale=np.exp(muH))
    G = lognorm(s=np.sqrt(s2G), scale=np.exp(muG))
    num = 0.5 * H.cdf(b) * (1.0 - G.cdf(b))
    den = H.pdf(b) * G.cdf(b) + H.cdf(b) * G.pdf(b)
    return b - num / den


b = np.linspace(0.02, 1.0, 20001)
v_l2 = inverse_bid(b, -2.0, 0.08)
v_ok = inverse_bid(b, -2.0, 0.5)

dec = np.diff(v_l2) < 0
lo, hi = b[1:][dec].min(), b[1:][dec].max()
say("Asker (2010) Lemma 2 example: H_r ~ LN(-2, 0.08), G ~ LN(-1, 1)")
say(f"  v(b) is decreasing for b in [{lo:.4f}, {hi:.4f}]")
i_peak = np.argmax(np.where((b > 0.1) & (b < 0.2), v_l2, -np.inf))
i_trough = np.argmin(np.where((b > 0.15) & (b < 0.3), v_l2, np.inf))
say(f"  local max of v(b): b = {b[i_peak]:.4f}, v = {v_l2[i_peak]:.4f}")
say(f"  local min of v(b): b = {b[i_trough]:.4f}, v = {v_l2[i_trough]:.4f}")
say(f"  overbidding check (v < b everywhere): {np.all(v_l2 < b)}")
say("Comparison example: H_r ~ LN(-2, 0.5), same G")
say(f"  v(b) strictly increasing on the grid: {np.all(np.diff(v_ok) > 0)}")
say(f"  overbidding check (v < b everywhere): {np.all(v_ok < b)}")
for bb in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
    say(f"  b = {bb:.2f}: v_Lemma2 = {inverse_bid(np.array([bb]), -2, 0.08)[0]:.4f}, "
        f"v_monotone = {inverse_bid(np.array([bb]), -2, 0.5)[0]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.75), constrained_layout=True)

ax1.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=":", label=r"$45^\circ$ line ($v = b$)")
ax1.plot(b, v_ok, color=BLUE, lw=1.6, label=r"$H_r$: $\sigma^2 = 0.5$ (monotone)")
ax1.plot(b, v_l2, color=ORANGE, lw=1.6, ls="--", label=r"$H_r$: $\sigma^2 = 0.08$ (Lemma 2)")
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.1, 1)
ax1.set_xlabel(r"knockout bid $b$")
ax1.set_ylabel(r"implied value $v(b)$")
ax1.set_title("(a) inverse bid functions")
ax1.legend(loc="upper left", frameon=False, handlelength=1.8)
ax1.grid(axis="y", color="#e4e3df", lw=0.6)

mask = (b > 0.08) & (b < 0.32)
ax2.axvspan(lo, hi, color="#f6d9cc", lw=0)
ax2.plot(b[mask], v_l2[mask], color=ORANGE, lw=1.6, ls="--")
vlev = 0.0535
ax2.axhline(vlev, color=GREY, lw=0.6, ls=":")
ax2.set_xlim(0.08, 0.32)
ax2.set_ylim(0.0, 0.11)
ax2.set_xlabel(r"knockout bid $b$")
ax2.set_title(r"(b) Lemma 2 example, detail")
ax2.text(0.5 * (lo + hi), 0.012, r"$\frac{dv}{db} < 0$", ha="center", fontsize=10, color=GREY)
ax2.text(0.226, 0.040, r"one value, three bids", fontsize=8.5, color=GREY)
ax2.grid(axis="y", color="#e4e3df", lw=0.6)

fig.savefig(OUT_PDF)
fig.savefig(os.path.join(HERE, "asker_vb_preview.png"), dpi=150)
say(f"saved {OUT_PDF}")
with open(OUT_TXT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
