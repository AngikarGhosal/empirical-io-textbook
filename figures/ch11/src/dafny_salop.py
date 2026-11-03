"""Salop-circle prices in Dafny's two hypothetical hospital markets (fig:11-dafny-prices).

Circle of circumference 1, unit mass of consumers, unit demand with value v, linear transport
cost t per unit distance, zero marginal cost. Hospital H sits at 1/2.
  Market 1: rivals R1 and R2 co-located at 0. Pre-merger they are undifferentiated Bertrand
            competitors and price at cost (0). Post-merger they act as one firm R at 0.
  Market 2: rivals at 1/6 and 5/6 (three firms evenly spaced). Post-merger one rival facility
            closes and the other stays at 1/6.
The script
  1. checks that the closed-form symmetric price profiles are Nash equilibria: given the other
     firms at their closed-form prices, each firm's best response (computed on a fine price grid
     with demand evaluated on a fine grid of consumers, allowing kinks and partial coverage)
     equals its closed-form price. (Best-response iteration is not used because best responses
     have slope -1 at demand kinks, so the iteration cycles there.)
  2. shows that Dafny's (2009, Appendix 1) post-merger formula v/2 for market 2 when
     3v/2 < t < 3v fails this check, and that the corrected piecewise form passes it:
        2v/5 + t/15   for 3v/2  < t < 18v/7  (compete on the short arc, monopoly on the long arc)
        v - t/6       for 18v/7 < t < 3v     (kink on the short arc)
        v/2           for t > 3v.
  3. reproduces the footnote-37 averages (v = 1): 66.7 percent (market 1) and 40 percent
     (market 2), the percentage gap between average post- and pre-merger prices over the range of
     t where the post-merger market is not a set of local monopolies (t < 2v and t < 3v/2).
  4. plots H's price against t in both markets (corrected market-2 post-merger curve).
Outputs: figures/ch11/dafny_salop_prices.pdf, figures/ch11/src/dafny_salop_numbers.txt
"""
from pathlib import Path

import numpy as np
from scipy.integrate import quad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "dafny_salop_prices.pdf"
TXT = HERE / "dafny_salop_numbers.txt"

V = 1.0
Z = (np.arange(60000) + 0.5) / 60000          # consumer locations
PGRID = np.linspace(0.0, V, 8001)


def circ_dist(z, x):
    d = np.abs(z - x)
    return np.minimum(d, 1.0 - d)


def best_response(t, locs, prices, own):
    """Profit-maximizing common price for the facilities in `own`, other prices fixed."""
    others = [V - p - t * circ_dist(Z, x) for k, (x, p) in enumerate(zip(locs, prices))
              if k not in own]
    other_best = np.max(np.stack(others), axis=0) if others else np.full(Z.shape, -np.inf)
    other_best = np.maximum(other_best, 0.0)          # no purchase gives zero surplus
    own_dist = np.min(np.stack([circ_dist(Z, locs[k]) for k in own]), axis=0)
    thresh = np.sort(V - t * own_dist - other_best)   # consumer buys iff p < thresh
    dem = 1.0 - np.searchsorted(thresh, PGRID, side="right") / len(thresh)
    return PGRID[np.argmax(PGRID * dem)]


def cf_m1_pre(t, v=V):
    return t / 4 if t < 8 * v / 3 else (2 * v - t / 2 if t < 3 * v else v / 2)


def cf_m1_post(t, v=V):
    return t / 2 if t < 4 * v / 3 else (v - t / 4 if t < 2 * v else v / 2)


def cf_m2_pre(t, v=V):
    return t / 3 if t < 2 * v else (v - t / 6 if t < 3 * v else v / 2)


def cf_m2_post_dafny(t, v=V):
    return t / 2 if t < 6 * v / 5 else (v - t / 3 if t < 3 * v / 2 else v / 2)


def cf_m2_post(t, v=V):
    if t < 6 * v / 5:
        return t / 2
    if t < 3 * v / 2:
        return v - t / 3
    if t < 18 * v / 7:
        return 2 * v / 5 + t / 15
    if t < 3 * v:
        return v - t / 6
    return v / 2


def eq_gap(t, market, merged, formula):
    """Max |best response - closed form| over the firms, all firms at the closed-form price."""
    p = formula(t)
    if market == 1 and not merged:
        locs, prices, groups = [0.5, 0.0, 0.0], [p, 0.0, 0.0], [{0}]  # rivals at cost
    elif market == 1:
        locs, prices, groups = [0.5, 0.0], [p, p], [{0}, {1}]
    elif not merged:
        locs, prices, groups = [0.5, 1 / 6, 5 / 6], [p, p, p], [{0}, {1}, {2}]
    else:
        locs, prices, groups = [0.5, 1 / 6], [p, p], [{0}, {1}]
    return max(abs(best_response(t, locs, prices, g) - p) for g in groups)


lines = ["Equilibrium check: max |best response - closed-form price| (grid step "
         f"{PGRID[1]-PGRID[0]:.6f})",
         "t     | M1 pre | M1 post | M2 pre | M2 post (Dafny) | M2 post (corrected)"]
tgrid = [0.4, 0.8, 1.2, 1.25, 1.4, 1.7, 1.9, 2.2, 2.5, 2.7, 2.9, 3.2]
for t in tgrid:
    g = [eq_gap(t, 1, False, cf_m1_pre), eq_gap(t, 1, True, cf_m1_post),
         eq_gap(t, 2, False, cf_m2_pre), eq_gap(t, 2, True, cf_m2_post_dafny),
         eq_gap(t, 2, True, cf_m2_post)]
    lines.append(f"{t:5.2f} | {g[0]:.4f} | {g[1]:.4f}  | {g[2]:.4f} | {g[3]:.4f}          | {g[4]:.4f}")

for name, pre, post, tmax in [("Market 1", cf_m1_pre, cf_m1_post, 2.0),
                              ("Market 2", cf_m2_pre, cf_m2_post, 1.5)]:
    pts = [4 / 3, 6 / 5]
    avg_pre = quad(pre, 0, tmax, points=pts)[0] / tmax
    avg_post = quad(post, 0, tmax, points=pts)[0] / tmax
    avg_pct = quad(lambda t: post(t) / pre(t) - 1, 1e-9, tmax, points=pts)[0] / tmax
    lines.append(f"{name}: t in (0,{tmax}): avg pre {avg_pre:.4f}, avg post {avg_post:.4f}, "
                 f"ratio-1 = {100*(avg_post/avg_pre-1):.1f}%, mean of pct increases = "
                 f"{100*avg_pct:.1f}%")
lines.append("dp/dv in kinked ranges: market 1 pre (8v/3<t<3v) = 2; market 2 pre (2v<t<3v) = 1; "
             "competitive (interior) ranges = 0")
TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# figure
ts = np.linspace(0.01, 3.4, 800)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
blue, orange = "#2a78d6", "#eb6834"
for ax, pre, post, title in [(axes[0], cf_m1_pre, cf_m1_post, "Market 1: co-located rivals"),
                             (axes[1], cf_m2_pre, cf_m2_post, "Market 2: evenly spaced")]:
    ax.plot(ts, [pre(t) for t in ts], color=blue, lw=2, label="Before the rivals merge")
    ax.plot(ts, [post(t) for t in ts], color=orange, lw=2, label="After the rivals merge")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"Transport cost $t$ (with $v=1$)", fontsize=10)
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 3.4)
axes[0].set_ylabel(r"Price of hospital $H$, $p_H^*$", fontsize=10)
axes[0].set_ylim(0, 0.75)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
