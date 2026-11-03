"""Multiple Bertrand-Nash equilibria in a two-segment logit duopoly (ex:04-multiple-eq, fig:04-multiple-eq).

Model (Section 4.4.5 of the book):
  * Two single-product firms, constant marginal cost c = 1, market size 1.
  * Consumer i in segment g gets u_ij = delta - alpha_g p_j + eps_ij (j = 1, 2) and u_i0 = eps_i0,
    with eps type-I extreme value and delta = 3 for both products.
  * Segment A (mass 0.9) has alpha_A = 2 (price sensitive, mass market).
    Segment B (mass 0.1) has alpha_B = 0.2 (price insensitive, niche).
  * Share of product j: s_j(p) = sum_g w_g exp(delta - alpha_g p_j) / (1 + sum_k exp(delta - alpha_g p_k)).
  * Profit of firm j: pi_j = (p_j - c) s_j(p).

Search:
  1. Best response BR(p_other): global maximization of profit over own price on a fine grid on
     [1, 25] (step 0.0005), then local refinement (bounded Brent) around the best grid point.
  2. Symmetric equilibria: roots of BR(p) - p on [1, 25].
  3. All equilibria (including asymmetric ones): fixed points of BR(BR(p)) on [1, 25].
  4. The rival price at which the best response jumps: the p_other at which the two local maxima
     of profit give the same profit (bisection).

Outputs: figures/ch04/multiple_equilibria.pdf and figures/ch04/src/multiple_equilibria_output.txt
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "multiple_equilibria.pdf"
OUT_TXT = HERE / "multiple_equilibria_output.txt"

C = 1.0
DELTA = 3.0
W = np.array([0.9, 0.1])
ALPHA = np.array([2.0, 0.2])
PMIN, PMAX = 1.0, 25.0
GRID = np.arange(PMIN, PMAX + 1e-12, 0.0005)


def seg_shares(p1, p2):
    """Segment-level shares of product 1 (arrays broadcast over p1)."""
    p1 = np.asarray(p1, dtype=float)
    e1 = np.exp(DELTA - ALPHA[:, None] * p1[None, ...].reshape(1, -1))
    e2 = np.exp(DELTA - ALPHA * p2)[:, None]
    return e1 / (1.0 + e1 + e2)            # shape (2, n)


def share(p1, p2):
    return (W[:, None] * seg_shares(p1, p2)).sum(axis=0)


def profit(p1, p2):
    return (np.asarray(p1, dtype=float).reshape(-1) - C) * share(p1, p2)


def local_maxima(p2):
    """Grid local maxima of profit in own price (index list) for rival price p2."""
    pr = profit(GRID, p2)
    idx = np.where((pr[1:-1] > pr[:-2]) & (pr[1:-1] >= pr[2:]))[0] + 1
    out = []
    for i in idx:
        lo, hi = GRID[max(i - 2, 0)], GRID[min(i + 2, len(GRID) - 1)]
        r = minimize_scalar(lambda x: -profit(x, p2)[0], bounds=(lo, hi), method="bounded",
                            options={"xatol": 1e-12})
        out.append((r.x, -r.fun))
    return out


def best_response(p2):
    lm = local_maxima(p2)
    return max(lm, key=lambda t: t[1])[0]


def main():
    log = []

    # ---------------------------------------------------------------- best responses
    p2_grid = np.arange(PMIN, 12.0 + 1e-12, 0.01)
    br = np.array([best_response(x) for x in p2_grid])
    jumps = np.where(np.abs(np.diff(br)) > 1.0)[0]
    log.append(f"BR computed on {len(p2_grid)} rival prices in [1, 12]; jumps at grid index {jumps.tolist()}")

    # exact switching points: two local maxima with equal profit
    def gap(x):
        lm = sorted(local_maxima(x), key=lambda t: t[0])
        return lm[0][1] - lm[-1][1]      # low-price max minus high-price max
    switches = []
    for j in jumps:
        xs = brentq(gap, p2_grid[j], p2_grid[j + 1], xtol=1e-10)
        lmj = sorted(local_maxima(xs), key=lambda t: t[0])
        switches.append((xs, lmj))
        log.append(f"BR jumps at p_other = {xs:.4f}: between {lmj[0][0]:.4f} (profit {lmj[0][1]:.5f}) "
                   f"and {lmj[-1][0]:.4f} (profit {lmj[-1][1]:.5f})")
        # an equilibrium at a switching point would need BR(p_own) = xs for one of the two maxima
        for cand, _ in (lmj[0], lmj[-1]):
            log.append(f"   check: BR({cand:.4f}) = {best_response(cand):.4f} (equals {xs:.4f}? no equilibrium here)")
    p2_switch, lm = switches[-1]
    # ---------------------------------------------------------------- equilibria
    fine = np.arange(PMIN, PMAX + 1e-12, 0.01)
    brf = np.array([best_response(x) for x in fine])
    # symmetric: sign changes of BR(p) - p, skipping the jump
    sym = []
    g = brf - fine
    for i in range(len(fine) - 1):
        if g[i] * g[i + 1] < 0 and abs(brf[i + 1] - brf[i]) < 1.0:
            root = brentq(lambda x: best_response(x) - x, fine[i], fine[i + 1], xtol=1e-12)
            sym.append(root)
    # all equilibria: fixed points of BR o BR
    brbr = np.array([best_response(b) for b in brf])
    h = brbr - fine
    allfp = []
    for i in range(len(fine) - 1):
        if h[i] == 0.0:
            allfp.append(fine[i])
        elif h[i] * h[i + 1] < 0 and abs(brbr[i + 1] - brbr[i]) < 1.0 and abs(brf[i + 1] - brf[i]) < 1.0:
            root = brentq(lambda x: best_response(best_response(x)) - x, fine[i], fine[i + 1], xtol=1e-12)
            allfp.append(root)
    log.append(f"Symmetric equilibria (roots of BR(p) - p): {[round(x, 4) for x in sym]}")
    log.append(f"Fixed points of BR(BR(p)) on [1, 25]: {[round(x, 4) for x in allfp]}")
    asym = [x for x in allfp if min(abs(x - s) for s in sym) > 1e-6]
    log.append(f"Asymmetric equilibria: {asym if asym else 'none'}")

    for ps in sym:
        s = share(ps, ps)[0]
        segs = seg_shares(ps, ps)[:, 0]
        pi = (ps - C) * s
        # slope of BR at the equilibrium (numerical)
        hh = 1e-5
        slope = (best_response(ps + hh) - best_response(ps - hh)) / (2 * hh)
        # profit from the best deviation to the other local maximum
        lms = sorted(local_maxima(ps), key=lambda t: t[0])
        other = [t for t in lms if abs(t[0] - ps) > 0.5]
        dev = other[0] if other else (np.nan, np.nan)
        log.append(
            f"Equilibrium p* = {ps:.4f}: share {s:.4f} (segment A {segs[0]:.4f}, segment B {segs[1]:.4f}), "
            f"profit {pi:.4f}, markup {ps - C:.4f}, BR slope {slope:.4f}, "
            f"best alternative local max at p = {dev[0]:.4f} with profit {dev[1]:.4f}"
        )
        log.append(f"   outside share {1 - 2 * s:.4f}")

    txt = "\n".join(log)
    print(txt)
    OUT_TXT.write_text(txt + "\n")

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.3))

    # panel (a): best responses
    cuts = [PMIN - 1] + [s for s, _ in switches] + [99.0]
    for k in range(len(cuts) - 1):
        m = (p2_grid > cuts[k]) & (p2_grid < cuts[k + 1])
        a1.plot(br[m], p2_grid[m], color=blue, lw=1.8, label=r"$BR_1(p_2)$" if k == 0 else None)
        a1.plot(p2_grid[m], br[m], color=orange, lw=1.8, label=r"$BR_2(p_1)$" if k == 0 else None)
    for s, lmj in switches:
        a1.plot([lmj[0][0], lmj[-1][0]], [s, s], color=blue, lw=0.9, ls=":")
        a1.plot([s, s], [lmj[0][0], lmj[-1][0]], color=orange, lw=0.9, ls=":")
    a1.plot([0.5, 12.5], [0.5, 12.5], color=muted, lw=0.7, ls="--")
    for ps, name in zip(sym, ["L", "H"]):
        a1.plot(ps, ps, "o", ms=6, color=ink, mec="white", mew=1.2, zorder=5)
        a1.annotate(name, (ps, ps), textcoords="offset points", xytext=(8, 5), fontsize=9, color=ink)
    a1.set_xlim(0.5, 12.5)
    a1.set_ylim(0.5, 12.5)
    a1.set_xlabel(r"$p_1$")
    a1.set_ylabel(r"$p_2$")
    a1.set_title("(a) Best responses", fontsize=10)
    a1.legend(frameon=False, fontsize=8, loc="center right")
    a1.grid(True, color=grid, lw=0.5)
    a1.set_axisbelow(True)

    # panel (b): profit in own price at the switching rival price
    pp = np.linspace(1.0, 16.0, 3001)
    a2.plot(pp, profit(pp, p2_switch), color=blue, lw=1.8)
    for x, v in lm:
        a2.plot(x, v, "o", ms=6, color=ink, mec="white", mew=1.2, zorder=5)
    a2.annotate(f"$p_1 = {lm[0][0]:.2f}$", (lm[0][0], lm[0][1]), textcoords="offset points",
                xytext=(6, 6), fontsize=8, color=ink)
    a2.annotate(f"$p_1 = {lm[-1][0]:.2f}$", (lm[-1][0], lm[-1][1]), textcoords="offset points",
                xytext=(-10, 8), fontsize=8, color=ink)
    a2.set_xlim(1, 16)
    a2.set_ylim(0, 0.45)
    a2.set_xlabel(r"$p_1$")
    a2.set_ylabel(r"$\pi_1(p_1, p_2)$")
    a2.set_title(f"(b) Profit of firm 1 at $p_2 = {p2_switch:.3f}$", fontsize=10)
    a2.grid(True, color=grid, lw=0.5)
    a2.set_axisbelow(True)
    for ax in (a1, a2):
        ax.spines["left"].set_color(muted)
        ax.spines["bottom"].set_color(muted)
        ax.tick_params(colors="#52514e")
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(HERE / "multiple_equilibria_preview.png", dpi=130)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
