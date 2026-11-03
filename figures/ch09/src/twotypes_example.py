"""Two-type identification example (ex:09-twotypes, fig:09-twotypes, tab:09-twotypes).

Two identical city pairs. Products differ only in price (in $100):
  market A: H at 2 and L at 1;  market B: H at 2, L1 at 1, L2 at 1.
Finite-mixture nested logit (all inside goods in one nest, nesting parameter sigma):
type r has population share psi_r and mean utility
  delta^r_j = b0 + xi_j - alpha_r p_j,   xi_L = 0,
and xi_H is chosen so that s_H = s_L in market A.
One-type benchmark: the nested logit with the same sigma and one type, calibrated to reproduce
market A's shares exactly (so H and L have equal mean utility, and B's three shares are equal).

The main example uses sigma = 0.4 (close to the nesting parameter Berry, Carnall and Spiller
estimate). The script also reports the sigma = 0 case used in a remark.

Output: figures/ch09/twotypes_example.pdf and figures/ch09/src/twotypes_numbers.txt
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "twotypes_example.pdf"
TXT = HERE / "twotypes_numbers.txt"

ALPHA = np.array([3.0, 0.1])   # type 1 (price sensitive), type 2 (price insensitive)
PSI = np.array([0.9, 0.1])
B0 = 0.0
P_H, P_L = 2.0, 1.0


def mixture(prices, xi, sigma, alpha=ALPHA, psi=PSI, b0=B0):
    """Aggregate shares s (J,), type shares sr (R, J), within-nest shares sjg (R, J)."""
    prices = np.asarray(prices, float)
    xi = np.asarray(xi, float)
    delta = b0 + xi[None, :] - alpha[:, None] * prices[None, :]
    e = np.exp(delta / (1.0 - sigma))
    D = e.sum(axis=1, keepdims=True)
    sjg = e / D
    sbar = D ** (1.0 - sigma) / (1.0 + D ** (1.0 - sigma))
    sr = sjg * sbar
    return psi @ sr, sr, sjg


def dsr_ddelta(sr, sjg, sigma):
    """(R, J, J) array: d s^r_j / d delta^r_k for each type."""
    R, J = sr.shape
    out = np.zeros((R, J, J))
    a = 1.0 / (1.0 - sigma)
    b = sigma / (1.0 - sigma)
    for r in range(R):
        for j in range(J):
            for k in range(J):
                if j == k:
                    out[r, j, k] = sr[r, j] * (a - b * sjg[r, j] - sr[r, j])
                else:
                    out[r, j, k] = -sr[r, j] * (b * sjg[r, k] + sr[r, k])
    return out


def price_jacobian(sr, sjg, sigma, alpha=ALPHA, psi=PSI):
    """J x J matrix with (j, k) element d s_j / d p_k."""
    d = dsr_ddelta(sr, sjg, sigma)
    return -np.einsum("r,rjk->jk", psi * alpha, d)


def calibrate_xi_h(sigma):
    def gap(xh):
        s, _, _ = mixture([P_H, P_L], [xh, 0.0], sigma)
        return s[0] - s[1]
    return brentq(gap, -20, 20, xtol=1e-14)


def one_type_B(sA_each, sigma):
    """One-type nested logit fitted to market A (two equal shares); returns each B share."""
    sbarA = 2.0 * sA_each
    k = sbarA / (1.0 - sbarA)            # = 2^{1-sigma} e^{d}
    kB = k * 1.5 ** (1.0 - sigma)        # = 3^{1-sigma} e^{d}
    sbarB = kB / (1.0 + kB)
    return sbarB / 3.0


def analyse(sigma, lines):
    xh = calibrate_xi_h(sigma)
    sA, srA, sjgA = mixture([P_H, P_L], [xh, 0.0], sigma)
    sB, srB, sjgB = mixture([P_H, P_L, P_L], [xh, 0.0, 0.0], sigma)
    sB1 = one_type_B(sA[0], sigma)
    lines.append(f"--- sigma = {sigma}: alpha = {ALPHA}, psi = {PSI}, b0 = {B0}, xi_H = {xh:.4f}")
    lines.append(f"A shares (H, L): {sA.round(4)}, outside {1 - sA.sum():.4f}")
    lines.append(f"B two-type shares (H, L1, L2): {sB.round(4)}, outside {1 - sB.sum():.4f}")
    lines.append(f"B one-type shares: {sB1:.4f} each, outside {1 - 3 * sB1:.4f}")
    lines.append(f"ratio s_H/s_L in B: two-type {sB[0] / sB[1]:.3f}, one-type 1.000")
    lines.append(f"type-specific shares A:\n{srA.round(4)}\nwithin-nest A:\n{sjgA.round(4)}")
    lines.append(f"type-specific shares B:\n{srB.round(4)}\nwithin-nest B:\n{sjgB.round(4)}")
    compA = PSI[:, None] * srA / sA[None, :]
    compB = PSI[:, None] * srB / sB[None, :]
    lines.append(f"Pr(type 2 | j): A {compA[1].round(3)}, B {compB[1].round(3)}")
    # Percentage changes from A to B.
    lines.append(f"pct change A->B: H {100 * (sB[0] / sA[0] - 1):.1f}%, L {100 * (sB[1] / sA[1] - 1):.1f}%; "
                 f"one-type {100 * (sB1 / sA[0] - 1):.1f}% for both")
    # FOC-implied single-product markups at the given prices.
    JA = price_jacobian(srA, sjgA, sigma)
    JB = price_jacobian(srB, sjgB, sigma)
    mA = sA / -np.diag(JA)
    mB = sB / -np.diag(JB)
    lines.append(f"single-product implied markups ($100): A {mA.round(3)}, B {mB.round(3)}")
    # Multi-product: one carrier owns L1 and L2 in B.
    idx = [1, 2]
    sub = JB[np.ix_(idx, idx)]
    mL = -np.linalg.solve(sub.T, sB[idx])
    lines.append(f"markups if one carrier owns L1 and L2 in B: {mL.round(3)}")
    # Own-price elasticities and diversion in B.
    el = -np.diag(JB) * np.array([P_H, P_L, P_L]) / sB
    lines.append(f"own-price elasticities B (H, L1, L2): {el.round(3)}")
    D_L1_L2 = -JB[2, 1] / JB[1, 1]
    D_L1_H = -JB[0, 1] / JB[1, 1]
    lines.append(f"diversion from L1 in B: to L2 {D_L1_L2:.3f}, to H {D_L1_H:.3f}, outside {1 - D_L1_L2 - D_L1_H:.3f}")
    # Hub semi-elasticities in market A: d ln s_j / d x_j = sum_r Pr(r|j) beta_r dln s^r_j/d delta^r_j.
    dA = dsr_ddelta(srA, sjgA, sigma)
    own = np.array([[dA[r, j, j] / srA[r, j] for j in range(2)] for r in range(2)])
    for label, bh in (("type 2 values hubs", np.array([0.2, 1.0])),
                      ("type 1 values hubs", np.array([1.0, 0.2]))):
        semi = (compA * bh[:, None] * own).sum(axis=0)
        lines.append(f"hub semi-elasticity A ({label}, beta^h = {bh}): H {semi[0]:.3f}, L {semi[1]:.3f}")
    return dict(sA=sA, sB=sB, sB1=sB1)


def main():
    lines = []
    res = analyse(0.4, lines)
    analyse(0.0, lines)
    text = "\n".join(lines)
    print(text)
    TXT.write_text(text + "\n")
    sA, sB, sB1 = res["sA"], res["sB"], res["sB1"]

    # ------------------------------------------------------------------ figure
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
        "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True,
                             gridspec_kw={"width_ratios": [2, 3]})
    w = 0.36
    ax = axes[0]
    xa = np.arange(2)
    ax.bar(xa - w / 2 - 0.01, [sA[0], sA[0]], w, color=BLUE, label="One type")
    ax.bar(xa + w / 2 + 0.01, sA, w, color=ORANGE, label="Two types")
    ax.set_xticks(xa)
    ax.set_xticklabels([r"$H$ (\$200)", r"$L$ (\$100)"])
    ax.set_title("(a) Market A (fitted)", fontsize=10)
    ax.set_ylabel("Market share")
    ax = axes[1]
    xb = np.arange(3)
    ax.bar(xb - w / 2 - 0.01, [sB1] * 3, w, color=BLUE, label="One type")
    ax.bar(xb + w / 2 + 0.01, sB, w, color=ORANGE, label="Two types")
    ax.set_xticks(xb)
    ax.set_xticklabels([r"$H$ (\$200)", r"$L_1$ (\$100)", r"$L_2$ (\$100)"])
    ax.set_title("(b) Market B (predicted)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for ax in axes:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylim(0, 0.075)
    fig.tight_layout()
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
