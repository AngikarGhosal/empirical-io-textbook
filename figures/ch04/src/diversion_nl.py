"""Diversion from product 1 under logit and nested logit with identical shares (fig:04-diversion).

Setting (Section 4.3.4 of the book):
  * Four inside products and an outside good. Nests {1, 2} and {3, 4}; the outside good is alone.
  * Every product has price 1; the price coefficient is alpha = 1 (diversion does not depend on it).
  * For each nesting parameter sigma in {0 (plain logit), 0.5, 0.7, 0.9} we calibrate one common
    mean utility delta so that every inside share equals 0.15 (outside share 0.40).

Nested logit (Berry 1994 form), product j in nest g:
    s_{j|g} = exp(delta_j / (1 - sigma)) / D_g,   D_g = sum_{k in g} exp(delta_k / (1 - sigma)),
    s_g     = D_g^{1 - sigma} / (1 + sum_h D_h^{1 - sigma}),   s_j = s_{j|g} s_g.
Diversion from j to k is D_jk = -(ds_k/dp_j) / (ds_j/dp_j). We compute the share Jacobian by
central finite differences of the share function (as a check) and compare it with the closed form
    ds_j/dp_j       = -alpha s_j [1/(1-sigma) - sigma/(1-sigma) s_{j|g} - s_j]
    ds_k/dp_j       =  alpha s_k [sigma/(1-sigma) s_{j|g} + s_j]      (k in the nest of j)
    ds_k/dp_j       =  alpha s_k s_j                                    (k in another nest, or k = 0)

Outputs: figures/ch04/diversion_nl.pdf and figures/ch04/src/diversion_nl_output.txt
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "diversion_nl.pdf"
OUT_TXT = HERE / "diversion_nl_output.txt"

ALPHA = 1.0
PRICE = 1.0
NESTS = [np.array([0, 1]), np.array([2, 3])]   # zero-based product indices
TARGET_SHARE = 0.15
SIGMAS = [0.0, 0.5, 0.7, 0.9]


def shares(p, xi, sigma):
    """Nested logit inside shares and outside share for prices p (length 4)."""
    delta = xi - ALPHA * p
    s = np.zeros(4)
    dg = []
    for g in NESTS:
        ev = np.exp(delta[g] / (1.0 - sigma))
        dg.append(ev.sum())
    denom = 1.0 + sum(d ** (1.0 - sigma) for d in dg)
    for g, d in zip(NESTS, dg):
        ev = np.exp(delta[g] / (1.0 - sigma))
        s[g] = ev / d * d ** (1.0 - sigma) / denom
    return s, 1.0 / denom


def calibrate(sigma):
    """Common quality xi so that each inside share equals TARGET_SHARE at price 1."""
    f = lambda x: shares(np.full(4, PRICE), np.full(4, x), sigma)[0][0] - TARGET_SHARE
    return brentq(f, -20.0, 20.0)


def diversion_fd(sigma, xi):
    """Diversion from product 1 to products 2, 3, 4 and the outside good by finite differences."""
    p0 = np.full(4, PRICE)
    h = 1e-6
    up, dn = p0.copy(), p0.copy()
    up[0] += h
    dn[0] -= h
    s_up, s0_up = shares(up, np.full(4, xi), sigma)
    s_dn, s0_dn = shares(dn, np.full(4, xi), sigma)
    ds = (s_up - s_dn) / (2 * h)
    ds0 = (s0_up - s0_dn) / (2 * h)
    own = ds[0]
    return np.array([-ds[1] / own, -ds[2] / own, -ds[3] / own, -ds0 / own])


def diversion_closed(sigma):
    s, s0, sjg = TARGET_SHARE, 1.0 - 4 * TARGET_SHARE, 0.5
    r = sigma / (1.0 - sigma)
    own = s * (1.0 / (1.0 - sigma) - r * sjg - s)
    same = s * (r * sjg + s)
    other = s * s
    out = s0 * s
    return np.array([same, other, other, out]) / own


def main():
    lines = []
    rows = {}
    for sig in SIGMAS:
        xi = calibrate(sig)
        d_fd = diversion_fd(sig, xi)
        d_cf = diversion_closed(sig)
        assert np.allclose(d_fd, d_cf, atol=1e-7), (sig, d_fd, d_cf)
        assert abs(d_cf.sum() - 1.0) < 1e-12
        rows[sig] = d_cf
        lines.append(
            f"sigma={sig:.1f}  xi={xi:.6f}  D12={d_cf[0]:.4f}  D13={d_cf[1]:.4f}  "
            f"D14={d_cf[2]:.4f}  D10={d_cf[3]:.4f}  sum={d_cf.sum():.12f}"
        )
    txt = "\n".join(lines)
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
    ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal blue ramp, light to dark
    labels = ["Logit", r"Nested logit, $\sigma^{\mathrm{NL}} = 0.5$",
              r"Nested logit, $\sigma^{\mathrm{NL}} = 0.7$", r"Nested logit, $\sigma^{\mathrm{NL}} = 0.9$"]
    groups = [r"$D_{12}$" + "\n(same nest)", r"$D_{13}$" + "\n(other nest)",
              r"$D_{14}$" + "\n(other nest)", r"$D_{10}$" + "\n(outside good)"]
    x = np.arange(4)
    width = 0.19
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for i, sig in enumerate(SIGMAS):
        xs = x + (i - 1.5) * (width + 0.01)
        bars = ax.bar(xs, rows[sig], width=width, color=ramp[i], label=labels[i],
                      edgecolor="white", linewidth=0.8)
        # direct labels only on the same-nest group, which carries the message
        b = bars[0]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{rows[sig][0]:.2f}",
                ha="center", va="bottom", fontsize=8, color="#0b0b0b")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Diversion ratio from product 1")
    ax.set_ylim(0, 1.0)
    ax.spines["left"].set_color("#898781")
    ax.spines["bottom"].set_color("#898781")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(HERE / "diversion_nl_preview.png", dpi=130)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
