"""Implied markups under nested-logit demand at fixed observed shares.

Illustrative market (no data): two inside nests. Nest A has three products
with share 0.15 each (nest share 0.45). Nest B has two products with share
0.10 each (nest share 0.20). The outside share is 0.35. alpha = 1.

For a firm f, the closed form of prop:07-nl-multi is
    m_fg = (1 - sigma) / (alpha (1 - sigma S_{f|g}) (1 - Phi_f)),
    Phi_f = sum_h (1 - sigma) S_fh / (1 - sigma S_{f|h}),
with S_fh the firm's combined share in nest h and S_{f|h} = S_fh / s_h.
We report the markup of a nest-A product under three ownership structures:
single-product firms, a firm owning two products in nest A, and a firm owning
one product in nest A and one in nest B. We also report the welfare
loss (in utils, so in dollars after dividing by alpha) from removing one
nest-A product at fixed prices, ln[1 - s_g + s_g (1 - s_{j|g})^{1 - sigma}]
(cor:07-removal). Output: printed table and
figures/ch07/src/nl_markup_numbers.txt.
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 1.0
S_A, S_B = 0.45, 0.20       # nest shares
SJ_A, SJ_B = 0.15, 0.10     # product shares


def markups(sigma, firm):
    """firm maps nest -> (S_fg, s_g). Returns markups by nest and Phi_f."""
    phi = sum((1 - sigma) * S / (1 - sigma * S / sg) for S, sg in firm.values())
    m = {g: (1 - sigma) / (ALPHA * (1 - sigma * S / sg) * (1 - phi))
         for g, (S, sg) in firm.items()}
    return m, phi


def main():
    lines = ["sigma | single A | single B | two in A (A) | A+B (A) | A+B (B) | % rise two-in-A | % rise A+B (A)"]
    for sigma in [0.0, 0.5, 0.8]:
        mA = markups(sigma, {"A": (SJ_A, S_A)})[0]["A"]
        mB = markups(sigma, {"B": (SJ_B, S_B)})[0]["B"]
        m2 = markups(sigma, {"A": (2 * SJ_A, S_A)})[0]["A"]
        mAB = markups(sigma, {"A": (SJ_A, S_A), "B": (SJ_B, S_B)})[0]
        lines.append(f"{sigma:.1f} | {mA:.3f} | {mB:.3f} | {m2:.3f} | {mAB['A']:.3f} | {mAB['B']:.3f} | "
                     f"{100 * (m2 / mA - 1):.1f} | {100 * (mAB['A'] / mA - 1):.1f}")
    lines.append("")
    lines.append("sigma | utility loss from removing one nest-A product, ln[1 - s_g + s_g (1 - s_j|g)^(1 - sigma)]")
    for sigma in [0.0, 0.5, 0.8]:
        loss = math.log(1 - S_A + S_A * (1 - SJ_A / S_A) ** (1 - sigma))
        lines.append(f"{sigma:.1f} | {loss:.4f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "nl_markup_numbers.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
