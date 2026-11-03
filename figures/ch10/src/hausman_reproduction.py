"""Reproduction checks for Hausman (1997), Tables 5.2-5.7 (Section sec:10-hausman).

All inputs are copied from the NBER PDF of Hausman (1997) and checked visually:
Table 5.3 (AIDS coefficients), Table 5.4 (segment demand), Tables 5.5-5.6
(elasticities), Table 5.7 (margins), Table 5.2 (prices and shares) and the text.

What the script does (every number it prints is "our computation"):
  1. Rebuilds Table 5.5 from Table 5.3 with the AIDS elasticity formula and fits the
     single unreported constant ln(y_G / P) by least squares.
  2. Shows that Table 5.6 minus Table 5.5 is rank one only after transposing Table 5.6,
     and recovers the common factor B of the multistage formula.
  3. Reproduces Table 5.7 from equation (14) and recomputes it with the orientation
     implied by step 2.
  4. Recomputes the price-change, welfare and price-index arithmetic in the text.

Output: figures/ch10/src/hausman_reproduction_numbers.txt
Run from the Textbook directory: python figures/ch10/src/hausman_reproduction.py
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
TXT = HERE / "hausman_reproduction_numbers.txt"

BRANDS = ["Cheerios", "HN Cheerios", "AC Cheerios", "Corn Flakes", "K Raisin Bran",
          "Rice Krispies", "Fr Mini-Wheats", "Fr Wheat Sq", "Post Raisin Bran"]

# ---------------------------------------------------------------- Table 5.3
LOW = {
    (1, 1): -0.18855,
    (2, 1): 0.02087, (2, 2): -0.13165,
    (3, 1): 0.00842, (3, 2): 0.01849, (3, 3): -0.07070,
    (4, 1): 0.04805, (4, 2): 0.00268, (4, 3): 0.00772, (4, 4): -0.14438,
    (5, 1): 0.02071, (5, 2): 0.02285, (5, 3): 0.01208, (5, 4): 0.03957, (5, 5): -0.12861,
    (6, 1): 0.02916, (6, 2): 0.03561, (6, 3): 0.01431, (6, 4): 0.01812, (6, 5): -0.00791,
    (6, 6): -0.14195,
    (7, 1): 0.03010, (7, 2): 0.00239, (7, 3): -0.00142, (7, 4): 0.01656, (7, 5): 0.03966,
    (7, 6): 0.02135, (7, 7): -0.13658,
    (8, 1): 0.00372, (8, 2): 0.00587, (8, 3): -0.00172, (8, 4): 0.00418, (8, 5): 0.00779,
    (8, 6): 0.00008, (8, 7): 0.01208, (8, 8): -0.03206,
}
BETA8 = [-0.03853, -0.01552, -0.00854, 0.02003, -0.01391, 0.02685, -0.01258, -0.00246]
# Mean revenue shares (bottom row of Table 5.5)
W = np.array([0.21617, 0.15026, 0.06193, 0.14243, 0.13117, 0.13539, 0.09067, 0.01475, 0.05722])

T55 = np.array([
    [-1.73851, 0.16166, 0.07110, 0.19818, 0.15355, 0.09268, 0.18649, 0.02593, 0.02716],
    [0.21637, -1.83838, 0.14169, 0.00390, 0.18550, 0.21253, 0.04330, 0.04414, 0.09425],
    [0.23945, 0.34899, -2.11677, 0.10597, 0.23973, 0.19848, 0.01366, -0.02100, 0.12936],
    [0.23185, -0.03254, 0.02883, -1.99465, 0.23222, 0.16056, 0.07898, 0.02246, 0.13165],
    [0.23744, 0.21291, 0.11121, 0.28729, -1.94608, -0.08546, 0.33045, 0.06454, -0.10626],
    [0.06656, 0.19055, 0.06997, 0.16068, -0.12272, -2.00148, 0.10508, -0.00909, 0.34211],
    [0.43609, 0.07708, 0.00939, 0.16386, 0.48235, 0.20255, -2.46950, 0.14003, 0.09692],
    [0.37740, 0.45906, -0.08636, 0.26062, 0.58179, -0.03396, 0.86314, -3.16485, -0.09011],
    [-0.10461, 0.11474, 0.08315, 0.23661, -0.35988, 0.73072, 0.07025, -0.03721, -2.51416]])
T56 = np.array([  # as printed
    [-1.92572, 0.01210, 0.04306, -0.02798, 0.03380, -0.20642, 0.23990, 0.18758, -0.51019],
    [0.03154, -1.98037, 0.21247, -0.21316, 0.07136, 0.00079, -0.05929, 0.32712, -0.16719],
    [0.01747, 0.08317, -2.17304, -0.04561, 0.05287, -0.00824, -0.04682, -0.14074, -0.03304],
    [0.07484, -0.13069, -0.02343, -2.16585, 0.15311, -0.01918, 0.03460, 0.13556, -0.03062],
    [0.03995, 0.06155, 0.12056, 0.07455, -2.06965, -0.28837, 0.36331, 0.46661, -0.60598],
    [-0.02457, 0.08459, 0.07548, -0.00219, -0.21300, -2.17246, 0.07967, -0.15285, 0.47670],
    [0.10797, -0.04239, -0.06872, -0.03001, 0.24504, -0.00943, -2.55178, 0.78352, -0.09987],
    [0.01315, 0.03020, -0.03440, 0.00473, 0.05064, -0.02772, 0.12664, -3.17781, -0.06489],
    [-0.02239, 0.04018, 0.07738, 0.06288, -0.16016, 0.26985, 0.04499, -0.14035, -2.62151]])

# Table 5.4 (family column), top level, Table 5.2, text
PHI_F, EPS_FF, EPS_U, W_F_TEXT = 0.99868, -1.82906, -0.90, 0.264
PRICE = {"C": 2.644, "HN": 3.605, "ACC": 3.480}
SH92 = np.array([21.62, 15.03, 6.19, 14.24, 13.11, 13.54, 9.07, 1.48, 5.72]) / 100
P92 = np.array([2.644, 3.605, 3.480, 1.866, 3.214, 2.475, 3.420, 3.262, 3.046])
POUNDS = {"C": 93738, "HN": 47215}
T57 = {"with": (0.5268, 0.5203), "without": (0.5251, 0.5096), "indep": (0.5193, 0.5050)}
PV, CS_WEEK = 7.14, 32268


def gamma_matrix():
    g = np.zeros((9, 9))
    for (i, j), v in LOW.items():
        g[i - 1, j - 1] = v
        g[j - 1, i - 1] = v
    for i in range(8):  # homogeneity gives the omitted ninth column
        g[i, 8] = -g[i, :8].sum()
        g[8, i] = g[i, 8]
    g[8, 8] = -g[8, :8].sum()
    return g


def main():
    out = []
    g = gamma_matrix()
    beta = np.array(BETA8 + [-sum(BETA8)])  # adding up
    out.append(f"Implied Post Raisin Bran: beta_9 = {beta[8]:.5f}, gamma_99 = {g[8, 8]:.5f}")

    # 1. Conditional elasticities, nonlinear AIDS formula
    def cond(lnyp):
        e = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                e[i, j] = -(i == j) + (g[i, j] - beta[i] * (W[j] - beta[j] * lnyp)) / W[i]
        return e
    fit = minimize_scalar(lambda x: ((cond(x) - T55) ** 2).sum(), bounds=(0, 30), method="bounded")
    lnyp = fit.x
    e_c = cond(lnyp)
    out.append(f"\n1. Fitted ln(y_G/P) = {lnyp:.3f} (exp = {np.exp(lnyp):,.0f});"
               f" max |rebuilt - Table 5.5| = {np.abs(e_c - T55).max():.5f}")
    stone = np.array([[-(i == j) + (g[i, j] - beta[i] * W[j]) / W[i] for j in range(9)]
                      for i in range(9)])
    out.append(f"   Stone-index formula instead: max abs gap {np.abs(stone - T55).max():.4f}")
    fitT = minimize_scalar(lambda x: ((cond(x) - T55.T) ** 2).sum(), bounds=(0, 30), method="bounded")
    out.append(f"   Transposed Table 5.5: best max abs gap {np.abs(cond(fitT.x) - T55.T).max():.4f}")
    out.append(f"   e(ACC,HN) rebuilt {e_c[2, 1]:.5f} vs printed {T55[2, 1]}; "
               f"e(HN,ACC) rebuilt {e_c[1, 2]:.5f} vs printed {T55[1, 2]}")

    # 2. Overall elasticities
    phi = 1 + beta / W
    d_print = T56 - T55
    d_trans = T56.T - T55
    out.append("\n2. Singular values of T56 - T55: " + np.array2string(np.linalg.svd(d_print)[1], precision=3))
    out.append("   Singular values of T56' - T55: " + np.array2string(np.linalg.svd(d_trans)[1], precision=3))
    ratio = d_trans / np.outer(phi, W)
    out.append(f"   (T56' - T55)_ij / (phi_i w_j): mean {ratio.mean():.4f}, sd {ratio.std():.5f}")
    out.append("   phi_i = 1 + beta_i/w_i: " + ", ".join(f"{b} {p:.3f}" for b, p in zip(BRANDS, phi)))
    b_formula = 1 + EPS_FF + PHI_F * (1 + EPS_U) * W_F_TEXT
    out.append(f"   Bracket from eq. (aggregation) with Table 5.4, -0.90 and w_F = 0.264: {b_formula:.4f}")
    wf_needed = (ratio.mean() - 1 - EPS_FF) / (PHI_F * EPS_U)
    out.append(f"   Bracket 1 + eps_FF + phi_F*eps_U*w_F matches {ratio.mean():.4f} at w_F = {wf_needed:.4f}")
    e_u = T56.T  # row = quantity
    out.append("   Overall Cheerios block (row quantity, column price):")
    for i in range(3):
        out.append("     " + BRANDS[i].ljust(14) + " ".join(f"{e_u[i, j]: .5f}" for j in range(3)))
    e_alt = T55 + np.outer(phi, W) * b_formula
    out.append(f"   Post RB w.r.t. Cheerios price: printed-transposed {e_u[8, 0]:.4f};"
               f" decomposition {T55[8, 0]:.4f} + {phi[8]:.3f}*{W[0]:.4f}*({ratio.mean():.4f})")
    out.append(f"   Same entry with bracket {b_formula:.3f}: {e_alt[8, 0]:.4f}; ACC own: {e_alt[2, 2]:.4f}")

    # 3. Table 5.7
    def margins(e, idx, s):
        ee = e[np.ix_(idx, idx)]
        w = -np.linalg.solve(ee.T, s[idx])  # s + E'w = 0 with E[k,j] = e_kj
        return w / s[idx]
    for label, e in [("Table 5.6 read with rows = quantities (as Hausman's FOC needs)", T56),
                     ("Table 5.6 transposed (orientation implied by step 2)", T56.T)]:
        l3 = margins(e, [0, 1, 2], W)
        l2 = margins(e, [0, 1], W)
        out.append(f"\n3. {label}:\n   with ACC  C {l3[0]:.4f}  HN {l3[1]:.4f}  ACC {l3[2]:.4f}"
                   f"\n   without   C {l2[0]:.4f}  HN {l2[1]:.4f}")
        dp_c = (1 - l2[0]) / (1 - l3[0]) - 1
        dp_h = (1 - l2[1]) / (1 - l3[1]) - 1
        loss = dp_c * PRICE["C"] * POUNDS["C"] + dp_h * PRICE["HN"] * POUNDS["HN"]
        out.append(f"   price change C {100 * dp_c:.2f}% (${dp_c * PRICE['C']:.4f}),"
                   f" HN {100 * dp_h:.2f}% (${dp_h * PRICE['HN']:.4f}); first-order CS loss ${loss:,.0f};"
                   f" net share of $32,000: {100 * (32000 - loss) / 32000:.1f}%")
    out.append(f"   Printed Table 5.7: {T57}")
    out.append(f"   Independent margins 1/|e_ii| (Table 5.6): C {1 / 1.92572:.4f}, HN {1 / 1.98037:.4f},"
               f" ACC {1 / 2.17304:.4f}")

    # 4. Text arithmetic
    lw, lwo = T57["with"], T57["without"]
    for k, name in enumerate(["C", "HN"]):
        r = (1 - lwo[k]) / (1 - lw[k]) - 1
        out.append(f"\n4. {name}: price ratio - 1 = {100 * r:.3f}%, dp = ${r * PRICE[name]:.4f},"
                   f" dCS = ${r * PRICE[name] * POUNDS[name]:,.0f}; margin ratio - 1 = {100 * (lw[k] / lwo[k] - 1):.2f}%")
    out.append(f"   0.0787/0.02231 implied HN price = {0.0787 / ((1 - lwo[1]) / (1 - lw[1]) - 1):.3f}")
    out.append(f"   Reported products: 0.0095*93738 = {0.0095 * 93738:.1f}; 0.0787*47215 = {0.0787 * 47215:.1f}")
    out.append(f"   27,395/32,000 = {27395 / 32000:.4f}; 78.1*0.856 = {78.1 * 0.856:.2f}; 0.3136*0.856 = {0.3136 * 0.856:.4f};"
               f" 78.1e6/0.3136 = {78.1e6 / 0.3136 / 1e6:.1f} million persons")
    seg_c = POUNDS["C"] * PRICE["C"] / W[0]
    seg_h = POUNDS["HN"] * PRICE["HN"] / W[1]
    seg = 0.5 * (seg_c + seg_h)
    acc_lb = W[2] * seg / PRICE["ACC"]
    rect = (PV - PRICE["ACC"]) * acc_lb
    out.append(f"   Family-segment spending per city-week: from Cheerios ${seg_c:,.0f}, from HN ${seg_h:,.0f}")
    out.append(f"   ACC pounds per city-week ~ {acc_lb:,.0f}; revenue ~ ${acc_lb * PRICE['ACC']:,.0f};"
               f" rectangle (7.14-3.48)q = ${rect:,.0f}; CS/rectangle = {CS_WEEK / rect:.3f}")
    ces = 1 - (1 - W[2]) ** (1 / 1.13)
    out.append(f"   CES gain 1-(1-0.06193)^(1/1.13) = {ces:.4f}; times segment spending = ${ces * seg:,.0f}"
               f" = {ces * seg / CS_WEEK:.2f} x AIDS CS")
    for pv in [5.55, 5.94]:
        out.append(f"   Lower-bound virtual price {pv}: {100 * (pv / PRICE['ACC'] - 1):.1f}% above $3.48;"
                   f" rectangle share rule 1.6%*(pv/p-1) = {1.6 * (pv / PRICE['ACC'] - 1):.2f}%")
    out.append(f"   $4.75 vs $3.48: {100 * (4.75 / 3.48 - 1):.1f}% above")
    # price-index thought experiment
    comp_before = (SH92[1] * P92[1] + SH92[2] * PV) / (SH92[1] + SH92[2])
    comp_after = (SH92[1] * P92[1] + SH92[2] * P92[2]) / (SH92[1] + SH92[2])
    fam_after = (SH92 * P92).sum() / SH92.sum()
    fam_before = fam_after + SH92[2] * (PV - P92[2]) / SH92.sum()
    out.append(f"   HN+ACC composite index: before ${comp_before:.3f}, after ${comp_after:.3f},"
               f" fall {100 * (1 - comp_after / comp_before):.1f}%")
    out.append(f"   Family index: before ${fam_before:.3f}, after ${fam_after:.3f},"
               f" fall {100 * (1 - fam_after / fam_before):.1f}%")
    out.append(f"   Shares: Cheerios brands {SH92[:3].sum() * 100:.2f}% of family; x0.264 = {SH92[:3].sum() * 26.4:.2f}%;"
               f" ACC x0.264 = {SH92[2] * 26.4:.2f}%")
    TXT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
