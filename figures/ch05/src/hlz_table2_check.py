"""Reading Hausman, Leonard and Zona (1994), Table 2 (premium segment).

Table 2 reports, for Budweiser, Molson, Labatts and Miller, the AIDS price
coefficients gamma_ij (symmetry imposed) and the conditional own-price
elasticities of all five brands; the Coors equation and the coefficients on the
Coors price are not reported.  This script
  1. backs out the within-segment revenue shares implied by the reported own
     elasticities through eps_ii = -1 + gamma_ii / w_i (the expenditure term
     beta_i is small and ignored);
  2. recovers the Coors column from homogeneity (sum_j gamma_ij = 0) and the
     Coors own coefficient from adding up, and checks the implied Coors
     own elasticity against the reported -4.641;
  3. computes conditional cross elasticities eps_ij = gamma_ij / w_i and
     compares them with the values quoted in the HLZ text (pp. 166 and 168):
     Budweiser cross elasticities "between 0.4 and 1.0"; Coors with respect to
     Labatts 1.0, Budweiser 1.25, Molson 0.86, Miller 0.52.
These are our calculations, not numbers reported by HLZ.
Output: figures/ch05/src/hlz_table2_check.txt
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
names = ["Budweiser", "Molson", "Labatts", "Miller", "Coors"]
G4 = np.array([[-0.936, 0.372, 0.243, 0.150],
               [0.372, -0.804, 0.183, 0.130],
               [0.243, 0.183, -0.588, 0.028],
               [0.150, 0.130, 0.028, -0.377]])
own = np.array([-3.527, -5.049, -4.277, -4.201, -4.641])
beta4 = np.array([-0.004, -0.011, -0.006, 0.017])

lines = []
w4 = np.diag(G4) / (own[:4] + 1.0)
w5 = 1.0 - w4.sum()
g_coors = -G4.sum(axis=1)                 # homogeneity in rows 1-4
g55 = -g_coors.sum()                      # adding up in the Coors column
G = np.zeros((5, 5))
G[:4, :4] = G4
G[:4, 4] = g_coors
G[4, :4] = g_coors
G[4, 4] = g55
w = np.append(w4, w5)
lines.append("Implied within-segment shares: " + ", ".join(f"{n} {x:.3f}" for n, x in zip(names, w)))
lines.append(f"Coors column from homogeneity: {np.round(g_coors, 3)}; gamma_55 = {g55:.3f}; beta_5 = {-beta4.sum():.3f}")
lines.append(f"Implied Coors own elasticity -1 + gamma_55 / w_5 = {-1 + g55 / w5:.3f} (reported -4.641)")
E = -np.eye(5) + G / w[:, None]
lines.append("Conditional elasticities eps_ij = -1(i=j) + gamma_ij / w_i (row i = quantity, column j = price):")
for n, row in zip(names, E):
    lines.append(f"  {n:10s} " + "  ".join(f"{x: .2f}" for x in row))
text = "\n".join(lines)
print(text)
with open(os.path.join(HERE, "hlz_table2_check.txt"), "w") as fh:
    fh.write(text + "\n")
