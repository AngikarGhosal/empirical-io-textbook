"""Two-type worked example for section 8.3 (ex:08-twotype). Numbers only.

Two consumer types with equal population weights: price sensitive (H, alpha = 4) and price
insensitive (L, alpha = 1). Two inside goods with quality indices q = (2, 3.5) and prices
p = (1, 2), and an outside good with utility zero. Type r has logit shares with V_rj = q_j - alpha_r p_j.
We compute type shares, aggregate shares, the composition of each good's buyers, own elasticities in
the two-type model, and the logit own elasticities alpha_bar p_j (1 - s_j) implied by a logit with the
same shares and the mean price coefficient alpha_bar = 2.5.

Output: src/twotype_numbers.txt
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TXT = HERE / "twotype_numbers.txt"

psi = np.array([0.5, 0.5])
alpha = np.array([4.0, 1.0])
p = np.array([1.0, 2.0])
q = np.array([2.0, 3.5])

V = q[None, :] - alpha[:, None] * p[None, :]
E = np.exp(V)
den = 1 + E.sum(axis=1, keepdims=True)
S = E / den                      # (type, good)
S0 = 1 / den[:, 0]
s = psi @ S
s0 = psi @ S0
lines = []
lines.append(f"V (rows H, L): {V.tolist()}")
lines.append(f"denominators: {den[:, 0].round(6).tolist()}")
lines.append(f"type shares H: s1 = {S[0,0]:.4f}, s2 = {S[0,1]:.4f}, s0 = {S0[0]:.4f}")
lines.append(f"type shares L: s1 = {S[1,0]:.4f}, s2 = {S[1,1]:.4f}, s0 = {S0[1]:.4f}")
lines.append(f"aggregate: s1 = {s[0]:.4f}, s2 = {s[1]:.4f}, s0 = {s0:.4f}")
compH = psi[0] * S[0] / s
lines.append(f"share of buyers who are type H: good 1 {compH[0]:.3f}, good 2 {compH[1]:.3f}")
# own derivative magnitude int alpha s (1 - s)
dd = np.array([np.sum(psi * alpha * S[:, j] * (1 - S[:, j])) for j in range(2)])
own = p * dd / s
lines.append(f"int alpha s(1-s): good 1 {dd[0]:.4f}, good 2 {dd[1]:.4f}")
lines.append(f"two-type own elasticities: good 1 {own[0]:.3f}, good 2 {own[1]:.3f}")
# type-specific own elasticities alpha_r p_j (1 - s_rj)
for r, name in enumerate(["H", "L"]):
    e = alpha[r] * p * (1 - S[r])
    lines.append(f"type {name} logit elasticities: good 1 {e[0]:.3f}, good 2 {e[1]:.3f}")
abar = psi @ alpha
lg = abar * p * (1 - s)
lines.append(f"logit with alpha_bar = {abar}: own elasticities good 1 {lg[0]:.3f}, good 2 {lg[1]:.3f}")
# cross derivatives and diversion
cross = np.sum(psi * alpha * S[:, 0] * S[:, 1])
d12 = cross / dd[0]
d21 = cross / dd[1]
d10 = np.sum(psi * alpha * S0 * S[:, 0]) / dd[0]
d20 = np.sum(psi * alpha * S0 * S[:, 1]) / dd[1]
lines.append(f"diversion 1->2 {d12:.3f}, 1->0 {d10:.3f}; 2->1 {d21:.3f}, 2->0 {d20:.3f}")
lines.append(f"logit diversion 1->0 {s0/(1-s[0]):.3f}, 2->0 {s0/(1-s[1]):.3f}")
# single-product markups 1 / (-(d ln s_j / d p_j)) = s_j / dd_j
lines.append(f"single-product markups p - mc = s/(-ds/dp): two-type good 1 {s[0]/dd[0]:.3f}, "
             f"good 2 {s[1]/dd[1]:.3f}; logit {1/(abar*(1-s[0])):.3f}, {1/(abar*(1-s[1])):.3f}")
text = "\n".join(lines)
print(text)
TXT.write_text(text + "\n")
