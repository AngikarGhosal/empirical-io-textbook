"""Likelihood-ratio tests from Deaton and Muellbauer (1980a), Table 4.

Table 4 reports 2 log L for the AIDS on British data, 1954-1974 (values read
from the paper, AER 70(3), p. 321).  The LR statistic for a set of r
restrictions is the difference in 2 log L between the less and the more
restricted model, asymptotically chi-squared with r degrees of freedom.
The script also reports the Table 3 gap between the Stone index P* and the
exact index P, and the counts used in the text.  Output: dm_lr_tests_numbers.txt.
"""
import os
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dm_lr_tests_numbers.txt")

t4 = {
    ("levels", "Pstar"): {"U": 1722.5, "H": 1579.6},
    ("levels", "P"): {"U": 1723.8, "H": 1585.1, "S": 1491.0},
    ("fd", "Pstar"): {"U": 1560.0, "H": 1546.6},
    ("fd", "P"): {"U": 1560.3, "H": 1547.9, "S": 1508.8},
}

lines = []


def test(label, a, b, df):
    lr = a - b
    p = chi2.sf(lr, df)
    lines.append(f"{label}: LR = {a:.1f} - {b:.1f} = {lr:.1f} on {df} df; "
                 f"5% cv = {chi2.ppf(0.95, df):.2f}, 1% cv = {chi2.ppf(0.99, df):.2f}, p = {p:.2e}")


for model in ("levels", "fd"):
    for idx in ("Pstar", "P"):
        d = t4[(model, idx)]
        test(f"{model:6s} {idx:5s} homogeneity", d["U"], d["H"], 7)
        if "S" in d:
            test(f"{model:6s} {idx:5s} symmetry | homogeneity", d["H"], d["S"], 21)
            test(f"{model:6s} {idx:5s} homogeneity + symmetry", d["U"], d["S"], 28)

# Table 3 (P* and P, 1970 = 1), read from the paper, p. 321
years = list(range(1954, 1975))
Pstar = [0.566, 0.587, 0.611, 0.631, 0.648, 0.655, 0.663, 0.684, 0.712, 0.729, 0.754,
         0.793, 0.827, 0.851, 0.894, 0.946, 1.000, 1.084, 1.161, 1.271, 1.465]
P = [0.571, 0.595, 0.617, 0.636, 0.653, 0.661, 0.666, 0.686, 0.715, 0.730, 0.758,
     0.797, 0.830, 0.852, 0.888, 0.944, 1.000, 1.084, 1.161, 1.279, 1.461]
gaps = [abs(a - b) for a, b in zip(Pstar, P)]
mx = max(gaps)
lines.append(f"Table 3: max |P* - P| = {mx:.3f} in years {[y for y, g in zip(years, gaps) if abs(g - mx) < 1e-9]}")
lines.append(f"Table 3: price level rises by factor {P[-1] / P[0]:.2f} (P) and {Pstar[-1] / Pstar[0]:.2f} (P*) from 1954 to 1974")

# parameter counts for J = 8 goods
J = 8
lines.append(f"J = {J}: homogeneity restrictions = {J - 1}; symmetry given homogeneity = {(J - 1) * (J - 2) // 2}")
lines.append(f"  free gammas under symmetry and homogeneity = {J * (J - 1) // 2} (upper triangle of 7x7 incl. diagonal)")
lines.append(f"  unrestricted gamma coefficients in Table 1 = {J * J}")

open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
