"""Arithmetic behind the numbers quoted in Section sec:09-bcs (Berry, Carnall and Spiller).

All inputs are printed values from the 1996 working paper (NBER WP 5561): Tables 3, 4, 5.
Output: figures/ch09/src/bcs_numbers.txt
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TXT = HERE / "bcs_numbers.txt"

T5 = {  # demand panel of Table 5: (type 1, type 2) or single value
    "I": {"lam": 0.605, "lam_se": 0.020, "psi": 1.000, "FARE": (0.455, None), "DIRECT": (0.656, None),
          "HUBSIZE": (0.437, None), "TEMPDIFF": (0.050, None), "TRIPS": (0.551, None)},
    "II": {"lam": 0.638, "lam_se": 0.020, "psi": 0.924, "FARE": (0.696, 0.068), "DIRECT": (0.542, None),
           "HUBSIZE": (0.213, None), "TEMPDIFF": (0.049, None), "TRIPS": (0.393, None)},
    "III": {"lam": 0.599, "lam_se": 0.027, "psi": 0.975, "FARE": (0.829, 0.077), "DIRECT": (0.414, 1.014),
            "HUBSIZE": (0.285, 0.761), "TEMPDIFF": (0.052, 0.049), "TRIPS": (0.329, 0.954)},
    "IV": {"lam": 0.617, "lam_se": 0.026, "psi": 0.952, "FARE": (0.986, 0.111), "DIRECT": (0.293, 0.876),
           "HUBSIZE": (0.615, 0.368), "TEMPDIFF": (0.049, 0.050), "TRIPS": (0.388, 0.493)},
}
# Table 4 bottom panel: passengers and population shares (type 1, type 2), percent.
T4 = {"II": ((70.99, 29.01), (92.35, 7.65)), "III": ((73.19, 26.81), (97.48, 2.52)),
      "IV": ((60.13, 39.87), (95.21, 4.79))}
T4_XI = {"II": (0.97, 1.19), "III": (0.86, 1.30), "IV": (0.87, 1.22)}
SD_HUB = 0.045  # Table 1
MAX_HUB = 0.210


def main():
    out = []
    for col, p in T5.items():
        sig = 1 - p["lam"]
        out.append(f"col {col}: sigma = 1 - lambda = {sig:.3f} (s.e. {p['lam_se']:.3f}); "
                   f"t(sigma=0) = {sig / p['lam_se']:.1f}")
    for col in ("II", "III", "IV"):
        a1, a2 = T5[col]["FARE"]
        out.append(f"col {col}: alpha1/alpha2 = {a1 / a2:.2f}")
    out.append("")
    for col in ("III", "IV"):
        a1, a2 = T5[col]["FARE"]
        for k in ("DIRECT", "HUBSIZE", "TRIPS", "TEMPDIFF"):
            b1, b2 = T5[col][k]
            w1, w2 = 100 * b1 / a1, 100 * b2 / a2
            line = f"col {col} WTP {k:8s}: type1 ${w1:8.1f}  type2 ${w2:8.1f}  ratio {w2 / w1:5.1f}"
            if k == "HUBSIZE":
                line += (f" | per sd (0.045): ${SD_HUB * w1:.2f} vs ${SD_HUB * w2:.2f}"
                         f" | at max (0.21): ${MAX_HUB * w1:.1f} vs ${MAX_HUB * w2:.1f}")
            out.append(line)
    out.append("")
    for col in ("I", "II", "III", "IV"):
        sig = 1 - T5[col]["lam"]
        a1, a2 = T5[col]["FARE"]
        s = f"col {col}: small-share semi-elasticity alpha/(1-sigma) per $100: type1 {a1 / (1 - sig):.3f}"
        if a2:
            s += f", type2 {a2 / (1 - sig):.3f}"
        out.append(s)
    out.append("")
    for col, ((p1, p2), (n1, n2)) in T4.items():
        ratio = (p2 / n2) / (p1 / n1)
        out.append(f"col {col}: purchase propensity type2/type1 = {ratio:.1f}; "
                   f"type-2 share of passengers / population = {p2 / n2:.1f}")
        x1, x2 = T4_XI[col]
        m = (p1 * x1 + p2 * x2) / 100
        out.append(f"   mean xi over tickets = {m:.3f}; type1 {100 * (x1 / m - 1):+.1f}%, "
                   f"type2 {100 * (x2 / m - 1):+.1f}%")
    out.append("")
    # Table 3
    out.append(f"Table 3 DENS: cost only -0.088 -> ${8.8:.1f} per 100,000; with demand -0.094 -> $9.4")
    out.append(f"Table 3 HUBSIZE 2.097: one sd {2.097 * SD_HUB * 100:.1f} dollars; at max {2.097 * MAX_HUB * 100:.1f}")
    out.append("DIRECT mean 0.4666602 + NSEGS mean 1.5333398 = "
               f"{0.4666602 + 1.5333398:.7f}; equal s.d. 0.4988892")
    # Footnote 10 mixture
    out.append("fn10 mixture: 0.5987 N(0.2375, 0.0781) + 0.4013 N(0.5946, 0.2752)")
    out.append(f"   mixture mean = {0.5987 * 0.2375 + 0.4013 * 0.5946:.4f}")
    # Table 4 average fare difference
    out.append(f"Table 4 col III fare gap: 351 - 204 = {351 - 204}")
    # Sampling error: Poisson approximation
    for n in (2, 5, 9, 30):
        out.append(f"sd of log count at expected count {n}: about {1 / n ** 0.5:.2f}")
    TXT.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
