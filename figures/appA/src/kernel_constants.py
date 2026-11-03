"""
Kernel constants for Appendix A, Section A.10 (tab:A-kernels), computed exactly with sympy.

For each second-order kernel K we compute
    int K            (must equal 1)
    mu2   = int u^2 K(u) du          (second moment)
    kappa2 = int K(u)^2 du           (roughness)
    rK1   = int K'(u)^2 du           (roughness of the derivative; enters Var of f-hat')
    delta = (kappa2 / mu2^2)^(1/5)   (canonical bandwidth factor: h*_K is proportional to delta)
    ratio = delta_K / delta_Gauss    (factor that converts a Gaussian rule of thumb to kernel K)
    eff   = sqrt(mu2_E) kappa2_E / (sqrt(mu2) kappa2)
            (AMISE efficiency relative to Epanechnikov, as a sample-size ratio)
Also: the normal-reference roughness int f''^2 = 3 / (8 sqrt(pi) sigma^5) and the Silverman
constant (4/3)^(1/5); the boundary constants int_0^1 u K(u) du used for the Nadaraya-Watson
boundary bias; and the asymptotic variances of the order-statistic inversion estimator of
Section A.10.1 compared with observing all N values.

Output: figures/appA/src/kernel_constants.json (every number quoted in the text).
"""
import os
import json
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
u, s, x = sp.symbols("u sigma x", real=True)
pos = sp.Symbol("sigma", positive=True)

ind = lambda expr: expr  # kernels below are defined on [-1, 1]
kernels = {
    "Uniform": (sp.Rational(1, 2), (-1, 1)),
    "Epanechnikov": (sp.Rational(3, 4) * (1 - u**2), (-1, 1)),
    "Biweight": (sp.Rational(15, 16) * (1 - u**2) ** 2, (-1, 1)),
    "Triweight": (sp.Rational(35, 32) * (1 - u**2) ** 3, (-1, 1)),
    "Gaussian": (sp.exp(-u**2 / 2) / sp.sqrt(2 * sp.pi), (-sp.oo, sp.oo)),
}

rows = {}
for name, (K, (a, b)) in kernels.items():
    one = sp.simplify(sp.integrate(K, (u, a, b)))
    mu2 = sp.simplify(sp.integrate(u**2 * K, (u, a, b)))
    kap = sp.simplify(sp.integrate(K**2, (u, a, b)))
    rk1 = sp.simplify(sp.integrate(sp.diff(K, u) ** 2, (u, a, b)))
    m1p = sp.simplify(sp.integrate(u * K, (u, 0, b)))   # int_0^1 u K(u) du
    delta = sp.simplify((kap / mu2**2) ** sp.Rational(1, 5))
    rows[name] = dict(int=one, mu2=mu2, kappa2=kap, rK1=rk1, m1plus=m1p, delta=delta)

dG = rows["Gaussian"]["delta"]
E = rows["Epanechnikov"]
out = {"kernels": {}}
for name, r in rows.items():
    eff = sp.sqrt(E["mu2"]) * E["kappa2"] / (sp.sqrt(r["mu2"]) * r["kappa2"])
    out["kernels"][name] = {
        "int": str(r["int"]),
        "mu2": str(r["mu2"]), "mu2_num": float(r["mu2"]),
        "kappa2": str(r["kappa2"]), "kappa2_num": float(r["kappa2"]),
        "rK1": str(r["rK1"]), "rK1_num": float(r["rK1"]),
        "m1plus": str(r["m1plus"]), "m1plus_num": float(r["m1plus"]),
        "delta_num": float(r["delta"]),
        "ratio_to_gauss": float(r["delta"] / dG),
        "eff_vs_epa": float(eff),
    }

# Normal reference: f = N(0, sigma^2); int f''^2 and the Silverman constant.
f = sp.exp(-x**2 / (2 * pos**2)) / (sp.sqrt(2 * sp.pi) * pos)
Rf2 = sp.simplify(sp.integrate(sp.diff(f, x, 2) ** 2, (x, -sp.oo, sp.oo)))
hstar5 = sp.simplify(rows["Gaussian"]["kappa2"] / (rows["Gaussian"]["mu2"] ** 2 * Rf2))  # times 1/n
out["normal_reference"] = {
    "int_f2pp": str(Rf2),
    "h5_times_n": str(hstar5),
    "silverman_constant": float(sp.Rational(4, 3) ** sp.Rational(1, 5)),
}

# Order-statistic inversion: asymptotic variance (per auction) of F-hat(v) from the price only,
# AV = F_w (1 - F_w) / phi'(F)^2, versus F (1 - F) / N when all N values are observed.
def phi(xx, N):
    return N * xx ** (N - 1) * (1 - xx) + xx**N


def dphi(xx, N):
    return N * (N - 1) * xx ** (N - 2) * (1 - xx)


osr = {}
for N in (2, 4, 6):
    for F in (0.2, 0.5, 0.8):
        Fw = phi(F, N)
        av = Fw * (1 - Fw) / dphi(F, N) ** 2
        allv = F * (1 - F) / N
        osr[f"N{N}_F{F}"] = {"Fw": Fw, "AV_price": av, "AV_all": allv, "ratio": av / allv}
out["orderstat"] = osr

with open(os.path.join(HERE, "kernel_constants.json"), "w") as fh:
    json.dump(out, fh, indent=2)

for name, r in out["kernels"].items():
    print(f"{name:13s} mu2={r['mu2']:>10s} kappa2={r['kappa2']:>22s} ({r['kappa2_num']:.4f}) "
          f"rK1={r['rK1_num']:.4f} m1+={r['m1plus_num']:.4f} ratio={r['ratio_to_gauss']:.4f} "
          f"eff={r['eff_vs_epa']:.4f}")
print(out["normal_reference"])
for k, v in osr.items():
    print(k, {kk: round(vv, 4) for kk, vv in v.items()})
