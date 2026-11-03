"""Back-of-envelope calculations from the published estimates of Athey, Levin and Seira (2011).

All inputs are read off ALS Tables III, IV, V and VII (QJE 126(1), 2011). No random numbers are used.
1. Within-auction correlation of log bids implied by the Gamma-Weibull model (Table IV, ln sigma^2).
2. Weibull shapes for a sale with two loggers and two mills, and the implied lower bound on the
   profit margin (v - b) / v >= 1 / (1 + sum of rivals' shapes).
3. Conversions of log effects (Table III) and dollar figures (Tables V and VII) used in the text.
Prints everything; the chapter quotes these numbers as "our calculation".
"""
import math
from scipy.special import polygamma

# 1. Correlation of log bids within an auction: psi_1(a) / (psi_1(a) + pi^2 / 6), a = 1 / sigma^2.
for region, ln_s2 in [("North", -0.502), ("California", -0.416)]:
    s2 = math.exp(ln_s2)
    a = 1.0 / s2
    psi1 = float(polygamma(1, a))
    corr = psi1 / (psi1 + math.pi ** 2 / 6)
    print(f"{region}: sigma^2 = {s2:.3f}, shape a = {a:.3f}, trigamma = {psi1:.3f}, corr = {corr:.3f}")

# 2. Shapes for (n_L, n_M) = (2, 2) from Table IV (ln rho equation).
coef = {"North": dict(const=1.134, minM=0.015, minL=0.016, mill=0.074, onemill=-0.386, nomill=-0.081),
        "California": dict(const=1.329, minM=-0.009, minL=-0.005, mill=-0.031, onemill=-0.076, nomill=-0.202)}
for region, c in coef.items():
    nL, nM = 2, 2
    lnrhoL = c["const"] + c["minL"] * min(nL, 5) + c["minM"] * min(nM, 5)
    lnrhoM = lnrhoL + c["mill"]
    rL, rM = math.exp(lnrhoL), math.exp(lnrhoM)
    sum_logger = (nL - 1) * rL + nM * rM      # rivals of a logger
    sum_mill = nL * rL + (nM - 1) * rM        # rivals of a mill
    print(f"{region}: rho_L = {rL:.3f}, rho_M = {rM:.3f}; margin bound logger = {1/(1+sum_logger):.3f}, "
          f"mill = {1/(1+sum_mill):.3f}")

# 3. Conversions.
for name, b in [("N logger entry OLS", 0.089), ("N price OLS", 0.094), ("N price match", 0.118),
                ("N price w/ entry OLS", 0.055), ("N price w/ entry match", 0.091), ("mill dummy N", 0.248)]:
    print(f"{name}: exp(b) - 1 = {math.exp(b) - 1:.3f}")
print("10% of $134,000 =", 0.10 * 134000, "; times 1,071 sales =", 0.10 * 134000 * 1071)
# Share of cooperation that rationalizes Table V Northern open prices by linear interpolation.
for col, comp, coll in [("bidding only", 67.9, 44.2), ("bidding + entry", 67.8, 44.1)]:
    q = (comp - 63.3) / (comp - coll)
    print(f"Interpolated cooperation share ({col}): {q:.3f}")
# Percent revenue effects in Table VII.
for name, d, base in [("N endog sealed-open", 3.17, 139.87), ("CA endog sealed-open", 14.18, 168.20),
                      ("N exog sealed-partcoll", 14.00, 124.33), ("N endog sealed-partcoll", 17.72, 125.32),
                      ("CA exog sealed-partcoll", 18.21, 154.43), ("CA endog sealed-partcoll", 29.55, 152.83)]:
    print(f"{name}: {100 * d / base:.1f} percent of the open-auction revenue")
print("N exog revenue difference from columns:", round(138.33 - 138.64, 2),
      "; CA:", round(172.64 - 172.10, 2))
print("Logger entry per 10 sales (Table VII):", round(10 * (3.10 - 2.76), 1))
