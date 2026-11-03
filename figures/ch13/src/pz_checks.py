"""Arithmetic checks for Sections 13.9 and 13.10 (Chapter 13, Part 5).

Every input number is copied from the papers' tables, read visually:
  * Porter and Zona (1993), NBER WP 4013 version, Tables 2 and 4-6 (PDF pp. 17, 33, 35, 36);
  * Porter and Zona (1999), NBER WP 6037 version, Tables 1-3, 8-10, 17 (PDF pp. 36-39, 46-48, 55);
  * other chapter cases: Asker (2010) Table 7, Porter (1983) Table 4, Levenstein (1997),
    Carlton, Bamberger and Epstein (1995) Table 6, Erutku and Hildebrand (2010) Table 6,
    Miller (2010) Table 2 (as reproduced in Parts 1-4 of the chapter).
The script recomputes every statistic that the text labels "our computation".
Output: figures/ch13/src/pz_checks_output.txt
"""
from pathlib import Path
import math

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "pz_checks_output.txt"
lines = []


def say(s=""):
    lines.append(s)
    print(s)


# ---------------------------------------------------------------- PZ (1993)
say("Porter and Zona (1993), NBER WP 4013")
# Table 4 degrees of freedom: observations - auction constants - reported slopes
say(f"  df all firms = 476 - 75 - 6 = {476 - 75 - 6} (Table 4: 395)")
say(f"  df competitive = 319 - 75 - 6 = {319 - 75 - 6} (Table 4: 238)")
say(f"  cartel parameters = 157 - 81 = {157 - 81}; slopes 4 -> auction constants {157 - 81 - 4}")
say(f"  Chow: unrestricted residual df = 238 + 81 = {238 + 81}; numerator df = 395 - 319 = {395 - 319}")
say(f"  Chow F(76,319) = 9.38: p = {stats.f.sf(9.38, 76, 319):.1e}")
say(f"  Wald 494.7 chi2(6) p = {stats.chi2.sf(494.7, 6):.1e}; cartel Wald 28.4 chi2(4) p = {stats.chi2.sf(28.4, 4):.1e}")
say(f"  all-firm Wald 21.9 chi2(6) p = {stats.chi2.sf(21.9, 6):.4f}")
# Table 5 (competitive ranks) and Table 6 (cartel ranks): log likelihoods
ll_c = {"all": -291.4, "low": -89.85, "high": -199.4}
ll_k = {"all": -73.97, "low": -44.58, "high": -24.92}
lr_c = 2 * (ll_c["low"] + ll_c["high"] - ll_c["all"])
lr_k = 2 * (ll_k["low"] + ll_k["high"] - ll_k["all"])
say(f"  LR competitive = {lr_c:.2f}; chi2(6) cdf = {stats.chi2.cdf(lr_c, 6):.3f}, p = {stats.chi2.sf(lr_c, 6):.3f}")
say(f"  LR cartel      = {lr_k:.2f}; chi2(4) cdf = {stats.chi2.cdf(lr_k, 4):.3f}, p = {stats.chi2.sf(lr_k, 4):.3f}")
say(f"  chi2 critical values: 5% df6 = {stats.chi2.ppf(.95, 6):.2f}, df4 = {stats.chi2.ppf(.95, 4):.2f}, 10% df4 = {stats.chi2.ppf(.90, 4):.2f}")
# choice-set bookkeeping
say(f"  competitive choice sets: 319 - 75 = {319 - 75} = 75 low + {319 - 75 - 75} higher (Table 5: 75, 169)")
say(f"  cartel: 85 choice sets in 50 auctions -> {85 + 50} cartel bids; higher = {85 - 50} (Table 6: 35)")
# Table 4, competitive column: UTIL and UTILSQ
b_u, b_u2 = -0.0973, 0.1720
for u in (0.3, 1.0):
    say(f"  log-bid effect of UTIL = {u}: {b_u * u + b_u2 * u * u:+.4f}")
say(f"  minimizing UTIL = {-b_u / (2 * b_u2):.3f}; value at min = {b_u ** 2 / (-4 * b_u2):+.4f}")
say(f"  ISLAND -0.0334 -> {100 * (math.exp(-0.0334) - 1):.2f} percent")
# scaling check: MNL coefficient = (pi/sqrt 6) x bid-equation coefficient
scale = math.pi / math.sqrt(6)
say(f"  pi/sqrt(6) = {scale:.4f}")
for name, gls, mnl in [("UTIL", -0.0973, -0.0070), ("UTILSQ", 0.1720, 0.0986),
                       ("CAP", -1.2691, -1.888), ("CAPSQ", 4.8519, 6.869)]:
    say(f"  {name}: GLS x scale = {gls * scale:+.3f} vs MNL all ranks {mnl:+.3f}")
# Table 2 diagonal (bids by each cartel firm)
diag = [48, 86, 71, 3, 20]
say(f"  Table 2 diagonal sum = {sum(diag)} (text: 228 cartel bids)")
say(f"  Firm 2 bids with Firm 3: 37/48 = {37 / 48:.3f}")
# uniform IPV gap between the two lowest bids: E = (n-1)/(n(n+1))
for n in range(2, 7):
    say(f"  uniform IPV procurement: E[b(2)-b(1)] with n={n}: {(n - 1) / (n * (n + 1)):.4f}")
# Monte Carlo check of the gap formula
rng = np.random.default_rng(1)
for n in (2, 4):
    c = np.sort(rng.random((200000, n)), axis=1)
    b = c + (1 - c) / n
    say(f"  simulated gap n={n}: {np.mean(b[:, 1] - b[:, 0]):.4f}")

# ---------------------------------------------------------------- PZ (1999)
say("")
say("Porter and Zona (1999), NBER WP 6037")
price = [0.12823, 0.12950, 0.12954, 0.12883, 0.13133, 0.13215, 0.13042, 0.13099, 0.13378, 0.13886, 0.15751]
say(f"  Table 1 average price per half pint 1980-1990 (simple mean) = {np.mean(price):.4f}")
say(f"  Table 1 price change 1980 -> 1990 = {100 * (price[-1] / price[0] - 1):.1f} percent")
cnt = {1: 1739, 2: 1300, 3: 702, 4: 106, 5: 18, 6: 1}
nb = sum(k * v for k, v in cnt.items())
say(f"  Table 3: auctions = {sum(cnt.values())}, bids = {nb}, mean bidders = {nb / sum(cnt.values()):.3f}")
say(f"  Table 2 control group: 2053/73965 = {2053 / 73965:.4f}; wins 1260/2053 = {1260 / 2053:.3f}")
say(f"  demand footnote 4: price coefficient 0.21, t = 1.4 -> se = {0.21 / 1.4:.3f}")
for name, eff in [("base, not closest", 0.00960348), ("bidder FE, not closest", 0.01535559),
                  ("bidder+district FE, not closest", 0.01941482)]:
    say(f"  {name}: 50 miles = {0.5 * eff * 100:.2f} cents = {100 * 0.5 * eff / np.mean(price):.1f} percent of mean price")
say(f"  Table 7 base: direct distance slope for processor = (0.000027 + 0.000009) per mile = {100 * (0.000027 + 0.000009):.4f} $ per 100 miles")
t9 = {"Beatrice": (6358.2, 6279.2, 18), "Borden": (7300.2, 7077.3, 20), "Coors Brothers": (6158.3, 6064.4, 19),
      "H. Meyer": (6552.7, 6313.9, 20), "Louis Trauth": (6480.1, 6320.5, 20), "Meadow Gold": (7527.6, 7275.7, 20)}
for k, (l0, l1, df) in t9.items():
    say(f"  Table 9 {k}: 2*(null - alt) = {2 * (l0 - l1):.1f}; 1% critical chi2({df}) = {stats.chi2.ppf(.99, df):.2f}")
t10 = {"Beatrice": (19, 78, 1.09), "Borden": (26, 386, 10.13), "Coors": (25, 102, 5.02),
       "Meyer": (26, 384, 11.53), "Trauth": (25, 345, 9.66), "Meadow Gold": (25, 771, 7.09)}
for k, (d1, d2, f) in t10.items():
    say(f"  Table 10 {k}: F({d1},{d2}) = {f}: p = {stats.f.sf(f, d1, d2):.4f}")
t17d = [3.0, 11.3, 8.6, 4.5, 6.7, 5.4, 6.5, 3.3, 2.9, -1.6, -0.3]
t17e = [3.2, 40.2, 23.2, 1.1, 19.7, 11.5, 20.5, 49.0, 29.4, 3.4, -8.3]
say(f"  Table 17 col (d) simple mean, 11 years = {np.mean(t17d):.2f}")
say(f"  Table 17 col (d) simple mean, 1980-81 to 1988-89 = {np.mean(t17d[:9]):.2f}")
say(f"  Table 17 col (d) simple mean excluding 1983-84 and 1989-91 = {np.mean([t17d[i] for i in (0, 1, 2, 4, 5, 6, 7, 8)]):.2f}")
say(f"  Table 17 col (e) simple mean, 11 years = {np.mean(t17e):.2f}")
# equivalent-firms example used in the text
shares_c = [1 / 6] * 6
shares_k = [3 / 6, 1 / 6, 1 / 6, 1 / 6]
hc, hk = sum(s * s for s in shares_c), sum(s * s for s in shares_k)
say(f"  example: HHI comp = {hc:.4f} (N^e = {1 / hc:.2f}); HHI col = {hk:.4f} (N^e = {1 / hk:.2f}); Delta = {1 / hc - 1 / hk:.2f}")
say(f"  Table 16: one more equivalent firm -> {100 * (math.exp(-0.0150) - 1):.2f} percent")

# ---------------------------------------------------------------- Section 13.10
say("")
say("Section 13.10")
say(f"  Porter Table 4: 0.2780/0.1673 = {0.2780 / 0.1673:.3f}; 0.2679/0.1612 = {0.2679 / 0.1612:.3f}")
say(f"  Porter Table 4 quantity: 25775/38680 = {25775 / 38680:.3f}; revenue 143309/129423 = {143309 / 129423:.3f}")
say(f"  Levenstein: 40.88/24.75 = {40.88 / 24.75:.3f}; 29.3/16.8 = {29.3 / 16.8:.3f}")
say(f"  Albaek: ln(1.19) - ln(1.019) = {math.log(1.19) - math.log(1.019):.3f}")
say(f"  Carlton et al. IVY 95% CI = [{-322.42 - 1.96 * 678.72:.0f}, {-322.42 + 1.96 * 678.72:.0f}]")
say(f"  Erutku-Hildebrand Montreal: t = {-1.75 / 0.80:.2f}; 95% CI = [{-1.75 - 1.96 * 0.80:.2f}, {-1.75 + 1.96 * 0.80:.2f}]")
say(f"  Erutku-Hildebrand: 1.75/64.0 = {1.75 / 64.0:.3f}")
say(f"  Miller column (4): -13.568 + 12.708 = {-13.568 + 12.708:.3f}")
say(f"  Asker: U.B. 36.99/74.21 = {36.99 / 74.21:.3f}; L.B. 26.50/74.21 = {26.50 / 74.21:.3f}")

# selection of detected cartels: two sampling schemes (Proposition prop:13-selection)
q = 0.2   # per-period hazard of natural breakdown: mean natural duration 1/q = 5
for h in (0.05, 0.2):
    r = (1 - q) * (1 - h)
    # scheme A: discovery after the fact, Pr(detected | L) = 1 - (1-h)^L
    pdet = h / (1 - r)
    ela = (1 / q - q * (1 - h) / (1 - r) ** 2) * (1 - r) / h
    L = np.arange(1, 2000)
    pL = q * (1 - q) ** (L - 1)
    det = 1 - (1 - h) ** L
    ela_num = np.sum(L * pL * det) / np.sum(pL * det)
    # scheme B: detection terminates the cartel (competing risks)
    elb = 1 / (1 - r)
    say(f"  h={h}: scheme A Pr(det) = {pdet:.3f}, E[L|det] = {ela:.2f} (numerical {ela_num:.2f}); "
        f"scheme B E[L|det] = {elb:.2f}; natural mean = {1 / q:.2f}")

OUT.write_text("\n".join(lines) + "\n")
