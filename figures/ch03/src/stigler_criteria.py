"""Recompute Stigler's (1964) detection criteria and check them against his tables.

Notation follows the textbook: h is the probability that an old buyer repeats with
his supplier (Stigler's p), n_o the number of old buyers, n_s the number of sellers,
lam the rate of appearance of new buyers, and zeta = sqrt(h n_s / ((1 - h) n_o))
(Stigler's theta).  Every function returns a percentage of average sales.
"""

import math


def zeta(h, n_o, n_s):
    return math.sqrt(h * n_s / ((1 - h) * n_o))


def criterion_I(h, n_o, n_s):
    """Largest undetected gain from one rival (Table 1)."""
    z = zeta(h, n_o, n_s)
    root = math.sqrt(n_s * (n_s - 2) * (1 - h) * (1 + z) / n_o)
    return 100 * (z * (1 - h) + root) / (n_s - 1)


def criterion_II(h, buyers_per_seller):
    """Old customers a price cutter can retain (Table 2), 2 sigma rule."""
    return 100 * 2 * math.sqrt((1 - h) * h / buyers_per_seller)


def criterion_III(lam, n_o, n_s):
    """Additional new customers obtainable (Table 3)."""
    return 100 * 2 / (1 + lam) * math.sqrt((n_s - 1) * lam / n_o)


def criterion_IV(h, n_small, buyers_per_small, K):
    """Gain of a firm against a rival K times as large (Table 4).

    Stigler's header: (1/(n_s - K)) [theta (1-h) sqrt(K)
    + sqrt(n_s K (1-h)(n_s - K - 1)(1 + theta/sqrt(K)) / n_o)], with
    n_s - K + 1 small firms counted as 'No. of firms' and n_o/n_s buyers per
    small seller.
    """
    n_s = n_small + K - 1
    n_o = buyers_per_small * n_s
    z = zeta(h, n_o, n_s)
    root = math.sqrt(n_s * K * (1 - h) * (n_s - K - 1) * (1 + z / math.sqrt(K)) / n_o)
    return 100 * (z * (1 - h) * math.sqrt(K) + root) / (n_s - K)


if __name__ == "__main__":
    # Table 1 cells as printed (PDF pp. 10-11 of the 2010 reprint).
    table1 = {
        (0.95, 20): [6.9, 11.3, 11.3, 11.4, 11.8, 12.7],
        (0.95, 100): [3.1, 4.4, 4.3, 4.3, 4.2, 4.4],
        (0.95, 400): [1.5, 2.1, 2.0, 1.9, 1.8, 1.8],
        (0.90, 100): [4.2, 6.0, 5.8, 5.6, 5.4, 5.5],
        (0.90, 200): [3.0, 4.1, 3.9, 3.8, 3.6, 3.6],
        (0.80, 20): [12.6, 19.3, 18.9, 18.7, 18.6, 19.4],
        (0.80, 100): [5.7, 8.0, 7.7, 7.4, 7.1, 7.1],
        (0.70, 20): [14.5, 22.3, 21.8, 21.5, 21.2, 21.9],
        (0.70, 100): [6.5, 9.3, 9.0, 8.7, 8.2, 8.2],
    }
    sellers = [2, 3, 4, 5, 10, 20]
    worst = 0.0
    for (h, n_o), printed in table1.items():
        calc = [criterion_I(h, n_o, s) for s in sellers]
        worst = max(worst, max(abs(a - b) for a, b in zip(calc, printed)))
        print(f"Table 1 h={h:.2f} n_o={n_o:3d}:", " ".join(f"{c:5.1f}" for c in calc),
              " printed:", printed)
    print(f"largest absolute gap, Table 1: {worst:.2f}")

    # Table 2 (h = 0.95, 0.90, 0.50)
    for h, printed in [(0.95, [13.8, 9.7, 6.2, 4.4]), (0.90, [19.0, 13.4, 8.5, 6.0]),
                       (0.50, [31.6, 22.4, 14.1, 10.0])]:
        calc = [criterion_II(h, b) for b in (10, 20, 50, 100)]
        print(f"Table 2 h={h:.2f}:", " ".join(f"{c:5.1f}" for c in calc), " printed:", printed)

    # Table 3 (lam = 1/100 and 1/10)
    for lam, n_o, printed in [(0.01, 20, [4.4, 6.3, 7.7, 8.9, 13.3, 19.3]),
                              (0.10, 20, [12.9, 18.2, 22.3, 25.7, 38.6, 56.0]),
                              (0.10, 100, [5.8, 8.1, 10.0, 11.5, 17.2, 25.1])]:
        calc = [criterion_III(lam, n_o, s) for s in sellers]
        print(f"Table 3 lam={lam:.2f} n_o={n_o}:", " ".join(f"{c:5.1f}" for c in calc),
              " printed:", printed)

    # Table 4 (h = 0.9, 10 firms, 10 buyers per small seller; K = 1, 2, 5, 9)
    for nf, b, printed in [(10, 10, [5.4, 7.2, 10.7, 14.0]), (2, 10, [9.5, 13.4, 21.2, 28.5])]:
        calc = [criterion_IV(0.9, nf, b, K) for K in (1, 2, 5, 9)]
        print(f"Table 4 h=0.90 firms={nf} buyers/small={b}:", " ".join(f"{c:5.1f}" for c in calc),
              " printed:", printed)

    # Text example: 100 customers, 10 firms, h = 0.9 -> 5.4 per rival, x9 rivals
    g = criterion_I(0.9, 100, 10)
    print(f"text example: one rival {g:.1f}, nine rivals {9 * g:.1f}")
    g = criterion_I(0.9, 200, 20)
    print(f"text example: 20 firms 200 customers one rival {g:.1f}, nine rivals {9 * g:.1f}")
    # Section II detection probability
    for n in (100, 1000):
        print(f"detection 1-(0.99)^{n} = {1 - 0.99 ** n:.5f}")
