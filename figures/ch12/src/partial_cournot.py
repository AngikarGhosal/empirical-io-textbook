"""Partial collusion in the linear Cournot supergame with Nash reversion (Section 12.3).

Units: a - c = 1, slope b = 1. N symmetric firms, each producing q on the
collusive path. Stage profit pi(q) = q (1 - N q), best deviation
pi_d(q) = (1 - (N - 1) q)^2 / 4, Cournot profit 1 / (N + 1)^2.

The text shows that the incentive constraint
    pi_d(q) - pi(q) <= delta / (1 - delta) * (pi(q) - pi_N)
holds for q in [q(delta), 1/(N+1)] with
    q(delta) = [(N+1)^2 (1-delta) + 4 delta] / {(N+1) [(N+1)^2 (1-delta) + 4 N delta]}.
This script checks the formula by brute force and prints the numbers quoted in
the text (N = 2, delta = 0.4), including the implied conduct parameter.
"""
import numpy as np


def pi(q, N):
    return q * (1 - N * q)


def pi_d(q, N):
    return np.maximum(1 - (N - 1) * q, 0.0) ** 2 / 4


def q_formula(delta, N):
    num = (N + 1) ** 2 * (1 - delta) + 4 * delta
    den = (N + 1) * ((N + 1) ** 2 * (1 - delta) + 4 * N * delta)
    return num / den


def q_bruteforce(delta, N):
    """Smallest (most collusive) sustainable q, not below monopoly sharing."""
    qm, qn = 1 / (2 * N), 1 / (N + 1)
    qs = np.linspace(qm, qn, 2_000_001)
    piN = 1 / (N + 1) ** 2
    ok = (1 - delta) * (pi_d(qs, N) - pi(qs, N)) <= delta * (pi(qs, N) - piN) + 1e-15
    return qs[np.argmax(ok)]


def main():
    for N in (2, 3, 5):
        dstar = (N + 1) ** 2 / (N ** 2 + 6 * N + 1)
        print(f"N = {N}: delta* = {dstar:.4f}, q^m = {1/(2*N):.4f}, q^N = {1/(N+1):.4f}")
        for delta in (0.1, 0.2, 0.3, 0.4, 0.5):
            if delta >= dstar:
                continue
            qf, qb = q_formula(delta, N), q_bruteforce(delta, N)
            print(f"   delta = {delta:.2f}: q formula = {qf:.6f}, brute force = {qb:.6f}")
    # numbers quoted in the text
    N, delta = 2, 0.4
    q = q_formula(delta, N)
    Q = N * q
    margin = 1 - Q
    piq = pi(q, N)
    ubar = 4 * delta * (N - 1) / ((N + 1) ** 2 * (1 - delta) + 4 * N * delta)
    theta = margin / Q
    print("\nWorked example N = 2, delta = 0.4 (units a - c = 1, b = 1)")
    print(f"  u-bar = {ubar:.6f} (= 16/86 = {16/86:.6f})")
    print(f"  q(delta) = {q:.6f} (= {(9*0.6+1.6)}/{3*(9*0.6+3.2)} = {7/25.8:.6f})")
    print(f"  total output Q = {Q:.6f}, price minus cost = {margin:.6f}")
    print(f"  profit per firm = {piq:.6f}; collusive 1/8 = {1/8:.6f}; Cournot 1/9 = {1/9:.6f}")
    print(f"  deviation profit = {pi_d(q, N):.6f}")
    print(f"  share of the gain over Cournot captured = {(piq - 1/9)/(1/8 - 1/9):.4f}")
    print(f"  implied conduct parameter theta = (P - c)/(b Q) = {theta:.4f}")
    lhs = pi_d(q, N) - piq
    rhs = delta / (1 - delta) * (piq - 1 / 9)
    print(f"  IC check: lhs = {lhs:.6f}, rhs = {rhs:.6f}")
    # theta(delta) along the frontier, N = 2
    print("\n  theta(delta) for N = 2:")
    for d in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 9 / 17):
        qq = q_formula(d, 2)
        print(f"    delta = {d:.4f}: q = {qq:.4f}, theta = {(1 - 2*qq)/(2*qq):.4f}")


if __name__ == "__main__":
    main()
