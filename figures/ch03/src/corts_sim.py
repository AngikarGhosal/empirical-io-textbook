"""Conduct-parameter estimate versus true conduct in a Bertrand supergame (fig:03-corts-sim).

Worked illustration for Section 3.6.2 (the Corts critique). The model is our own, in the
spirit of Rotemberg and Saloner (1986) and Corts (1999):

  * N symmetric Bertrand firms, constant marginal cost c, inverse demand P = a_t - Q.
  * The demand state a_t is i.i.d. over periods and observed by firms before they price.
  * Grim-trigger strategies: any deviation leads to P = c (zero profit) forever.
  * The most collusive stationary equilibrium caps industry profit at K:
        (N - 1) Pi(p, a) <= delta / (1 - delta) * V,  V = E[Pi(p(a'), a')],
    so Pi(p, a) <= K with K = delta V / ((1 - delta)(N - 1)).
  * Price p(a) = monopoly price (a + c)/2 if pi_m(a) = (a - c)^2/4 <= K, otherwise the
    LOWER root of (p - c)(a - p) = K.
  * K solves the fixed point K = s E[min(pi_m(a), K)], s = delta / ((1 - delta)(N - 1)).
    We select the largest fixed point. For s < 1 it is K = 0. At the cutoff s = 1
    (delta = 1 - 1/N) every K in [0, min_a pi_m(a)] is a fixed point, and the largest is
    min_a pi_m(a) = pi_m(A_LO). For s > 1 we iterate downward from a large starting value.

Conduct in state a: theta(a) = (p - c) / Q (inverse demand slope is -1, so the markup term
m = Q / |dQ/dP| equals Q).  theta(a) also equals the elasticity-adjusted Lerner index.
The researcher knows demand. Marginal cost is constant in output, but its level is unknown,
so the researcher estimates P = c + theta * m + error by IV with a constant and with the
demand state a as instrument (in the book the demand state is written A_t):
        theta_hat = Cov(a, P - c) / Cov(a, Q).

Population moments are computed by Gauss-Legendre quadrature over a ~ U[A_LO, A_HI].
A Monte Carlo IV regression on simulated data checks the quadrature numbers.

Outputs: figures/ch03/corts_sim.pdf and figures/ch03/src/corts_sim_output.txt
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "corts_sim.pdf"
OUT_TXT = HERE / "corts_sim_output.txt"

A_LO, A_HI = 6.0, 10.0   # support of the demand intercept a_t ~ U[A_LO, A_HI]
C = 0.0                  # constant marginal cost
NODES, WEIGHTS = np.polynomial.legendre.leggauss(400)
A_GRID = 0.5 * (A_HI - A_LO) * NODES + 0.5 * (A_HI + A_LO)
W_GRID = 0.5 * WEIGHTS   # weights of the uniform density on [A_LO, A_HI] (sum to 1)


def expect(f_vals):
    """E[f(a)] for a ~ U[A_LO, A_HI], f evaluated on A_GRID."""
    return float(np.sum(W_GRID * f_vals))


def pi_m(a):
    """Industry monopoly profit in state a."""
    return (a - C) ** 2 / 4.0


def largest_fixed_point(delta, n_firms, tol=1e-13, max_iter=200000, s_tol=1e-9):
    """Largest K with K = s E[min(pi_m(a), K)].

    s < 1: only K = 0 solves the equation.
    s = 1 (delta = 1 - 1/N): the solutions are [0, pi_m(A_LO)], and we return the upper endpoint.
    s > 1: the unique positive solution, found by iterating downward.
    """
    s = delta / ((1.0 - delta) * (n_firms - 1))
    if s < 1.0 - s_tol:
        return 0.0
    if s <= 1.0 + s_tol:
        return float(pi_m(A_LO))
    k = 10.0 * s * pi_m(A_HI) + 1.0          # start above the largest fixed point
    for _ in range(max_iter):
        k_new = s * expect(np.minimum(pi_m(A_GRID), k))
        if abs(k_new - k) < tol:
            return k_new
        k = k_new
    raise RuntimeError("fixed point iteration did not converge")


def price(a, k):
    """Collusive price in state a given the industry profit cap k."""
    a = np.asarray(a, dtype=float)
    pm = 0.5 * (a + C)
    if k <= 0.0:
        return np.full_like(a, C)
    gap = pi_m(a) - k
    return np.where(gap <= 0.0, pm, pm - np.sqrt(np.maximum(gap, 0.0)))


def outcomes(delta, n_firms):
    """Population summary statistics for one discount factor."""
    k = largest_fixed_point(delta, n_firms)
    p = price(A_GRID, k)
    q = A_GRID - p
    markup = p - C
    theta = markup / q
    cov_a_markup = expect(A_GRID * markup) - expect(A_GRID) * expect(markup)
    cov_a_q = expect(A_GRID * q) - expect(A_GRID) * expect(q)
    theta_hat = cov_a_markup / cov_a_q
    share_constrained = expect((pi_m(A_GRID) > k).astype(float)) if k > 0 else 1.0
    # cutoff state above which the cap binds
    a_star = C + 2.0 * np.sqrt(k) if k > 0 else A_LO
    return {
        "delta": delta, "K": k, "mean_theta": expect(theta), "theta_hat": theta_hat,
        "share_constrained": share_constrained, "a_star": a_star,
        "mean_markup": expect(markup), "mean_q": expect(q),
        "level_ratio": expect(markup) / expect(q),
    }


def monte_carlo_check(delta, n_firms, t_obs=200000, seed=257):
    """IV estimate on simulated data (just-identified, instrument a, with constant)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(A_LO, A_HI, size=t_obs)
    k = largest_fixed_point(delta, n_firms)
    p = price(a, k)
    q = a - p
    y = p - C
    za = a - a.mean()
    return float(np.sum(za * (y - y.mean())) / np.sum(za * (q - q.mean())))


def main():
    lines = []
    lines.append("Bertrand supergame with i.i.d. demand: a ~ U[%.0f, %.0f], c = %.0f" % (A_LO, A_HI, C))
    lines.append("theta(a) = (p - c)/Q; theta_hat = Cov(a, P - c)/Cov(a, Q)")
    lines.append("")
    results = {}
    for n_firms in (2, 3):
        d0 = 1.0 - 1.0 / n_firms
        lines.append("N = %d (collusion above cost is sustainable iff delta >= %.4f; at equality the "
                     "most collusive cap is K = pi_m(A_LO) = %.4f)" % (n_firms, d0, pi_m(A_LO)))
        # discount factor at which monopoly is sustainable in every state
        e_pim = expect(pi_m(A_GRID))
        s_full = pi_m(A_HI) / e_pim
        d_full = s_full * (n_firms - 1) / (1.0 + s_full * (n_firms - 1))
        lines.append("  E[pi_m] = %.4f, max pi_m = %.4f; monopoly in all states iff delta >= %.4f"
                     % (e_pim, pi_m(A_HI), d_full))
        lines.append("  %6s %9s %9s %10s %11s %9s %9s" % ("delta", "K", "a_star", "constr.sh",
                                                          "mean theta", "theta_hat", "MC check"))
        grid_report = [d0, d0 + 0.0001, d0 + 0.005, d0 + 0.01, d0 + 0.02, d0 + 0.03, d0 + 0.05,
                       d0 + 0.07, d_full, d_full + 0.05]
        for d in grid_report:
            r = outcomes(d, n_firms)
            mc = monte_carlo_check(d, n_firms)
            lines.append("  %6.4f %9.4f %9.4f %10.4f %11.4f %9.4f %9.4f"
                         % (d, r["K"], r["a_star"], r["share_constrained"], r["mean_theta"],
                            r["theta_hat"], mc))
        lines.append("")
        dgrid = np.concatenate([np.linspace(0.30, d0 - 1e-6, 40), [d0], np.linspace(d0 + 1e-4, 0.95, 400)])
        res = [outcomes(d, n_firms) for d in dgrid]
        results[n_firms] = (dgrid, res)

    # Locus of equilibria for the right panel.
    d_locus = 0.51
    r_loc = outcomes(d_locus, 2)
    th_grid = (price(A_GRID, r_loc["K"]) - C) / (A_GRID - price(A_GRID, r_loc["K"]))
    lines.append("Locus panel: N = 2, delta = %.2f: K = %.4f, a* = %.4f, mean theta = %.4f, "
                 "theta_hat = %.4f, E[P-c]/E[Q] = %.4f, theta range [%.4f, %.4f], "
                 "E[Q] = %.4f, E[P-c] = %.4f"
                 % (d_locus, r_loc["K"], r_loc["a_star"], r_loc["mean_theta"], r_loc["theta_hat"],
                    r_loc["level_ratio"], th_grid.min(), th_grid.max(), r_loc["mean_q"],
                    r_loc["mean_markup"]))
    # first discount factor at which the IV estimate turns positive (N = 2)
    dgrid2, res2 = results[2]
    pos = [r["delta"] for r in res2 if r["delta"] >= 0.5 and r["theta_hat"] > 0]
    lines.append("N = 2: theta_hat first positive at delta = %.4f" % min(pos))
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.75))

    ax = axes[0]
    dgrid, res = results[2]
    left = dgrid < 0.5
    right = ~left
    mt = np.array([r["mean_theta"] for r in res])
    th = np.array([r["theta_hat"] for r in res])
    for mask in (left, right):
        ax.plot(dgrid[mask], mt[mask], color="#1f4e79", lw=1.6)
        ax.plot(dgrid[mask], th[mask], color="#c55a11", lw=1.6, ls="--")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.axhline(1.0, color="0.6", lw=0.6, ls=":")
    ax.axvline(0.5, color="0.75", lw=0.6, ls=":")
    ax.text(0.62, 0.90, r"mean $\theta_t$", color="#1f4e79", fontsize=8)
    ax.text(0.515, -0.55, r"IV estimate $\hat\theta$", color="#c55a11", fontsize=8)
    ax.set_xlabel(r"discount factor $\delta$")
    ax.set_ylabel("conduct")
    ax.set_xlim(0.30, 0.80)
    ax.set_ylim(-0.75, 1.15)
    ax.set_title(r"(a) $N = 2$: level versus estimate", fontsize=9)

    ax = axes[1]
    k = r_loc["K"]
    a_line = np.linspace(A_LO, A_HI, 400)
    p_line = price(a_line, k)
    q_line = a_line - p_line
    ax.plot(q_line, p_line - C, color="#1f4e79", lw=2.0, label="equilibrium locus")
    qq = np.linspace(0.0, 9.0, 50)
    ax.plot(qq, qq, color="0.55", lw=0.8, ls=":")
    ax.text(1.5, 2.25, r"$\theta = 1$", color="0.4", fontsize=8)
    qbar, mbar = r_loc["mean_q"], r_loc["mean_markup"]
    ax.plot(qq, r_loc["level_ratio"] * qq, color="#548235", lw=1.0, ls="-.")
    ax.text(7.7, 2.65, "ray through\nmean point", color="#548235", fontsize=7, ha="center", va="top")
    qs = np.linspace(2.4, 8.6, 20)
    ax.plot(qs, mbar + r_loc["theta_hat"] * (qs - qbar), color="#c55a11", lw=1.2, ls="--")
    ax.text(5.0, 3.55, r"slope $\hat\theta$", color="#c55a11", fontsize=8)
    ax.plot([qbar], [mbar], "o", color="black", ms=3)
    ax.set_xlim(0.0, 9.0)
    ax.set_ylim(0.0, 5.2)
    ax.set_xlabel(r"quantity $Q_t$ (equal to $m_t$)")
    ax.set_ylabel(r"markup $P_t - c$")
    ax.set_title(r"(b) $N = 2$, $\delta = 0.51$", fontsize=9)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT_PDF)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
