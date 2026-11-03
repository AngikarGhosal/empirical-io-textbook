"""Most collusive sustainable price over a deterministic demand cycle (fig:13-bs-cycle).

Illustrative calibration of the model in Section 13.6.1 (not an estimate):
  - N symmetric firms, homogeneous good, constant marginal cost c.
  - Linear demand D_t(p) = a_t - p with a perfectly foreseen cycle
        a_t = 1 + AMP * sin(2 pi t / P).
  - A deviation is punished by T periods of Bertrand pricing (p = c), after which
    collusion resumes.
  - Industry profit at the most collusive sustainable path, V_t = pi_t(pbar_t), is the
    greatest fixed point of
        V_t = min{ pi^m_t , (1/(N-1)) * sum_{s=1}^{T} delta^s V_{t+s} },
    computed by iterating the operator downward from the monopoly profit path.
  - pbar_t solves (p - c)(a_t - p) = V_t on the increasing branch of profit.

Outputs: figures/ch13/bs_cycle.pdf and figures/ch13/src/bs_cycle_numbers.txt
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "bs_cycle.pdf"
TXT = HERE / "bs_cycle_numbers.txt"

N, T, DELTA = 8, 10, 0.99
AMP, C, P = 0.25, 0.2, 48
T1, T2 = 4, 20          # equal demand: a(4) = a(20) (symmetric around the peak at t = 12)


def solve():
    t = np.arange(P)
    a = 1.0 + AMP * np.sin(2.0 * np.pi * t / P)
    pim = (a - C) ** 2 / 4.0
    w = DELTA ** np.arange(1, T + 1)
    V = pim.copy()
    for _ in range(100000):
        R = np.array([w @ V[(tt + np.arange(1, T + 1)) % P] for tt in t]) / (N - 1)
        Vn = np.minimum(pim, R)
        if np.max(np.abs(Vn - V)) < 1e-15:
            V = Vn
            break
        V = Vn
    m = ((a - C) - np.sqrt(np.maximum((a - C) ** 2 - 4.0 * V, 0.0))) / 2.0
    pbar = C + m
    pm = (a + C) / 2.0
    binding = V < pim - 1e-12
    return t, a, pm, pbar, binding


def main():
    t, a, pm, pbar, binding = solve()
    # two cycles for the plot
    tt = np.arange(2 * P)
    A = np.tile(a, 2)
    PM = np.tile(pm, 2)
    PB = np.tile(pbar, 2)
    B = np.tile(binding, 2)

    lines = [
        f"N={N}, T={T}, delta={DELTA}, AMP={AMP}, c={C}, P={P}",
        f"periods with binding IC per cycle: {binding.sum()} of {P}",
        f"t1={T1}: a={a[T1]:.4f}, p^m={pm[T1]:.4f}, pbar={pbar[T1]:.4f}, margin={pbar[T1]-C:.4f}",
        f"t2={T2}: a={a[T2]:.4f}, p^m={pm[T2]:.4f}, pbar={pbar[T2]:.4f}, margin={pbar[T2]-C:.4f}",
        f"min pbar={pbar.min():.4f} at t={int(np.argmin(pbar))}; demand peak at t={int(np.argmax(a))}",
    ]
    TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    blue, gray, ink = "#2a78d6", "#52514e", "#0b0b0b"
    fig, ax = plt.subplots(figsize=(5.6, 2.9))

    # shade periods in which the incentive constraint binds
    for k in range(len(tt)):
        if B[k]:
            ax.axvspan(tt[k] - 0.5, tt[k] + 0.5, color="#e8eef7", lw=0, zorder=0)

    ax.plot(tt, PM, color=gray, lw=1.4, ls="--", label=r"monopoly price $p^m_t$", zorder=2)
    ax.plot(tt, PB, color=blue, lw=2.0, label=r"most collusive sustainable price $\bar p_t$", zorder=3)
    ax.axhline(C, color=gray, lw=0.8, zorder=1)
    ax.text(2 * P - 1, C + 0.008, r"marginal cost $c$", ha="right", va="bottom", color=gray, fontsize=8)

    for tk, lab in [(T1, r"$t_1$"), (T2, r"$t_2$")]:
        ax.plot([tk, tk], [C, PM[tk]], color=ink, lw=0.6, ls=":", zorder=2)
        ax.plot([tk], [PB[tk]], marker="o", ms=5, color=blue, mec="white", mew=1.2, zorder=4)
        ax.text(tk, C - 0.03, lab, ha="center", va="top", fontsize=9)
    ax.annotate(r"$\bar p_{t_1} = p^m_{t_1}$", xy=(T1, PB[T1]), xytext=(T1 + 1.5, PB[T1] - 0.085),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5, color=ink))
    ax.annotate(r"$\bar p_{t_2} < p^m_{t_2}$", xy=(T2, PB[T2]), xytext=(T2 + 2.2, PB[T2] - 0.075),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5, color=ink))
    ax.plot([T1, T2], [PM[T1], PM[T2]], color=ink, lw=0.6, ls=":")

    ax.set_xlim(-0.5, 2 * P - 0.5)
    ax.set_ylim(C - 0.08, 0.78)
    ax.set_xlabel(r"period $t$ (two demand cycles)")
    ax.set_ylabel("price")
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72, 84])
    ax.tick_params(length=2)
    ax.legend(loc="lower center", frameon=False, fontsize=7.5, ncol=1, bbox_to_anchor=(0.66, 0.16))
    fig.tight_layout()
    fig.savefig(OUT)
    fig.savefig(HERE / "bs_cycle_preview.png", dpi=150)


if __name__ == "__main__":
    main()
