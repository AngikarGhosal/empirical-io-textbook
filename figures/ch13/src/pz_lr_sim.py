"""Size and power of the rank-based test for phantom bidding (fig:13-pz-sim).

Monte Carlo illustration for Section 13.9.1 (exploded logit, Proposition prop:13-exploded-logit)
and Section 13.9.2 (Porter and Zona 1993).  The design is our own.

Model in standardized form.  In auction t there are n_t bidders with K = 2 standardized cost
shifters Z_it ~ N(0, I).  Bidder i's latent "competitiveness" is
        U_it = btilde' Z_it + zeta_it,   zeta_it i.i.d. standard type-I extreme value,
and the lowest bid belongs to the bidder with the highest U_it (Proposition prop:13-exploded-logit).

  * Competitive scenario: the whole ranking of bids is the ranking of U_it (exploded logit).
  * Phantom scenario: the lowest bid is still the bidder with the highest U_it (the designated
    winner is the lowest-cost member and bids competitively), but the order of the higher bids
    is a uniformly random permutation that ignores Z_it (complementary bids).

Test: estimate btilde from the low-rank choices only, from the higher-rank choices only, and
from all ranks; LR = 2 (l_low + l_high - l_all), compared with chi-square(K).

Design sizes mimic the cartel data of Porter and Zona (1993, Table 6): 50 auctions with two or
more cartel bids and about 35 higher-rank choices (n_t = 2 + Binomial(2, 0.35)).

Outputs: figures/ch13/pz_lr_sim.pdf and figures/ch13/src/pz_lr_sim_output.txt
"""
from pathlib import Path

import numpy as np
from scipy import optimize, stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE.parent / "pz_lr_sim.pdf"
OUT_PNG = HERE / "pz_lr_sim_preview.png"
OUT_TXT = HERE / "pz_lr_sim_output.txt"

K = 2
BTILDE = np.array([0.5, -0.5])
MAXN = 4                      # n_t = 2 + Binomial(2, 0.35) <= 4
SEED = 20261022
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"


def simulate(n_auctions, phantom, rng):
    """Return choice-set arrays (Z, mask, chosen, rank) for one simulated sample."""
    sets_z, sets_mask, rank_of_set = [], [], []
    n_bids = 2 + rng.binomial(2, 0.35, size=n_auctions)
    for n in n_bids:
        z = rng.standard_normal((n, K))
        u = z @ BTILDE + rng.gumbel(size=n)
        order = np.argsort(-u)                     # competitive ranking (low bid first)
        if phantom and n > 2:
            rest = order[1:].copy()
            rng.shuffle(rest)                      # higher bids ignore costs
            order = np.concatenate([order[:1], rest])
        for j in range(n - 1):                     # n - 1 nontrivial choices
            alts = order[j:]
            zz = np.zeros((MAXN, K))
            mm = np.zeros(MAXN, dtype=bool)
            zz[: len(alts)] = z[alts]
            mm[: len(alts)] = True
            sets_z.append(zz)                      # chosen alternative is in slot 0
            sets_mask.append(mm)
            rank_of_set.append(j)
    return np.array(sets_z), np.array(sets_mask), np.array(rank_of_set)


def negll(b, z, mask):
    v = z @ b
    v = np.where(mask, v, -np.inf)
    vmax = v.max(axis=1, keepdims=True)
    lse = vmax[:, 0] + np.log(np.exp(v - vmax).sum(axis=1))
    ll = v[:, 0] - lse
    p = np.exp(v - lse[:, None])
    grad = z[:, 0, :] - np.einsum("sk,skd->sd", p, z)
    return -ll.sum(), -grad.sum(axis=0)


def fit(z, mask):
    if len(z) == 0:
        return 0.0
    res = optimize.minimize(negll, np.zeros(K), args=(z, mask), jac=True, method="BFGS")
    return -res.fun


def lr_stat(z, mask, rank):
    low = rank == 0
    high = ~low
    l_all = fit(z, mask)
    l_low = fit(z[low], mask[low])
    l_high = fit(z[high], mask[high])
    return 2.0 * (l_low + l_high - l_all), int(high.sum())


def run(n_auctions, phantom, reps, rng):
    out = np.empty(reps)
    nh = np.empty(reps)
    for r in range(reps):
        z, m, rk = simulate(n_auctions, phantom, rng)
        out[r], nh[r] = lr_stat(z, m, rk)
    return out, nh


def main():
    rng = np.random.default_rng(SEED)
    lines = []
    crit05, crit10 = stats.chi2.ppf(0.95, K), stats.chi2.ppf(0.90, K)
    lines.append(f"K = {K}, btilde = {BTILDE.tolist()}, chi2 crit 5% = {crit05:.3f}, 10% = {crit10:.3f}")

    reps_a = 2000
    lr0, nh0 = run(50, False, reps_a, rng)
    lr1, nh1 = run(50, True, reps_a, rng)
    lines.append(f"Panel (a): T = 50 auctions, {reps_a} replications per scenario")
    lines.append(f"  mean number of higher-rank choices: {nh0.mean():.1f}")
    for name, lr in [("competitive", lr0), ("phantom", lr1)]:
        lines.append(
            f"  {name:11s}: mean LR = {lr.mean():.2f}, reject 5% = {np.mean(lr > crit05):.3f}, "
            f"reject 10% = {np.mean(lr > crit10):.3f}"
        )

    t_grid = [25, 50, 100, 200, 400]
    reps_b = 600
    pow05, size05 = [], []
    for t in t_grid:
        l0, _ = run(t, False, reps_b, rng)
        l1, _ = run(t, True, reps_b, rng)
        size05.append(np.mean(l0 > crit05))
        pow05.append(np.mean(l1 > crit05))
    lines.append(f"Panel (b): {reps_b} replications per point")
    for t, s, p in zip(t_grid, size05, pow05):
        lines.append(f"  T = {t:4d}: rejection rate competitive = {s:.3f}, phantom = {p:.3f}")

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax = axes[0]
    bins = np.linspace(0, 25, 51)
    ax.hist(lr0, bins=bins, density=True, color=BLUE, alpha=0.55, label="competitive bids",
            edgecolor="white", linewidth=0.4)
    ax.hist(lr1, bins=bins, density=True, color=ORANGE, histtype="step", linewidth=1.6,
            label="phantom higher bids")
    xs = np.linspace(0.01, 25, 400)
    ax.plot(xs, stats.chi2.pdf(xs, K), color=INK, lw=1.2, label=r"$\chi^2_2$ density")
    ax.axvline(crit05, color=INK2, lw=0.8, ls="--")
    ax.text(crit05 + 0.4, 0.33, "5% cutoff", color=INK2, fontsize=7.5, va="top")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("LR statistic")
    ax.set_ylabel("density")
    ax.set_title("(a) Distribution of LR, 50 auctions", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(t_grid, pow05, color=ORANGE, lw=2, marker="o", ms=4, label="phantom higher bids")
    ax.plot(t_grid, size05, color=BLUE, lw=2, marker="o", ms=4, label="competitive bids")
    ax.axhline(0.05, color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(t_grid)
    ax.set_xticklabels([str(t) for t in t_grid])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("number of auctions $T$")
    ax.set_ylabel("rejection rate, 5% test")
    ax.set_title("(b) Size and power", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=150)
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
