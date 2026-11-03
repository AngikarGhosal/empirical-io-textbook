"""Nocke and Whinston (2013 WP, Example 2 and Table 1): replication and figure.

Linear Cournot, P(Q) = 1 - Q, four firms with costs c0 = c2 = 0.5, c1 = 0.55,
c3 = 0.45. Firm 0 can merge with k = 1, 2, 3. Each merger is feasible with
probability 0.75. Conditional on feasibility the post-merger cost is
cbar = 0.2 + (naive cutoff - 0.2) * X with X ~ Beta(alpha, beta).

The script (i) computes the recursively defined cutoff policy, (ii) computes the
first-best, naive and optimal expected consumer-surplus gains (Table 1 of the
paper), and (iii) draws the (Delta Pi, Delta CS) merger curves with the naive and
optimal cutoffs marked (figure fig:11-nw).
"""
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {0: 0.5, 1: 0.55, 2: 0.5, 3: 0.45}
PHI = 0.75
LOW = 0.2


def cournot(costs):
    m = len(costs)
    P = (1.0 + sum(costs)) / (m + 1)
    q = [P - c for c in costs]
    return P, sum(q), q


P0, Q0, q0 = cournot([C[i] for i in range(4)])
CS0 = Q0 ** 2 / 2
pi0 = {i: q0[i] ** 2 for i in range(4)}


def outcome(k, cbar):
    """Delta Pi (bilateral) and Delta CS for merger of firm 0 and k at cost cbar."""
    others = [C[i] for i in range(1, 4) if i != k]
    P, Q, q = cournot([cbar] + others)
    dpi = (P - cbar) ** 2 - pi0[0] - pi0[k]
    dcs = Q ** 2 / 2 - CS0
    return dpi, dcs


naive = {k: C[k] - (P0 - C[0]) for k in (1, 2, 3)}


def draws(k, n, a, b, rng):
    x = rng.beta(a, b, size=n)
    return LOW + (naive[k] - LOW) * x


def vec_outcome(k, cb):
    others = sum(C[i] for i in range(1, 4) if i != k)
    P = (1.0 + cb + others) / 4.0
    Q = (3.0 - cb - others) / 4.0
    dpi = (P - cb) ** 2 - pi0[0] - pi0[k]
    dcs = Q ** 2 / 2 - CS0
    return dpi, dcs


def simulate(cutoffs, a, b, n=2_000_000, seed=1):
    """Return arrays of (dpi, dcs) for each merger, with -inf dpi if infeasible
    or not approved under the given cutoffs (cutoffs[k] = None means never)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in (1, 2, 3):
        cb = draws(k, n, a, b, rng)
        feas = rng.random(n) < PHI
        dpi, dcs = vec_outcome(k, cb)
        ok = feas & (cb <= (cutoffs[k] if cutoffs[k] is not None else -1))
        out[k] = (np.where(ok, dpi, -np.inf), dcs, np.where(feas, dcs, -np.inf))
    return out


def proposed_cs(sim, ks):
    """Delta CS of the proposed merger (max Delta Pi among acceptable ones)."""
    dpis = np.vstack([sim[k][0] for k in ks])
    dcss = np.vstack([sim[k][1] for k in ks])
    best = np.argmax(dpis, axis=0)
    bestpi = dpis[best, np.arange(dpis.shape[1])]
    cs = dcss[best, np.arange(dpis.shape[1])]
    cs = np.where(bestpi > 0, cs, 0.0)
    bestpi = np.where(bestpi > 0, bestpi, 0.0)
    return cs, bestpi


def recursive_cutoffs(a, b):
    cut = {1: naive[1], 2: None, 3: None}
    for k in (2, 3):
        sim = simulate(cut, a, b)
        cs_prev, pi_prev = proposed_cs(sim, [j for j in (1, 2, 3) if j < k])

        def gap(cbar):
            dpi, dcs = outcome(k, cbar)
            mask = pi_prev <= dpi
            return dcs - cs_prev[mask].mean()

        cut[k] = brentq(gap, LOW + 1e-6, naive[k])
    return cut


def table_row(a, b):
    cut = recursive_cutoffs(a, b)
    sim_opt = simulate(cut, a, b, seed=7)
    cs_opt, _ = proposed_cs(sim_opt, [1, 2, 3])
    sim_naive = simulate(naive, a, b, seed=7)
    cs_naive, _ = proposed_cs(sim_naive, [1, 2, 3])
    fb = np.maximum.reduce([np.maximum(sim_naive[k][2], 0.0) for k in (1, 2, 3)])
    fb_gain = fb.mean()
    dcs_cut = {k: outcome(k, cut[k])[1] for k in (1, 2, 3)}
    return cut, dcs_cut, 100 * fb_gain / CS0, 100 * cs_naive.mean() / fb_gain, 100 * cs_opt.mean() / fb_gain


if __name__ == "__main__":
    print("pre-merger P, Q, shares:", P0, Q0, [round(x / Q0, 4) for x in q0])
    print("naive cutoffs:", naive)
    rows = {}
    for a in (5, 3, 1):
        cut, dcs, fb, nv, op = table_row(a, 1)
        rows[a] = (cut, dcs)
        print(f"alpha={a}: cutoffs={ {k: round(v, 4) for k, v in cut.items()} }, "
              f"dCS={ {k: round(v, 5) for k, v in dcs.items()} }, first-best gain={fb:.2f}%, "
              f"naive={nv:.2f}%, optimal={op:.2f}%")

    # ---- figure: merger curves with naive and optimal (alpha = 5) cutoffs
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    colors = {1: "#2A78D6", 2: "#1BAF7A", 3: "#EB6834"}
    cut5 = rows[5][0]
    for k in (1, 2, 3):
        cb = np.linspace(LOW, C[k] + 0.02, 300)
        dpi, dcs = vec_outcome(k, cb)
        keep = dpi >= -0.002
        ax.plot(dpi[keep] * 1e3, dcs[keep] * 1e3, color=colors[k], lw=1.8, label=rf"$M_{k}$ (target cost {C[k]})")
        dpn, dcn = outcome(k, naive[k])
        ax.plot(dpn * 1e3, dcn * 1e3, "o", mfc="white", mec=colors[k], ms=6)
        dpo, dco = outcome(k, cut5[k])
        ax.plot(dpo * 1e3, dco * 1e3, "s", color=colors[k], ms=5)
    ax.axhline(0, color="#52514E", lw=0.8)
    ax.axvline(0, color="#52514E", lw=0.8)
    ax.set_xlabel(r"change in bilateral profit $\Delta\Pi$ ($\times 10^{-3}$)")
    ax.set_ylabel(r"change in consumer surplus $\Delta CS$ ($\times 10^{-3}$)")
    ax.plot([], [], "o", mfc="white", mec="k", label="naive cutoff ($\\Delta CS = 0$)")
    ax.plot([], [], "s", color="k", label="optimal cutoff, Beta(5, 1) draws")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(-3, 75)
    ax.set_ylim(-9, 25)
    fig.tight_layout()
    fig.savefig("../nw_curves.pdf")
    print("saved figure")
