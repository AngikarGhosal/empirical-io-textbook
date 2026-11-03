"""Appendix C.4: simulate 100 markets from a random coefficients logit.

Reads figures/appC/data/yogurt_markets.csv (characteristics, cost shifter,
owner, xi, omega, market size). For every market it
  1. draws n_s = 100 consumer types nu_i ~ N(0, I_4) (seed 2026),
  2. solves the multi-product Bertrand-Nash equilibrium by the
     Morrow-Skerlos fixed point p <- mc + zeta(p), and checks it with a
     Newton-type root finder and 20 random restarts (seed 2028),
  3. simulates realized shares from M_t consumers (seed 2027).
Writes figures/appC/data/simulated_market.csv and firm_draws.npz (inputs to
Sections C.5-C.7) and prints every number used in Section C.4, including
the new market-0 computations (buyers' mean price sensitivity, the Gamma
correction, sampling and simulation error of shares; seed 2029 for the
large-draw population shares).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DATA_FILE = DATA_DIR / "yogurt_markets.csv"

BETA = np.array([-1.0, 0.2, 0.15])
ALPHA = 3.0
SIGMA = np.diag([0.5, 0.2, 0.3, 0.5])
GAMMA1 = np.array([0.2, 0.25, 0.3])
GAMMA2 = 0.5
X_COLS = ["X0", "X1", "X2"]
OWNER_LABEL = {0: "A", 1: "B", 2: "C", 4: "D"}

N_DRAWS = 100
SEED_FIRM, SEED_CONSUMERS, SEED_STARTS, SEED_POP = 2026, 2027, 2028, 2029
N_STARTS = 20
N_POP = 100_000
TOL = 1e-12
MAX_ITER = 10_000
ROOT_TOL = 1e-14
FOC_OK = 1e-10
SAME_TOL = 1e-6


def choice_probs(delta, xtilde, nu, sigma):
    """B x J logit probabilities, overflow-safe (subtract max(0, max_k V))."""
    v = delta[None, :] + nu @ sigma.T @ xtilde.T
    m = np.maximum(v.max(axis=1, keepdims=True), 0.0)
    ev = np.exp(v - m)
    return ev / (np.exp(-m) + ev.sum(axis=1, keepdims=True))


def price_coefs(nu):
    return ALPHA - nu @ SIGMA[3, :]


def mean_utility(p, x, xi):
    return x @ BETA - ALPHA * p + xi


def share_moments(p, x, nu, xi):
    xtilde = np.column_stack([x, p])
    sij = choice_probs(mean_utility(p, x, xi), xtilde, nu, SIGMA)
    w = price_coefs(nu)[:, None] * sij
    lam = w.mean(axis=0)
    gam = w.T @ sij / sij.shape[0]
    return sij.mean(axis=0), lam, gam, sij


def foc(p, mc, own, x, nu, xi):
    s, lam, gam, _ = share_moments(p, x, nu, xi)
    return s + (own * (gam - np.diag(lam)).T) @ (p - mc)


def solve_zeta(mc, own, x, nu, xi, p0):
    p = p0.copy()
    for it in range(1, MAX_ITER + 1):
        s, lam, gam, _ = share_moments(p, x, nu, xi)
        p_new = mc + (s + (own * gam.T) @ (p - mc)) / lam
        if np.max(np.abs(p_new - p)) < TOL:
            return p_new, it, True
        p = p_new
    return p, MAX_ITER, False


def markup_gap(p, mc, own, x, nu, xi):
    s, lam, gam, _ = share_moments(p, x, nu, xi)
    with np.errstate(all="ignore"):
        return (s + (own * gam.T) @ (p - mc)) / lam - (p - mc)


def markup_residual(p, mc, own, x, nu, xi):
    r = np.max(np.abs(markup_gap(p, mc, own, x, nu, xi)))
    return float(r) if np.isfinite(r) else np.inf


def solve_root(fun, mc, own, x, nu, xi, p0):
    sol = root(fun, p0, args=(mc, own, x, nu, xi), method="hybr", tol=ROOT_TOL)
    return sol.x, sol.success


def realized_counts(delta, xtilde, m, rng):
    nu = rng.standard_normal((m, 4))
    eps = rng.gumbel(size=(m, len(delta) + 1))
    u = eps.copy()
    u[:, 1:] += delta[None, :] + nu @ SIGMA.T @ xtilde.T
    return np.bincount(u.argmax(axis=1), minlength=len(delta) + 1)


def multistart(mc, own, x, nu, xi, p_star, rng):
    keys = ("zeta", "root_foc", "root_scaled")
    out = {k: {"solved": 0, "max_dev": 0.0, "failed": 0} for k in keys}
    for _ in range(N_STARTS):
        p0 = rng.uniform(0.0, 4.0, size=len(mc))
        runs = {
            "zeta": solve_zeta(mc, own, x, nu, xi, p0)[0],
            "root_foc": solve_root(foc, mc, own, x, nu, xi, p0)[0],
            "root_scaled": solve_root(markup_gap, mc, own, x, nu, xi, p0)[0],
        }
        for k, ps in runs.items():
            if markup_residual(ps, mc, own, x, nu, xi) >= FOC_OK:
                out[k]["failed"] += 1
                continue
            out[k]["solved"] += 1
            out[k]["max_dev"] = max(out[k]["max_dev"], np.max(np.abs(ps - p_star)))
    return out


def ownership(owner):
    return (owner[:, None] == owner[None, :]).astype(float)


def main():
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values(["market_id", "product_id"]).reset_index(drop=True)
    df["owner"] = df["owner"].astype(int)
    df["mc_true"] = df[X_COLS].to_numpy() @ GAMMA1 + GAMMA2 * df["Z"] + df["omega"]
    markets = np.sort(df["market_id"].unique())

    # Design facts
    nprod = df.groupby("market_id").size()
    print(f"obs {len(df)}, markets {len(markets)}, products per market "
          f"{nprod.min()}-{nprod.max()} mean {nprod.mean():.2f}")
    print(f"xi sd {df['xi'].std(ddof=1):.3f}, omega sd {df['omega'].std(ddof=1):.4f}, "
          f"corr {np.corrcoef(df['xi'], df['omega'])[0, 1]:.3f}")
    Ms = df.groupby("market_id")["M"].first()
    print(f"M range {Ms.min()}-{Ms.max()}, mean {Ms.mean():.1f}")
    prod = df.groupby("product_id").agg(sugar=("X1", "first"), organic=("X2", "first"),
                                         milk=("Z", "first"), owner=("owner", "first"),
                                         nmk=("market_id", "size"))
    print(prod)

    rng_firm = np.random.default_rng(SEED_FIRM)
    nu_firm = rng_firm.standard_normal((len(markets), N_DRAWS, 4))
    np.savez(DATA_DIR / "firm_draws.npz", nu=nu_firm, market_id=markets)
    rng_starts = np.random.default_rng(SEED_STARTS)
    rng_cons = np.random.default_rng(SEED_CONSUMERS)
    rng_pop = np.random.default_rng(SEED_POP)

    n = len(df)
    price = np.zeros(n); s_exp = np.zeros(n); s_real = np.zeros(n)
    qty = np.zeros(n, dtype=int); delta_true = np.zeros(n); s_pop = np.zeros(n)
    iters, focs, diffs, ms_all, root_ok = [], [], [], [], []
    for t, mkt in enumerate(markets):
        rows = np.flatnonzero(df["market_id"].to_numpy() == mkt)
        g = df.iloc[rows]
        x = g[X_COLS].to_numpy(); xi = g["xi"].to_numpy(); mc = g["mc_true"].to_numpy()
        own = ownership(g["owner"].to_numpy())
        nu = nu_firm[t]
        p, it, ok = solve_zeta(mc, own, x, nu, xi, mc.copy())
        assert ok
        p_root, okr = solve_root(foc, mc, own, x, nu, xi, mc.copy())
        root_ok.append(okr)
        ms_all.append(multistart(mc, own, x, nu, xi, p, rng_starts))
        delta = mean_utility(p, x, xi)
        m = int(g["M"].iloc[0])
        counts = realized_counts(delta, np.column_stack([x, p]), m, rng_cons)
        price[rows] = p
        s_exp[rows] = share_moments(p, x, nu, xi)[0]
        s_real[rows] = counts[1:] / m
        qty[rows] = counts[1:]
        delta_true[rows] = delta
        # population shares with many draws (simulation-error check)
        nu_pop = rng_pop.standard_normal((N_POP, 4))
        s_pop[rows] = choice_probs(delta, np.column_stack([x, p]), nu_pop, SIGMA).mean(axis=0)
        iters.append(it)
        focs.append(np.max(np.abs(foc(p, mc, own, x, nu, xi))))
        diffs.append(np.max(np.abs(p - p_root)))

    df["price"], df["share_expected"], df["share_realized"] = price, s_exp, s_real
    df["quantity"], df["delta_true"] = qty, delta_true
    keep = ["market_id", "product_id", "owner", "X0", "X1", "X2", "Z", "M", "price",
            "share_expected", "share_realized", "quantity", "mc_true", "xi", "omega",
            "delta_true"]
    df[keep].to_csv(DATA_DIR / "simulated_market.csv", index=False)

    iters = np.array(iters)
    print("\nSolver checks")
    print(f"zeta iterations {iters.min()}-{iters.max()} mean {iters.mean():.2f}, market 0 {iters[0]}")
    print(f"max |FOC| zeta {max(focs):.2e}; max |p_zeta - p_root| {max(diffs):.2e}; "
          f"root success flags {sum(root_ok)} of {len(root_ok)}")
    for k in ("zeta", "root_foc", "root_scaled"):
        solved = sum(m[k]["solved"] for m in ms_all)
        dev = max(m[k]["max_dev"] for m in ms_all)
        print(f"multistart {k}: solved {solved} of {N_STARTS * len(markets)}, max dev {dev:.2e}")
    markup = df["price"] - df["mc_true"]
    print(f"prices {df['price'].min():.3f}-{df['price'].max():.3f} mean {df['price'].mean():.3f}")
    print(f"markups {markup.min():.3f}-{markup.max():.3f} mean {markup.mean():.3f}; "
          f"Lerner mean {(markup / df['price']).mean():.3f} min {(markup / df['price']).min():.3f} "
          f"max {(markup / df['price']).max():.3f}")
    alpha_i = price_coefs(nu_firm.reshape(-1, 4))
    print(f"min alpha_i firm draws {alpha_i.min():.3f}; max {alpha_i.max():.3f}; "
          f"P(alpha_i<=0) = {norm.sf(6):.3e}; P(alpha_i < 1.5) = {norm.sf(3):.4f}")

    # Realized vs expected shares
    gap = df["share_realized"] - df["share_expected"]
    s0e = 1 - df.groupby("market_id")["share_expected"].sum()
    s0r = 1 - df.groupby("market_id")["share_realized"].sum()
    print("\nRealized vs expected")
    print(f"total consumers {Ms.sum()}; zero shares {(df['quantity'] == 0).sum()}; "
          f"min realized {df['share_realized'].min():.4f} ({df['quantity'].min()} buyers); "
          f"min expected {df['share_expected'].min():.4f}")
    print(f"mean |gap| {gap.abs().mean():.4f}; max |gap| {gap.abs().max():.4f}; "
          f"corr {np.corrcoef(df['share_realized'], df['share_expected'])[0, 1]:.4f}")
    print(f"mean outside share expected {s0e.mean():.4f}, realized {s0r.mean():.4f}; "
          f"min outside realized {s0r.min():.4f}, max {s0r.max():.4f}")
    sd_samp = np.sqrt(s_pop * (1 - s_pop) / df["M"].to_numpy())
    z_samp = (s_real - s_pop) / sd_samp
    z_tot = (s_real - s_exp) / sd_samp
    z_sim = (s_exp - s_pop) / sd_samp
    print(f"z(realized - population): mean {z_samp.mean():.3f}, mean sq {np.mean(z_samp ** 2):.3f}")
    print(f"z(realized - expected): mean sq {np.mean(z_tot ** 2):.3f}")
    print(f"(expected - population)/sd_sampling: mean sq {np.mean(z_sim ** 2):.3f}; "
          f"rel sim err |s_exp/s_pop - 1|: median {np.median(np.abs(s_exp / s_pop - 1)):.3f} "
          f"max {np.max(np.abs(s_exp / s_pop - 1)):.3f}")
    se_log = np.sqrt((1 - s_real) / qty)
    print(f"log-share s.e. all obs: min {se_log.min():.3f} median {np.median(se_log):.3f} "
          f"max {se_log.max():.3f}; obs with fewer than 100 buyers {(qty < 100).sum()}; "
          f"buyers min {qty.min()} median {np.median(qty):.0f} max {qty.max()}")

    # Market 0
    print("\nMarket 0")
    g = df[df["market_id"] == markets[0]]
    x = g[X_COLS].to_numpy(); xi = g["xi"].to_numpy(); mc = g["mc_true"].to_numpy()
    owner = g["owner"].to_numpy(); own = ownership(owner); p = g["price"].to_numpy()
    nu = nu_firm[0]
    s, lam, gam, sij = share_moments(p, x, nu, xi)
    a_i = price_coefs(nu)
    m = p - mc
    ebar = lam / s
    first = s / lam
    gcorr = (own * gam) @ m / lam
    gdiag = np.diag(gam) * m / lam
    goff = gcorr - gdiag
    si0 = 1 - sij.sum(axis=1)
    a_out = np.sum(a_i * si0) / np.sum(si0)
    print(f"M0 = {int(g['M'].iloc[0])}; outside expected {1 - s.sum():.4f}, realized "
          f"{1 - g['share_realized'].sum():.4f}, population {1 - s_pop[g.index].sum():.4f}")
    print(f"E[alpha] over the 100 draws {a_i.mean():.4f}; E[alpha | outside] {a_out:.4f}")
    print("prod owner mc p m s_exp s_real q s_pop E[a|j] 1/E[a|j] Gamma_corr (diag, off) se_log Lerner")
    for k in range(len(g)):
        q = int(g["quantity"].iloc[k]); sr = g["share_realized"].iloc[k]
        sp = s_pop[g.index[k]]
        print(f"{g['product_id'].iloc[k]:>2} {OWNER_LABEL[owner[k]]} {mc[k]:.4f} {p[k]:.4f} "
              f"{m[k]:.4f} {s[k]:.4f} {sr:.4f} {q:>4} {sp:.4f} {ebar[k]:.3f} {first[k]:.4f} "
              f"{gcorr[k]:.4f} ({gdiag[k]:.4f}, {goff[k]:.4f}) {np.sqrt((1 - sr) / q):.3f} "
              f"{m[k] / p[k]:.3f}")
    print(f"max |m - (s/Lambda + corr)| = {np.max(np.abs(m - first - gcorr)):.1e}")
    print(f"Gamma correction share of markup: {np.min(gcorr / m):.3f}-{np.max(gcorr / m):.3f}")
    rel = (g["share_realized"].to_numpy() - s) / s
    print(f"relative gaps realized vs expected: {np.round(rel, 3)}")
    relp = (s - s_pop[g.index]) / s_pop[g.index]
    print(f"relative gaps expected vs population: {np.round(relp, 3)}")
    relr = (g["share_realized"].to_numpy() - s_pop[g.index]) / s_pop[g.index]
    print(f"relative gaps realized vs population: {np.round(relr, 3)}")
    sdlog_sim = np.sqrt(np.var(sij, axis=0, ddof=1) / N_DRAWS) / s
    print(f"simulation s.e. of log expected share (n_s=100): {np.round(sdlog_sim, 3)}")
    # Product summary over markets
    df["markup"] = df["price"] - df["mc_true"]
    summ = df.groupby("product_id").agg(nmk=("market_id", "size"), p=("price", "mean"),
                                        mk=("markup", "mean"), s=("share_realized", "mean"))
    print(summ.round(4))


if __name__ == "__main__":
    main()
