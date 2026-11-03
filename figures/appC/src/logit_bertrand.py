"""Appendix C.3: Bertrand pricing with logit demand in a 25-product market.

Reads figures/appC/data/logit_market.csv and prints every number used in
Section C.3 (equilibria with single- and multi-product firms, uniqueness
checks, the merger of owners 3 and 4 with UPP, and a common unit tax).
Writes figures/appC/psi.pdf and figures/appC/logit_merger.pdf.

Notation follows the book: a_j = x_j beta + xi_j, delta_j = a_j - alpha p_j,
Phi = 1 + sum_l exp(delta_l), v_j = a_j - alpha mc_j, E_f = sum_{k in f} e^{v_k}.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, root
from scipy.special import lambertw

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.serif": ["STIXGeneral", "DejaVu Serif"]})

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
DATA = FIG / "data" / "logit_market.csv"

df = pd.read_csv(DATA)
X = df[["X1", "X2", "X3"]].to_numpy()
BETA = np.array([0.1, 0.25, 0.1])
ALPHA = 2.0
GAMMA = np.array([0.2, 0.25, 0.3])
A = X @ BETA + df["xi"].to_numpy()
MC0 = X @ GAMMA + df["omega"].to_numpy()
J = len(df)
SINGLE = np.arange(J)
MULTI = df["owner"].to_numpy().astype(int)
MERGED = np.where(MULTI == 4, 3, MULTI)  # owners 3 and 4 merge
TOL = 1e-12


def shares(p):
    e = np.exp(A - ALPHA * p)
    return e / (1.0 + e.sum())


def profits(p, owner, mc=MC0):
    return np.bincount(owner, weights=(p - mc) * shares(p))


def best_response(p, owner, mc=MC0):
    """Closed-form multi-product Lambert-W best response of every firm."""
    e = np.exp(A - ALPHA * p)
    E = np.bincount(owner, weights=np.exp(A - ALPHA * mc))
    phi_minus = 1.0 + e.sum() - np.bincount(owner, weights=e)
    m = (1.0 + lambertw(E * np.exp(-1.0) / phi_minus).real) / ALPHA
    return mc + m[owner]


def omega_matrix(owner):
    return (owner[:, None] == owner[None, :]).astype(float)


def dsdp(p):
    s = shares(p)
    return ALPHA * (np.outer(s, s) - np.diag(s))  # [j,k] = d s_j / d p_k


def foc(p, owner, mc=MC0):
    s = shares(p)
    return s + (omega_matrix(owner) * dsdp(p).T) @ (p - mc)


def solve_br(owner, mc=MC0, p0=None, tol=TOL):
    p = mc + 1.0 / ALPHA if p0 is None else np.asarray(p0, float)
    for it in range(1, 100000):
        pn = best_response(p, owner, mc)
        if np.max(np.abs(pn - p)) < tol:
            return pn, it
        p = pn
    raise RuntimeError("BR iteration failed")


def solve_markup(owner, mc=MC0, tol=TOL):
    """p <- mc + 1/(alpha (1 - S_f(p))) (common-markup iteration)."""
    p = mc + 1.0 / ALPHA
    for it in range(1, 100000):
        s = shares(p)
        S = np.bincount(owner, weights=s)
        pn = mc + 1.0 / (ALPHA * (1.0 - S[owner]))
        if np.max(np.abs(pn - p)) < tol:
            return pn, it
        p = pn
    raise RuntimeError("markup iteration failed")


def solve_newton(owner, mc=MC0):
    sol = root(foc, mc + 1.0 / ALPHA, args=(owner, mc), method="hybr",
               tol=1e-14)
    return sol.x, sol.nfev


def S_of_phi(phi, E):
    """Solve h(S) = S exp(1/(1-S)) = E/phi in logs for each firm."""
    out = []
    for e in E:
        target = np.log(e / phi)
        out.append(brentq(lambda x: np.log(x) + 1 / (1 - x) - target,
                          1e-300, 1 - 1e-13, xtol=1e-15))
    return np.array(out)


def psi_minus_one(phi, owner, mc=MC0):
    E = np.bincount(owner, weights=np.exp(A - ALPHA * mc))
    return S_of_phi(phi, E).sum() + 1.0 / phi - 1.0


def report(owner, rng, label, mc=MC0):
    p, it_br = solve_br(owner, mc)
    p_mk, it_mk = solve_markup(owner, mc)
    p_nt, nfev = solve_newton(owner, mc)
    s = shares(p)
    S = np.bincount(owner, weights=s)
    s0 = 1 - s.sum()
    phi = 1 / s0
    # multistart
    starts = [mc + rng.uniform(0, 5, J) for _ in range(500)] + [mc, mc + 20]
    dev = 0.0
    its = []
    for p0 in starts:
        q, k = solve_br(owner, mc, p0)
        dev = max(dev, np.max(np.abs(q - p)))
        its.append(k)
    # aggregate equation on a log grid
    grid = np.logspace(0, 4, 2000)
    psi = np.array([psi_minus_one(g, owner, mc) for g in grid])
    crossings = int(np.sum(np.diff(np.sign(psi)) != 0))
    phi_root = brentq(lambda g: psi_minus_one(g, owner, mc), 1.0 + 1e-9, 1e4,
                      xtol=1e-14)
    # contraction moduli
    E = np.bincount(owner, weights=np.exp(A - ALPHA * mc))
    wz = lambertw(E * np.exp(-1.0)).real
    eta_global = np.max(wz / (1 + wz))
    eta_local = np.max(S * (1 - S - s0) / (1 - S))
    # exact global modulus: sup over Phi_{-f} >= 1 of the row sum
    # W(z)/(1+W(z)) * (1 - 1/Phi_{-f}), z = E_f e^{-1} / Phi_{-f}
    lgrid = np.linspace(0.0, 30.0, 300001)  # log Phi_{-f}
    eta_exact = 0.0
    for Ef in E[E > 0]:
        wg = lambertw(Ef * np.exp(-1.0) * np.exp(-lgrid)).real
        eta_exact = max(eta_exact, np.max(wg / (1 + wg) * (1 - np.exp(-lgrid))))
    # numerical check of the BR Jacobian at the equilibrium
    h = 1e-6
    jac_num = np.zeros((J, J))
    for l in range(J):
        dp = np.zeros(J)
        dp[l] = h
        jac_num[:, l] = (best_response(p + dp, owner, mc)
                         - best_response(p - dp, owner, mc)) / (2 * h)
    same = omega_matrix(owner).astype(bool)
    jac_th = np.where(same, 0.0, (S[owner] / (1 - S[owner]))[:, None] * s[None, :])
    jac_err = np.max(np.abs(jac_num - jac_th))
    out = dict(
        p=p, s=s, S=S, s0=s0, phi=phi, it_br=it_br, it_mk=it_mk, nfev=nfev,
        diff_mk=np.max(np.abs(p_mk - p)), diff_nt=np.max(np.abs(p_nt - p)),
        foc=np.max(np.abs(foc(p, owner, mc))), pi=profits(p, owner, mc),
        cs=np.log(phi) / ALPHA, dev=dev, its_max=max(its), crossings=crossings,
        phi_root=phi_root, eta_global=eta_global, eta_local=eta_local, eta_exact=eta_exact,
        jac_err=jac_err, grid=grid, psi=psi,
        pbar=p @ s / s.sum(),
    )
    print(f"--- {label} ---")
    print(f"BR iterations {it_br}; markup-iteration {it_mk}; newton nfev {nfev}")
    print(f"max|p_markup - p_BR| = {out['diff_mk']:.1e}; "
          f"max|p_newton - p_BR| = {out['diff_nt']:.1e}; max|FOC| = {out['foc']:.1e}")
    print(f"s0 = {s0:.4f}; Phi* = {phi:.4f}; CS = {out['cs']:.4f}; "
          f"total profit = {out['pi'].sum():.4f}; mean price (share-wtd) = {out['pbar']:.4f}")
    print(f"markups: min {np.min(p - mc):.4f} max {np.max(p - mc):.4f}")
    print(f"multistart (502 starts): max dev {dev:.1e}, max iterations {max(its)}")
    print(f"Psi-1 sign changes on grid: {crossings}; root {phi_root:.4f} vs 1/s0 {phi:.4f}")
    print(f"eta (global bound) = {eta_global:.4f}; local row-sum max = {eta_local:.4f}")
    print(f"exact global modulus (sup of row sum over Phi_-f >= 1) = {eta_exact:.4f}")
    print(f"BR Jacobian: max |numeric - formula| = {jac_err:.1e}")
    print(f"firm shares (%): {np.round(100 * S, 2)}")
    print(f"firm profits x100: {np.round(100 * out['pi'], 3)}")
    return out


def main():
    rng = np.random.default_rng(257)
    print("Data summary")
    print(f"x2 mean {X[:, 1].mean():.2f} sd {X[:, 1].std(ddof=1):.2f} "
          f"min {X[:, 1].min():.3f} max {X[:, 1].max():.2f}; "
          f"x3 ones {int(X[:, 2].sum())}")
    xi, om = df["xi"].to_numpy(), df["omega"].to_numpy()
    print(f"xi mean {xi.mean():.3f} sd {xi.std(ddof=1):.2f}; "
          f"omega mean {om.mean():.3f} sd {om.std(ddof=1):.3f}")
    print(f"owners: {np.bincount(MULTI)}; mc range {MC0.min():.3f} {MC0.max():.3f}")
    v = A - ALPHA * MC0
    print(f"v range {v.min():.3f} {v.max():.3f}")

    Sres = report(SINGLE, rng, "single-product (S)")
    Mres = report(MULTI, rng, "multi-product (M)")

    # Product table
    print("\nProduct table: j owner x2 x3 xi omega mc pS pM mS mM sS sM")
    rows_eq, rows_data = [], []
    for j in range(J):
        rows_data.append(
            f"{j} & {MULTI[j]} & {X[j, 1]:.3f} & {int(X[j, 2])} & {xi[j]:.3f} & "
            f"{om[j]:.3f} & {MC0[j]:.3f} \\\\")
        rows_eq.append(
            f"{j} & {MULTI[j]} & {MC0[j]:.3f} & {Sres['p'][j]:.3f} & {Mres['p'][j]:.3f} & "
            f"{Sres['p'][j] - MC0[j]:.3f} & {Mres['p'][j] - MC0[j]:.3f} & "
            f"{100 * Sres['s'][j]:.2f} & {100 * Mres['s'][j]:.2f} \\\\")
    (HERE / "logit_tables.tex").write_text(
        "%DATA\n" + "\n".join(rows_data) + "\n%EQ\n" + "\n".join(rows_eq) + "\n")
    dp = Mres["p"] - Sres["p"]
    print(f"price increase S->M: min {dp.min():.4f} (product {dp.argmin()}), "
          f"max {dp.max():.4f} (product {dp.argmax()})")
    print(f"total profit change S->M: {100 * (Mres['pi'].sum() / Sres['pi'].sum() - 1):.1f}%")
    print(f"total surplus S {Sres['cs'] + Sres['pi'].sum():.4f}, M {Mres['cs'] + Mres['pi'].sum():.4f}")
    # Logit diversion within firms
    s = Mres["s"]
    for f in range(5):
        k = MULTI == f
        Dwithin = (Mres["S"][f] - s[k]) / (1 - s[k])
        print(f"firm {f}: within-firm diversion range {Dwithin.min():.3f}-{Dwithin.max():.3f}")

    # ------------------------------------------------------------- merger
    print("\n--- merger of owners 3 and 4 ---")
    Gres = report(MERGED, rng, "post-merger (G)")
    pM, pG = Mres["p"], Gres["p"]
    dpG = pG - pM
    rows_g = [
        f"{j} & {MULTI[j]} & {MC0[j]:.3f} & {Sres['p'][j]:.3f} & {pM[j]:.3f} & {pG[j]:.3f} & "
        f"{Sres['p'][j] - MC0[j]:.3f} & {pM[j] - MC0[j]:.3f} & "
        f"{100 * Sres['s'][j]:.2f} & {100 * Mres['s'][j]:.2f} & {100 * Gres['s'][j]:.2f} \\\\"
        for j in range(J)]
    (HERE / "logit_eq_rows.tex").write_text("\n".join(rows_g) + "\n")
    merging = np.isin(MULTI, [3, 4])
    print(f"merging products: mean dp {dpG[merging].mean():.4f}, "
          f"max {dpG[merging].max():.4f}, min {dpG[merging].min():.4f}")
    print(f"  firm 3 products dp {np.unique(np.round(dpG[MULTI == 3], 4))}; "
          f"firm 4 products dp {np.unique(np.round(dpG[MULTI == 4], 4))}")
    print(f"non-merging: mean dp {dpG[~merging].mean():.4f}, "
          f"min {dpG[~merging].min():.4f}, max {dpG[~merging].max():.4f}")
    for f in [0, 1, 2]:
        print(f"  firm {f} dp {dpG[MULTI == f].mean():.4f}")
    print(f"CS {Mres['cs']:.4f} -> {Gres['cs']:.4f} (change {Gres['cs'] - Mres['cs']:.4f})")
    print(f"s0 {Mres['s0']:.4f} -> {Gres['s0']:.4f}")
    print(f"total profit {Mres['pi'].sum():.4f} -> {Gres['pi'].sum():.4f}")
    print(f"merged profit {Mres['pi'][3] + Mres['pi'][4]:.4f} -> {Gres['pi'][3]:.4f}")
    print(f"total surplus {Mres['cs'] + Mres['pi'].sum():.4f} -> {Gres['cs'] + Gres['pi'].sum():.4f}")
    mM = np.array([pM[MULTI == f][0] - MC0[MULTI == f][0] for f in range(5)])
    SM = Mres["S"]
    SG = Gres["S"]
    mG = np.array([pG[MERGED == f][0] - MC0[MERGED == f][0] for f in [0, 1, 2, 3]])
    print(f"pre markups by firm {np.round(mM, 4)}; pre shares {np.round(SM, 4)}")
    print(f"post markups by firm (0,1,2,M) {np.round(mG, 4)}; post shares {np.round(SG, 4)}")
    # post-merger shares of the old firm 3 and firm 4 product sets
    sG = Gres["s"]
    S3p, S4p = sG[MULTI == 3].sum(), sG[MULTI == 4].sum()
    print(f"post shares of old firm-3 set {S3p:.4f}, firm-4 set {S4p:.4f}")
    # UPP and GUPPI (product level, Farrell-Shapiro)
    upp = np.zeros(J)
    for j in np.flatnonzero(merging):
        partner = 4 if MULTI[j] == 3 else 3
        D = SM[partner] / (1 - s[j])
        upp[j] = D * mM[partner]
    guppi = upp / pM
    for f, partner in [(3, 4), (4, 3)]:
        k = MULTI == f
        print(f"firm {f}: diversion to partner {np.min(SM[partner] / (1 - s[k])):.4f}-"
              f"{np.max(SM[partner] / (1 - s[k])):.4f}; UPP {upp[k].min():.4f}-{upp[k].max():.4f}; "
              f"GUPPI {guppi[k].min():.4f}-{guppi[k].max():.4f}")
        print(f"   UPP / (S_g m_g) = 1/(1-s_j): {np.min(1 / (1 - s[k])):.4f}-{np.max(1 / (1 - s[k])):.4f}")
        # firm-level first-round tax and first-order change
        t_f = SM[partner] * mM[partner] / (1 - SM[f])
        print(f"   firm-level tax S_g m_g/(1-S_f) = {t_f:.4f}; "
              f"times (1-S_f) = {SM[partner] * mM[partner]:.4f}; "
              f"equilibrium change = {dpG[k][0]:.4f}")
        # exact decomposition of the change in markup
        Sfp = S3p if f == 3 else S4p
        Sgp = S4p if f == 3 else S3p
        mMg = mG[3]
        print(f"   decomposition: S_g' m_M = {Sgp * mMg:.4f}; "
              f"S_f' m_M - S_f m_f = {Sfp * mMg - SM[f] * mM[f]:.4f}; "
              f"sum = {Sgp * mMg + Sfp * mMg - SM[f] * mM[f]:.4f}")
    # Product-level UPP imposed as taxes, re-equilibrated (one round)
    tax = np.zeros(J)
    tax[merging] = upp[merging]
    pT, _ = solve_br(MULTI, MC0 + tax)
    print(f"one round with product-level taxes: merging dp "
          f"{(pT - pM)[MULTI == 3].mean():.4f} (firm 3), {(pT - pM)[MULTI == 4].mean():.4f} (firm 4); "
          f"non-merging mean {(pT - pM)[~merging].mean():.4f}")
    # Farrell-Shapiro iteration with firm-level taxes t_f = sum_{k in g} s_k m_k / (1 - S_f)
    p_it = pM.copy()
    for r in range(1, 60):
        s_it = shares(p_it)
        m_it = p_it - MC0
        S_it = np.bincount(MULTI, weights=s_it)
        V_it = np.bincount(MULTI, weights=s_it * m_it)
        tax = np.zeros(J)
        t3, t4 = V_it[4] / (1 - S_it[3]), V_it[3] / (1 - S_it[4])
        tax[MULTI == 3], tax[MULTI == 4] = t3, t4
        p_new, _ = solve_br(MULTI, MC0 + tax)
        d_it = p_new - pM
        if r <= 3:
            print(f"FS round {r}: t3 {t3:.4f} t4 {t4:.4f}; dp firm 3 {d_it[MULTI == 3][0]:.4f}, "
                  f"firm 4 {d_it[MULTI == 4][0]:.4f}, rivals mean {d_it[~merging].mean():.4f}; "
                  f"distance to merger eq {np.max(np.abs(p_new - pG)):.1e}")
        if np.max(np.abs(p_new - p_it)) < 1e-12:
            print(f"FS iteration converged after {r} rounds; distance {np.max(np.abs(p_new - pG)):.1e}")
            break
        p_it = p_new
    # firm-level table rows
    print("\nfirm-level merger table: firm, products, S_pre, S_post, m_pre, m_post, pi_pre, pi_post")
    for f in range(5):
        k = MULTI == f
        newf = 3 if f == 4 else f
        print(f"{f} & {k.sum()} & {100 * SM[f]:.2f} & {100 * sG[k].sum():.2f} & "
              f"{mM[f]:.3f} & {pG[k][0] - MC0[k][0]:.3f} & {100 * Mres['pi'][f]:.3f} & "
              f"{100 * np.sum((pG[k] - MC0[k]) * sG[k]):.3f}")
    print(f"merged & 13 & {100 * (SM[3] + SM[4]):.2f} & {100 * SG[3]:.2f} & - & {mG[3]:.3f} & "
          f"{100 * (Mres['pi'][3] + Mres['pi'][4]):.3f} & {100 * Gres['pi'][3]:.3f}")

    # merger figure
    order = np.lexsort((np.arange(J), MULTI))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    xs = np.arange(J)
    cols = np.where(merging[order], "#eb6834", "#2a78d6")
    ax.bar(xs, dpG[order], color=cols, width=0.7, zorder=2)
    um = merging[order]
    ax.scatter(xs[um], upp[order][um], s=42, facecolors="white",
               edgecolors="#0b0b0b", linewidths=1.2, zorder=3,
               label="first-round UPP $D_{j,g}\\,m_g$")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(j) for j in order], fontsize=7)
    # owner group separators and labels
    bounds = np.flatnonzero(np.diff(MULTI[order])) + 0.5
    for b in bounds:
        ax.axvline(b, color="#b5b4ad", lw=0.8, zorder=1)
    starts = np.concatenate([[-0.5], bounds])
    ends = np.concatenate([bounds, [J - 0.5]])
    for f, (a0, a1) in enumerate(zip(starts, ends)):
        ax.text((a0 + a1) / 2, 0.205, f"owner {f}", ha="center", va="bottom",
                fontsize=8, color="#52514e")
        if f in (3, 4):
            g_ = 4 if f == 3 else 3
            ax.hlines(SM[g_] * mM[g_], a0 + 0.1, a1 - 0.1, colors="#0b0b0b",
                      linestyles="dashed", lw=1.2, zorder=3,
                      label="first-order effect $S_g\\,m_g$" if f == 3 else None)
    ax.set_ylim(0, 0.225)
    ax.set_xlim(-0.6, J - 0.4)
    ax.set_xlabel("product $j$ (grouped by pre-merger owner)")
    ax.set_ylabel("price change $p_j^{G} - p_j^{M}$")
    ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color="#eb6834", label="merging owners 3 and 4"),
               Patch(color="#2a78d6", label="non-merging owners"),
               ax.collections[0],
               Line2D([0], [0], color="#0b0b0b", ls="--", lw=1.2,
                      label="first-order effect $S_g\\,m_g$")]
    ax.legend(handles=handles, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "logit_merger.pdf", bbox_inches="tight")
    fig.savefig(HERE / "logit_merger_preview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # psi figure
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    for res, lab, col, mk in [(Sres, "single-product firms (S)", "#2a78d6", "o"),
                              (Mres, "five multi-product owners (M)", "#eb6834", "s")]:
        ax.plot(res["grid"], res["psi"], color=col, lw=2, label=lab)
        ax.plot([res["phi_root"]], [0.0], marker=mk, color=col, ms=8,
                markeredgecolor="white", markeredgewidth=1.5, zorder=4)
    ax.annotate(r"$\Phi^* = %.3f$" % Sres["phi_root"], (Sres["phi_root"], 0),
                xytext=(Sres["phi_root"] * 2.2, 0.35), fontsize=8,
                arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.8))
    ax.annotate(r"$\Phi^* = %.3f$" % Mres["phi_root"], (Mres["phi_root"], 0),
                xytext=(Mres["phi_root"] * 0.18, -0.45), fontsize=8,
                arrowprops=dict(arrowstyle="-", color="#52514e", lw=0.8))
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_xlabel(r"logit denominator $\Phi$ (log scale)")
    ax.set_ylabel(r"$\Psi(\Phi) - 1$")
    ax.grid(color="#e6e5e0", lw=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "psi.pdf")
    fig.savefig(HERE / "psi_preview.png", dpi=110)
    plt.close(fig)
    print(f"Psi(1)-1: S {Sres['psi'][0]:.4f}, M {Mres['psi'][0]:.4f}")

    # ------------------------------------------------------------- tax
    print("\n--- common unit tax t = 0.1 ---")
    t = 0.1
    for res, owner, lab in [(Sres, SINGLE, "S"), (Mres, MULTI, "M")]:
        pt, _ = solve_br(owner, MC0 + t)
        st = shares(pt)
        rho = (pt - res["p"]) / t
        s0 = res["s"]
        Sf = res["S"]
        partial = 1 - Sf[owner]
        pbar0 = res["p"] @ s0 / s0.sum()
        pbar1 = pt @ st / st.sum()
        fixedw = (pt - res["p"]) @ s0 / s0.sum() / t
        # closed-form derivative at t = 0
        w = Sf * (1 - Sf) ** 2 / ((1 - Sf) ** 2 + Sf)
        W = w.sum()
        rho_cf = 1 - Sf * res["s0"] / (((1 - Sf) ** 2 + Sf) * (res["s0"] + W))
        # numerical derivative
        hh = 1e-6
        ph, _ = solve_br(owner, MC0 + hh, tol=1e-14)
        rho_num = (ph - res["p"]) / hh
        print(f"{lab}: product pass-through min {rho.min():.4f} max {rho.max():.4f} "
              f"mean {rho.mean():.4f}; partial 1-S_f min {partial.min():.4f} max {partial.max():.4f}")
        print(f"   avg-price pass-through {(pbar1 - pbar0) / t:.4f}; fixed weights {fixedw:.4f}; "
              f"composition term {(pbar1 - pbar0) / t - fixedw:.4f}")
        print(f"   closed-form rho at t=0: min {rho_cf.min():.4f} max {rho_cf.max():.4f}; "
              f"max |cf - numeric derivative| = {np.max(np.abs(rho_cf[owner] - rho_num)):.1e}")
        print(f"   s0 {res['s0']:.4f} -> {1 - st.sum():.4f}; W = {W:.4f}; "
              f"CS {res['cs']:.4f} -> {np.log(1 / (1 - st.sum())) / ALPHA:.4f}; "
              f"profit {res['pi'].sum():.4f} -> {profits(pt, owner, MC0 + t).sum():.4f}")
        if lab == "M":
            for f in range(5):
                print(f"   firm {f}: rho {rho[owner == f][0]:.4f}, partial {1 - Sf[f]:.4f}, "
                      f"closed form {rho_cf[f]:.4f}")
        else:
            print(f"   rho vs s_j corr {np.corrcoef(rho, s0)[0, 1]:.3f}")


if __name__ == "__main__":
    main()
