"""Bertrand-Nash equilibrium in a simulated 25-product logit market (ex:04-logit25).

Primitives (data_simulations.csv in the course folder): 25 products with characteristics
x_j = (1, X2_j, X3_j), unobserved quality xi_j, cost shock omega_j and an owner id (five
owners holding 3, 3, 6, 6 and 7 products). Parameters: beta = (0.1, 0.25, 0.1), alpha = 2,
gamma = (0.2, 0.25, 0.3), mc_j = x_j gamma + omega_j, market size 1.

Two ownership structures:
  S  every product is owned by a separate firm (Omega = identity),
  M  products are owned by the five owners in the data.

Solvers (both are run and must agree):
  (1) best-response iteration with the Lambert-W closed form
        S: p_j = mc_j + (1/alpha)[1 + W(exp(v_j - 1) / Phi_{-j})]
        M: m_f = (1/alpha)[1 + W(E_f exp(-1) / Phi_{-f})], E_f = sum_{k in f} exp(v_k)
      started at p = mc + 1/alpha, stopped when the sup-norm change is below 1e-12;
  (2) markup iteration p <- mc - (Omega o Delta(p)^T)^{-1} s(p).

Outputs:
  figures/ch04/logit25_markups.pdf     (fig:04-logit25-markups)
  figures/ch04/src/logit25_numbers.txt (all numbers quoted in the text and tab:04-logit25)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import lambertw
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                      # .../ECON_257
DATA = ROOT / "ProblemSet_1" / "data_simulations.csv"
OUT_PDF = HERE.parent / "logit25_markups.pdf"
OUT_TXT = HERE / "logit25_numbers.txt"

BETA = np.array([0.1, 0.25, 0.1])
ALPHA = 2.0
GAMMA = np.array([0.2, 0.25, 0.3])
TOL = 1e-12

df = pd.read_csv(DATA).sort_values("prod_id").reset_index(drop=True)
X = df[["X1", "X2", "X3"]].to_numpy()
XI = df["xi"].to_numpy()
OMEGA_C = df["omega"].to_numpy()
OWNER = df["owner"].to_numpy()
J = len(df)
MC = X @ GAMMA + OMEGA_C
A = X @ BETA + XI                            # quality index, delta_j = A_j - alpha p_j
V = A - ALPHA * MC                           # mean utility at marginal-cost pricing


def shares(p):
    e = np.exp(A - ALPHA * p)
    phi = 1.0 + e.sum()
    return e / phi, 1.0 / phi, phi


def jacobian(p):
    """Delta[j, k] = d s_j / d p_k for the logit."""
    s, _, _ = shares(p)
    D = ALPHA * np.outer(s, s)
    D[np.diag_indices(J)] = -ALPHA * s * (1.0 - s)
    return D


def omega_matrix(owner):
    return (owner[:, None] == owner[None, :]).astype(float)


def br_iteration(owner, maxit=10_000):
    """Best-response iteration with the Lambert-W formula (firm-level for multi-product)."""
    p = MC + 1.0 / ALPHA
    firms = np.unique(owner)
    for it in range(1, maxit + 1):
        e = np.exp(A - ALPHA * p)
        p_new = p.copy()
        for f in firms:
            own = owner == f
            phi_minus = 1.0 + e[~own].sum()
            E_f = np.exp(V[own]).sum()
            z = E_f * np.exp(-1.0) / phi_minus
            m_f = (1.0 + lambertw(z).real) / ALPHA
            p_new[own] = MC[own] + m_f
        diff = np.max(np.abs(p_new - p))
        p = p_new
        if diff < TOL:
            return p, it
    raise RuntimeError("best-response iteration did not converge")


def markup_iteration(owner, maxit=10_000):
    """p <- mc - (Omega o Delta^T)^{-1} s."""
    Om = omega_matrix(owner)
    p = MC + 1.0 / ALPHA
    for it in range(1, maxit + 1):
        s, _, _ = shares(p)
        D = jacobian(p)
        p_new = MC - np.linalg.solve(Om * D.T, s)
        diff = np.max(np.abs(p_new - p))
        p = p_new
        if diff < TOL:
            return p, it
    raise RuntimeError("markup iteration did not converge")


def foc_residual(p, owner):
    s, _, _ = shares(p)
    return s + (omega_matrix(owner) * jacobian(p).T) @ (p - MC)


def summarize(p, owner_for_firms):
    s, s0, phi = shares(p)
    m = p - MC
    profit_j = m * s
    out = {"p": p, "m": m, "s": s, "s0": s0, "phi": phi,
           "cs": np.log(phi) / ALPHA, "profit": profit_j.sum(),
           "pbar": (s * p).sum() / s.sum()}
    rows = []
    for f in np.unique(owner_for_firms):
        own = owner_for_firms == f
        Sf = s[own].sum()
        rows.append(dict(firm=f, n=own.sum(), S=Sf, mbar=(s[own] * m[own]).sum() / Sf,
                         profit=profit_j[own].sum()))
    out["firms"] = pd.DataFrame(rows)
    return out


single_owner = np.arange(J)
pS_br, itS_br = br_iteration(single_owner)
pM_br, itM_br = br_iteration(OWNER)
pS_mk, itS_mk = markup_iteration(single_owner)
pM_mk, itM_mk = markup_iteration(OWNER)

S = summarize(pS_br, OWNER)     # firm rows group products by the owner in the data
M = summarize(pM_br, OWNER)

# Common-markup check for M and the closed form 1 / (alpha (1 - S_f)).
cm_err = 0.0
for _, r in M["firms"].iterrows():
    own = OWNER == r.firm
    cm_err = max(cm_err, np.max(np.abs(M["m"][own] - 1.0 / (ALPHA * (1.0 - r.S)))))
# Single-product closed form 1 / (alpha (1 - s_j)).
sp_err = np.max(np.abs(S["m"] - 1.0 / (ALPHA * (1.0 - S["s"]))))

# Robustness: random starting values for best-response iteration.
rng = np.random.default_rng(257)


def br_from(p0, owner):
    p = p0.copy()
    firms = np.unique(owner)
    for _ in range(10_000):
        e = np.exp(A - ALPHA * p)
        p_new = p.copy()
        for f in firms:
            own = owner == f
            z = np.exp(V[own]).sum() * np.exp(-1.0) / (1.0 + e[~own].sum())
            p_new[own] = MC[own] + (1.0 + lambertw(z).real) / ALPHA
        if np.max(np.abs(p_new - p)) < TOL:
            return p_new
        p = p_new
    return p


dev_S = max(np.max(np.abs(br_from(MC + rng.uniform(0, 5, J), single_owner) - pS_br)) for _ in range(500))
dev_M = max(np.max(np.abs(br_from(MC + rng.uniform(0, 5, J), OWNER) - pM_br)) for _ in range(500))

# Strategic complementarity in isolation: only owner 4 consolidates its seven products,
# every other product stays with a separate firm.
owner_4only = np.where(OWNER == 4, 100, np.arange(J))
p4, it4 = br_iteration(owner_4only)
s4, s04, _ = shares(p4)
own4 = OWNER == 4
dp4_own = p4[own4] - pS_br[own4]
dp4_riv = p4[~own4] - pS_br[~own4]
S4_f = s4[own4].sum()

# ----------------------------------------------------------------- write numbers
lines = []
w = lines.append
w("25-product logit market: Bertrand-Nash equilibria")
w(f"alpha = {ALPHA}, beta = {BETA.tolist()}, gamma = {GAMMA.tolist()}")
w(f"owners and product counts: {dict(zip(*np.unique(OWNER, return_counts=True)))}")
w(f"mc range: {MC.min():.3f} to {MC.max():.3f}")
w("")
w(f"BR iteration: S converged in {itS_br} iterations, M in {itM_br}")
w(f"Markup iteration: S converged in {itS_mk} iterations, M in {itM_mk}")
w(f"max |p_BR - p_markup|: S {np.max(np.abs(pS_br - pS_mk)):.2e}, M {np.max(np.abs(pM_br - pM_mk)):.2e}")
w(f"max FOC residual: S {np.max(np.abs(foc_residual(pS_br, single_owner))):.2e}, "
  f"M {np.max(np.abs(foc_residual(pM_br, OWNER))):.2e}")
w(f"closed-form checks: single-product markup err {sp_err:.2e}, common markup err {cm_err:.2e}")
w(f"500 random starts mc + U[0,5]^25: max deviation S {dev_S:.2e}, M {dev_M:.2e}")
w("")
w("Market level                 S          M")
w(f"outside share            {S['s0']:.4f}     {M['s0']:.4f}")
w(f"consumer surplus         {S['cs']:.4f}     {M['cs']:.4f}")
w(f"total profit             {S['profit']:.4f}     {M['profit']:.4f}")
w(f"share-weighted price     {S['pbar']:.4f}     {M['pbar']:.4f}")
w(f"Phi* = 1/s0              {S['phi']:.3f}      {M['phi']:.3f}")
w(f"profit change (%)        {100 * (M['profit'] / S['profit'] - 1):.2f}")
w(f"CS change                {M['cs'] - S['cs']:.4f}  ({100 * (M['cs'] / S['cs'] - 1):.2f}%)")
w(f"total surplus            {S['cs'] + S['profit']:.4f}     {M['cs'] + M['profit']:.4f}")
w("")
w("Firm level: firm, n, S_f^S(%), S_f^M(%), mbar^S, m^M, 100 pi^S, 100 pi^M, 1/(alpha(1-S_f^M))")
for (_, a), (_, b) in zip(S["firms"].iterrows(), M["firms"].iterrows()):
    w(f"{int(a.firm)}  {int(a.n)}  {100 * a.S:6.2f}  {100 * b.S:6.2f}  {a.mbar:.3f}  {b.mbar:.3f}  "
      f"{100 * a.profit:7.3f}  {100 * b.profit:7.3f}  {1 / (ALPHA * (1 - b.S)):.3f}")
w("")
w(f"single-product markups: {S['m'].min():.3f} to {S['m'].max():.3f}")
dp = M["p"] - S["p"]
w(f"price increase S->M: min {dp.min():.3f} (product {dp.argmin()}), max {dp.max():.3f} (product {dp.argmax()})")
w(f"products whose share falls S->M: {int((M['s'] < S['s']).sum())} of {J}")
w("")
w("Only owner 4 consolidates (others single-product):")
w(f"  converged in {it4} iterations; owner-4 share {100 * S4_f:.2f}% (S: {100 * S['firms'].S[4]:.2f}%), "
  f"common markup {(p4 - MC)[own4][0]:.3f} = 1/(alpha(1-S)) {1 / (ALPHA * (1 - S4_f)):.3f}")
w(f"  owner-4 price increases {dp4_own.min():.4f} to {dp4_own.max():.4f}")
w(f"  rival price increases {dp4_riv.min():.4f} to {dp4_riv.max():.4f} (mean {dp4_riv.mean():.4f})")
w(f"  outside share {s04:.4f}; owner-4 profit {100 * ((p4 - MC) * s4)[own4].sum():.3f} (x100); "
  f"rivals' total profit {((p4 - MC) * s4)[~own4].sum():.4f} vs S {((pS_br - MC) * S['s'])[~own4].sum():.4f}")
w("")
w("Product level: j, owner, mc, pS, pM, mS, mM, sS(%), sM(%)")
for j in range(J):
    w(f"{j:2d} {OWNER[j]} {MC[j]:.3f} {S['p'][j]:.3f} {M['p'][j]:.3f} {S['m'][j]:.3f} {M['m'][j]:.3f} "
      f"{100 * S['s'][j]:6.2f} {100 * M['s'][j]:6.2f}")
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

# ----------------------------------------------------------------- figure
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
C_S = "#2a78d6"     # categorical slot 1
C_M = "#eb6834"     # categorical slot 2
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
})

order = []
bounds = []
for f in np.unique(OWNER):
    prods = np.where(OWNER == f)[0]
    prods = prods[np.argsort(-S["s"][prods])]   # within an owner, largest share first
    start = len(order)
    order.extend(prods.tolist())
    bounds.append((f, start, len(order) - 1))
pos = np.arange(J)

fig, ax = plt.subplots(figsize=(6.2, 3.5))
ax.axhline(1.0 / ALPHA, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.text(J - 0.4, 1.0 / ALPHA - 0.003, r"$\dfrac{1}{\alpha} = 0.5$", ha="right", va="top",
        fontsize=10, color=INK2)
for f, a, b in bounds:
    if f > 0:
        ax.axvline(a - 0.5, color=GRID, lw=0.8, zorder=0)
    ax.text((a + b) / 2, 0.705, f"Owner {f}", ha="center", va="bottom", fontsize=9, color=INK2)
ax.scatter(pos, S["m"][order], s=30, marker="o", color=C_S, edgecolor="#fcfcfb", linewidth=1.0,
           zorder=3, label="Single-product firms (S)")
ax.scatter(pos, M["m"][order], s=30, marker="s", color=C_M, edgecolor="#fcfcfb", linewidth=1.0,
           zorder=3, label="Five multi-product owners (M)")
ax.set_xticks(pos)
ax.set_xticklabels([str(j) for j in order], fontsize=7.5)
ax.set_xlabel("Product (grouped by owner, largest share first)")
ax.set_ylabel(r"Markup $p_j - mc_j$")
ax.set_xlim(-0.7, J - 0.3)
ax.set_ylim(0.465, 0.74)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
# legend above the axes, clear of the owner separators and of the data
ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          handletextpad=0.3, columnspacing=1.5)
fig.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(HERE / "logit25_markups_preview.png", dpi=130)
print(f"wrote {OUT_PDF}")
