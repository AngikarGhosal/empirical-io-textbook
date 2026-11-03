"""
Appendix C, Problem C.1(i): merger predictions from a cross-industry SCP regression.

Design. 1,000 industries with linear inverse demand P = a - bQ, constant marginal cost c and
fixed cost F. Primitives are drawn independently,
    a ~ U[8, 12], c ~ U[1, 4], b ~ U[0.5, 2], log F ~ U[log 0.1, log 10],
and kept if the free-entry Cournot number of firms is at least one, x = (a - c)/sqrt(bF) >= 2.
The number of firms N ~ U{1, ..., 20} is drawn independently of the primitives and firms play
the symmetric Cournot game (design (1) of the Monte Carlo experiment in Chapter 1). The seed
and the order of the random draws are those of figures/ch01/src/scp_montecarlo.py, so the SCP
regression below reproduces column (1) of Table 1.x in Chapter 1 exactly.

Steps.
  1. OLS of L on a constant and HHI with HC1 standard errors (numpy only).
  2. For each industry with N >= 2, a merger of two firms without cost savings moves the
     industry to the symmetric Cournot equilibrium with N - 1 firms.
       actual change          dL  = L(N - 1) - L(N) = c(a - c) / ((a + (N-1)c)(a + Nc))
       naive prediction       b1 * 2 s_i s_j = b1 * 2 / N^2
       equilibrium-HHI pred.  b1 * (1/(N-1) - 1/N) = b1 / (N(N-1))
       local-slope prediction (dL/dHHI at the industry's own HHI) * 1/(N(N-1))
  3. Summary statistics overall and by groups of N; figure.

Outputs (relative to Textbook/):
  figures/appC/merger_mc.pdf
  figures/appC/src/cournot_merger_mc_summary.json
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)

SEED = 20261022          # same seed and draw order as the Chapter 1 experiment
NIND = 1000
rng = np.random.default_rng(SEED)

A_RNG = (8.0, 12.0)
C_RNG = (1.0, 4.0)
B_RNG = (0.5, 2.0)
F_RNG = (0.1, 10.0)      # log-uniform
NFIX_MAX = 20


def draw_primitives(n, rng):
    """Independent draws of (a, c, b, F), kept if x = (a - c)/sqrt(bF) >= 2."""
    keep = {k: [] for k in ("a", "c", "b", "F")}
    accepted = 0
    while accepted < n:
        m = 2 * (n - accepted) + 10
        a = rng.uniform(*A_RNG, m)
        c = rng.uniform(*C_RNG, m)
        b = rng.uniform(*B_RNG, m)
        F = np.exp(rng.uniform(np.log(F_RNG[0]), np.log(F_RNG[1]), m))
        ok = (a - c) / np.sqrt(b * F) >= 2.0
        idx = np.where(ok)[0][: n - accepted]
        for k, v in zip(("a", "c", "b", "F"), (a, c, b, F)):
            keep[k].extend(v[idx].tolist())
        accepted += len(idx)
    return pd.DataFrame(keep)


def lerner_cournot(a, c, N):
    P = (a + N * c) / (N + 1.0)
    return (P - c) / P


def cournot_slope(a, c, H):
    """dL/dHHI along the Cournot curve L = (a - c) H / (a H + c)."""
    return c * (a - c) / (a * H + c) ** 2


def ols_hc1(y, h):
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(y), np.asarray(h, float)])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    meat = (X * (e ** 2)[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv * n / (n - k)
    r2 = 1.0 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(b0=float(beta[0]), se0=float(np.sqrt(V[0, 0])), b1=float(beta[1]),
                se1=float(np.sqrt(V[1, 1])), r2=float(r2), n=int(n))


# ------------------------------------------------------------------ simulate
prim = draw_primitives(NIND, rng)
a, c, b, F = (prim[k].to_numpy() for k in ("a", "c", "b", "F"))
Nfix = rng.integers(1, NFIX_MAX + 1, NIND).astype(float)

H = 1.0 / Nfix
L = lerner_cournot(a, c, Nfix)
reg = ols_hc1(L, H)
b1 = reg["b1"]

# ------------------------------------------------------------------ mergers
m = Nfix >= 2
Nm, am, cm, Hm = Nfix[m], a[m], c[m], H[m]
dL_true = lerner_cournot(am, cm, Nm - 1.0) - lerner_cournot(am, cm, Nm)
assert np.allclose(dL_true, cm * (am - cm) / ((am + (Nm - 1) * cm) * (am + Nm * cm)))
dH_naive = 2.0 / Nm ** 2
dH_eq = 1.0 / (Nm * (Nm - 1.0))
preds = {
    "naive": b1 * dH_naive,
    "eq": b1 * dH_eq,
    "local": cournot_slope(am, cm, Hm) * dH_eq,
}


def summarize(sel):
    out = dict(n=int(sel.sum()),
               dHHI_naive=float(1e4 * dH_naive[sel].mean()),
               dHHI_eq=float(1e4 * dH_eq[sel].mean()),
               dL_true=float(dL_true[sel].mean()))
    for k, p in preds.items():
        err = p[sel] - dL_true[sel]
        out["pred_" + k] = float(p[sel].mean())
        out["corr_" + k] = float(np.corrcoef(p[sel], dL_true[sel])[0, 1]) if sel.sum() > 2 \
            and np.std(p[sel]) > 0 else float("nan")
        out["ratio_" + k] = float(p[sel].mean() / dL_true[sel].mean())
        out["share50_" + k] = float(np.mean(np.abs(err) > 0.5 * dL_true[sel]))
        out["share_over_" + k] = float(np.mean(err > 0))
        out["mape_" + k] = float(np.mean(np.abs(err) / dL_true[sel]))
    return out


allsel = np.ones(m.sum(), bool)
groups = {"all": allsel, "N2_4": (Nm >= 2) & (Nm <= 4), "N5_10": (Nm >= 5) & (Nm <= 10),
          "N11_20": Nm >= 11}
summ = {g: summarize(s) for g, s in groups.items()}

# correlation across industries within each N is driven only by (a, c)
summary = dict(seed=SEED, n_industries=NIND, scp=reg, merger=summ,
               n_N1=int((Nfix == 1).sum()),
               mean_local_slope=float(cournot_slope(a, c, H).mean()))
with open(os.path.join(HERE, "cournot_merger_mc_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

print("SCP regression (design 1): b0 = %.3f (%.3f), b1 = %.3f (%.3f), R2 = %.3f, n = %d"
      % (reg["b0"], reg["se0"], reg["b1"], reg["se1"], reg["r2"], reg["n"]))
print("industries with N = 1 (no merger possible):", summary["n_N1"])
print("mean local Cournot slope over all industries: %.3f" % summary["mean_local_slope"])
for g, s in summ.items():
    print("\n--- group", g, "n =", s["n"])
    print("  mean dHHI points: naive %.0f, equilibrium %.0f" % (s["dHHI_naive"], s["dHHI_eq"]))
    print("  mean actual dL: %.4f" % s["dL_true"])
    for k in preds:
        print("  %-6s mean pred %.4f  ratio %.3f  corr %.3f  share|err|>50%% %.3f  share over %.3f"
              "  mean |err|/dL %.3f"
              % (k, s["pred_" + k], s["ratio_" + k], s["corr_" + k], s["share50_" + k],
                 s["share_over_" + k], s["mape_" + k]))

# ------------------------------------------------------------------ figure
INK, MUTED, GRID, ORANGE = "#0b0b0b", "#52514e", "#d9d8d4", "#eb6834"
ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
bounds = [1.5, 3.5, 5.5, 8.5, 11.5, 15.5, 20.5]
cmap = ListedColormap(ramp)
norm = BoundaryNorm(bounds, cmap.N)
plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
lims = (1.0e-3, 0.5)
titles = {"naive": r"(a) $\Delta\mathit{HHI} = 2 s_i s_j = \frac{2}{N^2}$",
          "eq": r"(b) $\Delta\mathit{HHI} = \frac{1}{N - 1} - \frac{1}{N}$"}
for ax, k in zip(axes, ("naive", "eq")):
    ax.plot(lims, lims, color=ORANGE, lw=1.2, zorder=1, label="45-degree line")
    sc = ax.scatter(preds[k], dL_true, c=Nm, cmap=cmap, norm=norm, s=9, lw=0.25,
                    edgecolors="white", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.set_title(titles[k], fontsize=9, color=INK)
    ax.set_xlabel(r"predicted $\Delta L = \hat\beta_1 \Delta\mathit{HHI}$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel(r"equilibrium $\Delta L$")
axes[0].legend(loc="upper left", frameon=False, fontsize=8)
cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02, ticks=[2.5, 4.5, 7, 10, 13.5, 18])
cb.ax.set_yticklabels(["2-3", "4-5", "6-8", "9-11", "12-15", "16-20"])
cb.set_label("number of firms $N$ before the merger")
cb.outline.set_edgecolor(MUTED)
fig.savefig(os.path.join(FIGDIR, "merger_mc.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "merger_mc_preview.png"), dpi=150, bbox_inches="tight")
print("\nwrote", os.path.join(FIGDIR, "merger_mc.pdf"))
