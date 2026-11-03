"""Monte Carlo for other-city price instruments with national demand shocks
(Figure fig:10-hausman-iv-mc, Proposition prop:10-hausman-iv-bias).

Model (one product, N cities, T weeks), as in Section sec:10-bresnahan:
  demand   q_nt = d_n - b p_nt + xi_nt,      xi_nt = chi_t + e_nt
  cost     c_nt = c_n + k_t + v_nt
  pricing  single-product Bertrand with linear demand: p_nt = (c_nt + (d_n + xi_nt) / b) / 2
National components chi_t (variance s * sigma_xi^2) and k_t (variance sigma_k^2) are common
to all cities; local components e_nt and v_nt are independent across cities.

Estimators (city fixed effects removed by within-city demeaning):
  OLS of q on p;
  IV of q on p with the average price of the same product in the other N-1 cities
  in the same week (a Hausman instrument).
Probability limits (Proposition prop:10-hausman-iv-bias):
  b_IV  -> b (1 - 2 lam),     lam     = s sigma_xi^2 / (b^2 sigma_k^2 + s sigma_xi^2)
  b_OLS -> b (1 - 2 lam_OLS), lam_OLS = sigma_xi^2 / (b^2 (sigma_k^2 + sigma_v^2) + sigma_xi^2)
Implied consumer surplus with linear demand at the observed point, q^2 / (2 b_hat),
relative to the truth, equals b / b_hat.

Outputs: figures/ch10/hausman_iv_bias.pdf and figures/ch10/src/hausman_iv_bias_numbers.txt
Run from the Textbook directory: python figures/ch10/src/hausman_iv_bias.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "hausman_iv_bias.pdf"
TXT = HERE / "hausman_iv_bias_numbers.txt"

# Design: mirrors Hausman's panel (7 cities, 137 weeks).
N_CITY, T_WEEK, R = 7, 137, 500
B, D_MEAN, C_MEAN = 2.0, 10.0, 2.0
SIG_XI, SIG_K, SIG_V = 1.0, 0.5, 0.5
SHARES = np.linspace(0.0, 0.5, 11)
SEED = 20261022

BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1-2 (validated)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plims(s):
    lam = s * SIG_XI**2 / (B**2 * SIG_K**2 + s * SIG_XI**2)
    lam_ols = SIG_XI**2 / (B**2 * (SIG_K**2 + SIG_V**2) + SIG_XI**2)
    return 1 - 2 * lam, 1 - 2 * lam_ols


def one_draw(rng, s):
    d_n = D_MEAN + rng.normal(0, 1.0, N_CITY)          # city demand effects
    c_n = C_MEAN + rng.normal(0, 0.2, N_CITY)          # city cost effects
    chi = rng.normal(0, np.sqrt(s) * SIG_XI, T_WEEK)
    e = rng.normal(0, np.sqrt(1 - s) * SIG_XI, (T_WEEK, N_CITY))
    k = rng.normal(0, SIG_K, T_WEEK)
    v = rng.normal(0, SIG_V, (T_WEEK, N_CITY))
    xi = chi[:, None] + e
    c = c_n[None, :] + k[:, None] + v
    p = 0.5 * (c + (d_n[None, :] + xi) / B)
    q = d_n[None, :] - B * p + xi
    pt = p - p.mean(axis=0)                            # within-city demeaning
    qt = q - q.mean(axis=0)
    z = (p.sum(axis=1, keepdims=True) - p) / (N_CITY - 1)   # other-city average price
    zt = z - z.mean(axis=0)
    b_iv = -(zt * qt).sum() / (zt * pt).sum()
    b_ols = -(pt * qt).sum() / (pt * pt).sum()
    first_stage_r = np.corrcoef(zt.ravel(), pt.ravel())[0, 1]
    return b_iv, b_ols, first_stage_r


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for s in SHARES:
        draws = np.array([one_draw(rng, s) for _ in range(R)])
        iv, ols, fs = draws[:, 0] / B, draws[:, 1] / B, draws[:, 2]
        a_iv, a_ols = plims(s)
        rows.append(dict(s=s, iv_mean=iv.mean(), iv_p10=np.percentile(iv, 10), iv_p90=np.percentile(iv, 90),
                         ols_mean=ols.mean(), a_iv=a_iv, a_ols=a_ols,
                         cs_iv=np.median(1 / iv), cs_ols=np.median(1 / ols),
                         cs_iv_p10=np.percentile(1 / iv, 10), cs_iv_p90=np.percentile(1 / iv, 90),
                         fs=fs.mean()))
    lines = ["s  IVmean  IVp10  IVp90  plimIV  OLSmean  plimOLS  CSratioIV(median)  CS_plim  firststage_corr"]
    for r in rows:
        lines.append(f"{r['s']:.2f} {r['iv_mean']:.3f} {r['iv_p10']:.3f} {r['iv_p90']:.3f} {r['a_iv']:.3f} "
                     f"{r['ols_mean']:.3f} {r['a_ols']:.3f} {r['cs_iv']:.3f} {1 / r['a_iv']:.3f} {r['fs']:.3f}")
    TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.75))
    s = np.array([r["s"] for r in rows])
    sf = np.linspace(0, 0.5, 200)
    a_iv_f = np.array([plims(x)[0] for x in sf])
    a_ols_f = np.array([plims(x)[1] for x in sf])

    ax = axes[0]
    ax.fill_between(s, [r["iv_p10"] for r in rows], [r["iv_p90"] for r in rows], color=BLUE,
                    alpha=0.15, linewidth=0)
    ax.plot(sf, a_iv_f, color=BLUE, lw=1.6)
    ax.plot(sf, a_ols_f, color=ORANGE, lw=1.6, ls="--")
    ax.plot(s, [r["iv_mean"] for r in rows], "o", color=BLUE, ms=4.5, mec="white", mew=0.8)
    ax.plot(s, [r["ols_mean"] for r in rows], "s", color=ORANGE, ms=4.5, mec="white", mew=0.8)
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    ax.text(0.30, 0.66, "IV, other-city prices", color=INK, fontsize=8)
    ax.text(0.02, 0.24, "OLS", color=INK, fontsize=8)
    ax.set_ylabel(r"$\frac{\hat b}{b}$", rotation=0, labelpad=10, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Estimated demand slope", fontsize=9, color=INK)

    ax = axes[1]
    ax.fill_between(s, [r["cs_iv_p10"] for r in rows], [r["cs_iv_p90"] for r in rows], color=BLUE,
                    alpha=0.15, linewidth=0)
    ax.plot(sf, 1 / a_iv_f, color=BLUE, lw=1.6)
    ax.plot(sf, 1 / a_ols_f, color=ORANGE, lw=1.6, ls="--")
    ax.plot(s, [r["cs_iv"] for r in rows], "o", color=BLUE, ms=4.5, mec="white", mew=0.8)
    ax.plot(s, [r["cs_ols"] for r in rows], "s", color=ORANGE, ms=4.5, mec="white", mew=0.8)
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    ax.text(0.27, 1.55, "IV", color=INK, fontsize=8)
    ax.text(0.02, 3.18, "OLS", color=INK, fontsize=8)
    ax.set_ylabel(r"$\frac{\widehat{CS}}{CS}$", rotation=0, labelpad=14, fontsize=12)
    ax.set_ylim(0.8, 4.8)
    ax.set_title("Implied consumer surplus", fontsize=9, color=INK)

    for ax in axes:
        ax.set_xlabel(r"national share of demand variance, $\frac{\sigma_\chi^2}{\sigma_\chi^2+\sigma_e^2}$")
        ax.set_xlim(-0.01, 0.51)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT)
    fig.savefig(HERE / "hausman_iv_bias_preview.png", dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
