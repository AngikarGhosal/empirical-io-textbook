"""Illustrative two-regime (collusion / price war) market: why switching
regressions can detect regimes from the shape of the joint distribution of
price and quantity.

Linear inverse demand  P = A - B Q + u_d,
marginal cost          mc = c0 + b1 Q + u_c,
supply relation        P = mc + theta * B * Q   (perceived MR = mc),
with theta = theta_c (collusion, probability varpi) or theta_w (price war).

All parameter values are illustrative and chosen by the author.
Output: figures/ch01/regime_mixture.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)

# ------------------------------------------------------------- parameters
A, B = 10.0, 1.0          # inverse demand intercept and slope
C0, B1 = 2.0, 0.25        # marginal cost intercept and slope
THETA_C, THETA_W = 0.5, 0.0
VARPI = 0.7               # probability of the collusive regime
SD_D, SD_C = 0.8, 0.4     # standard deviations of demand and cost errors
T = 400
rng = np.random.default_rng(20261022)

collude = rng.random(T) < VARPI
theta = np.where(collude, THETA_C, THETA_W)
u_d = rng.normal(0.0, SD_D, T)
u_c = rng.normal(0.0, SD_C, T)
k = B * (1.0 + theta) + B1
Q = (A - C0 + u_d - u_c) / k
P = A - B * Q + u_d


def regime_moments(th):
    """Mean and standard deviation of equilibrium price in one regime."""
    kk = B * (1.0 + th) + B1
    mean_q = (A - C0) / kk
    mean_p = A - B * mean_q
    var_p = SD_D ** 2 * (1.0 - B / kk) ** 2 + SD_C ** 2 * (B / kk) ** 2
    return mean_q, mean_p, np.sqrt(var_p)


qc, pc, sc = regime_moments(THETA_C)
qw, pw, sw = regime_moments(THETA_W)
print("collusion: E[Q]=%.3f E[P]=%.3f sd(P)=%.3f" % (qc, pc, sc))
print("price war: E[Q]=%.3f E[P]=%.3f sd(P)=%.3f" % (qw, pw, sw))
print("share of collusive draws in sample: %.3f" % collude.mean())

# ------------------------------------------------------------- figure
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral"})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.7))

# Panel (a): equilibria by regime with the deterministic curves.
qg = np.linspace(2.0, 8.6, 50)
ax1.plot(qg, A - B * qg, color=INK, lw=1.2)
ax1.plot(qg, C0 + B1 * qg, color=INK2, lw=1.0, ls=(0, (4, 2)))
ax1.plot(qg, C0 + (B1 + THETA_C * B) * qg, color=INK2, lw=1.0, ls=(0, (1, 1.5)))
ax1.scatter(Q[collude], P[collude], s=9, color=BLUE, alpha=0.55, linewidths=0,
            label="Collusive weeks")
ax1.scatter(Q[~collude], P[~collude], s=11, marker="^", color=ORANGE, alpha=0.65,
            linewidths=0, label="Price-war weeks")
ax1.text(2.75, A - B * 2.75 + 0.2, "Demand", ha="left", va="bottom", color=INK,
         fontsize=7.2)
ax1.text(2.1, C0 + B1 * 2.1 - 0.2, r"$P = mc$ ($\theta = 0$)", ha="left", va="top",
         color=INK2, fontsize=7.2)
ax1.text(5.9, C0 + (B1 + THETA_C * B) * 5.9 + 0.35,
         r"$P = mc + \theta^{c} b Q$", ha="right", va="bottom", color=INK2, fontsize=7.2)
ax1.set_xlim(2.0, 8.6)
ax1.set_ylim(1.5, 8.0)
ax1.set_xlabel(r"Quantity $Q$")
ax1.set_ylabel(r"Price $P$")
ax1.set_title("(a) Equilibria by regime", loc="left")
ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.72), frameon=False,
           handletextpad=0.3, borderaxespad=0.2)

# Panel (b): marginal distribution of price, which the analyst sees without labels.
bins = np.linspace(2.0, 7.0, 31)
ax2.hist(P, bins=bins, density=True, color="#b7d3f6", edgecolor="white", lw=0.6,
         label="Simulated prices")
pg = np.linspace(2.0, 7.0, 400)


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


mix = VARPI * npdf(pg, pc, sc) + (1.0 - VARPI) * npdf(pg, pw, sw)
ax2.plot(pg, mix, color=INK, lw=1.3, label="Two-regime mixture")
m_all, s_all = P.mean(), P.std()
ax2.plot(pg, npdf(pg, m_all, s_all), color=INK2, lw=1.0, ls=(0, (4, 2)),
         label="Single normal")
ax2.set_xlim(2.0, 7.0)
ax2.set_xlabel(r"Price $P$")
ax2.set_ylabel("Density")
ax2.set_title("(b) Distribution of price", loc="left")
ax2.legend(loc="upper left", frameon=False, handletextpad=0.4, borderaxespad=0.2)

for a_ in (ax1, ax2):
    a_.grid(True, color=GRID, lw=0.5)
    a_.set_axisbelow(True)

fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIGDIR, "regime_mixture.pdf"))
fig.savefig(os.path.join(HERE, "regime_mixture_preview.png"), dpi=150)
print("saved")
