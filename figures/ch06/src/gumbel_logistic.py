"""Standard Gumbel (type-I extreme value) and standard logistic densities,
each compared with a normal density with the same mean and variance.

Panel (a): Gumbel density f(x) = exp(-x) exp(-exp(-x)), mean gamma_E,
           variance pi^2/6, against N(gamma_E, pi^2/6).
Panel (b): logistic density (difference of two independent standard Gumbels),
           variance pi^2/3, against N(0, pi^2/3); the inset shows both
           densities on a log scale to display the tails.

No random numbers are used. Output: figures/ch06/gumbel_logistic.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)

GAMMA_E = np.euler_gamma
VAR_G = np.pi ** 2 / 6.0
VAR_L = np.pi ** 2 / 3.0

# --------------------------------------------------------------- numbers
g = stats.gumbel_r()
lg = stats.logistic()
print("Gumbel mean      %.6f (gamma_E = %.6f)" % (g.mean(), GAMMA_E))
print("Gumbel variance  %.6f (pi^2/6 = %.6f)" % (g.var(), VAR_G))
print("Gumbel skewness  %.6f" % float(g.stats(moments="s")))
print("Gumbel median    %.6f (-ln ln 2 = %.6f)" % (g.median(), -np.log(np.log(2.0))))
print("Logistic var     %.6f (pi^2/3 = %.6f)" % (lg.var(), VAR_L))
print("Logistic excess kurtosis %.6f" % float(lg.stats(moments="k")))
z = 6.0
print("Tail P(Z > 6): logistic %.3e, normal(var pi^2/3) %.3e"
      % (lg.sf(z), stats.norm(0, np.sqrt(VAR_L)).sf(z)))

# --------------------------------------------------------------- figure
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    "mathtext.fontset": "stix", "font.family": "STIXGeneral"})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.75))

# Panel (a): Gumbel versus normal with the same mean and variance.
x = np.linspace(-3.0, 7.0, 600)
ax1.plot(x, g.pdf(x), color=BLUE, lw=1.6, label="Standard Gumbel")
ax1.plot(x, stats.norm(GAMMA_E, np.sqrt(VAR_G)).pdf(x), color=ORANGE, lw=1.4,
         ls=(0, (4, 2)), label=r"Normal, mean $\gamma_E$, variance $\frac{\pi^2}{6}$")
ax1.axvline(GAMMA_E, color=INK2, lw=0.8, ls=(0, (1, 1.5)))
ax1.annotate(r"mean $\gamma_E \approx 0.577$", xy=(GAMMA_E, 0.06),
             xytext=(2.3, 0.10), fontsize=7.2, color=INK,
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax1.annotate("mode 0", xy=(0.0, g.pdf(0.0)), xytext=(-2.8, 0.36), fontsize=7.2,
             color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
ax1.set_xlim(-3.0, 7.0)
ax1.set_ylim(0.0, 0.47)
ax1.set_xlabel(r"$\epsilon$")
ax1.set_ylabel("Density")
ax1.set_title("(a) Gumbel: right skewed", loc="left")
ax1.legend(loc="upper right", frameon=False, handlelength=2.2, borderaxespad=0.2)

# Panel (b): logistic versus normal with variance pi^2/3.
zz = np.linspace(-8.0, 8.0, 800)
ax2.plot(zz, lg.pdf(zz), color=BLUE, lw=1.6, label="Standard logistic")
ax2.plot(zz, stats.norm(0.0, np.sqrt(VAR_L)).pdf(zz), color=ORANGE, lw=1.4,
         ls=(0, (4, 2)), label=r"Normal, variance $\frac{\pi^2}{3}$")
ax2.set_xlim(-8.0, 8.0)
ax2.set_ylim(0.0, 0.30)
ax2.set_xlabel(r"$z = \epsilon_1 - \epsilon_2$")
ax2.set_ylabel("Density")
ax2.set_title("(b) Logistic: fatter tails", loc="left")
ax2.legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.2)

ins = ax2.inset_axes([0.69, 0.40, 0.29, 0.38])
zt = np.linspace(0.0, 12.0, 300)
ins.semilogy(zt, lg.pdf(zt), color=BLUE, lw=1.3)
ins.semilogy(zt, stats.norm(0.0, np.sqrt(VAR_L)).pdf(zt), color=ORANGE, lw=1.2,
             ls=(0, (4, 2)))
ins.set_xlim(0.0, 12.0)
ins.set_ylim(1e-9, 1.0)
ins.set_xticks([0, 6, 12])
ins.set_yticks([1e-8, 1e-4, 1.0])
ins.tick_params(labelsize=6, length=2)
ins.set_title("log scale, $z > 0$", fontsize=6.5, pad=2)
for s in ("top", "right"):
    ins.spines[s].set_visible(False)

for a_ in (ax1, ax2):
    a_.grid(True, color=GRID, lw=0.5)
    a_.set_axisbelow(True)

fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(FIGDIR, "gumbel_logistic.pdf"))
fig.savefig(os.path.join(HERE, "gumbel_logistic_preview.png"), dpi=150)
print("saved")
