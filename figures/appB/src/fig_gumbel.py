"""
Figure fig:B-gumbel for Appendix B, Part 2 (Section B.5.2).

Produces (paths relative to Textbook/):
    figures/appB/gumbel_logistic.pdf
    figures/appB/src/gumbel_logistic_preview.png   (for checking only)

Left panel:  standard Gumbel density f(x) = exp(-x) exp(-exp(-x)) and the normal density with the
             same mean (Euler's constant) and variance (pi^2 / 6).
Right panel: standard logistic density exp(-x) / (1 + exp(-x))^2 (the difference of two independent
             standard Gumbels) and the normal density with mean 0 and variance pi^2 / 3.
Run standalone from Textbook/:  python figures/appB/src/fig_gumbel.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
BLUE, ORANGE = '#2a78d6', '#eb6834'      # categorical slots 1 and 2
INK, MUTED = '#0b0b0b', '#52514e'

gE = np.euler_gamma
x1 = np.linspace(-4, 8, 801)
gumbel = np.exp(-x1) * np.exp(-np.exp(-x1))
norm1 = stats.norm.pdf(x1, loc=gE, scale=np.pi / np.sqrt(6))

x2 = np.linspace(-8, 8, 801)
logistic = np.exp(-x2) / (1 + np.exp(-x2)) ** 2
norm2 = stats.norm.pdf(x2, loc=0, scale=np.pi / np.sqrt(3))

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
ax = axes[0]
ax.plot(x1, gumbel, color=BLUE, lw=2, label=r'Gumbel $\mathrm{EV}(0,1)$')
ax.plot(x1, norm1, color=ORANGE, lw=2, ls='--', label=r'$N\!\left(\gamma_E, \frac{\pi^2}{6}\right)$')
ax.axvline(0, color=MUTED, lw=0.6, ls=':')
ax.axvline(gE, color=MUTED, lw=0.6, ls=':')
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('density')
ax.set_xlim(-4, 8)
ax.set_ylim(0, 0.42)
ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.07, 1.0), handlelength=1.6)
ax.text(0.0, 1.02, '(a)', transform=ax.transAxes, fontsize=9, color=INK)

ax = axes[1]
ax.plot(x2, logistic, color=BLUE, lw=2, label=r'logistic, $\epsilon_1 - \epsilon_2$')
ax.plot(x2, norm2, color=ORANGE, lw=2, ls='--', label=r'$N\!\left(0, \frac{\pi^2}{3}\right)$')
ax.set_xlabel(r'$\epsilon_1 - \epsilon_2$')
ax.set_xlim(-8, 8)
ax.set_ylim(0, 0.36)
ax.legend(frameon=False, loc='upper center', ncol=1)
ax.text(0.0, 1.02, '(b)', transform=ax.transAxes, fontsize=9, color=INK)

fig.tight_layout(w_pad=2.0)
fig.savefig(os.path.join(FIGDIR, 'gumbel_logistic.pdf'))
fig.savefig(os.path.join(HERE, 'gumbel_logistic_preview.png'), dpi=150)
print('saved', os.path.join(FIGDIR, 'gumbel_logistic.pdf'))
