"""Welfare calculations and the virtual-price figure for Chapter 5, Section 5.7.

Uses the running AIDS example of the chapter (three goods):
    alpha_0 = 0, alpha = (0.5, 0.3, 0.2), beta = (-0.10, 0.04, 0.06),
    Gamma = [[-0.20, 0.12, 0.08], [0.12, -0.22, 0.10], [0.08, 0.10, -0.18]],
    base prices p = (1, 1, 1), ln I = 1, beta_0 = 1.

It computes
  1. CV and EV of a 10 percent rise in p_1 (closed form), the Laspeyres
     (fixed-weight) approximation and the second-order approximation;
  2. the Marshallian and Hicksian virtual prices of good 3 (other prices fixed),
     the exact CV of removing good 3, and the CV evaluated at the Marshallian
     choke price (which understates the loss);
  3. the closed-form beta_i = 0 approximation, and the CES calibrated to the
     same share and own-price elasticity (sigma, CV);
  4. a product-withdrawal counterfactual: good 3 leaves, the single-product
     firms selling goods 1 and 2 re-optimize (marginal costs recovered from
     their pre-withdrawal first-order conditions), and the CV is decomposed
     into a pure variety effect and a price-adjustment effect;
  5. the figure figures/ch05/virtual_price.pdf (Hicksian demand for good 3
     under AIDS and under the calibrated CES).
Numbers are written to figures/ch05/src/virtual_price_numbers.txt.
"""
import os
import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.dirname(HERE)
OUT = os.path.join(HERE, "virtual_price_numbers.txt")

alpha0 = 0.0
alpha = np.array([0.5, 0.3, 0.2])
beta = np.array([-0.10, 0.04, 0.06])
Gamma = np.array([[-0.20, 0.12, 0.08],
                  [0.12, -0.22, 0.10],
                  [0.08, 0.10, -0.18]])
J = 3
lnI = 1.0
I = np.exp(lnI)


def lnP(lnp):
    return alpha0 + alpha @ lnp + 0.5 * lnp @ Gamma @ lnp


def b_of(lnp):
    return np.exp(beta @ lnp)


def shares(lnp, lnX=lnI):
    return alpha + Gamma @ lnp + beta * (lnX - lnP(lnp))


def quantities(lnp, lnX=lnI):
    return shares(lnp, lnX) * np.exp(lnX) / np.exp(lnp)


def ln_e(lnp, u):
    return lnP(lnp) + u * b_of(lnp)


def indirect_u(lnp, lnX=lnI):
    return (lnX - lnP(lnp)) / b_of(lnp)


def hicks_share(lnp, u):
    return alpha + Gamma @ lnp + beta * u * b_of(lnp)


def elasticities(lnp, lnX=lnI):
    w = shares(lnp, lnX)
    dlnP = alpha + Gamma @ lnp
    dw = Gamma - np.outer(beta, dlnP)
    eps = -np.eye(J) + dw / w[:, None]
    eta = 1.0 + beta / w
    return w, eps, eta


lines = []
out = lambda s="": lines.append(s)

lnp0 = np.zeros(J)
u0 = indirect_u(lnp0)
w0, eps0, eta0 = elasticities(lnp0)
out(f"Base: u0 = {u0:.4f}, I = {I:.4f}, shares = {np.round(w0, 4)}")
out(f"  own elasticities = {np.round(np.diag(eps0), 3)}")

# ---------------------------------------------------------------- 1. CV and EV of a 10% rise in p1
lnp1 = np.array([np.log(1.1), 0.0, 0.0])
u1 = indirect_u(lnp1)
lne_p1_u0 = lnP(lnp1) + (lnI - lnP(lnp0)) * b_of(lnp1) / b_of(lnp0)
CV = np.exp(lne_p1_u0) - I
lne_p0_u1 = lnP(lnp0) + (lnI - lnP(lnp1)) * b_of(lnp0) / b_of(lnp1)
EV = I - np.exp(lne_p0_u1)
dlnPidx = lnP(lnp1) - lnP(lnp0)
nonhom = (lnI - lnP(lnp0)) * (b_of(lnp1) / b_of(lnp0) - 1.0)
laspeyres = I * (w0 @ (np.exp(lnp1) - 1.0))          # sum_i w_i (p_i'/p_i - 1) I
first_order = w0 @ lnp1
C = Gamma + np.outer(beta, beta) * (lnI - lnP(lnp0))  # Hicksian share derivatives
second_order = first_order + 0.5 * lnp1 @ C @ lnp1
# Marshallian surplus: integral of q1(p1, 1, 1, I) from 1 to 1.1
grid = np.linspace(0.0, np.log(1.1), 2001)
q1 = np.array([quantities(np.array([g, 0.0, 0.0]))[0] for g in grid])
p1 = np.exp(grid)
dCS = np.trapezoid(q1, p1)
out()
out("1. Ten percent rise in p1 (closed form)")
out(f"  u1 = {u1:.4f}")
out(f"  Delta ln P (translog) = {dlnPidx:.5f}; non-homothetic term = {nonhom:.5f}; ln(e'/I) = {lne_p1_u0 - lnI:.5f}")
out(f"  CV = {CV:.4f} ({100 * CV / I:.3f} percent of I); EV = {EV:.4f} ({100 * EV / I:.3f} percent)")
out(f"  Marshallian surplus loss = {dCS:.4f} ({100 * dCS / I:.3f} percent)")
out(f"  Laspeyres approximation w'(p'/p - 1) I = {laspeyres:.4f} ({100 * laspeyres / I:.3f} percent)")
out(f"  first-order ln approximation sum w dln p = {first_order:.5f}; second order = {second_order:.5f}")
out(f"  exact COL index e(p',u0)/e(p0,u0) = {np.exp(lne_p1_u0 - lnI):.5f}")

# ---------------------------------------------------------------- 2. virtual prices of good 3
fM = lambda x: shares(np.array([0.0, 0.0, x]))[2]
lnpM = brentq(fM, 0.0, 5.0)
# check with the quadratic of the text: -1/2 b g x^2 + [g - b(a+G)] x + [a + G + b(lnX - A)]
a3, g33, b3 = alpha[2], Gamma[2, 2], beta[2]
G3 = 0.0           # other log prices are zero
A = alpha0         # other log prices are zero
coefs = [-0.5 * b3 * g33, g33 - b3 * (a3 + G3), a3 + G3 + b3 * (lnI - A)]
roots = np.roots(coefs)
fH = lambda x: hicks_share(np.array([0.0, 0.0, x]), u0)[2]
lnpH = brentq(fH, 0.0, 5.0)
lnpV_beta0 = -(a3 + G3) / g33   # if beta_3 were zero
CV_remove = np.exp(ln_e(np.array([0.0, 0.0, lnpH]), u0)) - I
CV_at_M = np.exp(ln_e(np.array([0.0, 0.0, lnpM]), u0)) - I
# check: area under the Hicksian demand curve
gH = np.linspace(0.0, lnpH, 4001)
hq = np.array([hicks_share(np.array([0.0, 0.0, g]), u0)[2] * np.exp(ln_e(np.array([0.0, 0.0, g]), u0)) / np.exp(g) for g in gH])
area = np.trapezoid(hq, np.exp(gH))
out()
out("2. Virtual prices of good 3 (p1 = p2 = 1)")
out(f"  Marshallian: ln p3V = {lnpM:.4f}, p3V = {np.exp(lnpM):.3f}; quadratic roots = {np.round(roots, 4)}")
out(f"  Hicksian at u0: ln p3V = {lnpH:.4f}, p3V = {np.exp(lnpH):.3f}")
out(f"  ln p3V if beta_3 = 0: {lnpV_beta0:.4f}")
out(f"  CV of removal (Hicksian virtual price) = {CV_remove:.4f} = {100 * CV_remove / I:.2f} percent of I")
out(f"  area under Hicksian demand = {area:.4f}")
out(f"  e(p3 = Marshallian choke, u0) - I = {CV_at_M:.4f} = {100 * CV_at_M / I:.2f} percent (understates)")

# ---------------------------------------------------------------- 3. beta = 0 formula and CES
s3 = w0[2]
e33 = eps0[2, 2]
lnCV_b0 = s3 / (2.0 * (-e33 - 1.0))
sigma = (-e33 - s3) / (1.0 - s3)
CV_ces = (1.0 - s3) ** (1.0 / (1.0 - sigma)) - 1.0
lnCV_ces = -(1.0 - s3) * np.log(1.0 - s3) / (-e33 - 1.0)
out()
out("3. Closed forms at the same share and own elasticity")
out(f"  share s3 = {s3:.4f}, own Marshallian elasticity = {e33:.4f}")
out(f"  AIDS beta=0 formula: ln(p^V/p) = {1.0 / (-e33 - 1.0):.4f}, ln(1+CV/I) = {lnCV_b0:.4f}, CV/I = {100 * (np.exp(lnCV_b0) - 1):.2f} percent")
out(f"  CES: sigma = {sigma:.4f}, CV/I = {100 * CV_ces:.2f} percent, ln(1+CV/I) = {np.log(1 + CV_ces):.4f} (formula {lnCV_ces:.4f})")
out(f"  full AIDS: CV/I = {100 * CV_remove / I:.2f} percent, ln(1+CV/I) = {np.log(1 + CV_remove / I):.4f}")

# ---------------------------------------------------------------- 4. withdrawal counterfactual
c = np.exp(lnp0) * (1.0 + 1.0 / np.diag(eps0))      # single-product FOC costs
out()
out("4. Withdrawal of good 3 with re-pricing of goods 1 and 2")
out(f"  marginal costs from FOCs c = {np.round(c, 4)}, Lerner = {np.round(-1 / np.diag(eps0), 4)}")


def choke_M(lp1, lp2):
    f = lambda x: shares(np.array([lp1, lp2, x]))[2]
    lo = max(lp1, lp2, 0.0) - 2.0
    return brentq(f, lo, 8.0)


def q_without3(lp):
    x = choke_M(lp[0], lp[1])
    q = quantities(np.array([lp[0], lp[1], x]))
    return q[:2]


def foc_without3(lp):
    q = q_without3(lp)
    h = 1e-6
    res = np.zeros(2)
    for k in range(2):
        d = np.zeros(2); d[k] = h
        dq = (q_without3(lp + d)[k] - q_without3(lp - d)[k]) / (2 * h)   # dq_k/dln p_k
        dq_dp = dq / np.exp(lp[k])
        res[k] = q[k] + (np.exp(lp[k]) - c[k]) * dq_dp
    return res


lp_new = fsolve(foc_without3, np.zeros(2), xtol=1e-12)
p_new = np.exp(lp_new)
x_new = choke_M(*lp_new)
q_new = q_without3(lp_new)
# own elasticities in the counterfactual
h = 1e-6
own_new = []
for k in range(2):
    d = np.zeros(2); d[k] = h
    own_new.append((np.log(q_without3(lp_new + d)[k]) - np.log(q_without3(lp_new - d)[k])) / (2 * h))
out(f"  new prices p1' = {p_new[0]:.4f}, p2' = {p_new[1]:.4f} (percent change {100 * (p_new[0] - 1):.2f}, {100 * (p_new[1] - 1):.2f})")
out(f"  Marshallian choke price of good 3 at new prices = {np.exp(x_new):.4f}")
out(f"  own elasticities after withdrawal = {np.round(own_new, 4)}; residual FOC = {foc_without3(lp_new)}")


def hicks_choke(lp1, lp2, u):
    f = lambda x: hicks_share(np.array([lp1, lp2, x]), u)[2]
    return brentq(f, -2.0, 8.0)


xv0 = hicks_choke(0.0, 0.0, u0)
xv1 = hicks_choke(lp_new[0], lp_new[1], u0)
e_variety = np.exp(ln_e(np.array([0.0, 0.0, xv0]), u0))
e_total = np.exp(ln_e(np.array([lp_new[0], lp_new[1], xv1]), u0))
CV_var = e_variety - I
CV_tot = e_total - I
CV_price = CV_tot - CV_var
out(f"  Hicksian virtual price of good 3 at new prices = {np.exp(xv1):.4f}")
out(f"  CV total = {CV_tot:.4f} ({100 * CV_tot / I:.2f} percent of I)")
out(f"    variety effect = {CV_var:.4f} ({100 * CV_var / I:.2f} percent), price-adjustment effect = {CV_price:.4f} ({100 * CV_price / I:.2f} percent)")
pi_old = (np.exp(lnp0) - c) * quantities(lnp0)
pi_new = (p_new - c[:2]) * q_new
out(f"  profits before = {np.round(pi_old, 4)}, after (goods 1, 2) = {np.round(pi_new, 4)}")
out(f"  change in profits of firms 1 and 2 = {pi_new.sum() - pi_old[:2].sum():.4f}; lost profit of firm 3 = {pi_old[2]:.4f}")

text = "\n".join(lines)
print(text)
with open(OUT, "w") as fh:
    fh.write(text + "\n")

# ---------------------------------------------------------------- 5. figure
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "mathtext.fontset": "cm"})

p_aids = np.exp(np.linspace(-0.25, lnpH, 600))
h_aids = np.array([hicks_share(np.array([0.0, 0.0, np.log(pp)]), u0)[2] *
                   np.exp(ln_e(np.array([0.0, 0.0, np.log(pp)]), u0)) / pp for pp in p_aids])
a3c = s3
p_ces = np.linspace(np.exp(-0.25), 9.0, 800)
P_ces = ((1 - a3c) + a3c * p_ces ** (1 - sigma)) ** (1.0 / (1 - sigma))
h_ces = a3c * p_ces ** (-sigma) * P_ces ** sigma * I       # u0 = I at the base (P = 1)

fig, ax = plt.subplots(figsize=(5.6, 4.0))
mask = p_aids >= 1.0
ax.fill_betweenx(p_aids[mask], 0, h_aids[mask], color=BLUE, alpha=0.16, linewidth=0)
ax.plot(h_aids, p_aids, color=BLUE, lw=2.0)
ax.plot(h_ces, p_ces, color=ORANGE, lw=2.0, ls="--")
q0 = h_aids[np.argmin(np.abs(p_aids - 1.0))]
ax.plot([q0], [1.0], "o", color=INK, ms=4.5, zorder=5)
ax.axhline(1.0, color=INK2, lw=0.6, ls=":")
ax.axhline(np.exp(lnpH), color=INK2, lw=0.6, ls=":")
ax.text(1.02, 1.0, r"$p_3^0 = 1$", color=INK, va="center", fontsize=9, transform=ax.get_yaxis_transform())
ax.text(1.02, np.exp(lnpH), r"$p_3^V = %.2f$" % np.exp(lnpH), color=INK, va="center", fontsize=9,
        transform=ax.get_yaxis_transform())
ax.text(0.02, 1.25, "CV (AIDS)\n%.1f%% of $I$" % (100 * CV_remove / I), color=INK, fontsize=9)
ia = np.argmin(np.abs(p_aids - 2.6))
ax.annotate("AIDS", xy=(h_aids[ia], p_aids[ia]), xytext=(0.36, 3.1), color=INK, fontsize=9.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
ic = np.argmin(np.abs(p_ces - 5.5))
ax.annotate(r"CES, $\sigma = %.2f$" % sigma, xy=(h_ces[ic], p_ces[ic]), xytext=(0.22, 6.2), color=INK,
            fontsize=9.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
ax.set_xlim(0, 0.95)
ax.set_ylim(0.75, 9.0)
ax.set_xlabel("quantity of good 3 (Hicksian demand at $u^0$)", color=INK)
ax.set_ylabel("price of good 3", color=INK)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(INK2)
ax.tick_params(colors=INK2)
fig.subplots_adjust(left=0.11, right=0.82, bottom=0.13, top=0.97)
fig.savefig(os.path.join(FIGDIR, "virtual_price.pdf"))
fig.savefig(os.path.join(HERE, "virtual_price_preview.png"), dpi=150)
print("figure written")
