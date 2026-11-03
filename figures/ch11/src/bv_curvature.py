"""Unit demand versus constant expenditures in a stylized analgesics merger (fig:11-bvcurv).

A purely illustrative calibration (NOT a replication of Bjornerstedt and Verboven 2016).
Four products: two paracetamol products owned by firms A and B (nest P), one ibuprofen
product (nest I) and one ASA product (nest S), plus an outside good. One-level nested logit
with nesting parameter sigma. Pre-merger prices are normalized to 1, so value and volume
shares coincide at the calibration point. Inside shares: 50 percent of the potential market,
split 24/24/27/25 percent (paracetamol A, paracetamol B, ibuprofen, ASA), roughly the
substance shares around December 2008 with paracetamol split equally (an assumption).

Two demand specifications:
  unit demand:          delta_j = xi_j - alpha * p_j,      q_j = M * s_j(delta)
  constant expenditure: delta_j = xi_j - alpha * ln p_j,   q_j = B * w_j(delta) / p_j
For each target pair (own-price elasticity of paracetamol A, diversion ratio from A to B)
both specifications are calibrated to the same pre-merger elasticity and diversion, so they
imply the same pre-merger margins, the same upward pricing pressure and the same GUPPI.
Marginal costs come from the pre-merger single-product FOCs. The merger joins A and B, with
no efficiencies. Differences in predicted price effects therefore come only from curvature.

Outputs: figures/ch11/bv_curvature.pdf, figures/ch11/src/bv_curvature_numbers.txt
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, fsolve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "bv_curvature.pdf"
TXT = HERE / "bv_curvature_numbers.txt"

NEST = np.array([0, 0, 1, 2])                  # nest of each product
S_IN = 0.5 * np.array([0.24, 0.24, 0.27, 0.25])
S0 = 1.0 - S_IN.sum()
P0 = np.ones(4)


def nl_shares(delta, sigma):
    ev = np.exp(delta / (1.0 - sigma))
    Dg = np.array([ev[NEST == g].sum() for g in range(3)])
    s_cond = ev / Dg[NEST]
    s_g = Dg ** (1.0 - sigma) / (1.0 + (Dg ** (1.0 - sigma)).sum())
    return s_cond * s_g[NEST], s_cond


def nl_jac(s, s_cond, sigma):
    """J[k, j] = d s_k / d delta_j for the one-level nested logit."""
    J = np.zeros((4, 4))
    for k in range(4):
        for j in range(4):
            if k == j:
                J[k, j] = s[k] * (1 / (1 - sigma) - sigma / (1 - sigma) * s_cond[k] - s[k])
            elif NEST[k] == NEST[j]:
                J[k, j] = -s[k] * (sigma / (1 - sigma) * s_cond[j] + s[j])
            else:
                J[k, j] = -s[k] * s[j]
    return J


def A1(sigma):
    s_cond = S_IN[0] / S_IN[:2].sum()
    return 1 / (1 - sigma) - sigma / (1 - sigma) * s_cond - S_IN[0]


def d_unit(sigma):
    s_g = S_IN[:2].sum()
    return S_IN[1] * (sigma / ((1 - sigma) * s_g) + 1) / A1(sigma)


class Model:
    def __init__(self, kind, eps, D):
        self.kind = kind
        target = D if kind == "unit" else D / (1 - 1 / eps)
        self.sigma = brentq(lambda sg: d_unit(sg) - target, 0.0, 0.9999)
        self.alpha = eps / A1(self.sigma) if kind == "unit" else (eps - 1) / A1(self.sigma)
        s_cond = S_IN / np.array([S_IN[NEST == g].sum() for g in range(3)])[NEST]
        delta = np.log(S_IN / S0) - self.sigma * np.log(s_cond)
        self.xi = delta + (self.alpha * P0 if kind == "unit" else self.alpha * np.log(P0))
        q, dq = self.demand(P0)
        self.c = P0 + q / np.diag(dq)                # single-product FOC: q + (p-c) dq/dp = 0
        self.own_eps = np.diag(dq) * P0 / q
        self.div12 = -dq[0, 1] / dq[0, 0]            # dq[j,k] = d q_k / d p_j

    def demand(self, p):
        if self.kind == "unit":
            delta = self.xi - self.alpha * p
        else:
            delta = self.xi - self.alpha * np.log(p)
        s, s_cond = nl_shares(delta, self.sigma)
        J = nl_jac(s, s_cond, self.sigma)            # J[k, j] = ds_k/ddelta_j
        if self.kind == "unit":
            q = s
            dq = (-self.alpha * J).T                 # dq[j, k] = d q_k / d p_j
        else:
            q = s / p
            dq = np.zeros((4, 4))
            for j in range(4):
                for k in range(4):
                    dq[j, k] = J[k, j] * (-self.alpha / p[j]) / p[k]
                    if j == k:
                        dq[j, k] -= s[k] / p[k] ** 2
        return q, dq

    def foc(self, p, owners):
        q, dq = self.demand(p)
        out = np.zeros(4)
        for j in range(4):
            out[j] = q[j] + sum((p[k] - self.c[k]) * dq[j, k] for k in range(4)
                                if owners[k] == owners[j])
        return out

    def merge(self):
        owners = [0, 0, 1, 2]
        p = fsolve(self.foc, P0 * 1.05, args=(owners,), xtol=1e-13)
        assert np.abs(self.foc(p, owners)).max() < 1e-10
        return p


lines = ["Stylized calibration: inside shares", str(S_IN), f"outside share {S0}"]
EPS = 3.0
lines.append(f"Target own elasticity of paracetamol A: -{EPS}")
lines.append("D12 | model | sigma | alpha | own eps (P_A, I, S) | margin P_A | GUPPI | "
             "dp P_A % | dp I % | dp S %")
grid = np.linspace(0.15, 0.5, 36)
res = {"unit": [], "ce": []}
for D in grid:
    for kind in ("unit", "ce"):
        m = Model(kind, EPS, D)
        p = m.merge()
        res[kind].append(100 * (p - 1))
        if abs(D - 0.30) < 1e-9 or abs(D - 0.45) < 1e-9 or abs(D - 0.15) < 1e-9 or abs(D - 0.5) < 1e-9:
            L = (1 - m.c[0]) / 1.0
            lines.append(f"{D:.2f} | {kind:4s} | {m.sigma:.3f} | {m.alpha:.3f} | "
                         f"{m.own_eps[0]:.2f}, {m.own_eps[2]:.2f}, {m.own_eps[3]:.2f} | "
                         f"{L:.3f} | {m.div12*L:.4f} | {100*(p[0]-1):.2f} | "
                         f"{100*(p[2]-1):.2f} | {100*(p[3]-1):.2f}  (check D={m.div12:.4f})")
res = {k: np.array(v) for k, v in res.items()}
ratio = res["ce"][:, 0] / res["unit"][:, 0]
lines.append(f"ratio CE/unit paracetamol effect over D grid: min {ratio.min():.2f}, max {ratio.max():.2f}")
TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(6.0, 3.6))
blue, orange = "#2a78d6", "#eb6834"
ax.plot(grid, res["ce"][:, 0], color=orange, lw=2, label="Constant expenditures")
ax.plot(grid, res["unit"][:, 0], color=blue, lw=2, label="Unit demand")
ax.text(grid[-1] + 0.005, res["ce"][-1, 0], "Constant\nexpenditures", fontsize=9, va="center")
ax.text(grid[-1] + 0.005, res["unit"][-1, 0], "Unit\ndemand", fontsize=9, va="center")
ax.set_xlabel(r"Pre-merger diversion ratio between the paracetamol products, $D_{AB}$", fontsize=10)
ax.set_ylabel("Predicted paracetamol price rise (%)", fontsize=10)
ax.set_xlim(0.14, 0.575)
ax.set_ylim(0, None)
ax.grid(True, color="#e6e5e0", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
