"""Most collusive price schedules in the Rotemberg-Saloner continuum model (Section 12.4.2).

N = 2 Bertrand firms, marginal cost c = 0, demand state omega ~ U[1, 2], iid.
Panel (a): multiplicative shocks, D(p, omega) = omega (1 - p), p^m(omega) = 1/2.
Panel (b): additive shocks,       D(p, omega) = omega - p,     p^m(omega) = omega / 2.

The most collusive scheme caps industry profit at kappa * E_star, where
kappa = delta / ((1 - delta)(N - 1)) and E_star is the largest fixed point of
h(E) = E_omega[ min{ pi^m(omega), kappa E } ] (Proposition prop:12-rs-threshold).
We find E_star by iterating E <- h(E) from E_0 = E[pi^m], a monotone decreasing
sequence that converges to the largest fixed point. Prices: p(omega) = p^m(omega)
below the cutoff omega_star, and the lower root of pi(p, omega) = kappa E_star above.
Also computes the restricted collude-or-compete scheme for comparison.
Output: figures/ch12/rs_continuum.pdf. Prints every number quoted in the text.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#7a7974"

N = 2
W_LO, W_HI = 1.0, 2.0
grid = np.linspace(W_LO, W_HI, 200001)  # fine grid for expectations under U[1, 2]


def kappa(delta):
    return delta / ((1 - delta) * (N - 1))


MODELS = {
    "multiplicative": dict(
        pim=lambda w: w / 4.0,
        pm=lambda w: 0.5 + 0 * w,
        # lower root of w p (1 - p) = v
        price=lambda v, w: 0.5 * (1 - np.sqrt(np.maximum(1 - 4 * v / w, 0))),
        deltas=(0.52, 0.56),
    ),
    "additive": dict(
        pim=lambda w: w ** 2 / 4.0,
        pm=lambda w: w / 2.0,
        # lower root of p (w - p) = v
        price=lambda v, w: 0.5 * (w - np.sqrt(np.maximum(w ** 2 - 4 * v, 0))),
        deltas=(0.54, 0.60),
    ),
}


def largest_fixed_point(pim_vals, k, tol=1e-13, maxit=200000):
    E = pim_vals.mean()
    for _ in range(maxit):
        En = np.minimum(pim_vals, k * E).mean()
        if abs(En - E) < tol:
            return En
        E = En
    return E


def restricted_fixed_point(pim_vals, k, tol=1e-13, maxit=200000):
    """Collude (monopoly) in states with pi^m <= k E_R, compete (p = c) otherwise."""
    E = pim_vals.mean()
    for _ in range(maxit):
        En = np.where(pim_vals <= k * E, pim_vals, 0.0).mean()
        if abs(En - E) < tol:
            return En
        E = En
    return E


results = {}
for name, m in MODELS.items():
    pim_vals = m["pim"](grid)
    Epim = pim_vals.mean()
    kfull = m["pim"](W_HI) / Epim
    dfull = kfull * (N - 1) / (1 + kfull * (N - 1))
    print(f"[{name}] E[pi^m] = {Epim:.6f}; monopoly in all states iff kappa >= {kfull:.6f}, "
          f"i.e. delta >= {dfull:.6f}")
    for d in m["deltas"]:
        k = kappa(d)
        E = largest_fixed_point(pim_vals, k)
        cap = k * E
        # cutoff state: pi^m(omega*) = cap
        wstar = grid[np.argmin(np.abs(pim_vals - cap))] if cap < m["pim"](W_HI) else W_HI
        p_top = m["price"](cap, W_HI)
        ER = restricted_fixed_point(pim_vals, k)
        print(f"  delta = {d:.2f}: kappa = {k:.6f}, E* = {E:.6f}, cap kappa E* = {cap:.6f}, "
              f"omega* = {wstar:.4f}, price at omega* = {m['pm'](wstar):.4f}, "
              f"price at omega = 2: {p_top:.4f}, p^m(2) = {m['pm'](W_HI):.4f}; "
              f"collude-or-compete E_R = {ER:.6f}")
        results[(name, d)] = (k, E, cap, wstar)

# closed-form check for the multiplicative model: omega*^2 - (4 - 2/kappa) omega* + 1 = 0
for d in MODELS["multiplicative"]["deltas"]:
    k = kappa(d)
    bq = 4 - 2 / k
    print(f"  closed form omega* (multiplicative, delta = {d}): {(bq + np.sqrt(bq ** 2 - 4)) / 2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=False)
w = np.linspace(W_LO, W_HI, 801)
titles = {"multiplicative": r"(a) $D(p,\omega) = \omega(1-p)$",
          "additive": r"(b) $D(p,\omega) = \omega - p$"}
for ax, (name, m) in zip(axes, MODELS.items()):
    ax.plot(w, m["pm"](w), color=GRAY, ls="--", lw=1.4, label=r"monopoly $p^m(\omega)$")
    for d, col in zip(m["deltas"], (BLUE, ORANGE)):
        k, E, cap, wstar = results[(name, d)]
        p = np.where(m["pim"](w) <= cap, m["pm"](w), m["price"](cap, w))
        ax.plot(w, p, color=col, lw=2, label=rf"collusive, $\delta = {d:.2f}$")
        ax.axvline(wstar, color=col, ls=":", lw=0.9)
    ax.set_title(titles[name], fontsize=9.5)
    ax.set_xlabel(r"demand state $\omega$")
    ax.set_xlim(W_LO, W_HI)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
axes[0].set_ylabel("price")
axes[0].set_ylim(0, 0.6)
axes[1].set_ylim(0, 1.05)
axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "rs_continuum.pdf")
fig.savefig(out)
print("saved", os.path.abspath(out))
