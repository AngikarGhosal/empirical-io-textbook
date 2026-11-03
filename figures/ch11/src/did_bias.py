"""Control-group bias of difference in differences in a linear Bertrand merger (fig:11-didbias).

Model (Section 11.6.1, prop:11-didbias): N symmetric single-product firms with linear demand
    q_i = a - b p_i + d sum_{j != i} p_j,   b > (N-1) d > 0,
constant marginal cost c. Firms 1 and 2 merge; the N-2 outsiders keep pricing alone.

The script
  1. solves the pre- and post-merger Bertrand equilibria numerically (linear FOC systems),
  2. checks the closed forms of the proposition:
        p0 = (a + b c) / (2b - (N-1) d),
        lambda = dp_out / dp_in = 2d / (2b - (N-3) d),
        dp_in = d (p0 - c) / (2(b - d) - (N-2) d lambda),
  3. reports the worked example (N = 3, a = 10, b = 3, d = 1, c = 2),
  4. plots the share of the true effect that a DiD with an outsider as control recovers,
     1 - lambda, against the diversion ratio D = d / b, for N = 3, 4, 6.

Outputs: figures/ch11/did_bias.pdf, figures/ch11/src/did_bias_numbers.txt
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "did_bias.pdf"
TXT = HERE / "did_bias_numbers.txt"


def equilibrium(a, b, d, c, N, merged):
    """Solve the linear FOC system. merged=True: firms 0 and 1 maximize joint profit."""
    A = np.zeros((N, N))
    rhs = np.zeros(N)
    for i in range(N):
        # d/dp_i of own profit: a - 2b p_i + b c + d sum_{j!=i} p_j (+ joint terms)
        A[i, i] = -2 * b
        for j in range(N):
            if j != i:
                A[i, j] = d
        rhs[i] = -(a + b * c)
        if merged and i in (0, 1):
            k = 1 - i
            # + (p_k - c) dq_k/dp_i = d (p_k - c)
            A[i, k] += d
            rhs[i] += d * c
    return np.linalg.solve(A, rhs)


def closed_form(a, b, d, c, N):
    p0 = (a + b * c) / (2 * b - (N - 1) * d)
    lam = 2 * d / (2 * b - (N - 3) * d)
    x = d * (p0 - c) / (2 * (b - d) - (N - 2) * d * lam)
    return p0, lam, x, lam * x


lines = []
# 1-2. numerical check of the closed forms on random admissible parameters
rng = np.random.default_rng(0)
maxerr = 0.0
for _ in range(2000):
    N = int(rng.integers(3, 9))
    d = rng.uniform(0.1, 1.0)
    b = (N - 1) * d * rng.uniform(1.01, 4.0)
    a = rng.uniform(5, 20)
    # demand at p = c must be positive: a > (b - (N-1) d) c
    c = rng.uniform(0, 0.99) * a / (b - (N - 1) * d)
    pre = equilibrium(a, b, d, c, N, False)
    post = equilibrium(a, b, d, c, N, True)
    p0, lam, x, y = closed_form(a, b, d, c, N)
    err = max(abs(pre - p0).max(), abs(post[0] - p0 - x), abs(post[2:] - p0 - y).max())
    maxerr = max(maxerr, err)
    assert 0 < lam < 1 and x > 0
lines.append(f"Max abs error closed form vs numerical (2000 random draws): {maxerr:.2e}")

# 3. worked example
a, b, d, c, N = 10.0, 3.0, 1.0, 2.0, 3
pre = equilibrium(a, b, d, c, N, False)
post = equilibrium(a, b, d, c, N, True)
p0, lam, x, y = closed_form(a, b, d, c, N)
lines.append("Worked example N=3, a=10, b=3, d=1, c=2")
lines.append(f"  pre-merger price p0 = {p0:.4f} (numerical {pre[0]:.4f})")
lines.append(f"  post-merger insider price = {post[0]:.4f}, outsider = {post[2]:.4f}")
lines.append(f"  dp_in = {x:.4f}  dp_out = {y:.4f}  lambda = {lam:.4f} (= d/b = {d/b:.4f})")
lines.append(f"  pct change insiders = {100*x/p0:.2f}%, outsider = {100*y/p0:.2f}%")
lines.append(f"  log change insiders = {np.log(post[0]/p0):.4f}, outsider = {np.log(post[2]/p0):.4f}")
lines.append(f"  DiD (levels) = {x - y:.4f} = {(1-lam):.4f} x dp_in; DiD (logs) = "
             f"{np.log(post[0]/p0) - np.log(post[2]/p0):.4f}")
# margins and diversion
lines.append(f"  pre-merger margin (p0-c)/p0 = {(p0-c)/p0:.4f}; diversion D = d/b = {d/b:.4f}")
q0 = a - b * p0 + d * (N - 1) * p0
lines.append(f"  pre-merger quantity per firm = {q0:.4f}")
TXT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# 4. figure: share of true insider effect recovered by DiD with an outsider as control
fig, ax = plt.subplots(figsize=(6.0, 3.6))
colors = {3: "#2a78d6", 4: "#eb6834", 6: "#1baf7a"}
for N in (3, 4, 6):
    Dmax = 1.0 / (N - 1)            # b > (N-1) d  <=>  D < 1/(N-1)
    D = np.linspace(0.0, Dmax * 0.999, 300)
    lam = 2 * D / (2 - (N - 3) * D)
    ax.plot(D, 1 - lam, color=colors[N], lw=2, label=f"$N={N}$")
    ax.text(D[-1] + 0.006, (1 - lam)[-1], f"$N={N}$", color="#0b0b0b", fontsize=9,
            va="center", ha="left")
ax.set_xlabel(r"Diversion ratio to each rival, $D=\dfrac{d}{b}$", fontsize=10)
ax.set_ylabel(r"Share of $\Delta p_I$ recovered, $1-\lambda$", fontsize=10)
ax.set_xlim(0, 0.56)
ax.set_ylim(0, 1.02)
ax.grid(True, color="#e6e5e0", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
