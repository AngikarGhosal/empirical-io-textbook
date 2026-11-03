"""Check of the IV bias formula for other-city price instruments (Section 5.8).

Model (Bresnahan 1997, appendix, written with column vectors):
  demand   q_nt = a + Gamma p_nt + xi_nt,         xi_nt = eta_t + e_nt
  supply   p_jnt = c_jnt + q_jnt / (-Gamma_jj),   c_nt  = cbar_t + v_nt
(single-product Bertrand pricing with linear demand and constant marginal cost).
eta_t is a national demand shock (covariance Sigma_eta), cbar_t a national cost
shock (covariance Sigma_c); e_nt and v_nt are local and independent across
cities.  Each demand equation is estimated by IV, using the prices of all J
products in another city as instruments.

Analytic probability limit derived in the text:
  plim Gamma_hat = Gamma + Lambda (D - Gamma),
  Lambda = Sigma_eta (D Sigma_c D + Sigma_eta)^{-1},  D = diag(-Gamma_jj).
The script compares it with a large simulation, for a one-product case and for
two-product cases with uncorrelated and correlated national demand shocks.
"""
import numpy as np

rng = np.random.default_rng(257)


def analytic(Gam, Sc, Se):
    D = np.diag(-np.diag(Gam))
    Lam = Se @ np.linalg.inv(D @ Sc @ D + Se)
    return Gam + Lam @ (D - Gam), Lam


def simulate(Gam, Sc, Se, T=200000, sd_local_xi=0.5, sd_local_c=0.5):
    J = Gam.shape[0]
    D = np.diag(-np.diag(Gam))
    M = np.linalg.inv(D - Gam)
    a = np.full(J, 10.0)
    Lc = np.linalg.cholesky(Sc)
    Le = np.linalg.cholesky(Se)
    cbar = rng.standard_normal((T, J)) @ Lc.T
    eta = rng.standard_normal((T, J)) @ Le.T
    data = []
    for n in range(2):                              # two cities
        xi = eta + sd_local_xi * rng.standard_normal((T, J))
        cc = 1.0 + cbar + sd_local_c * rng.standard_normal((T, J))
        p = (a + cc @ D.T + xi) @ M.T                # (D - Gamma) p = a + D c + xi
        q = a + p @ Gam.T + xi
        data.append((p, q))
    (p1, q1), (p2, q2) = data
    # IV for city 1 using city-2 prices (demeaned)
    Z = p2 - p2.mean(0)
    X = p1 - p1.mean(0)
    Y = q1 - q1.mean(0)
    B = np.linalg.solve(Z.T @ X, Z.T @ Y)            # column j = coefficients of equation j
    return B.T


cases = {
    "one product": (np.array([[-1.0]]), np.array([[1.0]]), np.array([[0.5]])),
    "two products, rho = 0": (np.array([[-2.0, 0.5], [0.5, -1.5]]), np.diag([0.3, 0.3]), np.diag([0.4, 0.4])),
    "two products, rho = 0.8": (np.array([[-2.0, 0.5], [0.5, -1.5]]), np.diag([0.3, 0.3]),
                                0.4 * np.array([[1.0, 0.8], [0.8, 1.0]])),
}
for name, (Gam, Sc, Se) in cases.items():
    plim, Lam = analytic(Gam, Sc, Se)
    sim = simulate(Gam, Sc, Se)
    print(name)
    print("  true Gamma:\n", Gam)
    print("  Lambda:\n", np.round(Lam, 4))
    print("  analytic plim:\n", np.round(plim, 4))
    print("  simulated IV:\n", np.round(sim, 4))
    if Gam.shape[0] == 1:
        g, sc, se = Gam[0, 0], Sc[0, 0], Se[0, 0]
        print("  scalar formula gamma*(g^2 sc - se)/(g^2 sc + se) =", g * (g * g * sc - se) / (g * g * sc + se))
