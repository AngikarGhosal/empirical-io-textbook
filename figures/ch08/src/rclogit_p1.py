"""Shared helpers for the Chapter 8, Part 1 figures (sections 8.2-8.4).

Random coefficients logit with K_r characteristics that carry random coefficients.
Notation follows the book:
  V_ij = delta_j + mu_ij,   mu_ij = xr_j . (sigma * nu_i)   (diagonal Sigma, no demographics)
  s_ij = exp(V_ij) / (1 + sum_k exp(V_ik)),   s_j = E_nu[s_ij].
The integral over nu is approximated by a set of nodes with weights (pseudo-random draws,
quasi-random draws, or a Gauss-Hermite product rule).
"""
import itertools

import numpy as np
from scipy.stats import norm, qmc


def individual_shares(delta, xr, sigma, nodes):
    """Return s_ij (n_nodes, J) and s_i0 (n_nodes,) with an overflow-safe log-sum-exp.

    delta : (J,) mean utilities
    xr    : (J, K) characteristics with random coefficients
    sigma : (K,) standard deviations of the random coefficients
    nodes : (n, K) standard normal nodes (draws or quadrature abscissae)
    """
    mu = nodes @ (xr * sigma).T            # (n, J): mu_ij = sum_k xr_jk sigma_k nu_ik
    v = delta[None, :] + mu                # (n, J)
    m = np.maximum(0.0, v.max(axis=1, keepdims=True))   # subtract max(0, max_k V_ik)
    ev = np.exp(v - m)
    e0 = np.exp(-m)
    den = e0 + ev.sum(axis=1, keepdims=True)
    return ev / den, (e0 / den)[:, 0]


def shares(delta, xr, sigma, nodes, weights=None):
    sij, si0 = individual_shares(delta, xr, sigma, nodes)
    if weights is None:
        return sij.mean(axis=0), si0.mean()
    return weights @ sij, weights @ si0


def contraction(s_obs, xr, sigma, nodes, weights=None, tol=1e-14, maxit=100000):
    """Solve s(delta) = s_obs by the BLP fixed point delta <- delta + ln s_obs - ln s(delta)."""
    s0 = 1.0 - s_obs.sum()
    delta = np.log(s_obs) - np.log(s0)     # logit starting value
    for it in range(maxit):
        s, _ = shares(delta, xr, sigma, nodes, weights)
        new = delta + np.log(s_obs) - np.log(s)
        if np.max(np.abs(new - delta)) < tol:
            return new, it + 1
        delta = new
    raise RuntimeError("contraction did not converge")


def gauss_hermite_nodes(K, q):
    """Product Gauss-Hermite rule for a K-dimensional standard normal: q**K nodes."""
    x, w = np.polynomial.hermite.hermgauss(q)
    x1 = np.sqrt(2.0) * x
    w1 = w / np.sqrt(np.pi)
    nodes = np.array(list(itertools.product(x1, repeat=K)))
    weights = np.array([np.prod(c) for c in itertools.product(w1, repeat=K)])
    return nodes, weights


def halton_normal(n, K, seed):
    """Scrambled Halton points mapped to standard normals by the inverse CDF."""
    u = qmc.Halton(d=K, scramble=True, seed=seed).random(n)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return norm.ppf(u)


def derivative_matrix(delta, xr, sigma, alpha_i_fn, nodes, weights=None):
    """Matrix D with D[j, k] = d s_j / d p_k, where alpha_i_fn(nodes) gives alpha_i per node.

    d s_j / d p_k = int alpha_i s_ij (s_ik - 1{j = k}) dF.
    """
    sij, _ = individual_shares(delta, xr, sigma, nodes)
    a = alpha_i_fn(nodes)
    n = sij.shape[0]
    w = np.full(n, 1.0 / n) if weights is None else weights
    wa = w * a
    gamma = (sij * wa[:, None]).T @ sij        # int alpha_i s_ij s_ik
    lam = (sij * wa[:, None]).sum(axis=0)       # int alpha_i s_ij
    return gamma - np.diag(lam)
