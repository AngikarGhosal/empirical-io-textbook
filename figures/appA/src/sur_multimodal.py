"""
Iterated SUR can stop at a local maximum of the Gaussian likelihood (Appendix A, Section A.2.5).

Design: two equations y_g = x_g b_g + e_g, g = 1, 2, one regressor per equation and no intercept,
n = 4 observations, standard normal regressors, true b = (0, 0) and errors with correlation 0.8.
For each draw we run iterated feasible SUR (recompute Sigma from the residuals, then GLS) from the
equation-by-equation OLS estimate and from every point of a grid of starting values. Every reported
limit is checked to be a fixed point and a strict local maximum of the concentrated likelihood
-n/2 log det S(b), S(b) the residual covariance matrix. Limits whose residual correlation exceeds
0.95 in absolute value are discarded, so that no reported maximum sits near a singular S(b). The
script stops at the first draw in which the limit reached from OLS has a log likelihood at least
0.05 below that of another limit. A grid over [-6, 6]^2 with step 0.01 then checks that the other
limit is the highest point of the likelihood there. All 2 x 2 algebra is written out in closed form
for speed.

Output: figures/appA/src/sur_multimodal.json (the numbers quoted in the text).
"""
import os
import json
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261103
N = 4
RHO = 0.8
GRID = np.linspace(-3.0, 3.0, 7)
MAX_CORR = 0.95
MIN_GAP = 0.05


class System:
    def __init__(self, X, Y):
        self.n = len(X)
        self.xx = X.T @ X
        self.xy = X.T @ Y
        self.yy = Y.T @ Y

    def cov(self, b):
        b1, b2 = b
        xx, xy, yy, n = self.xx, self.xy, self.yy, self.n
        s11 = (yy[0, 0] - 2 * b1 * xy[0, 0] + b1 * b1 * xx[0, 0]) / n
        s22 = (yy[1, 1] - 2 * b2 * xy[1, 1] + b2 * b2 * xx[1, 1]) / n
        s12 = (yy[0, 1] - b1 * xy[0, 1] - b2 * xy[1, 0] + b1 * b2 * xx[0, 1]) / n
        return s11, s12, s22

    def logdet(self, b):
        s11, s12, s22 = self.cov(b)
        det = s11 * s22 - s12 * s12
        return math.log(det) if det > 0 else math.inf

    def corr(self, b):
        s11, s12, s22 = self.cov(b)
        return s12 / math.sqrt(s11 * s22)

    def gls_step(self, b):
        s11, s12, s22 = self.cov(b)
        det = s11 * s22 - s12 * s12
        w11, w12, w22 = s22 / det, -s12 / det, s11 / det
        xx, xy = self.xx, self.xy
        a11, a12, a22 = w11 * xx[0, 0], w12 * xx[0, 1], w22 * xx[1, 1]
        c1 = w11 * xy[0, 0] + w12 * xy[0, 1]
        c2 = w12 * xy[1, 0] + w22 * xy[1, 1]
        d = a11 * a22 - a12 * a12
        return ((a22 * c1 - a12 * c2) / d, (a11 * c2 - a12 * c1) / d)

    def iterate(self, b, tol=1e-12, max_iter=5000):
        path = [self.logdet(b)]
        for it in range(1, max_iter + 1):
            new = self.gls_step(b)
            path.append(self.logdet(new))
            if max(abs(new[0] - b[0]), abs(new[1] - b[1])) < tol:
                return new, it, path
            b = new
        return b, max_iter, path

    def hessian(self, b, h=1e-4):
        b = np.asarray(b)
        E = np.eye(2)
        H = np.zeros((2, 2))
        for j in range(2):
            for k in range(2):
                H[j, k] = (self.logdet(b + h * E[j] + h * E[k]) - self.logdet(b + h * E[j] - h * E[k])
                           - self.logdet(b - h * E[j] + h * E[k]) + self.logdet(b - h * E[j] - h * E[k])) / (4 * h * h)
        return H


def main():
    rng = np.random.default_rng(SEED)
    L = np.linalg.cholesky(np.array([[1.0, RHO], [RHO, 1.0]]))
    for draw in range(1, 100001):
        X = rng.standard_normal((N, 2))
        Y = rng.standard_normal((N, 2)) @ L.T
        sys_ = System(X, Y)
        ols = (sys_.xy[0, 0] / sys_.xx[0, 0], sys_.xy[1, 1] / sys_.xx[1, 1])
        try:
            b_ols, it_ols, path = sys_.iterate(ols)
            limits = [sys_.iterate((s, t))[0] for s in GRID for t in GRID]
        except (ZeroDivisionError, ValueError):
            continue
        if not all(math.isfinite(v) for v in path) or abs(sys_.corr(b_ols)) >= MAX_CORR:
            continue
        limits = [b for b in limits if math.isfinite(sys_.logdet(b)) and abs(sys_.corr(b)) < MAX_CORR]
        if not limits:
            continue
        best = min(limits, key=sys_.logdet)
        gap = -N / 2 * (sys_.logdet(best) - sys_.logdet(b_ols))
        if gap < MIN_GAP:
            continue
        H_ols, H_best = sys_.hessian(b_ols), sys_.hessian(best)
        if min(np.linalg.eigvalsh(H_ols)) <= 0 or min(np.linalg.eigvalsh(H_best)) <= 0:
            continue
        rise = [-N / 2 * (path[i + 1] - path[i]) for i in range(len(path) - 1)]
        g = np.linspace(-6.0, 6.0, 1201)
        B1, B2 = np.meshgrid(g, g, indexing="ij")
        s11, s12, s22 = sys_.cov((B1, B2))
        surface = np.log(np.where(s11 * s22 - s12 ** 2 > 0, s11 * s22 - s12 ** 2, np.nan))
        k = np.unravel_index(np.nanargmin(surface), surface.shape)
        out = {
            "seed": SEED, "draw": draw, "n": N, "error_correlation": RHO,
            "X": X.tolist(), "Y": Y.tolist(),
            "ols": list(ols),
            "limit_from_ols": list(b_ols), "iterations_from_ols": it_ols,
            "loglik_never_falls_from_ols": bool(min(rise) > -1e-12),
            "other_limit": list(best),
            "loglik_gap": gap,
            "residual_correlations": [sys_.corr(b_ols), sys_.corr(best)],
            "fixed_point_errors": [max(abs(a - c) for a, c in zip(sys_.gls_step(b), b)) for b in (b_ols, best)],
            "hessian_eigenvalues_logdet": [np.linalg.eigvalsh(H_ols).tolist(), np.linalg.eigvalsh(H_best).tolist()],
            "grid_argmax_loglik_on_[-6,6]^2": [float(B1[k]), float(B2[k])],
            "grid_max_minus_other_limit": float(-N / 2 * (surface[k] - sys_.logdet(best))),
        }
        with open(os.path.join(HERE, "sur_multimodal.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        return
    raise SystemExit("no draw found")


if __name__ == "__main__":
    main()
