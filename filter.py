from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

VERY_SMALL_POSITIVE = 1E-300
SMALL_POSITIVE = 1E-9
LARGE_POSITIVE = 1E100


class Filter:
    def __init__(self, obs: Iterable[float]):
        self._obs = obs
        self._theta = [np.mean(obs),
                       np.std(obs),
                       0.5,
                       -np.mean(obs),
                       2 * np.std(obs)]

    def params(self):
        return {'mu': self._theta[0],
                'sigma': self._theta[1],
                'lambda': self._theta[2],
                'muJ': self._theta[3],
                'sigmaJ': self._theta[4]}

    def calibrate(self, theta0=None, **kwargs):
        if theta0 is None:  theta0 = self._theta

        bnds = [(-10, 10),
                (SMALL_POSITIVE, 10),
                (0, 1),
                (-10, 10),
                (SMALL_POSITIVE, 10)]
        res = minimize(self._computeCalibrationLogLikelihood,
                       theta0,
                       bounds=bnds,
                       options=kwargs)

        self._theta = res.x

    def _computeLogLikelihood(self, theta=None):
        if theta is None:   theta = self._theta

        mu, sigma, lmbda, muJ, sigmaJ = theta
        ls = []
        for r in self._obs:
            l = (1 - lmbda) * norm.pdf(r, loc=mu, scale=sigma) \
                + lmbda * norm.pdf(r, loc=mu + muJ, scale=np.sqrt(sigma ** 2 + sigmaJ ** 2))
            ls.append(l)
        ls = np.log(np.maximum(ls, VERY_SMALL_POSITIVE))
        return ls

    def _computeCalibrationLogLikelihood(self, theta=None):
        if theta is None:   theta = self._theta
        L = sum(self._computeLogLikelihood(theta=theta))
        return np.min([-L, LARGE_POSITIVE])

    def inferJumps(self, theta=None):
        if theta is None:   theta = self._theta

        mu, sigma, lmbda, muJ, sigmaJ = theta
        ps = []
        for r in self._obs:
            p = lmbda * norm.pdf(r, loc=mu + muJ, scale=np.sqrt(sigma ** 2 + sigmaJ ** 2))
            p /= (p + (1 - lmbda) * norm.pdf(r, loc=mu, scale=sigma))
            ps.append(p)
        return ps

    def _computeOPG(self, theta=None):
        if theta is None:   theta = self._theta
        gs = []
        for i, elt in enumerate(theta):
            bump = np.abs(elt) / 100
            theta1 = theta.copy()
            theta1[i] = elt + bump

            theta2 = theta.copy()
            theta2[i] = elt - bump

            g = 1 / (2 * bump) * (self._computeLogLikelihood(theta=theta1)
                                  - self._computeLogLikelihood(theta=theta2))
            gs.append(g)

        gs = np.array(gs.T)
        I = 1 / gs.shape[0] * sum(np.apply_along_axis(lambda x: np.outer(x, x), 1, gs))
        V = np.linalg.inv(I)
        return V
