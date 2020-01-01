from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from accelerators import computeLLF, computeLLV
from numdiff import D1, D2

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
        return res

    def _computeLogLikelihood(self, theta=None):
        if theta is None:   theta = self._theta

        # mu, sigma, lmbda, muJ, sigmaJ = theta
        # ls = (1 - lmbda) * norm.pdf(self._obs, loc=mu, scale=sigma) \
        #      + lmbda * norm.pdf(self._obs, loc=mu + muJ, scale=np.sqrt(sigma ** 2 + sigmaJ ** 2))
        # ls = np.log(np.maximum(ls, VERY_SMALL_POSITIVE))
        # return ls
        return computeLLV(self._obs, theta)

    def _computeCalibrationLogLikelihood(self, theta=None):
        if theta is None:   theta = self._theta
        # L = np.sum(self._computeLogLikelihood(theta=theta))
        # return np.min([-L, LARGE_POSITIVE])
        return -computeLLF(self._obs, theta)

    def inferJumps(self, theta=None):
        if theta is None:   theta = self._theta

        mu, sigma, lmbda, muJ, sigmaJ = theta
        ps = []
        for r in self._obs:
            p = lmbda * norm.pdf(r, loc=mu + muJ, scale=np.sqrt(sigma ** 2 + sigmaJ ** 2))
            p /= (p + (1 - lmbda) * norm.pdf(r, loc=mu, scale=sigma))
            ps.append(p)
        return ps

    def estimateVariance(self, theta=None):
        if theta is None:   theta = self._theta
        V1 = self._estimateVarianceOPG(theta)
        V2 = self._estimateVarianceHessian(theta)
        V = V2.dot(np.linalg.solve(V1, V2))
        return V

    def _estimateVarianceOPG(self, theta=None):
        if theta is None:   theta = self._theta
        G = D1(self._computeLogLikelihood, theta)
        V = np.linalg.inv(G.dot(G.T))
        return V

    def _estimateVarianceHessian(self, theta=None):
        if theta is None:   theta = self._theta
        H = D2(self._computeCalibrationLogLikelihood, theta)
        V = np.linalg.inv(H)
        return V
