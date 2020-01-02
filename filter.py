from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from likelihood import computeLLF, computeLLV, computeLLVM
from numdiff import D1, D2

VERY_SMALL_POSITIVE = 1E-300
SMALL_POSITIVE = 1E-9
LARGE_POSITIVE = 1E100


class Filter:
    def __init__(self, obs: Iterable[float]):
        self._obs = obs
        self._theta = [0,
                       np.std(obs),
                       0.5,
                       np.mean(obs),
                       2 * np.std(obs)]

    def params(self):
        V = self.estimateVariance()
        stdev = np.sqrt(np.diag(V))
        labels = ['mu', 'sigma', 'lambda', 'muJ', 'sigmaJ']
        return dict(zip(labels, zip(self._theta, stdev)))

    def calibrate(self, theta0=None, bnds=None, **kwargs):
        if theta0 is None:  theta0 = self._theta
        if bnds is None:    bnds = [(-10, 10),
                                    (SMALL_POSITIVE, 10),
                                    (0, 1),
                                    (-10, 10),
                                    (SMALL_POSITIVE, 10)]
        res = minimize(lambda x: -self._computeLogLikelihoodFunction(x),
                       theta0,
                       bounds=bnds,
                       options=kwargs)

        self._theta = res.x

    def _computeLogLikelihoodVector(self, theta=None):
        if theta is None:   theta = self._theta
        return computeLLV(self._obs, theta)

    def _computeLogLikelihoodFunction(self, theta=None):
        if theta is None:   theta = self._theta
        return computeLLF(self._obs, theta)

    def estimateJumps(self, theta=None):
        if theta is None:   theta = self._theta

        lamb = theta[2]
        l0, l1 = computeLLVM(self._obs, theta)
        ps = lamb * l1 / ((1 - lamb) * l0 + lamb * l1)
        return ps

    def estimateVariance(self, theta=None):
        if theta is None:   theta = self._theta
        V1 = self._estimateVarianceOPG(theta)
        V2 = self._estimateVarianceHessian(theta)
        V = V2.dot(np.linalg.solve(V1, V2))
        return V

    def _estimateVarianceOPG(self, theta=None):
        if theta is None:   theta = self._theta
        G = D1(self._computeLogLikelihoodVector, theta)
        V = np.linalg.inv(G.dot(G.T))
        return V

    def _estimateVarianceHessian(self, theta=None):
        if theta is None:   theta = self._theta
        H = D2(self._computeLogLikelihoodFunction, theta)
        V = np.linalg.inv(-H)
        return V
