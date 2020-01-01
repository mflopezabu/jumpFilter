import numpy as np
from numba import jit

VERY_SMALL_POSITIVE = 1E-300
SMALL_POSITIVE = 1E-9
LARGE_POSITIVE = 1E100


@jit(nopython=True)
def normpdf(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


@jit(nopython=True)
def computeLLV(rs, theta):
    mu, sigma, lamb, muJ, sigmaJ = theta
    ls = (1 - lamb) * normpdf(rs, mu, sigma) + lamb * normpdf(rs, mu + muJ, np.sqrt(sigma ** 2 + sigmaJ ** 2))
    return np.log(np.maximum(ls, VERY_SMALL_POSITIVE))


@jit(nopython=True)
def computeLLF(rs, theta):
    return np.sum(computeLLV(rs, theta))
