import numpy as np
from numba import jit

VERY_SMALL_POSITIVE = 1E-300
SMALL_POSITIVE = 1E-9
LARGE_POSITIVE = 1E100


@jit(nopython=True)
def normpdf(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


@jit(nopython=True)
def computeLLVM(rs, theta):
    mu, sigma, _, muJ, sigmaJ = theta
    l0 = normpdf(rs, mu, sigma)
    l1 = normpdf(rs, mu + muJ, np.sqrt(sigma ** 2 + sigmaJ ** 2))
    return l0, l1


@jit(nopython=True)
def computeLLV(rs, theta):
    _, _, lamb, _, _ = theta
    l0, l1 = computeLLVM(rs, theta)
    return np.log(np.maximum((1 - lamb) * l0 + lamb * l1, VERY_SMALL_POSITIVE))


@jit(nopython=True)
def computeLLF(rs, theta):
    return np.sum(computeLLV(rs, theta))
