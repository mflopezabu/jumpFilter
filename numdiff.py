import numpy as np


def vecCanonical(n, i):
    e = np.zeros(n)
    e[i] = 1
    return e


def D1f(f, x, absStep=1e-2, relStep=1e-2):
    L = []
    dx = np.minimum(absStep, np.abs(x) * relStep)
    for i, dxi in enumerate(dx):
        ei = vecCanonical(len(x), i)
        fp = f(x + dxi * ei)
        fn = f(x - dxi * ei)
        L.append((fp - fn) / (2 * dxi))

    L = np.array(L)
    return L


def D2f(f, x, absStep=1e-2, relStep=1e-2):
    L = []
    dx = np.minimum(np.abs(x) * relStep, absStep)
    for i, dxi in enumerate(dx):
        ei = vecCanonical(len(x), i)
        for j, dxj in enumerate(dx):
            if j < i:   continue
            ej = vecCanonical(len(x), j)
            fpp = f(x + dxi * ei + dxj * ej)
            fpn = f(x + dxi * ei - dxj * ej)
            fnp = f(x - dxi * ei + dxj * ej)
            fnn = f(x - dxi * ei - dxj * ej)
            L.append((fpp - fpn - fnp + fnn) / (4 * dxi * dxj))

    iu = np.triu_indices(len(x))
    H = np.zeros(2 * [len(x)])
    H[iu] = L

    il = np.tril_indices(len(x), -1)
    H[il] = H.T[il]
    return H
