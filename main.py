from filter import Filter
import numpy as np
import matplotlib.pyplot as plt

nobs = int(1e5)
obs = np.concatenate([np.random.normal(0.5, 1, nobs),
                      np.random.normal(0.5 + -2.0, np.sqrt(1.0 ** 2 + 3.0 ** 2), nobs)])
F = Filter(obs=obs)
res = F.calibrate(disp=True)
V = F.estimateVariance()

# ps = F.inferJumps()
# plt.stem(np.round(ps, 0))
