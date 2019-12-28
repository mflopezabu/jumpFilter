from filter import Filter
import numpy as np
import matplotlib.pyplot as plt

obs = np.concatenate([np.random.normal(0.5, 1, 1000), np.random.normal(-2.0, 3.0, 1000)])
F = Filter(obs=obs)
F.calibrate(disp=True)

# ps = F.inferJumps()
# plt.stem(np.round(ps, 0))
