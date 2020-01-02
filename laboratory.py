from filter import Filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

nobs = int(1e4)
mu, sigma, lamb, muJ, sigmaJ = 0.25, 1.0, 0.5, -10.0, 3.0

np.random.seed(42)
u = np.random.normal(0, sigma, nobs)
z = np.random.normal(muJ, sigmaJ, nobs)
xi = np.random.binomial(1, lamb, nobs)
r = mu + u + z * xi

F = Filter(obs=r)
F.calibrate()

df = pd.DataFrame({'Returns': r, 'JumpProb': F.estimateJumps()})
maskActual = xi > 0
maskPredicted = (df['JumpProb'] > 0.50).values

confMatrix = pd.DataFrame(index=['No Jump', 'Jump'], columns=['No Jump', 'Jump'])
confMatrix.loc['No Jump', 'No Jump'] = df['JumpProb'][~maskActual & ~maskPredicted].count()
confMatrix.loc['Jump', 'No Jump'] = df['JumpProb'][~maskActual & maskPredicted].count()
confMatrix.loc['No Jump', 'Jump'] = df['JumpProb'][maskActual & ~maskPredicted].count()
confMatrix.loc['Jump', 'Jump'] = df['JumpProb'][maskActual & maskPredicted].count()
confMatrix['Total'] = confMatrix.sum(axis=1)
confMatrix.loc['Total'] = confMatrix.sum(axis=0)

sns.set_style("ticks")
_, axs = plt.subplots(figsize=(12, 8), ncols=2, sharex='all', sharey='all')
dictOptions = {'alpha': 1.0, 's': 1}
axs[0].scatter(df['JumpProb'][~maskActual], df['Returns'][~maskActual], color='#003262', label='No Jump', **dictOptions)
axs[0].set_title('No Jump')
axs[1].scatter(df['JumpProb'][maskActual], df['Returns'][maskActual], color='#FDB515', label='Jump', **dictOptions)
axs[1].set_title('Jump')
for ax in axs:
    ax.grid()
    ax.axvspan(0.5, 1, color='#53626F', alpha=0.25, lw=0)
    ax.set_xlim([-0.05, 1.05])
