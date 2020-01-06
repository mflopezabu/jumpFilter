from filter import Filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

nobs = int(1e4)
dt = 1 / 252
mu, sigma, lamb, muJ, sigmaJ = 5.0 * dt, 20.0 * np.sqrt(dt), 0.05, -6.0, 3

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
_, ax = plt.subplots(figsize=(9, 4))
ax.hist(r, density=True, bins=20, color='#46535E', alpha=0.75, edgecolor='k', linewidth=0.5)
ax.grid()
ax.set_xticklabels(["{:,.0f}%".format(x) for x in ax.get_xticks()])
ax.set_yticklabels(["{:,.0f}%".format(100 * y) for y in ax.get_yticks()])
ax.set_xlabel('Return')
ax.set_ylabel('Relative Frequency')
plt.savefig('./figs/hist.png', bbox_inches='tight', dpi=1200)

sns.set_style("ticks")
_, axs = plt.subplots(ncols=2, figsize=(9, 4), sharex="all", sharey="all")
dictOptions = {'alpha': 1.0, 's': 1}
axs[0].scatter(df['JumpProb'][~maskActual], df['Returns'][~maskActual], color='k', label='No Jump', **dictOptions)
axs[1].scatter(df['JumpProb'][maskActual], df['Returns'][maskActual], color='r', label='Jump', **dictOptions)
for ax in axs:
    ax.set_xticklabels(["{:,.0f}%".format(100 * x) for x in ax.get_xticks()])
    ax.set_yticklabels(["{:,.0f}%".format(y) for y in ax.get_yticks()])
    ax.grid()
    ax.axvspan(0.5, 1, color='#46535E', alpha=0.25, lw=0)
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlabel('Inferred Probability of Jump')
    ax.legend(loc = 1)
axs[0].set_ylabel('Return')
plt.savefig('./figs/inferredJumps.png', bbox_inches='tight', dpi=1200)
