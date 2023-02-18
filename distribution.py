import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st


x = st.skewnorm.rvs(10, size=100000)

def sample_mean(dist, sample_size, n_samples):
    sample_means = []
    for i in range(n_samples):
        sample = np.random.choice(dist, sample_size, replace=False)
        sample_means.append(sample.mean())
    return sample_means

sns.displot(sample_mean(x, 1000, 1000), color='green')
plt.xlim(-1.5, 1.5)

plt.show()
