import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

x = st.skewnorm.rvs(10, size=1000)
xbar = np.mean(x)
print('xbar', xbar)

squared_diff = [(x_i - xbar)**2 for x_i in x]

variance = np.sum(squared_diff) / len(x)
print(variance)
print(np.var(x))
print(variance**(1/2))
print(np.std(x))
