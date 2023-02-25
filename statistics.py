import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
x = st.skewnorm.rvs(10, size=1000)
fig, ax = plt.subplots()

xbar = x.mean()
median = np.median(x)
std = x.std()
stde = st.sem(x, ddof=0)

plt.axvline(x = xbar, color='orange')
plt.axvline(x = xbar+std, color='green')
plt.axvline(x = xbar-std, color='green')
plt.hist(x, color='lightgray')
plt.show()

print(1-st.norm.cdf(2.5))
