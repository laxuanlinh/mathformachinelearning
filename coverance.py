import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

iris = sns.load_dataset('iris')
print(iris)

x = iris.sepal_length
y = iris.petal_length

sns.set(style='whitegrid')
sns.scatterplot(x=x, y=y)
print(y)
plt.show()

xbar = np.mean(x)
ybar = np.mean(y)

product = []
for i in range(len(x)):
    product.append((x[i]-xbar)*(y[i]-ybar))

cov = sum(product)/len(x)

print(cov)
print(np.cov(x, y, ddof=0))
print(np.cov(iris.sepal_length, iris.petal_width, ddof=0))
print(st.pearsonr(x, y))
