import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')

x = iris.sepal_length
y = iris.petal_length
n = iris.sepal_width.size

sns.scatterplot(x=x, y=y)
xbar, ybar = x.mean(), y.mean()
product = []
for i in range(n):
    product.append((x[i]-xbar)*(y[i]-ybar))

cov = sum(product)/n
beta1 = cov/np.var(x)
print(beta1)
beta0 = ybar - beta1*xbar
print(beta0)

xline = np.linspace(4, 8, 1000)
yline = beta0+beta1*xline

plt.plot(xline, yline, color='orange')
plt.show()
