import numpy as np
import matplotlib.pyplot as plt

x1 = [0, 1, 2, 3, 4, 5, 6, 7]
y = [1.86, 1.31, 0.62, 0.33, 0.09, -0.67, -1.23, -1.37]
title = 'Clinical Trial'
xlabel = 'Drug doseage'
ylabel = 'Forgetfulness'


x0 = np.ones(8)

X = np.concatenate((np.matrix(x1).T, np.matrix(x0).T), axis=1)

w = np.dot(np.linalg.pinv(X), y)


b = np.asarray(w).reshape(-1)[0]
m = np.asarray(w).reshape(-1)[1]

fig, ax = plt.subplots()
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
_ = ax.scatter(x1, y)
x_min, x_max = ax.get_xlim()
y_at_xmin = m*x_min + b
y_at_xmax = m*x_max + b

ax.set_xlim([x_min, x_max])
_ = ax.plot([x_min, x_max], [y_at_xmin, y_at_xmax], c='C01')
plt.show()
