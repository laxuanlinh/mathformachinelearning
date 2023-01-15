import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)

y = x**2 + 2*x +2

fig, ax = plt.subplots()
_ = ax.plot(x, y)
ax.set_xlim(-1.01, -0.99)
ax.set_ylim(0.99, 1.01)
plt.show()
