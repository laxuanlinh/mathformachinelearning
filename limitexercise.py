import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-50, 50, 1000)
y = (x**2 -2*x-8)/(x-4)

fig, avx = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')

plt.xlim(-50, 50)
plt.ylim(-50, 50)

plt.axvline(x=4, color='purple', linestyle='--')
plt.axhline(y=6, color='purple', linestyle='--')

plt.plot(x, y)

plt.show()
