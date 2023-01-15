import numpy as np
import matplotlib.pyplot as plt

def sin_fxn(x):
	return 25/x
x = np.linspace(-50, 50, 500)
y = sin_fxn(x)

fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')
plt.xlim(-50, 50)
plt.ylim(-50, 50)

plt.axvline(x=0, color='purple', linestyle='--')
plt.axhline(y=1, color='purple', linestyle='--')
plt.plot(x, y)

plt.show()
