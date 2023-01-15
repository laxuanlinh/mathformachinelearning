import numpy as np
import matplotlib.pyplot as plt

def f(x):
	return x**2+2*x+2

def slope(x, y, delta):
	x_delta = x + delta
	y_delta = f(x_delta) 
	return [(y_delta - y)/(x_delta-x), x_delta, y_delta]
def b(x, y, slope):
	return y - slope*x


x = np.linspace(-50, 50, 1000)
y = f(x)

fig, ax = plt.subplots()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')

plt.scatter(2, 10)
_ = ax.plot(x, y)

result = slope(2, 10, 0.00001)

x_delta = result[1]
y_delta = result[2]
slope = result[0]
print(slope)
b = b(x_delta, y_delta, slope)

line_y = slope*x+b

plt.plot(x, line_y, color='orange')
plt.scatter(x_delta, y_delta, color='orange')

plt.show()
