import numpy as np
import matplotlib.pyplot as plt
import torch

def f(x, y):
    return x**2 - y**2

def f_prime(x, y):
    return -2*y

def draw_tangent_line(ys, x, y, f, f_prime, col):
    z = f(x, y)
    #display the point of interest
    plt.scatter(y, z, c=col, zorder=3) 

    tangent_m = f_prime(x, y)
    tangent_b = z - tangent_m*y
    tangent_line = tangent_m * ys + tangent_b

    plt.plot(ys, tangent_line, c=col, linestyle='dashed', linewidth=0.7, zorder=3)
    

ys = np.linspace(-3, 3, 1000)
y_samples = [-2, -1, 0, 1, 2]
colors = ['red', 'orange', 'green', 'blue', 'purple']
z_wrt_y = f(0, ys)

fig = plt.figure()
ax = plt.axes()
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')

plt.xlabel('y')
plt.ylabel('z', rotation=0)
ax.plot(ys, z_wrt_y)

for i, y in enumerate(y_samples):
    draw_tangent_line(ys, 0, y, f, f_prime, colors[i])

plt.show()

