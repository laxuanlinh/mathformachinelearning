import numpy as np
import matplotlib.pyplot as plt
import torch
import math #for constant pi

def f(x, y):
    return x**2 - y**2

def delz_delx(x, y):
    return 2*x

def point_and_tangent_wrt_x(xs, x, y, f, fprime, col):
    z = f(x, y)
    plt.scatter(x, z, c=col, zorder=3)
    
    tangent_m = fprime(x, y)
    tangent_b = z - tangent_m*x #Because the tangent line on x-z graph is z = m*x + b, so b = z - m*x
    tangent_line = tangent_m*xs + tangent_b
    
    plt.plot(xs, tangent_line, c=col, linestyle='dashed', linewidth=0.7, zorder=3)

xs = np.linspace(-3, 3, 1000)
zs_wrt_x = f(xs, 0)

x_samples = [-2, -1, 0, 1, 2]

colors = ['red', 'orange', 'green', 'blue', 'purple']

fig = plt.figure()
ax = plt.axes()
plt.axvline(x = 0, color='lightgray')
plt.axhline(y = 0, color='lightgray')

plt.xlabel('x')
plt.ylabel('z', rotation=0)

ax.plot(xs, zs_wrt_x)

for i, x in enumerate(x_samples):
    point_and_tangent_wrt_x(xs, x, 0, f, delz_delx, colors[i])

plt.show()
