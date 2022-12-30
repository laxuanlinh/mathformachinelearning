import numpy as np
import matplotlib.pyplot as plt
import torch

v = np.array([3, 1])
def plot_vectors(vectors, colors):
    plt.figure()
    plt.axvline(x=0, color='lightgrey')
    plt.axhline(y=0, color='lightgrey')

    for i in range(len(vectors)):
            x = np.concatenate([[0,0], vectors[i]])
            plt.quiver([x[0]], [x[1]], [x[2]], [x[3]], angles='xy', scale_units='xy', scale=1, color=colors[i])


I = np.array([[-1, 0], [0, 1]])
Iv = np.dot(I, v)
print(Iv)
plot_vectors([v, Iv], ['lightblue', 'blue'])
plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.show()
