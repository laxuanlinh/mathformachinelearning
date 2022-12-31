import numpy as np
import torch
import matplotlib.pyplot as plt

A = np.array([[-1, 4], [2, -2]])

lambdas, V = np.linalg.eig(A)
print('lambdas')
print(lambdas)
print('V')
print(V)
print('A')
print(A)
v = V[:, 0]
Av = np.dot(A, v)
print('Av')
print(Av)
lambda_v = np.dot(lambdas[0], v)
print('lambdas v is ')
print(lambda_v)
print('magnitude of v')
print(np.linalg.norm(v))
print('magnitude of Av')
print(np.linalg.norm(Av))

origin = np.array([[0,0], [0,0]])
plt.quiver(v, Av, color=['r', 'b'])
plt.xlim(-1, 2)
plt.ylim(-1, 2) 
plt.show()
