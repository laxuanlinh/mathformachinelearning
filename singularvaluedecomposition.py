import numpy as np
import torch

A = np.array([[-1, 2], [3, -2], [5, 7]])
print(A)

U, D, Vt = np.linalg.svd(A)
print('U')
print(U)

print('Vt')
print(Vt)
D = np.concatenate((np.diag(D), [[0, 0]]), axis=0)
print('D')
print(D)

print(np.dot(np.dot(U, D), Vt))
