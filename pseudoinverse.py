import numpy as np

A = np.array([[-1, 2], [3, -2], [5, 7]])

U, D, Vt = np.linalg.svd(A)

V = np.transpose(Vt)
Ut = np.transpose(U)
D = np.diag(D)
Dinv = np.linalg.inv(D)
Dplus = np.concatenate((Dinv, np.array([[0, 0]]).T), axis=1)
Aplus = np.matrix(V) * np.matrix(Dplus) * np.matrix(Ut)
print('Aplus')
print(Aplus)
