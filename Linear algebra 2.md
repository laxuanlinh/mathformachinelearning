# `Linear Algebra II`

## `Affine transformation`

- Is the transformation that changes the geometry of vectors, adjusts their angles or distances but still preserves their parallelism
  $$v = \begin{bmatrix}
    3, 1
  \end{bmatrix}$$ 
  $$I = \begin{bmatrix}
    1&0\\
    0&-1
  \end{bmatrix}$$
  $$
    I_v = v.I = \begin{bmatrix}
    3, -1
    \end{bmatrix}
  $$
- `Affine transformation` could include:
  - Flipping a vector over axes
  - Scaling
  - Shearing
  - Rotation
- Many transformations can be applied to a vector simultaneously 

## `Eigenvectors` and `Eigenvalues`
- Given a matrix $A_{nn}$, a vector $v_{1n}$ is considered `Eigenvector` and a scalar $\lambda$ is considered `Eigenvalue` if
  $$Av = \lambda v$$
- If we transform vector $v$ using matrix $A$, the product $Av$ has the same direction as $v$
  - `Eigenvalue` is 1 if the magnitude doesn't change
  - If it doubles then the `Eigenvalue` is 2
  - -1 means backward
  - 0.5 means cut in half.
  ```python
  import numpy as np

  A = np.array([[-1, 4], [2, -2]])

  lambdas, V = np.linalg.eig(A)
  print(lambdas)
  #[ 1.37228132 -4.37228132]
  print(V)
  #[[ 0.86011126 -0.76454754][ 0.51010647  0.64456735]]
  #the column is the vector v
  print(A)
  #[[-1  4][ 2 -2]]
  v = V[:, 0]
  Av = np.dot(A, v)
  print(Av)
  #[1.18031462 0.70000958]
  lambda_v = np.dot(lambdas[0], v)
  print(lambda_v)
  #[1.18031462 0.70000958]
  print(np.linalg.norm(v))
  #1.0
  print(np.linalg.norm(Av))
  #1.3722813232690143
  ```
## `Matrix determinant`
- Determinant represents the change of space when we transform the space using the matrix A
- Example: let say we have a matrix $\begin{bmatrix}3&0\\0&2\end{bmatrix}$. When we apply this matrix for transformation, the basis vector i$\begin{bmatrix}1&0\end{bmatrix}$ increases magnitude by 3 but keeps direction, j$\begin{bmatrix}0&1\end{bmatrix}$ increases magnitude by 2 but keeps direction, the area of these 2 increases by 6. So 6 is the determinant and it's the product of Eigenvalues of i and j
- Enable us to determine wheter the matrix can be inverted
- If $det(X)=0$, it cannot be inverted
  $$X = \begin{bmatrix}
    a&b\\
    c&d
  \end{bmatrix}$$
  $$
    \vert X \vert = ad-bc
  $$
  ```python
  import numpy as np
  X = np.array([[4,2], [-5, -3]])
  np.linalg.det(X)
  #-2.0000000000000013
  ```
- For larger matrices we use recursion
  $$
  \vert X \vert = x_{1,1}det(X_{1,1}) - x{1,2}det(X_{1,2}) + ... -/+ x_{1,n}det(X_{1,n})
  $$

## `Determinants` and `Eigenvalues`
- `Determinant` of matrix X is product of all of its `Eigenvalues`
- $\vert det(X) \vert$ quantifies volumn change as the result of applying $X$
  - If $det(X) = 0$ then X collapses space completely in at least 1 dimension thus eliminates all volumn
  - If $0 < \vert det(X) \vert < 1$ then it contracts volume to some extent


## `Eigenvalue decomposition`
- `Eigendecomposition` is a factorization of a matrix into canonical form when the matrix is represented by its `Eigenvectors` and `Eigenvalues`
  $$A = V \Lambda V^{-1}$$
  - V is the matrix of all `eigenvectors` (columns)
  - $\Lambda$ is the matrix with all corresponding `eigenvalues` on its diagonal
  ```python
  import numpy as np
  A = np.array([[4, 2], [-5, -3]])
  A
  #array([[ 4.,  2.], [-5., -3.]])
  lambdas, V = np.linalg.eig(A)
  lambdA = np.diag(lambdas)
  V_inv = np.linalg.inv(V)
  np.dot(np.dot(V, lambdA), V_inv)
  #array([[ 4.,  2.], [-5., -3.]])
  ```
- Not all matrices can be decomposed, even when it can be, the decomposition may involve complex numbers
- In machine learning we usually work with real symmetric metrices which are easier to decompose
- If A is symmetric then
  $$A = Q\Lambda Q^{T}$$
  Q is the analogous to V but it's a special case because it's an orthogonal matrix
- Because $Q^{-1}$ is $Q^T$, it's more efficient to compute eigendecomposition of a symmetric matrix
