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


## `Singular value decomposition`
- Given a random matrix A, products of $AA^T$ and $A^TA$ are always symmetric
- We call 
  - $AA^T$ the left-singular matrix $U$
  - $A^TA$ the right-singular matrix $V$
- Both $U$ and $V$ consist of perpendicular eigenvectors of A (columns)
- Both $U$ and $V$ are called semi-infinite because their eigenvalues $\sigma$ are $\geq 0$
- If we sort the eigenvectors of both matrices in descending order, the eigenvalues $\sigma$ of 2 are identical, the left over values are always 0
- The square root of these eigenvalues are called the singular values of matrix A
- Given any matrix A
  $$
  A_{mn} = U_{mm}\Sigma_{mn} V_{nn}^T
  $$
  - $U$ and $V$ are left and right singular matrices of A. They are both orthogonal and the vectors are sorted in descending order
  - $\Sigma$ is a regtangular diagonal matrix with singular values sorted in descending order
  ```python
  import numpy as np
  A = np.array([[-1, 2], [3, -2], [5, 7]])
  print(A)
  #[[-1  2][ 3 -2][ 5  7]]
  U, D, Vt = np.linalg.svd(A)
  D = np.concatenate((np.diag(D), [[0, 0]]), axis=0)
  print(np.dot(np.dot(U, D), Vt))
  #[[-1  2][ 3 -2][ 5  7]]
  ```
  - Left-singular vector of A = eigenvectors of $AA^T$
  - Right-singular vector of A = eigenvectors of $A^TA$
  - By decomposing a matrix, we can significantly reduce the size of data while still retaining a large fraction of data

## `Compress data using SVD`
- If we consider an image as a matrix, we can decompose it into 2 orthogonal and 1 diagonal matrices.
- Since the $\Sigma$ matrix has singular values sorted descendingly, the first columns of $U$ and $V$ have the most information about the original image.
- We can compress the image by just taking a limit number of columns from both matrices because the further to the end of the matrices, the lower the corresponding singular values are and the less info which we can discard without reducing too much of image quality.
  ```python 
  import numpy as np
  from PIL import Image
  from matplotlib import pyplot as plt

  url = 'D:\\Video\\160677131_1713657132147314_1730400168674006607_n.jpg'
  img = Image.open(url)
  imggray = img.convert('LA')
  imgmat = np.array(list(imggray.getdata(band=0)), float)
  imgmat.shape = (imggray.size[1], imggray.size[0])
  imgmat = np.matrix(imgmat)
  U, sigma, Vt = np.linalg.svd(imgmat)
  num = 200 

  reconstruction = np.matrix(U[:, :num]) * np.diag(sigma[:num]) * np.matrix(Vt[:num, :])
  _ = plt.imshow(reconstruction, cmap='gray')
  plt.show()
  ```
## `The Moore-Penrose Pseudoinverse`
- Matrix cannot be inverted if it's not square, singular or dependent
- A pseudo-inverse of a matrix is a fake inverse matrix
- For matrix $A$, the pseudoinverse $A^+$ can be calculated
  $$A^+ = VD^+U^T$$
  $U, D$ and $V$ are SVD of A
  
  $D^+$ is D with reciprocal of all non-zero elements
  ```python
  import numpy as np

  A = np.array([[-1, 2], [3, -2], [5, 7]])

  U, D, Vt = np.linalg.svd(A)

  V = np.transpose(Vt)
  Ut = np.transpose(U)
  D = np.diag(D)
  Dinv = np.linalg.inv(D)
  Dplus = np.concatenate((Dinv, np.array([[0, 0]]).T), axis=1)
  Aplus = np.matrix(V) * np.matrix(Dplus) * np.matrix(Ut)
  print(Aplus)
  ```
- $A^+$ is useful in machine learning because non-square matrices are common
- $y = Xw$ can now be solved even when X is not a inversible matrix
- If $X$ is `overdetermined` (n > m), $X^+$ provides $Xy$ as close to $w$ as possible (in term of Euclidean distance)
- If $X$ is `underdetermined` (n < m), $X^+$ provides the $w=X^+y$ solution that has the smallest Euclidean norm from all the possible solutions

## `Pseudo-inverse matrix in linear regression`
- Given a set of points with $x$ and $y$ that represent the correlation between dependent and independent variables, most of the time we can't draw a line that goes through all the points.
- We can use pseudo-inverse matrix to find a line that has the smallest norm from the points.
- Assume we have slope $m$ and intercept $b$ that somehow satify $y=mx+b$ (well that's not gonna happen), all points can be expressed as matrix equation
  $$
  \begin{bmatrix}
    x_1&1\\x_2&1\\...&...\\x_n&1
  \end{bmatrix}
  \begin{bmatrix}
    m\\b
  \end{bmatrix}
  =
  \begin{bmatrix}
    y_1\\y_2\\...\\y_n
  \end{bmatrix}
  $$
  $$
  <=>\begin{bmatrix}
    m\\b
  \end{bmatrix}
  = \begin{bmatrix}
    x_1&1\\x_2&1\\...&...\\x_n&1
  \end{bmatrix}^+
  \begin{bmatrix}
    y_1\\y_2\\...\\y_n
  \end{bmatrix}
  $$

  ```python 
  import numpy as np
  import matplotlib.pyplot as plt

  x1 = [0, 1, 2, 3, 4, 5, 6, 7]
  y = [1.86, 1.31, 0.62, 0.33, 0.09, -0.67, -1.23, -1.37]

  x0 = np.ones(8)

  X = np.concatenate((np.matrix(x1).T, np.matrix(x0).T), axis=1)

  w = np.dot(np.linalg.pinv(X), y)
  ```

## `Trace operator`
- `Trace operator` gives the sum of all elements on diagonal line
  ```python
  import numpy as np

  A=np.array([[1,2,3], [3,4,3], [5,6,4]])
  np.trace(A)
  #9
  ```
## `Principle component analysis`
- An algorithm that enables identification of structure in unlabeled data
- Enables lossy compression because the first principal component contains the most variance(data structure), second PC contains the next most ...
- Given a set of data with n number of rows and m number of columns, when we plot them on a 2D or 3D graph with each axis being a feature. This should give us an idea of how some points are related from the aspect of these features.
- The problem is that we usually have more than 2 or 3 features per dataset and we can't just plot the data on n-dimension graph.
- PCA solves this issue by drawing Principal Component lines (PC) that have the best fit
- A line that has the best fit when the it is closest to all the points, this means the projections of all points have the largest distance to the origin (Pythagoras theorem)
- These PC lines don't eliminate dimensions but they have eigenvalues (sum of square distances from the projected points to the origin), the ones that have the highest eigenvalue have the highest variance thus we can calculate how much variance portion a PC line account for and only choose the highest portion lines.
- Let's say we have 10 features, for each feature, we have a PC line. We can choose the 2 or 3 highest variance lines to represent the best of how all the points spread out thus we can detect how some of them are related to each other.
  ```python
  from sklearn import datasets
  from sklearn.decomposition import PCA 
  import matplotlib.pyplot as plt

  iris = datasets.load_iris()
  #set of 150 flowers of different species

  pca = PCA(n_components=2)
  #select only 2 PC lines of the most variance

  X = pca.fit_transform(iris.data)

  plt.scatter(X[:, 0], X[:, 1], c=iris.target)
  #iris.target is an array of 150 numbers, each number represent a specie of the respective point, we can use this to color points
  plt.show()
  ```