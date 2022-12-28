# `Linear Algebra`

Jill generator:
- Mark 1: b1 = 1*a
- Mark 2: b2 = 4*a
- 1.a+30.1 = 4.a
  <=> a+30=4a
  <=> a = 10

## `Tensor`
- Is a generalization of vectors and matrices to any number of dimensions
- For example
  - 0: `Scalar`, no direction, just magnitude
  - 1: `Vector`
  - 2: `Matrix`
  - `3-tensor`, a 3D table or cube
  - `n-tensor`,. higher dimensional

### `Scalar tensors`

- `Pytorch`: tensors are designed to behave like `NumPy` arrays, easier to be used on GPU, to do differentiation, easier to code overall.
- `TensorFlow`: bigger community, older, in a lot of prod env
- How to create scalar in `Pytorch`:
    ```python
    import torch

    x_pt = torch.tensor(25)
    y_pt = torch.tensor(5)
    x_pt+y_pt
    #tensor(30)
    ```
- How to create scalar in `TensorFlow`
    ```python
    import tensorflow as tf

    x_pt = tf.Variable(25, dtype=tf.int16)
    y_pt = tf.Variable(5, dtype=tf.float16)
    #cast an int to float tensor
    x_pt = tf.cast(x_pt, tf.float16)
    x_pt+y_pt
    #<tf.Tensor: shape=(), dtype=float16, numpy=30.0>
    ```
----
### `Vector tensor`
- Row vector (1, 3)
$$\begin{bmatrix}x_{1}\\
x_{2}\\
x_{3}
\end{bmatrix}$$

- Column vector (3, 1)
$$\begin{bmatrix} 
x_{1} & x_{1} & x_{1} 
\end{bmatrix}$$
- How to create vector in torch
    ```python
    import torch
    import numpy as np

    #create numpy array
    x = np.array([[25, 5, 2]])

    #create torch vector
    x_pt = torch.tensor([[25, 5, 2]])
    #transpose of vector
    y_pt = x_pt.mT

    x_pt.shape
    #torch.Size([1, 3])

    y_pt.shape
    #torch.Size([3, 1])
    ```
- To display the magnitude of a vector, we use Norm functions
- The most popular Norm function is $L^2$ Norm
- The $L^2$ is described as sum of all squared elements, then square roots
    $$ \lVert x \rVert_2 = \sqrt{\sum_{i} x_i^2}$$
- $L^2$ Norm measures the normal distance between the origin to the point
- $\lVert x \rVert_2$ can be denoted as $\lVert x \rVert$
- In python, $L^2$ Norm is 
    ```python
    np.linalg.norm(x)
    ```
- If magnitude of a vector is 1, then it's called a `unit vector`
- $L^1$ Norm ($\lVert x \rVert_1$) is sum of abs values of all elements. It's used when differences between zero and non-zero are keys
- Squared $L^2$ Norm is similar to $L^2$ Norm but without the square root, this is computationally cheaper to use because it's simply $x^Tx$.
  - Derivatives of element **x** requires that element alone instead of the whole vector like $L^2$
  - The down side is it grows slowly near the origin so hard to distinguish between zero and non-zero. If that's the case, just use $L^1$
- `Max Norm` returns the max abs value
- `Generalized Norm`:
    $$\Vert x \Vert_p = (\sum_{i} \vert x_i \vert^p)^\frac{1}{p}$$
- `Basis vectors` are i(1, 0) and j(0, 1), can be scaled to any vector
- `Orthogonal` vectors are vectors that $x^Ty=0$, are 90 degree angle to each other.
- `Orthonormal` are `orthogonal` and have unit vectors. Ex: `basis vectors`

### `Matrix`
- A matrix is a 2-dimensional array of numbers.
    ```python
    import torch 
    import tensorflow as np

    x_pt = torch.tensor([[25, 5], [3, 6]])
    x_pt.shape
    #torch.Size([2, 2])

    y_tf = tf.Variable([[25, 5], [3, 6]])
    y_tf.shape
    #TensorShape([3, 2])
    ```
- Higher-rank tensors are common for images where each dimension corresponds to:
  - Number of images
  - Image height
  - Image width
  - number of color channel

## `Tensor Operation`

### `Tensor transposition`
- Transpose of a vector is to convert columns to rows
- `Tensor transposition` is a special case when we flip the axes over the main diagonal so that every element that is not on the main diagonal:
    $$(X^T)_{i,j}=X_{j, i}$$

### `Hadamard product`
- `Hadamard product` is when 2 matrices have the same size, the operations are often applied element-wise by default.
- This results a vector that has the same size as 2 operands
  ```python
  x_pt = np.array([[3, 6], [7, 2]])
  y_pt = np.array([[2, 2], [4, 6]])
  x_pt*y_pt
  #array([[ 6, 12],[28, 12]])
  x_pt = torch.tensor([[3, 6], [7, 2]])
  y_pt = torch.tensor([[2, 2], [4, 6]])
  torch.mul(x_pt, y_pt)
  #tensor([[ 6, 12],[28, 12]])
  ```

### `The Dot Product`
- It's basically `Hadamard product` but we sum all the elements at the end
  - $x\cdot x$
  - $x^Tx$
  - $(x, y)$
- Very common operator in deep learning
  ```python
  #only works with 1D tensor 
  x_pt = torch.tensor([25, 2, 5])
  y_pt = torch.tensor([2, 5, 10])
  torch.dot(x_pt, y_pt)
  #tensor(110)
  25*2+2*5+5*10
  #110
  ```

  ### `Fribenius Norm`
    $$\Vert X \Vert_F = \sqrt{\sum_{i,j}X_{i,j}^2}$$
  - It's similar to $L^2$ Norm of vector but for matrix, to measure the size in term of `Euclidean` distance.
  - It's also the sum of magnitude all vectors
  ```python
  x = np.array([[25, 2], [3, 9]])
  np.linalg.norm(x)
  #26.814175355583846
  x = torch.tensor([[25, 2], [3, 9]], dtype=torch.double)
  torch.norm(x)
  #tensor(26.8142, dtype=torch.float64)
  ```

  ### `Matrix multiplication`
    $$C_{i,k} = \sum_{j}a_{i,j}b_{j,k}$$
  - Can on only multiply 2 matrices if the number of columns of the first matrix is equal to the number of columns of the second matrix
  - The result of the mul is a matrix that has the same number of rows of the first matrix and number of columns of the second matrix
    ```python
    import torch
    import numpy as np
    import tensorflow as tf

    a = np.array([[3, 4], [5, 6], [7, 8]])
    b = np.array([[1, 3], [2, 4]])
    np.matmul(a, b)
    #array([[11, 25], [17, 39], [23, 53]])
    a_pt = torch.from_numpy(a)
    b_pt = torch.from_numpy(b)
    torch.matmul(a_pt, b_pt)
    #tensor([[11, 25], [17, 39],[23, 53]], dtype=torch.int32)
    a_tf = tf.convert_to_tensor(a)
    b_tf = tf.convert_to_tensor(b)
    tf.matmul(a_tf, b_tf)
    #array([[11, 25], [17, 39], [23, 53]])>
    ```
