# `Partial Derivatives and Intergrals`

## `Partial derivatives`
- Allows us to calculate derivatives of multivarite equations
- The partial derivative of $z$ with respect to $x$ is obtained by considering $y$ to be a constant
    $$
    z = x^2 - y^2
    $$
    $$
    {dz\over dx} = 2x
    $$
    $$
    {dz\over dy} = -2y
    $$

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def f(x, y):
        return x**2 - y**2

    def delz_delx(x, y):
        return 2*x

    def point_and_tangent_wrt_x(xs, x, y, f, fprime, col):
        z = f(x, y)
        plt.scatter(x, z, c=col, zorder=3)
        #The slope of the tangent line is the partial derivative of z with respect to x
        tangent_m = fprime(x, y)
        #Because the tangent line on x-z graph is z=m*x+b, so b=z-m*x
        tangent_b = z - tangent_m*x 
        tangent_line = tangent_m*xs + tangent_b
        
        plt.plot(xs, tangent_line, c=col, linestyle='dashed', linewidth=0.7, zorder=3)

    xs = np.linspace(-3, 3, 1000)
    zs_wrt_x = f(xs, 0)
    #points of interest on the curve
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
    ```
## `Find partial derivatives with Auto differentiation`
- Can use backward and grad to find the partial derivatives
    ```python
    import numpy as np
    import torch 
    import math

    def f(x, y):
        return x**2-y**2

    x = torch.tensor(3.).requires_grad_()
    y = torch.tensor(5.).requires_grad_()

    z = f(x, y)
    #backward to all functions we have, in this case is only 1
    z.backward()

    print(x.grad)
    #tensor(6.)
    print(y.grad)
    #tensor(-10.)
    ```
## `Advanced partial derivatives`
- Calculate the change of volume of a cylinder of its length changes
    ```python
    import torch
    import math

    def cylinder_vol(r, l):
        return math.pi * r**2 * l

    #given a cynlinder with radius r and length l
    r = torch.tensor(3.).requires_grad_()
    l = torch.tensor(5.).requires_grad_()

    v = cylinder_vol(r, l)
    v.backward()

    print(l.grad)

    #the derivative of volumne with respect to length is the change of volume if length changes
    print(cylinder_vol(3, 6) - cylinder_vol(3, 5))
    ```
## `The chain rule`
- Let's say
    $$
    {dy\over dx} = {dy\over du} {du\over dx}
    $$
- With a multivariate function
    $$y = f(u) $$
    $$u = g(x, z)$$
    $${dy\over dx} = {dy\over du}{du\over dx}$$
    $${dy\over dz} = {dy\over du}{du\over dz}$$
- With multiple multivariate functions
    $$y=f(u,v)$$
    $$u=g(x,z)$$
    $$v=f(x,z)$$
    $${dy\over dx}={dy\over du}+{du\over dx} + {dy\over dv}{dv\over dx}$$
- Generalized:
  $$y=f(u_1, u_2,..., u_m)$$
  $$u_j=f(x_1, x_2,..., x_n)$$
  for i=1,2,...,n
  $$
  {dy\over dx_i}={dy\over du_1}{du_1\over dx_i}+{dy\over du_2}{du_2\over dx_i}+...+{dy\over du_m}{du_m\over dx_i}
  $$

## `Calculate partial derivatives of cost C of a single point` 
- The quadratic cost in regression
  $$C=(\hat{y} - y)^2$$
- These are nested equations
  $$C=u^2$$
  $$u=\hat{y}-y$$
- Calculate partial derivatives of $C$ with respect to $u$ and $u$ with respect to $\hat{y}$
  $${dC\over du}=2u=2(\hat{y}-y)$$
  $${du\over d\hat{y}}=1-0=1$$
- Use the chain rule to calculate the partial derivative of $C$ with respect to $\hat{y}$
  $${dC\over d\hat{y}}={dC\over du}{du\over d\hat{y}}=2(\hat{y}-y)$$
- Calculate partial derivatives of $\hat{y}$ with respect to $m$ and $b$
  $$\hat{y} = mx+b$$
  $${d\hat{y}\over db}=0+1=1$$
  $${d\hat{y}\over dm}=1x+0=x$$
- From there can calculate partial derivatives of $C$ with respect to $m$ and $b$
  $${dC\over dm}={dC\over d\hat{y}}{d\hat{y}\over dm}=2x(\hat{y}-y)$$
  $${dC\over db}={dC\over d\hat{y}}{d\hat{y}\over db}=2(\hat{y}-y)$$
- Python code:
    ```python
    import torch

    xs = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.])
    ys = torch.tensor([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])

    def regression(x, m, b):
        return x*m + b

    def squared_error(y_hat, y):
        return (y_hat - y)**2

    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()

    i = 7
    x = xs[i]
    y = ys[i]
    #-1.3700

    y_hat = regression(x, m, b)
    #6.4000
    #No where near the value of 1.37 of the actual y

    C = squared_error(y_hat, y)
    #60.3729]

    #partial derivatives of C wrt m and b by autodiff
    C.backward()
    m.grad
    #108.7800
    b.grad
    #15.5400

    #by hand
    2*x*(y_hat-y)
    #108.7800
    2*(y_hat-y)
    #15.5400
    ```
## `The gradient of cost` 
- The gradient of cost denoted $\nabla{C}$, is a vector of all partial derivatives of C with repect to each of parameters
  $$\nabla{C}=\nabla_p{C}=\begin{bmatrix}
    {dC\over dp_1},{dC\over dp_2},...,{dC\over dp_n}
  \end{bmatrix}^T$$
- In this case, we only have 2 parameters $b$ and $m$:
  $$\nabla{C}=\begin{bmatrix}
    {dC\over db}\\{dC\over dm}
  \end{bmatrix}$$

## `Calculate partial derivatives of cost C of multiple points`
- The formula of C for multiple points:
  $$C={1\over n} \sum(\hat{y}-y_i)^2$$
- These are nested equations
  $$C={1\over n} \sum u^2$$
  $$u=\hat{y_i}-y_i$$
- Derivatives of $C$ with respect to $u$ and $u$ with respect to $\hat{y_i}$
  $${dC\over du}={1\over n}\sum 2u={2\over n}\sum u={2\over n}\sum (\hat{y_i}-y_i)$$
  $${du\over d\hat{y_i}}=1-0=1$$
- The partial derivatives of $\hat{y_i}$ with respect to $m$ and $b$
  $${d\hat{y_i}\over dm}=x_i$$
  $${d\hat{y_i}\over db}=1$$
- The partial derivatives of $C$ with repect to $m$ and $b$
  $${dC\over dm}={dC\over du}{du\over d\hat{y_i}}{d\hat{y_i}\over dm}={2\over n}\sum(\hat{y_i}-y_i)x_i$$
  $${dC\over db}={dC\over du}{du\over d\hat{y_i}}{d\hat{y_i}\over db}={2\over n}\sum(\hat{y_i}-y_i)$$
```python
import torch
import matplotlib.pyplot as plt

xs = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.])
ys = torch.tensor([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])

def regression(x, m, b):
    return m*x+b
def mse(y_hat, y):
    sigma = torch.sum((y_hat - y)**2)
    return sigma/len(y)

m = torch.tensor([0.9]).requires_grad_()
b = torch.tensor([0.1]).requires_grad_()

#forward pass
y_hats = regression(xs, m, b)
#[0.1000, 1.0000, 1.9000, 2.8000, 3.7000, 4.6000, 5.5000, 6.4000]
C = mse(y_hats, ys)
#19.6755

#autodiff
C.backward()
print('auto diff')
m.grad
#36.3050
b.grad
#6.2650

#by hand
print('by hand')
n = len(ys)
2/n * torch.sum((y_hats - ys)*xs)
#36.3050
2/n * torch.sum(y_hats - ys)
#6.2650

gradient = torch.tensor([b.grad.item(), m.grad.item()]).T
#tensor([ 6.2650, 36.3050])
```

- Visualize:
```python
import matplotlib.pyplot as plt

def visualize(my_x, my_y, my_m, my_b, C, include_grad=True):
    x = my_x.detach().numpy() 
    y = my_y.detach().numpy()
    m = my_m.detach().numpy()
    b = my_b.detach().numpy()
    title = 'Cost = {}'.format('%.3g' % C.item())
    if include_grad:
        xlabel = 'm = {}, m grad = {}'.format('%.3g' % m.item(), '%.3g' % my_m.grad.item())
        ylabel = 'b = {}, b grad = {}'.format('%.3g' % b.item(), '%.3g' % my_b.grad.item())
    else:
        xlabel = 'm = {}'.format('%.3g' % m.item())
        ylabel = 'b = {}'.format('%.3g' % b.item())
    
    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax.scatter(x, y)
    
    x_min, x_max = ax.get_xlim()
    y_min = regression(x_min, m, b)
    y_max = regression(x_max, m, b)

    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_min, y_max], c='C01')
    plt.show()
```
- Decend gradient:
```python
visualize(xs, ys, m, b, C)

optimizer = torch.optim.SGD([m, b], lr=0.01)
epochs = 1008
for epoch in range(epochs):
    optimizer.zero_grad() 
    C = mse(regression(xs, m, b), ys)
    C.backward()
    optimizer.step()
visualize(xs, ys, m, b, C)
```