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

    #the derivative of volumne in respect to length is the change of volume if length changes
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
