# `Calculus`

- The study of change of funtions when variables change
- There are 2 branches of calculus:
  - `Differential calculus`: the focus of Calculus 1, concerning rate of change and slopes of curves
  - `Integral calculus`: concerning the accumulation and area under the curves

## `Method of exhaustion`
- Method of exhaution is to calculate the area of curved shape by inscribing inside it a sequence of polygons whose area cover or is close to the area of the shape
- Integral calculus is entirely based on this method
- Calculus overall relies on the idea of approaching infinity
- As we approach the infinity-sided polygon, so too do the differential accuracy and integral accuracy improve
- If we draw a diagram for a curve and zoom in on it, the more we zoom in, the closer for the curve to being straignt
  ```python
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-10, 10, 1000)

    y = x**2 + 2*x +2

    fig, ax = plt.subplots()
    _ = ax.plot(x, y)
    ax.set_xlim(-1.01, -0.99)
    ax.set_ylim(0.99, 1.01)
    plt.show()
  ```
## `Application of Calculus in Machine Learning`
- Differentials: optimize by finding the minima and maxima of curves
- In engineering to find max strength, in finance to find min cost
- In machine learning and deep learning
  - `Gradient decent` to minimize cost
  - High-order derivatives used in other optimizers

## `Calculating limits`
- Limits are trivially easy to calculate for continious functions

## `The Delta method`
- Given a curve, if we have 2 points on that curve, we can calculate the difference between 2 points to calculate the slope
  $$
    m = {\Delta y\over\Delta x} = {y_2-y_1\over x_2-x_1}
  $$
- If the delta is small enough and 2 points are very close, the slope of 2 points can be considered the slope of the curve at 1 point
  ```python
  import numpy as np

  def f(x):
    return x**2+2*x+2

  def calculate(x, y, delta):
    x_delta = x + delta
    y_delta = f(x_delta) 
    return [(y_delta - y)/(x_delta-x), x_delta, y_delta]

  x = np.linspace(-50, 50, 1000)
  y = f(x)
  delta= 0.00001
  result = calculate(2, 10, 0.delta)
  x_delta = result[1]
  y_delta = result[2]
  slope = result[0]

  print(slope)
  #6.00000999991201 => slope at (2, 10) is 6
  ```
## `Derivatives`
- Derivative of a function is the instantaneous rate of change at one point.
- For example, we have function $f(x)$, if x changes, output of $f(x)$ also changes, so the derivative of $f(x)$ is how fast it changes at $x$.
- Obviously we cannot calculate the rate of change at $x$ because it's only 1 point. But if we select $x1$ and $x2$ being extremely close, the rate of change between $f(x_1)$ and $f(x_2)$ can be considered the derivative at $x1$
  $$
  {dy\over dx} = \lim_{\Delta x\to0}{y_2-y_1\over \Delta x}
  $$
  $$
  \leftrightarrow {dy\over dx} = \lim_{\Delta x\to0} {f(x+\Delta x) - f(x)\over \Delta x}
  $$
- Given a curve $y=x^2$, as the $x$ changes, the area under the curve also changes. The change of $x$ is $dx$, the change of area is $dA$.
- The rate $dA\over dx$ is called the derivative of area $A$ or how fast $A$ changes when $x$ changes

## `Derivative notation`
- First derivative operator:
  $d\over dx$
- Second derivative operator:
  $d^2\over dx$

## `Derivative of a Constant`
- Assuming $c$ is a constant, ${d\over dx}c=0$
## `The Power rule`
- The formula:
  $$
    {d\over dx}x^n = nx^{n-1}
  $$
- For example:
  $$
  {d\over dx}x^4 = 4x^3
  $$
## `The Constant multiple rule`
- The formula:
  $$
  {d\over dx}(cy) = c{d\over dx}(y) = c {dy\over dx}
  $$
- For example:
  $$y = x^4$$
  $${dy\over dx}=4x^3$$
  $${d\over dx}2y=2{dy\over dx}=8x^3$$
## `The Sum rule`
- The formula:
  $$
  {d(y+w)\over dx} = {dy\over dx}+{dw\over dx}
  $$
## `The Product rule`
- The formula:
  $$
  {d(wz)\over dx} = w{dz\over dx} + z{dw\over dx}
  $$
## `The Quotient (Fraction) rule`
- The formula:
  $$
  {d\over dx}({w\over y}) = {z{dw\over dx} - w{dz\over dx}\over z^2}
  $$
## `The Chain rule`
- Applications in:
  - Gradient decent 
  - Backpropagation
- The formula:
  $$
  {dy\over dx} = {dy\over du}{du\over dx}
  $$
- For example:
  - $y=(5x+25)^3$
  - Let $u = 5x+25$
  - $y = u^3$
  - $y$ is a function of $u$ and $u$ is a function of $x$
  $$y = (2x^2+8)^2$$
  $$u = 2x^2+8$$
  $$y=u^2$$
  $${dy\over dx} = {dy\over du}{du\over dx} = 2u 4x = 16x^3+64x$$
## `The Power rule on a function chain`
- The formula:
  $$
  {d\over dx}u^n = nu^{n-1}{du\over dx}
  $$
- For example:
  $$y=(3x+1)^2$$
  $${dy\over dx} = 2(3x+1)3$$
  $${dy\over dx} = 18x+6$$

## `Gradient`
- If we have a function that has multiple variables $x, y, z...$, we can calculate partial derivatives of the function with the respect to one variable with all other variables as constants.
- A vector that contains all partial derivatives of a function is called gradient.
- The gradient points in the direction of maximum change
- Its magnitude is equal to the maxinum rate of change
- Basically gradient is the derivative of a multivariable function

## `Automatic Differentiation`
- There are many way to do differentiation
  - Manual
  - Numeric
  - Symbolic
  - Automatic
  ```python
  import torch
  import numpy as np

  #require the gradient to be tracked on this tensor
  #we don't do this by default to save memory but when we need to do operations on x, we need to track its gradient
  #we track this contagiously, meaning any variables created as a result of x are also tracked
  x = torch.tensor(5.0, requires_grad=True)

  y = x**2

  #autodiff in backward mode
  y.backward()

  print(x.grad)
  #10
  ```