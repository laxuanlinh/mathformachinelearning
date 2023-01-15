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
  $dy\over dx$
- Second derivative operator:
  $d^2y\over dx$