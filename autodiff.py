import torch
import numpy as np

#require the gradient to be tracked on this tensor
#we don't do this by default to save memory but when we need to do operations on x, we need to track its gradient
#we track this contagiously, meaning any variables created as a result of x are also tracked
x = torch.tensor(5.0, requires_grad=True)

y = x**2

#autodiff
y.backward()

print(x.grad)

