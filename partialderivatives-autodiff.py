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
print(y.grad)
