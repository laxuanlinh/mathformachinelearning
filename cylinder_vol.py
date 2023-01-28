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
