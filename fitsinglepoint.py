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
print('y is', y)

y_hat = regression(x, m, b)
#No where near the value of 1.37 of the actual y
print('y_hat is', y_hat)

C = squared_error(y_hat, y)
print('C is', C)

C.backward()

print('partial derivative of C with respect to m', m.grad)
print('partial derivative of C with respect to b', b.grad)
print('partial derivative of C with respect to m by hand', 2*x*(y_hat-y))
print('partial derivative of C with respect to b by hand', 2*(y_hat-y))

