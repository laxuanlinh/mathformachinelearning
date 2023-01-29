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
print(y_hats)
C = mse(y_hats, ys)
print(C)

#autodiff
C.backward()
print('auto diff')
print(m.grad)
print(b.grad)

#by hand
print('by hand')
n = len(ys)
print(2/n * torch.sum((y_hats - ys)*xs))
print(2/n * torch.sum(y_hats - ys))

gradient = torch.tensor([b.grad.item(), m.grad.item()]).T
print(gradient)

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

visualize(xs, ys, m, b, C)

optimizer = torch.optim.SGD([m, b], lr=0.01)
epochs = 1008
for epoch in range(epochs):
    optimizer.zero_grad() 
    C = mse(regression(xs, m, b), ys)
    C.backward()
    optimizer.step()
visualize(xs, ys, m, b, C)
