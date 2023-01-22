import torch
import matplotlib.pyplot as plt

def regression_plot(x, y, my_m, my_b):
    m = my_m.detach().numpy()
    b = my_b.detach().numpy()
    fig, ax = plt.subplots()
    _ = ax.scatter(x, y)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = m*x_min + b, m*x_max + b    
    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_min, y_max])

def regression(x, m, b):
    return x*m+b

def mse(y_hat, y):
    sigma = torch.sum((y_hat - y)**2)
    return sigma/len(y)
    
x=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
#add noise to the equation
#y = -0.5*x + 2 + torch.normal(mean=torch.zeros(8), std=0.2)
y = torch.tensor([1.86, 1.31, 0.62, 0.33, 0.09, -0.67, -1.23, -1.37])

m = torch.tensor([0.9], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

#the line does not fit very well
regression_plot(x, y, m, b)
plt.show()

#begin forward pass
#pass x, m, b to the function to give an estimate of y
y_hat = regression(x, m, b)
print(y_hat)
print(y)
#calculate the error 
C = mse(y_hat, y)
print(C)

C.backward()
#the slope of cost C with the respect of m and b
#the derivatives of C with the respect of m and b are positive, we could reduce m and b to reduce C
print(m.grad)
print(b.grad)
#because the derivative of m is much bigger than b, m has more impact on C than b

#gradient decent
#lr is learning rate, how much we want to change m and b
optimizer = torch.optim.SGD([m, b], lr=0.01)
optimizer.step()
C = mse(regression(x, m, b), y)
#now C is much smaller than the first C
print(C)

epochs = 1000
for epoch in range(epochs):
    #reset gradient to 0, else they will accumulate
    optimizer.zero_grad()
    y_hat = regression(x, m, b)
    C = mse(y_hat, y)
    C.backward()
    optimizer.step()
    
print('Epoch {}, cost {}, m grad {}, b grad {}'.format(epoch, '%.3g'%C.item(), '%.3g'%m.grad.item(), '%.3g'%b.grad.item()))
print(m.item())
print(b.item())
regression_plot(x, y, m, b)
plt.show()
