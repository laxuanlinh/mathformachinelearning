import torch
import matplotlib.pyplot as plt

def regression(x, m, b):
    return m*x + b

def regression_plot(x, y, my_m, my_b, title):
    m = my_m.detach().numpy() 
    b = my_b.detach().numpy() 
    fig, ax = plt.subplots()
    _ = ax.scatter(x, y)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = m*x_min + b, m*x_max + b
    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_min, y_max])    
    plt.title(title)
    plt.show()

def mse(y_hat, y):
    sigma = torch.sum((y_hat - y)**2)
    return sigma/len(y)

x = torch.linspace(1, 11, steps=11) 
y = 2*x + 2 + torch.normal(mean = torch.zeros(11), std=0.2) 

m = torch.tensor([0.3], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

regression_plot(x, y, m, b, 'First graph')

optimizer = torch.optim.SGD([m, b], lr=0.01)

epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_hat = regression(x, m, b)
    C = mse(y_hat, y)
    C.backward()
    optimizer.step()
print(C)
print(m)
print(b)
regression_plot(x, y, m,b, 'Optmized graph')
