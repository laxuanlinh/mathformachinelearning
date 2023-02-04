from scipy.integrate import quad

def g(x):
    return 2*x 

print(quad(g, 3, 4))
