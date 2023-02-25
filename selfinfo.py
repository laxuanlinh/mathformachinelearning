import numpy as np

def self_info(p):
    return -1*np.log(p)


def binary_entropy(p):
    return (p-1)*np.log(1-p)-p*np.log(p)

print(binary_entropy(0.99999))
print(binary_entropy(0.0001))
print(binary_entropy(0.5))
print(binary_entropy(0.9))
