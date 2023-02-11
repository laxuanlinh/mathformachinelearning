import numpy as np
from math import factorial

def number_of_outcomes(n, k):
    return factorial(n)/(factorial(k)*factorial(n-k))

def probability(number_of_outcomes, n):
    return number_of_outcomes/2**n

outcomes = number_of_outcomes(5, 2)
P = probability(outcomes, 5)

print(P)
