from math import factorial 

def coinflip_prob(n, k):
    n_choose_k = factorial(n)/(factorial(k)*factorial(n-k))
    return n_choose_k/2**n
    
P = [coinflip_prob(2, x) for x in range(3)]    

print(P)
#E = sum(P[x]*x for x in range(2))
#print(E)
