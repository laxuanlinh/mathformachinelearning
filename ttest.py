import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

x = [48, 50, 54, 60]

xbar = np.mean(x)
sx = st.sem(x)
t = (xbar - 50)/sx

print(t)

def p_from_t(t, n):
    return 2*st.t.cdf(-abs(t), n-1) # 2nd arg of t.cdf() is degrees of freedom

print(p_from_t(t, 4))
    
