import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

n_experiments = 1000
heads_count = np.random.binomial(5, 0.5, n_experiments)

heads, event_count = np.unique(heads_count, return_counts=True)

print(heads)
print(event_count)

