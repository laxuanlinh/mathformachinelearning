import numpy as np

heads_count = np.random.binomial(5, 0.5, 1000)

heads_count.sort()

median = np.median(heads_count)
print(median)
