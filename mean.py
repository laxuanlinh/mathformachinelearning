from math import factorial
import numpy as np

heads_count = np.random.binomial(5, 0.5, 1000)

mean = sum(heads_count)/len(heads_count)
print(mean)
