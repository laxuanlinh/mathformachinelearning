import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

sns.set(style='whitegrid')
tips = sns.load_dataset('tips')
print(tips)
sns.boxplot(x='day', y='total_bill', hue='smoker', data=tips)
plt.show()
