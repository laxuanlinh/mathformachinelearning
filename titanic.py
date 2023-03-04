import numpy as np
import seaborn as sns
import scipy.special as sc
import pandas as pd
import statsmodels.api as sm

titanic = sns.load_dataset('titanic')

gender = pd.get_dummies(titanic['sex'])
clas = pd.get_dummies(titanic['class'])
y = titanic.survived
print(y)
X = pd.concat([clas.First, clas.Second, gender.female, titanic.age], axis=1)
X = sm.add_constant(X)
print(X)
model = sm.Logit(y, X, missing = 'drop')
result = model.fit()
beta = result.params
print(beta)
linear_out = beta[0] + beta[1]*1 + beta[3]*1 + beta[4]*17
print(linear_out)
print(sc.expit(linear_out))
