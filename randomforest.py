import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

iris = sns.load_dataset('iris')
y = iris.species
X = iris[['sepal_width', 'petal_width']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)
iris_dt = DecisionTreeClassifier().fit(X_train, y_train)
iris_dt_yhat = iris_dt.predict(X_test)
print(accuracy_score(iris_dt_yhat, y_test))

iris_rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
iris_rf_yhat = iris_rf.predict(X_test)
print(accuracy_score(iris_rf_yhat, y_test))
