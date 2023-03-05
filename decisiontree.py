import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, roc_auc_score

titanic = sns.load_dataset('titanic')

y=titanic.survived
gender=pd.get_dummies(titanic['sex'])
clas=pd.get_dummies(titanic['class'])
X = pd.concat([clas.First, clas.Second, gender.female], axis=1)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

#plot_tree(dt_model)

rose = np.array([[1, 0, 1]])
jack = np.array([[0, 0, 0]])

print(dt_model.predict_proba(rose))
print(dt_model.predict_proba(jack))

dt_yhat = dt_model.predict(X_test)
print(dt_yhat[0:6])
print(y_test[0:6])

print(accuracy_score(dt_yhat, y_test))
print(roc_auc_score(dt_yhat, y_test))
