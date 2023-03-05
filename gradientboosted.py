import seaborn as sns
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

iris = sns.load_dataset('iris')
y = iris.species
X = iris[['sepal_width', 'petal_width']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)
# XGBoost Dmatrix() method requires numeric inputs, not string

y_train_int = y_train.replace(['setosa', 'versicolor', 'virginica'], [0, 1, 2])
y_test_int = y_test.replace(['setosa', 'versicolor', 'virginica'], [0, 1, 2])

D_train = xgb.DMatrix(X_train, label=y_train_int)
D_test = xgb.DMatrix(X_test, label=y_test_int)

param = {
    'eta': 0.1,
    'max_depth': 2,
    'objective': 'multi:softprob',
    'num_class': 3
}
steps = 10
xg_model = xgb.train(param, D_train, steps)
xg_yhats = xg_model.predict(D_test)
print(xg_yhats[0])
xg_yhat = np.asarray([np.argmax(line) for line in xg_yhats])


print(accuracy_score(xg_yhat, y_test_int))


