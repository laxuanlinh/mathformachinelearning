# `Data Structure and Algorithms`

## `Tree`
- Extension of linked-list

## `Decision tree`
- Binary tree, nax 2 child nodes per parent
- Classification tree: predict categorical outcome
- Regression tree: predict continuous outcome
- CART: classification and regression tree
  ```python
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import accuracy_score, roc_auc_score
    titanic = sns.load_dataset('titanic')
    y=titanic.survived
    gender=pd.get_dummies(titanic['sex'])
    clas=pd.get_dummies(titanic['class'])
    X = pd.concat([clas.First, clas.Second, gender.female], axis=1)
    #   First  Second  female
    # 0    0     0     0
    # 1    1     0     1
    # 2    0     0     1
    # 3    1     0     1
    # 4    0     0     0 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # split dataset into train and test set
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    rose = np.array([[1, 0, 1]])
    jack = np.array([[0, 0, 0]])
    dt_model.predict_proba(rose)
    # [[0.05084746 0.94915254]]
    dt_model.predict_proba(jack)
    # [[0.85232068 0.14767932]]
    dt_yhat = dt_model.predict(X_test)
    # use the model to predict test data set
    # test the survived column of the first 6 rows
    dt_yhat[0:6]
    # [0 0 0 1 0 1]
    y_test[0:6]
    # 709    1
    # 439    0
    # 840    0
    # 720    1
    # 39     1
    # 290    1
    # data does not match as first passenger actually survived even though the prediction says otherwise
    accuracy_score(dt_yhat, y_test)
    # 0.7728813559322034
    roc_auc_score(dt_yhat, y_test)
    # 0.826696770662288
  ```

## `Random forests`
- Tends to outperform any decision tree single model
- Randomly creates decision trees, each node of the decision trees works on a random subset of features to calculate the output, finally the results are combined
- Bootstraping is the process to randomly select items from the training dataset
- After running the test 100 times, we can average the results to:
  - Reduce variance
  - Cancel out errors
  ```python
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    iris_rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    iris_rf_yhat = iris_rf.predict(X_test)
  ```

## `XGBoost: Gradient-Boosted Trees`
- 1st tree is a regular decision tree
- 2nd tree trained on 1st's errors
- 3re tree trained on 2nd's errors
- Repeat until no more improvements
  ```python
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
    xg_yhat = np.asarray([np.argmax(line) for line in xg_yhats])
    accuracy_score(xg_yhat, y_test_int)
    # 0.98
  ```