from sklearn import datasets
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

iris = datasets.load_iris()

pca = PCA(n_components=2)
X = pca.fit_transform(iris.data)

plt.scatter(X[:, 0], X[:, 1], c=iris.target)
plt.show()