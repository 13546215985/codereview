import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, 2:4]
kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(X)
color = ("red", "blue")
colors = np.array(color)[kmeans]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.show()