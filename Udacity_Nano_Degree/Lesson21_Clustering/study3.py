#https://mubaris.com/posts/kmeans-clustering/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs

plt.rcParams['figure.figsize']=(16, 9)

X, y = make_blobs(n_samples=800, n_features=3, centers=4)

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:,1], X[:,2])

Kmeans = KMeans(n_clusters=4)

Kmeans = Kmeans.fit(X)

labels = Kmeans.predict(X)

C = Kmeans.cluster_centers_

fig2 = plt.figure()

ax2 = Axes3D(fig2)

ax2.scatter(X[:,0],X[:,2],X[:,2],c=y)
ax2.scatter(C[:,0],C[:,1],C[:,2],marker='*',c=[1,2,3,4],s=1000)


plt.show()

