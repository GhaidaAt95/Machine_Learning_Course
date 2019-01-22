# https://mubaris.com/posts/kmeans-clustering/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.cluster import KMeans

### Read Data
data = pd.read_csv('xclara.csv')
print("Data shape {}".format(data.shape))
print(data.head())

## Plot Data
x1 = data['V1'].values
x2 = data['V2'].values
X = np.array(list(zip(x1,x2)))
plt.scatter(x1,x2,c='black',s=7)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def manuall():
    k = 4
    C_x = np.random.randint(0, np.max(X) - 20, size=k)
    C_y = np.random.randint(0, np.max(X) - 20, size=k)
    C = np.array(list(zip(C_x,C_y)), dtype=np.float32)

    plt.scatter(C_x, C_y, marker='*', s=200, c='g')

    C_old = np.zeros(C.shape)

    clusters = np.zeros(len(X))

    error = dist(C, C_old, None)

    while error != 0:
        for i in range(len(X)):
            distances = dist(X[i],C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        C_old = deepcopy(C)

        for i in range(k):
            points = [ X[j] for j in range(len(X)) if clusters[j] == i ]
            C[i] = np.mean(points, axis=0)
        error = dist(C,C_old,None)

    colors = ['r','g','b','y','c','m']
    fig, ax = plt.subplots()

    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j]==i])
        ax.scatter(points[:,0], points[:,1], s=7, c=colors[i])
    ax.scatter(C[:,0],C[:,1], marker='*',s=200,c='#050505')

    print(C)

def sklearn_kmeans():
    ## Set the number of clusters
    Kmeans = KMeans(n_clusters=3)

    ## Fitt
    Kmeans = Kmeans.fit(X) 

    labels = Kmeans.predict(X)

    centroids = Kmeans.cluster_centers_

    print(centroids)

manuall()
sklearn_kmeans()

plt.show()

