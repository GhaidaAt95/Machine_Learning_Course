import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, mixture, datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
'''
    # Generate a gaussian dataset and attempt to cluster it
        * Check if the clustering matches the ground truth
'''

n_samples = 1000

varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[5, 1, 0.5],
                             random_state=3)

X, y = varied[0], varied[1]

plt.figure( figsize=(16,12))
plt.title("Ground Truth")
plt.scatter(X[:,0],X[:,1],c=y,edgecolors='black',lw=1.5,s=100,cmap=plt.get_cmap('viridis'))

# 1st KMeans
Kmeans = KMeans(n_clusters=3)
pred = Kmeans.fit_predict(X)

plt.figure(figsize=(16,12))
plt.title("Kmeans Clustering")
plt.scatter(X[:,0],X[:,1],c=pred,edgecolors='black',lw=1.5,s=100,cmap=plt.get_cmap('viridis'))

# 2nd Gaussian Mixture Model Clustering
gmm = GaussianMixture(n_components=3)
gmm = gmm.fit(X)

pred_gmm = gmm.predict(X)

plt.figure(figsize=(16,12))
plt.title("Gaussian Mixture Model Clustering")
plt.scatter(X[:,0],X[:,1],c=pred_gmm,edgecolors='black',lw=1.5,s=100,cmap=plt.get_cmap('viridis'))

### GMM predicted better than Kmeans

plt.show()