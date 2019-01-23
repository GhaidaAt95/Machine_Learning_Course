import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import preprocessing

'''
    # Clustering on the iris dataset which contains:
        - 4 dimensions/attributes
        - 150 samples
        - Each sample is labeld as one of the 3 Iris flowers
'''

iris = load_iris()
# Features ; ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print('*'*70)
print('Iris dataSet\n',iris.data[:10])
print(type(iris.data))
print('*'*70)
print('Iris dataSet\n',iris.target[:10])
print(type(iris.target))
print('*'*70)

ward_pred = np.empty(iris.data.shape[0])
complete_pred = np.empty(iris.data.shape[0])
avg_pred = np.empty(iris.data.shape[0])

def calc_pred(X, n):
    ## Ward Clustering 
    ward = AgglomerativeClustering(n_clusters=3)
    ward_pred[:,] = ward.fit_predict(X)

    ## Complete Clustering
    complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
    complete_pred[:,] = complete.fit_predict(X)

    ## Average Clustering
    avg = AgglomerativeClustering(n_clusters=3,linkage='average')
    avg_pred[:,] = avg.fit_predict(X)

calc_pred(iris.data, 3)

## Evaluating
'''
    # Metric: adjusted_rand_score
    - Computes the similarity between two clusters
    - It considers all pairs of samples AND 
        counts pairs that are assigned in the same or different clusters in the predicted and true clusters
    - 0 Lowest, 1: Highest

'''

def calc_scores():
    ward_ar_score = adjusted_rand_score(iris.target, ward_pred)

    complete_ar_score = adjusted_rand_score(iris.target, complete_pred)

    avg_ar_score = adjusted_rand_score(iris.target, avg_pred)

    print("Scores:\nWard: {}\nComplete: {}\nAverage: {}".format(ward_ar_score,complete_ar_score,avg_ar_score))

calc_scores()
## Normalizing the Data
'''
    Examining the data we notice that the forth column [petal width (cm)] is smaller
        than the rest of the columns. ---> Variance counts for less in the clustering process
        [Clustering is based on distance]

    Normalize the dataset such that each feature/dimension lies between 0 and 1
        - All have equal weight in the clustering process

        - By: 
            1. Subtracting the minimum from each column 
            2. Dividing the difference by the range
    
    -Use sklearn Normalize
'''

normalized_X = preprocessing.normalize(iris.data)

print('*'*70)
print("Normalized Data Set:\n",normalized_X[:10])
print('*'*70)

calc_pred(normalized_X, 3)
calc_scores()

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

linkage_type = 'ward'

linkage_matrix = linkage(normalized_X, linkage_type)

plt.figure(figsize=(22,18))
dendrogram(linkage_matrix)

import seaborn as sns

sns.clustermap(normalized_X, figsize=(18,50), method=linkage_type, cmap='viridis')

plt.show()


