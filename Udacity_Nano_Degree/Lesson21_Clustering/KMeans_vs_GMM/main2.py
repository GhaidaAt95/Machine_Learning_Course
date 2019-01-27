import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
'''
    # Test a dataset that has more than 2 features
    # Iris Dataset --> We can assume that it is 
        distributed according to Gaussian distribution
'''
iris = sns.load_dataset("iris")

print("-"*70)
print("\t\t Iris Head")
print(iris.head())
print("-"*70)

'''
    #There are few ways to visualize a dataset with 4 dimensions:
        1) PairGrid
            https://seaborn.pydata.org/generated/seaborn.PairGrid.html
        2) t-SNE
            https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        3) project into a lower number dimensions using PCA
            https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py
    
    # We will use PairGrid since it does not distort the dataset 
        It merely plots every pair of features against each other in a subplot
'''

g = sns.PairGrid(iris, hue="species", palette=sns.color_palette("cubehelix",3),vars=['sepal_length','sepal_width','petal_length','petal_width'])
g.map(plt.scatter)
g.add_legend(title="Ground Truth")

# Kmeans 

Kmeans_iris = KMeans(n_clusters=3)
pred_kmeans_iris = Kmeans_iris.fit_predict(iris[['sepal_length','sepal_width','petal_length','petal_width']])

iris['kmeans_pred'] = pred_kmeans_iris
g = sns.PairGrid(iris, hue="kmeans_pred", palette=sns.color_palette("cubehelix", 3), vars=['sepal_length','sepal_width','petal_length','petal_width'])
g.map(plt.scatter)
g.add_legend(title="kmeans")

# To evaluate we'll use the external cluster validation method ARI

iris_kmeans_score = adjusted_rand_score(iris['species'],iris['kmeans_pred'])
print('-'*70)
print("Kmeans Adjusted Rand Score = {}".format(iris_kmeans_score))
print('-'*70)

# Gaussian Mixture Model Clustering

gmm_iris = GaussianMixture(n_components=3).fit(iris[['sepal_length','sepal_width','petal_length','petal_width']])
pred_gmm_iris = gmm_iris.predict(iris[['sepal_length','sepal_width','petal_length','petal_width']])

iris['gmm_pred'] = pred_gmm_iris

g = sns.PairGrid(iris, hue="gmm_pred", palette=sns.color_palette("cubehelix", 3), vars=['sepal_length','sepal_width','petal_length','petal_width'])
g.map(plt.scatter)
g.add_legend(title="GMM")

iris_gmm_score = adjusted_rand_score(iris['species'],iris['gmm_pred'])

print('-'*70)
print("GMM Adjusted Rand Score = {}".format(iris_gmm_score))
print('-'*70)
plt.show()