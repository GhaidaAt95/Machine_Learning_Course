import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dbscan_lab_helper as helper
from sklearn import cluster

## Import Dataset 
dataset_1 = pd.read_csv('blobs.csv')[:80].values

#Visualize Dataset
helper.plot_dataset(dataset_1)

dbscan = cluster.DBSCAN()
dbscan.fit(dataset_1)

clustering_labels_1 = dbscan.labels_

helper.plot_clustered_dataset(dataset_1, clustering_labels_1)

## It was not able to cluster the 3 clusters with default eps and min_samples

helper.plot_clustered_dataset(dataset_1, clustering_labels_1, neighborhood=True)

## eps = 0.5 is very small for this dataset 

epsilon = 1.6

dbscan = cluster.DBSCAN(eps=epsilon)
dbscan.fit(dataset_1)

clustering_labels_2 = dbscan.labels_
helper.plot_clustered_dataset(dataset_1, clustering_labels_2, neighborhood=True)

'''
    after testing multiple values we can see that from 1.6 to 5
    it is able to cluster 3 blobs.
'''

###### Second DataSET 

dataset_2 = pd.read_csv('varied.csv')[:300].values
helper.plot_dataset(dataset_2, xlim=(-14, 5), ylim=(-12, 7))

dbscan = cluster.DBSCAN()

clustering_labels_1 = dbscan.fit_predict(dataset_2)

helper.plot_clustered_dataset(dataset_2, 
                              clustering_labels_3, 
                              xlim=(-14, 5), 
                              ylim=(-12, 7), 
                              neighborhood=True, 
                              epsilon=0.5)

'''
    * The efault clustering seems arbitrary 
    * Scenarios of clustering that we might want:
        1) Break the dataset into 3 clusters [left, right, middle]
        2) Break the dataset into 2 cluster [left, right] and the middle as noise
'''

## Test different values 
eps= 1
min_samples= 4


# Cluster with DBSCAN
dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
clustering_labels_4 = dbscan.fit_predict(dataset_2)

# Plot
helper.plot_clustered_dataset(dataset_2, 
                              clustering_labels_4, 
                              xlim=(-14, 5), 
                              ylim=(-12, 7), 
                              neighborhood=True, 
                              epsilon=0.5)


eps_values = [0.3, 0.5, 1, 1.3, 1.5]
min_samples_values = [2, 5, 10, 20, 80]

helper.plot_dbscan_grid(dataset_2, eps_values, min_samples_values)

'''
    scenario 1: eps = 1.3 and min_samples = 5
    scenario 2: eps = 1.3 and min_samples = 20
'''