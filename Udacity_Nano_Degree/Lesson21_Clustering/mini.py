import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
import helper

from sklearn.cluster import KMeans

### Movies DataSet 
movies = pd.read_csv('MiniProjectData\movies.csv')
print('*'*100)
print("\t\t\t\tMovies DataSet: of shape {}".format(movies.shape))
print(movies.head(n=10))
print('*'*100)

### Ratings DataSet
ratings = pd.read_csv('MiniProjectData\\ratings.csv')
print('*'*100)
print("\t\t\t\tRatings DataSet: of shape {}".format(ratings.shape))
print(ratings.head(n=20))
print('*'*100)