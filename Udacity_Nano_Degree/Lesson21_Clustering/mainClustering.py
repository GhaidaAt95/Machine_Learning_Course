#https://scikit-learn.org/stable/modules/clustering.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
import helper

from sklearn.cluster import KMeans

## Import Movies dataset
movies = pd.read_csv('MiniProjectData\movies.csv')
print('*'*70)
print(movies.head())
print('*'*70)

## Import Ratings dataset
ratings = pd.read_csv('MiniProjectData\\ratings.csv')
print('*'*70)
print(ratings.head())
print('*'*70)

print('*'*70)
print('The dataset conatins: {} ratings of {} movies'.format(len(ratings), len(movies)))
print('*'*70)

##### Romance vs Scifi

## Calculate the avg rating of romance and scifi movies
genre_ratings = helper.get_genre_ratings(ratings, movies, ['Romance','Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating'])
print('*'*70)
print("Number of records ", len(genre_ratings))
print(genre_ratings.head())
'''
    Each user's average rating of all romance movies and all scifi movies
'''
print('*'*70)

## Remove people who like both genres
biased_dataset = helper.bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)
print('*'*70)
print("Number of records ", len(biased_dataset))
print(biased_dataset.head())
print('*'*70)

helper.draw_scatterplot(biased_dataset['avg_scifi_rating'],'Avg Scifi rating',biased_dataset['avg_romance_rating'], 'Avg romance rating')

X = biased_dataset[ ['avg_romance_rating', 'avg_scifi_rating'] ].values

Kmeans_1 = KMeans(n_clusters=2)
Kmeans_1.fit(X)
predictions1 = Kmeans_1.predict(X)
helper.draw_clusters(biased_dataset, predictions1)


Kmeans_2 = KMeans(n_clusters=3)
Kmeans_2.fit(X)
predictions2 = Kmeans_2.predict(X)
helper.draw_clusters(biased_dataset, predictions2)

Kmeans_3 = KMeans(n_clusters=4)
Kmeans_3.fit(X)
predictions3 = Kmeans_3.predict(X)
helper.draw_clusters(biased_dataset, predictions3)

## Elbow Method
'''
    To calculate the error we subtract the point coodinate from the centroid coordinate that it belongs to
    Then we square that difference

    The sum of he values gives us the error for all points when k = #
'''
possible_k_values = range(2, len(X)+1, 5)

errors_per_k = [helper.clustering_errors(k, X) for k in possible_k_values]

# print(list(zip(possible_k_values, errors_per_k)))

fig, ax = plt.subplots(figsize=(16,6))
ax.set_xlabel('K- numer of clusters')
ax.set_ylabel('Silhouette Score (higher is better)')
ax.plot(possible_k_values,errors_per_k)

xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
ax.set_xticks(xticks, minor=False)
ax.set_xticks(xticks, minor=True)
ax.xaxis.grid(True, which='both')
yticks = np.arange(round(min(errors_per_k),2), max(errors_per_k), 0.5)
ax.set_yticks(yticks, minor=False)
ax.set_yticks(yticks, minor=True)
ax.yaxis.grid(True, which='both')

kmeans_4 = KMeans(n_clusters=7)
kmeans_4.fit(X)
predictions_4 = kmeans_4.predict(X)
helper.draw_clusters(biased_dataset, predictions_4)


### Add Action genre

biased_dataset_3_genres = helper.get_genre_ratings(ratings, movies,
                                                   ['Romance','Sci-Fi', 'Action'],
                                                   ['avg_romance_rating', 'avg_scifi_rating', 'avg_action_rating'])

biased_dataset_3_genres = helper.bias_genre_rating_dataset(biased_dataset_3_genres, 3.2,2.5).dropna()

print('*'*70)
print("Number of records ", len(biased_dataset_3_genres))
print(biased_dataset_3_genres.head())
print('*'*70)

X_with_action = biased_dataset_3_genres[['avg_romance_rating', 'avg_scifi_rating', 'avg_action_rating']].values

Kmeans_5 = KMeans(n_clusters=7)
Kmeans_5.fit(X_with_action)
predictions5 = Kmeans_5.predict(X_with_action)
helper.draw_clusters_3d(biased_dataset_3_genres, predictions5)

##### Movie-Level Clustering
## How users rated individual movies
# Form data in userId vs user rating for each movie

ratings_title = pd.merge(ratings, movies[['movieId','title']], on='movieId')
user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns='title', values='rating')

print('Dataset Dimensions {} \n\nSubset example: {}'.format(user_movie_ratings.shape, user_movie_ratings.iloc[:6,:10]))

## Problem Nan: 
'''
    - Most users havent watched and rated most movies
    - This Dataset is called SPARSE because only a small # of cells have values

    - Solution: 
        - Sort by :
            - the most rated movies
            - the users who rated the most
'''
# Most-rated movies vs users with the most ratings 
n_movies = 30
n_users = 18
most_rated_movies_users_selection = helper.sort_by_rating_density(user_movie_ratings,n_movies, n_users)
print('Dataset Dimensions: ', most_rated_movies_users_selection.shape)
print(most_rated_movies_users_selection.head())

helper.draw_movies_heatmap(most_rated_movies_users_selection)

# For performance reasons, we will only use ratings for 1000 movies

user_movie_ratings = pd.pivot_table(ratings_title, index='userId',columns='title',values='rating')
most_rated_movies_1k = helper.get_most_rated_movies(user_movie_ratings, 1000)

#https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.sparse.csr_matrix.html

sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())

######### Clustering
# TEST K = 20
predictions = KMeans(n_clusters=20, algorithm='full').fit_predict(sparse_ratings)

## To visualize plot each cluster as a heat map
max_users = 70
max_movies = 50

clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
helper.draw_movie_clusters(clustered,max_users,max_movies)

plt.show()