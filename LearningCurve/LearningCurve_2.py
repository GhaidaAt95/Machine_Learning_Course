"""
	Resource :https://www.dataquest.io/blog/learning-curves-machine-learning/
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC


def plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean, name):

	plt.plot(train_sizes, train_scores_mean, label='Training Error')
	plt.plot(train_sizes, validation_scores_mean, label='Validation Error')
	
	plt.ylabel('MSE', fontsize = 14)
	plt.xlabel('Training set size', fontsize = 14)
	title = 'Learning curves for a ' + name + ' model'
	plt.title(title, fontsize = 18, y=1.00)
	plt.legend()
	plt.ylim(0,40)

def learning_curve_ghaida(estimator, data, features, target, train_sizes, cv):
	
	train_sizes, train_scores, validation_scores = learning_curve(
                                                 estimator, data[features], data[target], train_sizes = train_sizes,
                                                 cv = cv, scoring = 'neg_mean_squared_error')	
	
	train_scores_mean = (-1)*train_scores.mean(axis = 1)
	validation_scores_mean = (-1)*validation_scores.mean(axis = 1)
	
	title = str(estimator).split('(')[0] + ' model'
	plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean,title)



# Model that predicts the Hourly Electrical Energy of a Power Plant

electricity = pd.read_excel('Folds5x2_pp.xlsx')

train_sizes=[1, 100, 500, 2000, 5000, 7654]

features = ['AT','V','AP','RH']
target = 'PE'


plt.figure(figsize=(16,5))

plt.subplot(2,2,1)
learning_curve_ghaida(RandomForestRegressor(),electricity, features, target, train_sizes, 5)

plt.subplot(2,2,2)
learning_curve_ghaida(LinearRegression(),electricity, features, target, train_sizes, 5)

plt.subplot(2,2,3)
learning_curve_ghaida(RandomForestRegressor(max_leaf_nodes = 350),electricity, features, target, train_sizes, 5)


"""
	The model achieved from RandomForestRegressor is a high variance one, 
		and low bias --> Over fitting
		
	This one by adding more training instances it is likely to lead to a better 
	model with the same algorithm
	
	Also increase the regulrization (should increase the bias and decrese the variance)
	
	reduce the numbers of features 
"""

"""
	One way ti regularize is by limiting the max_leaf_nodes parameter of the the algorithm
"""
plt.show()