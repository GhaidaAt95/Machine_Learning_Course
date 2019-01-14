"""
	Resource :https://www.dataquest.io/blog/learning-curves-machine-learning/
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

# Model that predicts the Hourly Electrical Energy of a Power Plant

electricity = pd.read_excel('Folds5x2_pp.xlsx')

print(electricity.info())
print('Head : \n',electricity.head(3))

###### Train/Test Split

"""
	K-Fold cross validation is used for each training size

	Scikit-Learn has a built in learning_curve() function that will take care of the validation set
		- This function returns a tuple containing three elements:
				1) Training set sizes
				2) Error scores for the validation sets 
				3) Error scores for the trainings sets
		
		- It takes the following parameters:
			1) estimator   : Learning Alg. we use to estimate the true model
			2) 		x	   : Data containg features
			3)		y	   : Data containg the Target
			4) Train_Sizes : The training sizes to be used
			5) 		CV	   : Determines the cross-validation splitting startegy
			6) scoring	   : Error Metric to use {MSE not possible, nearest proxy negative MSE and we flip signs later on}
"""
def plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean, name=" "):
	
	plt.figure()
	plt.style.use('seaborn')
	
	plt.plot(train_sizes, train_scores_mean, label='Training Error')
	plt.plot(train_sizes, validation_scores_mean, label='Validation Error')
	
	plt.ylabel('MSE', fontsize = 14)
	plt.xlabel('Training set size', fontsize = 14)
	title = 'Learning curves for a linear regression model ' + name
	plt.title(title, fontsize = 18, y=1.00)
	plt.legend()
	plt.ylim(0,40)
	
def Learning_Curve_Train(randomize=False):
	"""
		The data set has 9568 instances
		80% Training = 7654 instances
		20% Validation = 20%
	"""
	train_sizes=[1, 100, 500, 2000, 5000, 7654]

	features = ['AT','V','AP','RH']
	target = 'PE'


	train_sizes, train_scores, validation_scores = learning_curve( estimator = LinearRegression(),\
																   X = electricity[features],\
																   y = electricity[target],\
																   train_sizes = train_sizes,\
																   cv = 5,\
																   shuffle=randomize,\
																   scoring = 'neg_mean_squared_error')

	print('\n','-'*70)
	print('Training scores:\n\n', train_scores)
	print('\n','-'*70)
	print('\nValidation scores:\n\n',validation_scores)

	"""
		For each score we have 6 rows = 6 train_sizes and 5 columns = 5-fold cross-validation
		
		Since we want to plot the learning curve for each training size we take the mean for
			each row across the 5-folds
	"""

	train_scores_mean = (-1)*train_scores.mean(axis = 1)
	validation_scores_mean = (-1)*validation_scores.mean(axis = 1)

	print('Mean training scores\n\n',pd.Series(train_scores_mean,index=train_sizes))
	print('\n','-'*20)
	print('\nMean Validation scores\n\n',pd.Series(validation_scores_mean,index=train_sizes))
	if randomize :
		plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean,name="Shuffle On")
	else:
		plot_learning_curve(train_sizes, train_scores_mean, validation_scores_mean)


Learning_Curve_Train()

Notice =\
"""
	---------------------------------------------------------------
	| Notice that the error scores for the training sets are same |
 	| across each row or explicitly from the 2nd split and onward |
	| To stop this we need to shuffle the instances in data set   |
	| to ensure rrandomizing indices for the training data in eah |
	| split 													  |
	---------------------------------------------------------------
"""
print(Notice)

Learning_Curve_Train(randomize=True)

"""
	When the line stay still, that meas adding more points wont 
	lead to any significantly better models.
	Then things like switching algorithm can be tried,
		adding more features( lowering the bias by adding compleity to the model),
		DECREASING Regulization of the learning algorithm:
			Preventing it from fitting the training data too well --> decreasing bias and incresing variance
	
	Our Model is High Bias Since the Tarining and validation error is High, and 
		Low variance since the validation error - training error is verry low and the model doesnt
		fit the training data very well
"""

plt.show()
""" Features :
	AT : Ambiental Temperature
	V  : Exhaust Vacuum
	AP : Ambiental Pressure
	RH : Relative Humidity
	
	Target/Label:
	PE : Electrical Energy Output
"""