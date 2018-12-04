import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from my_input_fn import my_input_fn

#Sets the threshold for what messages will be logged.
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format ='{:.1f}'.format

cal_housing_DF = pd.read_csv('california_housing_train.csv',sep=',')

cal_housing_DF = cal_housing_DF.reindex(np.random.permutation(cal_housing_DF.index))

cal_housing_DF['median_house_value'] /= 1000.0
print(cal_housing_DF)
print('------------------------------------------\n')
print(cal_housing_DF.describe())

def train_model(learning_rate, steps, batch_size, input_feature='total_rooms'):
	""" Train a Linear Regression Model of 1 feature;
		* steps : INT, total number of training steps. 
		* A training step consists of a forward and backward pass using a single batch.
	"""
	#for each period, the training loss will be computed
	periods = 10
	steps_per_period = steps /periods
	
	my_feature = input_feature
	my_feature_data = cal_housing_DF[[my_feature]]
	my_label = 'median_house_value'
	targets = cal_housing_DF[my_label]
	
	feature_columns = [tf.feature_column.numeric_column(my_feature)]
	
	training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)
	prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
	
	my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) 
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
	linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,optimizer=my_optimizer)

	plt.figure(figsize=(15, 6))
	plt.subplot(1, 2, 1)
	plt.title("Learned Line by Period")
	plt.ylabel(my_label)
	plt.xlabel(my_feature)
	sample = cal_housing_DF.sample(n=300)
	plt.scatter(sample[my_feature], sample[my_label])
	colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
	
	print("Training model...")
	print("RMSE (on training data):")
	root_mean_squared_errors = []

	for period in range(0, periods):
		linear_regressor.train(input_fn=training_input_fn, steps= steps_per_period)
	
		predictions = linear_regressor.predict(input_fn=prediction_input_fn)
		predictions = np.array([item['predictions'][0] for item in predictions])
	
		root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))
		print("  period %02d : %0.2f" % (period, root_mean_squared_error))
		root_mean_squared_errors.append(root_mean_squared_error)
		y_extents = np.array([0, sample[my_label].max()])
		
		weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
		bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')		
		
		x_extents = (y_extents - bias) / weight
		x_extents = np.maximum(np.minimum(x_extents,sample[my_feature].max()),sample[my_feature].min())
		
		y_extents = weight * x_extents + bias
		plt.plot(x_extents, y_extents, color=colors[period])
	print("Model training finished.")
	
	plt.subplot(1, 2, 2)
	plt.ylabel('RMSE')
	plt.xlabel('Periods')
	plt.title("Root Mean Squared Error vs. Periods")
	plt.tight_layout()
	plt.plot(root_mean_squared_errors)
	
	calibration_data = pd.DataFrame()
	calibration_data["predictions"] = pd.Series(predictions)
	calibration_data["targets"] = pd.Series(targets)
	print(calibration_data.describe())

	print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
	
train_model(learning_rate=0.00001,steps=1000,batch_size=50)
plt.show()