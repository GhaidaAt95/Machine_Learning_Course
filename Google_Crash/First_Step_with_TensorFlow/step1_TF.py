import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from my_input_fn import my_input_fn

#Sets the threshold for what messages will be logged.
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format ='{:.1f}'.format

cal_housing_DF = pd.read_csv('california_housing_train.csv',sep=',')
#print(cal_housing_DF.describe())

"""
	1) We need to randomize the data to ensure not to get any pathological 
	ordering effect -> that might effect the performance of Stochastic Gradient
	
	2) we will scale median_house_value to be in units of thousands --> learned easily
"""

cal_housing_DF = cal_housing_DF.reindex(np.random.permutation(cal_housing_DF.index))

cal_housing_DF['median_house_value'] /= 1000.0
print(cal_housing_DF)
print('------------------------------------------\n')
print(cal_housing_DF.describe())

#################### Step 1: Define Features and Configure Feature Columns ####################
"""In TF we indicate the features data type using feature column
	feature column store only a description of the feature data
"""
#dataframe
my_feature = cal_housing_DF[['total_rooms']]
#print(type(my_feature))

#list
feature_columns = [tf.feature_column.numeric_column('total_rooms')]
#print(type(feature_columns))
# print(feature_columns)

#################### Step 2: Define Traget/label
targets = cal_housing_DF['median_house_value']

#################### Step 3; Configure the LinearRegressor
"""
	Linear Regression Model, the model will be trained using GradientDescentOptimizer in which 
	it will imlement the Mini-Batch Stochastic Gradient Descent
	
	The learning rate should control the gradient step
	 Ans. to; How large of a step should we make in the direction of Gradient Descent
"""
# learning rate of 1*10e-7
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001) 

#limitting the magnitude of the gradient
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model 
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,optimizer=my_optimizer)

#################### Step 4: Traing the Model
_ = linear_regressor.train(input_fn = lambda:my_input_fn(my_feature,targets),steps=100)
#################### Step 5: Evaluate the Model
# Uses the same input function with 1epoch (no repeat needed) and no shuffle needed
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Make predictions
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# convert predictions as numpy arrays to pass it to error metic from sklearn
predictions = np.array([item['predictions'][0] for item in predictions])


# calculate error
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

# We get MSE = 56367.025 and RMSE = 237.417

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print('------------------------------------------\n')
print(calibration_data.describe())

###################### Draw the Line we got
sample = cal_housing_DF.sample(n=300)

weight_model = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

pt_sample_0_x = sample['total_rooms'].min()
pt_sample_1_x = sample['total_rooms'].max()

pt_sample_0_y = weight_model * pt_sample_0_x + bias
pt_sample_1_y = weight_model * pt_sample_1_x + bias

plt.plot([pt_sample_0_x,pt_sample_1_x],[ pt_sample_0_y,pt_sample_1_y], c='r', linewidth=2)

plt.ylabel('Median House Vlue')
plt.xlabel('Total Rooms In A block')

plt.scatter(sample['total_rooms'],sample['median_house_value'])

plt.show()


################### Step6: Tweak the Model HyperParameters 
"""
Great Resource:
	1) A Gentle Introduction to Exploding Gradients in Neural Networks
		https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
	2) Epoch vs Batch Size vs Iterations
		https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9

"""